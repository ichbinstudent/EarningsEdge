"""Data-coverage measurement and repair-target selection.

The FF ladder hist gate (``hist_rms_move``) needs >= MIN_HIST_EVENTS usable
realized outcomes per ticker, and the training pipeline needs dense
has_options=1 rows. This module measures both gaps against the snapshots
table and selects repair candidates — liquid names first, cheapest top-ups
first — so the backfill scripts (warm_hist_coverage.py,
backfill_snapshot_iv_polygon.py) can run as a targeted repair drive instead
of a blind re-scrape.

Pure read queries + list selection: no network, no writes.
"""
from __future__ import annotations

import sqlite3
from typing import Optional

from earnings_edge.db.repositories import (
    snapshots_distinct_tickers,
    snapshots_hist_repair_stats,
    snapshots_usable_counts_by_ticker,
)

# Mirrors fwd_factor_ladder.MIN_HIST_EVENTS; kept as a parameter everywhere
# so tests can shrink it.
DEFAULT_MIN_EVENTS = 3

# A "usable" outcome for the hist gate — same predicate as
# fwd_factor_ladder.hist_rms_move / ensure_hist_moves.
_USABLE_OUTCOME = (
    "actual_move_pct IS NOT NULL AND outcome_fetched_at IS NOT NULL "
    "AND outcome_fetched_at != 'unavailable'"
)


def hist_move_coverage(
    *args,
    universe: Optional[list[str]] = None,
    min_events: int = DEFAULT_MIN_EVENTS,
) -> dict:
    """Fraction of the earnings universe passing the FF hist gate.

    Universe defaults to every distinct ticker in snapshots. Returns counts
    plus a small distribution so repair drives can report before/after.
    A leading DB-API connection is still accepted (ignored when the engine
    is already configured to the same file).
    """
    from earnings_edge.db.repositories import _split_conn

    conn, rest = _split_conn(args)
    if universe is None and rest:
        universe = rest[0]
    if universe is None:
        universe = snapshots_distinct_tickers(*((conn,) if conn is not None else ()))
    if not universe:
        return {"universe": 0, "covered": 0, "pct": 0.0, "distribution": {}}

    counts = snapshots_usable_counts_by_ticker(
        *((conn,) if conn is not None else ()), universe=list(universe)
    )
    distribution: dict[int, int] = {}
    covered = 0
    for t in universe:
        n = counts.get(t, 0)
        distribution[n] = distribution.get(n, 0) + 1
        if n >= min_events:
            covered += 1
    return {
        "universe": len(universe),
        "covered": covered,
        "pct": 100.0 * covered / len(universe),
        "distribution": dict(sorted(distribution.items())),
    }


def repair_candidates(
    *args,
    min_events: int = DEFAULT_MIN_EVENTS,
    liquid_only: bool = False,
) -> list[str]:
    """Tickers below the hist gate, ordered for a repair drive.

    Ordering: liquid (any has_options=1 snapshot) first, then most existing
    outcomes first (cheapest top-up to the gate), then ticker. With
    ``liquid_only`` the OTC/zero-option tail is skipped entirely — those
    rows never feed option-feature training and are never FF candidates.
    """
    from earnings_edge.db.repositories import _split_conn

    conn, _rest = _split_conn(args)
    rows = snapshots_hist_repair_stats(*((conn,) if conn is not None else ()))
    cands = [
        (r["ticker"], int(r["usable"] or 0), int(r["liquid"] or 0))
        for r in rows if int(r["usable"] or 0) < min_events
    ]
    if liquid_only:
        cands = [c for c in cands if c[2] == 1]
    cands.sort(key=lambda c: (-c[2], -c[1], c[0]))
    return [t for t, _, _ in cands]


def iv_null_stats(conn: sqlite3.Connection) -> dict:
    """NULL rates of option-feature columns among has_options=1 rows.

    Includes the training-critical subset (labeled rows) since those are the
    rows the model actually learns from.
    """
    cols = ["atm_iv_near", "atm_call_iv", "atm_put_iv", "rv30", "hist_vol_3m",
            "iv30_rv30", "sigma_short_leg"]
    total = conn.execute(
        "SELECT COUNT(*) FROM snapshots WHERE has_options=1"
    ).fetchone()[0]
    labeled = conn.execute(
        "SELECT COUNT(*) FROM snapshots WHERE has_options=1 "
        "AND actual_move_pct IS NOT NULL"
    ).fetchone()[0]
    out: dict = {"has_options_rows": total, "labeled_rows": labeled, "columns": {}}
    for col in cols:
        n_null = conn.execute(
            f"SELECT COUNT(*) FROM snapshots WHERE has_options=1 AND {col} IS NULL"
        ).fetchone()[0]
        n_null_labeled = conn.execute(
            f"SELECT COUNT(*) FROM snapshots WHERE has_options=1 "
            f"AND actual_move_pct IS NOT NULL AND {col} IS NULL"
        ).fetchone()[0]
        out["columns"][col] = {
            "null": n_null,
            "pct": 100.0 * n_null / total if total else 0.0,
            "null_labeled": n_null_labeled,
        }
    return out


def dense_training_rows(conn: sqlite3.Connection) -> int:
    """Rows the trainer can actually use: options + outcome + core vol fields."""
    return conn.execute(
        """SELECT COUNT(*) FROM snapshots WHERE has_options=1
           AND actual_move_pct IS NOT NULL
           AND atm_iv_near IS NOT NULL AND rv30 IS NOT NULL"""
    ).fetchone()[0]
