"""Integration: Polygon IV backfill -> COALESCE-only DB write path.

The feature collection itself is out of scope (covered elsewhere); this tests
that apply_features() fills NULL option-feature columns without ever
overwriting existing values, and that fetch_rows() targets exactly the
has_options=1 NULL-IV gap with training-critical rows ordered first.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))


def _seed(ticker, ed, sd, labeled=True, has_options=1, **cols):
    from sqlalchemy import text
    from earnings_edge.db import engine as db_engine

    base = dict(
        price=50.0, has_options=has_options,
        actual_move_pct=7.5 if labeled else None,
        outcome_fetched_at="2026-08-01T00:00:00" if labeled else None,
        data_source="integration_fixture",
    )
    base.update(cols)
    names = ", ".join(["ticker", "earnings_date", "scan_date", "timing", *base])
    params = {"ticker": ticker, "earnings_date": ed, "scan_date": sd, "timing": "Post Market"}
    params.update(base)
    placeholders = ", ".join(f":{n}" for n in ["ticker", "earnings_date", "scan_date", "timing", *base])
    with db_engine.session_scope() as s:
        result = s.execute(
            text(f"INSERT INTO snapshots ({names}) VALUES ({placeholders})"),
            params,
        )
        return result.lastrowid


def test_apply_features_fills_nulls_and_preserves_existing(tmp_db_path):
    from sqlalchemy import text
    from earnings_edge.db import engine as db_engine
    import backfill_snapshot_iv_polygon as ivbf

    sid = _seed("IVCO", "2026-07-30", "2026-07-29",
                atm_iv_near=None, rv30=None, sigma_short_leg=0.31)

    filled = ivbf.apply_features(sid, {
        "atm_iv_near": 0.45, "rv30": 0.28, "sigma_short_leg": 0.99,
        "atm_call_iv": None,  # absent values are skipped
    })
    with db_engine.get_session() as s:
        row = s.execute(
            text(
                "SELECT atm_iv_near, rv30, sigma_short_leg, atm_call_iv "
                "FROM snapshots WHERE id = :id"
            ),
            {"id": sid},
        ).first()
    assert row[0] == 0.45      # NULL -> filled
    assert row[1] == 0.28      # NULL -> filled
    assert row[2] == 0.31      # existing value preserved (COALESCE)
    assert row[3] is None      # None feature never written
    assert "atm_iv_near" in filled and "atm_call_iv" not in filled


def test_fetch_rows_targets_gap_and_orders_training_first(tmp_db_path):
    import backfill_snapshot_iv_polygon as ivbf

    # gap rows: has_options=1, atm_iv_near NULL
    old_labeled = _seed("OLDLB", "2026-05-06", "2026-05-05", labeled=True)
    unlab = _seed("UNLAB", "2026-08-04", "2026-08-03", labeled=False)
    new_labeled = _seed("NEWLB", "2026-07-30", "2026-07-29", labeled=True)
    # non-gap rows that must NOT be selected
    _seed("HASIV", "2026-07-30", "2026-07-29", labeled=True, atm_iv_near=0.4)
    _seed("NOOPT", "2026-07-30", "2026-07-29", labeled=True, has_options=0)

    rows = ivbf.fetch_rows()
    ids = [r["id"] for r in rows]
    assert set(ids) == {old_labeled, new_labeled, unlab}  # exactly the gap rows
    # labeled first, most recent scan_date first within the labeled group
    assert ids[0] == new_labeled
    assert ids[1] == old_labeled
    assert ids[2] == unlab  # unlabeled row is last

    labeled_only = ivbf.fetch_rows(with_outcomes_only=True)
    assert {r["id"] for r in labeled_only} == {old_labeled, new_labeled}
