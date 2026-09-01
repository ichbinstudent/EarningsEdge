#!/usr/bin/env python3
"""Backfill NULL IV/option feature fields on snapshots from historical Polygon data.

Targets rows where atm_iv_near IS NULL (the ~50% feature-coverage gap).
Uses polygon_backfill.collect_polygon_features, which reconstructs as-of-scan-date
values from historical contracts + closes — no current-data leakage.

Resumable: rows are committed one at a time; re-running skips filled rows.

Usage:
  PYTHONUNBUFFERED=1 .venv/bin/python3.12 scripts/backfill_snapshot_iv_polygon.py \
      [--with-outcomes-only] [--limit N] [--sleep 12.5]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import os

from sqlalchemy.exc import OperationalError

from polygon_backfill import PolygonClient, EarningsEvent, collect_polygon_features
from earnings_edge.db import snapshots_coalesce_features, snapshots_iv_gap_rows

FIELDS = [
    "price", "avg_volume_30d", "rv30", "hist_vol_3m", "has_options",
    "nearest_expiry", "days_to_expiry",
    "atm_call_iv", "atm_put_iv", "atm_iv_near", "atm_call_delta", "atm_put_delta",
    "straddle_price", "expected_move_dollars", "expected_move_pct",
    "iv30_rv30", "term_slope", "term_structure_valid",
    "sigma_baseline_1y", "sigma_short_leg", "sigma_short_leg_fair",
    "actual_to_fair_ratio",
]


def apply_features(snapshot_id: int, feats: dict,
                   lock_retries: int = 5) -> list[str]:
    """COALESCE-only write of collected features onto one snapshot row.

    Existing non-NULL values are never overwritten (repair, not re-scrape).
    Writers contend on the WAL DB (bot, other backfills), so a locked
    database is retried with backoff instead of killing the run.
    Returns the list of columns that were candidates for filling.
    """
    for attempt in range(lock_retries):
        try:
            return snapshots_coalesce_features(snapshot_id, feats)
        except OperationalError as exc:
            if "locked" not in str(exc).lower() or attempt == lock_retries - 1:
                raise
            time.sleep(5 * (attempt + 1))
    return []


def fetch_rows(with_outcomes_only: bool = False,
               scan_date_since: str | None = None,
               limit: int | None = None) -> list:
    """Target rows for IV repair, training-critical first.

    Ordering: labeled rows (feed the model) before unlabeled, then most
    recent scan_date first — a bounded run repairs the highest-value rows
    first and stays resumable.
    """
    return snapshots_iv_gap_rows(
        with_outcomes_only=with_outcomes_only,
        scan_date_since=scan_date_since,
        limit=limit,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-outcomes-only", action="store_true",
                    help="Only rows with a realized outcome (training-critical)")
    ap.add_argument("--scan-date-since", default=None,
                    help="Only rows scanned on/after YYYY-MM-DD (prioritize recent)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sleep", type=float, default=12.5)
    args = ap.parse_args()

    api_key = os.environ.get("POLYGON_API_KEY", "")
    if not api_key:
        raise SystemExit("POLYGON_API_KEY not set")

    rows = fetch_rows(with_outcomes_only=args.with_outcomes_only,
                      scan_date_since=args.scan_date_since, limit=args.limit)

    # Structural failures (Polygon has no historical options data for the
    # name at all) are deterministic — reprocessing them on every resume
    # just burns rate limit. Recorded once, skipped thereafter. Transient
    # EXCEPTIONs are NOT recorded and always retry.
    skip_log = Path(__file__).resolve().parent.parent / "data" / ".iv_backfill_structural.jsonl"
    skipped_ids: set[int] = set()
    if skip_log.exists():
        for line in skip_log.read_text().splitlines():
            try:
                skipped_ids.add(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                continue
    before = len(rows)
    rows = [r for r in rows if r["id"] not in skipped_ids]
    if before != len(rows):
        print(f"(skipped {before - len(rows)} rows with known structural gaps)", flush=True)
    print(f"{len(rows)} snapshots to backfill", flush=True)

    pg = PolygonClient(api_key, sleep=args.sleep)
    updated = failed = 0
    t0 = time.time()
    for i, r in enumerate(rows, 1):
        sid, ticker, ed_s, sd_s = r["id"], r["ticker"], r["earnings_date"], r["scan_date"]
        ed = date.fromisoformat(str(ed_s)[:10])
        sd = date.fromisoformat(str(sd_s)[:10])
        offset = max((ed - sd).days, 1)  # align as_of with the original scan_date
        try:
            feats = collect_polygon_features(
                pg, EarningsEvent(ticker=ticker, earnings_date=ed, timing="unknown"),
                scan_offset_days=offset,
            )
        except Exception as exc:
            print(f"[{i}/{len(rows)}] {ticker} {sd}: EXCEPTION {exc}", flush=True)
            failed += 1
            continue

        apply_features(sid, feats)
        err = feats.get("collection_error")
        if err:
            failed += 1
            if err.startswith(("no historical option contracts", "no ATM near option prices")):
                with open(skip_log, "a") as fh:
                    fh.write(json.dumps({"id": sid, "ticker": ticker,
                                         "scan_date": str(sd_s)[:10], "err": err}) + "\n")
            print(f"[{i}/{len(rows)}] {ticker} {sd}: {err}", flush=True)
        else:
            updated += 1
            rate = (time.time() - t0) / i
            eta_h = rate * (len(rows) - i) / 3600
            print(f"[{i}/{len(rows)}] {ticker} {sd}: OK "
                  f"iv={feats.get('atm_iv_near')} em={feats.get('expected_move_pct'):.1f}% "
                  f"(ETA {eta_h:.1f}h)", flush=True)

    print(f"\nDone: {updated} updated, {failed} failed of {len(rows)}", flush=True)


if __name__ == "__main__":
    main()
