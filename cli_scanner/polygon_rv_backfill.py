#!/usr/bin/env python3
"""
Backfill rv30 and hist_vol_3m from historical stock bars.
Only needs 1 API call per unique ticker (daily bars for 120 days).

--source polygon (default): Polygon.io, 13s rate limit.
--source lse:               London Strategic Edge vault, ~0.35s pacing.
"""
import argparse
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from polygon_backfill import PolygonClient, realized_vol_30d, hist_vol
from earnings_edge.db import (
    snapshots_apply_rv,
    snapshots_count_null_rv30,
    snapshots_rv_pending_pairs,
)

load_dotenv(Path(__file__).resolve().parent / ".env")


def main():
    parser = argparse.ArgumentParser(description="Historical RV backfill")
    parser.add_argument("--source", choices=["polygon", "lse"], default="polygon")
    args = parser.parse_args()

    if args.source == "lse":
        from earnings_edge.collectors.lse import LSECollector
        if not os.environ.get("LSE_API_KEY"):
            raise RuntimeError("LSE_API_KEY not set")
        pg = LSECollector()
        per_call = pg.sleep
    else:
        key = os.environ.get("POLYGON_API_KEY")
        if not key:
            raise RuntimeError("POLYGON_API_KEY not set")
        pg = PolygonClient(key, sleep=13)
        per_call = 13

    rows = snapshots_rv_pending_pairs()

    print(f"Unique (ticker, scan_date) pairs needing rv30: {len(rows)}")
    ok = 0
    failed = 0
    start = time.time()

    for i, row in enumerate(rows, 1):
        ticker = row["ticker"]
        sd = datetime.strptime(row["scan_date"], "%Y-%m-%d").date()
        remaining = (len(rows) - i) * per_call / 60

        print(f"[{i}/{len(rows)}] {ticker} scan={row['scan_date']} ~{remaining:.0f}min left")

        try:
            bars = pg.daily_bars(ticker, sd - timedelta(days=120), sd)
            if not bars:
                print(f"  SKIP: no stock bars")
                failed += 1
                continue

            rv = realized_vol_30d(bars)
            hv = hist_vol(bars, 63)

            if rv is None and hv is None:
                print(f"  SKIP: insufficient data")
                failed += 1
                continue

            snapshots_apply_rv(ticker, row["scan_date"], rv, hv)
            ok += 1
            print(f"  OK: rv30={rv:.4f} hv3m={hv:.4f}" if rv and hv else f"  OK: rv30={rv} hv3m={hv}")

        except Exception as exc:
            print(f"  ERROR: {exc}")
            failed += 1

    elapsed = (time.time() - start) / 60
    print(f"\nDone in {elapsed:.1f}min: {ok} updated, {failed} failed")
    
    # Summary
    remaining = snapshots_count_null_rv30()
    print(f"Remaining with null rv30: {remaining}")


if __name__ == "__main__":
    main()
