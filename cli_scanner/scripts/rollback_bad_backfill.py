#!/usr/bin/env python3
"""Roll back the yfinance-based IV backfill and prepare for Polygon-based re-backfill."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from earnings_edge.db import (
    calendar_call_trades_count,
    snapshots_clear_iv_since,
    snapshots_iv_presence_counts,
)


def main() -> None:
    n = snapshots_clear_iv_since("2026-05-01")
    print(f"Nulled IV fields on {n} May+ rows")

    counts = snapshots_iv_presence_counts()
    print(f"Pre-May rows with IV: {counts['iv_before']}")
    print(f"Post-May rows with IV: {counts['iv_after']} (should be 0)")
    print(f"Total May+ snapshots: {counts['total_may']}")
    print(f"May+ snapshots needing Polygon backfill: {counts['need_backfill']}")

    print(f"Calendar trades (untouched): {calendar_call_trades_count()}")
    print("Rollback complete.")


if __name__ == "__main__":
    main()
