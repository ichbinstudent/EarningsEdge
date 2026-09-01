#!/usr/bin/env python3
"""CLI entrypoint for generating top plays (picks) from earnings_ml.db."""
import argparse
import sys
from pathlib import Path
from datetime import datetime

from earnings_edge.db import configure, snapshots_max_scan_date
from earnings_edge.picks import generate_picks

DEFAULT_DB = Path(__file__).parent / "data" / "earnings_ml.db"

def main() -> int:
    parser = argparse.ArgumentParser(description="Generate options picks from DB")
    parser.add_argument("--db", default=str(DEFAULT_DB), help="Path to earnings_ml.db")
    parser.add_argument("--date", help="As-of date (YYYY-MM-DD), default latest snapshot")
    parser.add_argument("--limit", type=int, default=20, help="Max rows per strategy (default 20)")
    parser.add_argument("--output", help="Optional CSV output prefix")
    parser.add_argument("--persist", action="store_true",
                        help="Persist picks into the picks table (opens the DB read-write)")
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"database not found: {db_path}")
        return 1

    configure(db_path)
    if args.date:
        as_of = datetime.strptime(args.date, "%Y-%m-%d").date()
    else:
        latest_str = snapshots_max_scan_date()
        if not latest_str:
            print("No data in snapshots table.")
            return 1
        as_of = datetime.strptime(latest_str[:10], "%Y-%m-%d").date()
        print(f"Using latest snapshot date: {as_of}")

    picks = generate_picks(as_of)

    if args.persist:
        from earnings_edge.picks import persist_picks
        n = persist_picks(picks, as_of)
        print(f"persisted {n} picks for {as_of}")

    print("\n" + "=" * 50)
    print(f"OQUANTS PICKS AS OF {as_of}")
    print("=" * 50)

    for name, df in picks.items():
        print(f"\n--- Strategy: {name.upper()} ---")
        if df.empty:
            print("No picks found.")
        else:
            display_df = df.head(args.limit) if args.limit > 0 else df
            print(display_df.to_string(index=False))
            if len(df) > args.limit > 0:
                print(f"... and {len(df) - args.limit} more rows.")

        if args.output and not df.empty:
            out_path = f"{args.output}_{name}.csv"
            df.to_csv(out_path, index=False)
            print(f"-> Saved full {name} list to {out_path}")

    return 0

if __name__ == "__main__":
    sys.exit(main())
