import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import text

from earnings_edge.db import get_session

if __name__ == "__main__":
    with get_session() as session:
        # Check: how many unique tickers (not groups) need backfill?
        tickers = session.execute(text("""
            SELECT DISTINCT ticker FROM snapshots
            WHERE scan_date >= '2026-05-01' AND has_options = 1 AND collection_error IS NULL AND atm_iv_near IS NULL
        """)).fetchall()
        print(f"Unique tickers: {len(tickers)}")

        # Check: how many have scan_date = earnings_date - 1? (We can use collect_polygon_features directly)
        exact_minus_1 = session.execute(text("""
            SELECT COUNT(DISTINCT ticker || earnings_date) FROM snapshots
            WHERE scan_date >= '2026-05-01' AND has_options = 1 AND collection_error IS NULL AND atm_iv_near IS NULL
            AND date(earnings_date, '-1 day') = scan_date
        """)).scalar()
        total_groups = session.execute(text("""
            SELECT COUNT(DISTINCT ticker || earnings_date || scan_date) FROM snapshots
            WHERE scan_date >= '2026-05-01' AND has_options = 1 AND collection_error IS NULL AND atm_iv_near IS NULL
        """)).scalar()
        print(f"Groups where scan=ed-1: {exact_minus_1}/{total_groups}")

        # Check: how many unique tickers with scan != ed-1?
        special = session.execute(text("""
            SELECT DISTINCT ticker, earnings_date, scan_date FROM snapshots
            WHERE scan_date >= '2026-05-01' AND has_options = 1 AND collection_error IS NULL AND atm_iv_near IS NULL
            AND date(earnings_date, '-1 day') != scan_date
            LIMIT 10
        """)).fetchall()
        print(f"Tickers where scan != ed-1 (first 10):")
        for r in special:
            print(f"  {r[0]} ed={r[1]} scan={r[2]}")

        # How many API calls if we optimize: 1 stock_bars + 1 contracts + 2 option_close per ticker
        # But each ticker might have multiple earnings dates
        unique_ticker_dates = session.execute(text("""
            SELECT COUNT(DISTINCT ticker || '|' || earnings_date || '|' || scan_date) FROM snapshots
            WHERE scan_date >= '2026-05-01' AND has_options = 1 AND collection_error IS NULL AND atm_iv_near IS NULL
        """)).scalar()
        print(f"\nTotal unique (ticker, ed, sd) tuples: {unique_ticker_dates}")
        print(f"Est API calls @ 5 per tuple: {unique_ticker_dates * 5}")
        print(f"Est time @ 13s/call: {unique_ticker_dates * 5 * 13 / 3600:.1f} hours")
