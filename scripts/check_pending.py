import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import text

from earnings_edge.db import get_session

if __name__ == "__main__":
    with get_session() as session:
        total = session.execute(text("SELECT COUNT(*) FROM snapshots WHERE scan_date >= '2026-05-01'")).scalar()
        has_opts = session.execute(text("SELECT COUNT(*) FROM snapshots WHERE scan_date >= '2026-05-01' AND has_options = 1")).scalar()
        no_opts = total - has_opts
        has_error = session.execute(text("SELECT COUNT(*) FROM snapshots WHERE scan_date >= '2026-05-01' AND collection_error IS NOT NULL")).scalar()
        pending = session.execute(text("SELECT COUNT(*) FROM snapshots WHERE scan_date >= '2026-05-01' AND has_options = 1 AND collection_error IS NULL AND atm_iv_near IS NULL")).scalar()
        tickers = session.execute(text("SELECT COUNT(DISTINCT ticker) FROM snapshots WHERE scan_date >= '2026-05-01' AND has_options = 1 AND collection_error IS NULL AND atm_iv_near IS NULL")).scalar()

        print(f"May+ total: {total}")
        print(f"  has_options=1: {has_opts}")
        print(f"  has_options=0: {no_opts}")
        print(f"  has collection_error: {has_error}")
        print(f"  pending backfill: {pending}")
        print(f"  unique tickers: {tickers}")
        print(f"  est time @13s/ticker: {tickers * 13 / 60:.0f} min")
