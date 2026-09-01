import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import text

from earnings_edge.db import get_session

if __name__ == "__main__":
    with get_session() as session:
        # How many unique (ticker, scan_date) pairs?
        pairs = session.execute(text("""
            SELECT COUNT(DISTINCT ticker || scan_date) FROM snapshots
            WHERE scan_date >= '2026-05-01' AND has_options = 1 AND collection_error IS NULL AND atm_iv_near IS NULL
        """)).scalar()
        print(f"Unique (ticker, scan_date) pairs: {pairs}")
        print(f"Est time @13s: {pairs * 13 / 60:.0f} min ({pairs * 13 / 3600:.1f} hr)")

        # But really, IV data only depends on ticker + scan_date
        # So we can fetch once per pair and apply to all matching snapshot IDs
        rows = session.execute(text("""
            SELECT ticker, scan_date, COUNT(*) as cnt
            FROM snapshots
            WHERE scan_date >= '2026-05-01' AND has_options = 1 AND collection_error IS NULL AND atm_iv_near IS NULL
            GROUP BY ticker, scan_date
            HAVING cnt > 1
        """)).fetchall()
        print(f"Duplicate (ticker, scan_date) groups: {len(rows)} (saving {sum(r[2]-1 for r in rows)} API calls)")
