import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import text

from earnings_edge.db import get_session

if __name__ == "__main__":
    with get_session() as session:
        total = session.execute(text("SELECT COUNT(*) FROM snapshots WHERE scan_date >= '2026-05-01' AND has_options = 1")).scalar()
        iv_null = session.execute(text("SELECT COUNT(*) FROM snapshots WHERE scan_date >= '2026-05-01' AND has_options = 1 AND atm_iv_near IS NULL")).scalar()
        rv_null = session.execute(text("SELECT COUNT(*) FROM snapshots WHERE scan_date >= '2026-05-01' AND has_options = 1 AND rv30 IS NULL")).scalar()
        both_null = session.execute(text("SELECT COUNT(*) FROM snapshots WHERE scan_date >= '2026-05-01' AND has_options = 1 AND atm_iv_near IS NULL AND rv30 IS NULL")).scalar()
        # Pre-May comparison
        pre_iv = session.execute(text("SELECT COUNT(*) FROM snapshots WHERE scan_date < '2026-05-01' AND has_options = 1 AND atm_iv_near IS NOT NULL")).scalar()
        pre_total = session.execute(text("SELECT COUNT(*) FROM snapshots WHERE scan_date < '2026-05-01' AND has_options = 1")).scalar()
        print(f"May+ has_options=1: {total}")
        print(f"  IV null: {iv_null} ({iv_null*100//total}%)")
        print(f"  RV null: {rv_null} ({rv_null*100//total}%)")
        print(f"  Both null: {both_null}")
        print(f"Pre-May has_options=1: {pre_total}, IV present: {pre_iv}")
