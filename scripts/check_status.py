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
        print(f"Total has_options=1: {total}")
        print(f"IV null: {iv_null}")
        print(f"RV null: {rv_null}")
        print(f"Both null: {both_null}")
