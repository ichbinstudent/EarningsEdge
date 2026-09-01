import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import text

from earnings_edge.db import get_session

if __name__ == "__main__":
    with get_session() as session:
        # Check pre-May rows where both exist — are they the same?
        rows = session.execute(text("""
            SELECT atm_iv_near, sigma_short_leg FROM snapshots
            WHERE atm_iv_near IS NOT NULL AND sigma_short_leg IS NOT NULL
            LIMIT 20
        """)).fetchall()
        print("Comparing atm_iv_near vs sigma_short_leg:")
        for r in rows:
            print(f"  iv={r[0]:.6f}  sigma={r[1]:.6f}  diff={abs(r[0]-r[1]):.8f}")

        # Quick fix: copy sigma_short_leg -> atm_iv_near where null
        fixable = session.execute(text("""
            SELECT COUNT(*) FROM snapshots
            WHERE scan_date >= '2026-05-01' AND atm_iv_near IS NULL AND sigma_short_leg IS NOT NULL
        """)).scalar()
        print(f"\nRows fixable by copying sigma_short_leg -> atm_iv_near: {fixable}")

        # How many rv30/hist_vol_3m null?
        no_rv = session.execute(text("""
            SELECT COUNT(DISTINCT ticker) FROM snapshots
            WHERE scan_date >= '2026-05-01' AND has_options = 1 AND rv30 IS NULL AND sigma_short_leg IS NOT NULL
        """)).scalar()
        print(f"Unique tickers needing rv30 backfill: {no_rv}")
