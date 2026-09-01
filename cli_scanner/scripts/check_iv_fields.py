import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import text

from earnings_edge.db import get_session

# Check which IV fields are missing for May+ has_options=1
iv_fields = ["atm_iv_near", "atm_call_iv", "atm_put_iv", "rv30", "hist_vol_3m",
             "iv30_rv30", "term_slope", "term_structure_valid", "expected_move_pct",
             "expected_move_dollars", "straddle_price", "atm_call_delta", "atm_put_delta",
             "sigma_baseline_1y", "sigma_short_leg", "sigma_short_leg_fair", "actual_to_fair_ratio"]

if __name__ == "__main__":
    with get_session() as session:
        for f in iv_fields:
            non_null = session.execute(text(f"""
                SELECT COUNT(*) FROM snapshots
                WHERE scan_date >= '2026-05-01' AND has_options = 1 AND {f} IS NOT NULL
            """)).scalar()
            print(f"  {f}: {non_null}/890")

        # Sample a row that has term_slope but NOT atm_iv_near
        sample = session.execute(text("""
            SELECT * FROM snapshots
            WHERE scan_date >= '2026-05-01' AND has_options = 1 AND term_slope IS NOT NULL AND atm_iv_near IS NULL
            ORDER BY scan_date DESC LIMIT 1
        """)).mappings().fetchone()
        if sample:
            print(f"\nSample: {sample['ticker']} {sample['earnings_date']}")
            for f in iv_fields:
                print(f"  {f} = {sample[f]}")
