"""Analyze ff_backfill results: premium_ratio distribution + validation vs realized moves."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import text

from earnings_edge.db import get_session

if __name__ == "__main__":
    with get_session() as session:
        print("=== SKIP REASONS ===")
        for r in session.execute(text("""
            SELECT COALESCE(skip_reason,'?') s, COUNT(*) n FROM ff_snapshots
            WHERE selector_version=2 AND skip_reason IS NOT NULL
            GROUP BY s ORDER BY n DESC""")).mappings():
            print(f"  {r['n']:4d}  {r['s']}")

        rows = session.execute(text("""
            SELECT ticker, scan_date, earnings_date, implied_event_move_pct, hist_rms_move_pct,
                   premium_ratio, sigma_fwd, t1_dte
            FROM ff_snapshots
            WHERE selector_version=2 AND premium_ratio IS NOT NULL""")).mappings().fetchall()
        pr = sorted(r['premium_ratio'] for r in rows)
        n = len(pr)
        def q(p): return pr[min(n-1, int(p*n))]
        print(f"\n=== PREMIUM RATIO (n={n}) ===")
        print(f"  min {pr[0]:.2f} | p25 {q(.25):.2f} | median {q(.5):.2f} | p75 {q(.75):.2f} | p90 {q(.9):.2f} | max {pr[-1]:.2f}")
        for th in (1.0, 1.1, 1.2, 1.3, 1.5, 2.0):
            c = sum(1 for x in pr if x >= th)
            print(f"  >= {th:.1f}: {c:4d} ({c/n:.0%})")

        print("\n=== VALIDATION: implied vs realized by premium bucket ===")
        buckets = {'<1.0': [], '1.0-1.2': [], '>=1.2': []}
        joined = 0
        for r in rows:
            out = session.execute(text("""
                SELECT ABS(actual_move_pct) FROM snapshots
                WHERE ticker=:ticker AND scan_date=:scan_date AND actual_move_pct IS NOT NULL
                  AND outcome_fetched_at IS NOT NULL AND outcome_fetched_at != 'unavailable'"""),
                {"ticker": r['ticker'], "scan_date": r['scan_date']}).fetchone()
            if not out or out[0] is None:
                continue
            actual = float(out[0])
            implied = float(r['implied_event_move_pct'])
            if implied <= 0:
                continue
            joined += 1
            ratio = actual / implied
            b = '<1.0' if r['premium_ratio'] < 1.0 else ('1.0-1.2' if r['premium_ratio'] < 1.2 else '>=1.2')
            buckets[b].append(ratio)

        print(f"joined outcomes: {joined}/{n}")
        for b, rs in buckets.items():
            if not rs:
                print(f"  {b:8s}: no data")
                continue
            rs = sorted(rs)
            m = len(rs)
            win = sum(1 for x in rs if x < 1.0)
            mean = sum(rs) / m
            med = rs[m//2] if m % 2 else (rs[m//2-1] + rs[m//2]) / 2
            print(f"  {b:8s}: n={m:3d} | overpriced (actual<implied) {win/m:.0%} | "
                  f"mean actual/implied {mean:.2f} | median {med:.2f}")
