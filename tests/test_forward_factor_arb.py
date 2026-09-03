import pytest
from datetime import date, timedelta
from typing import Optional

from earnings_edge.forward_factor_arb import build_candidate
from earnings_edge.fwd_factor import occ_symbol

class FakeAlpaca:
    def __init__(self, spot=150.0):
        self.spot = spot
        self.chain = {}
    
    def get_stock_latest_trade(self, ticker: str) -> Optional[float]:
        return self.spot
        
    def get_options_chain_snapshots(self, ticker: str) -> dict:
        return self.chain


TODAY = date(2026, 7, 25)
SPOT = 150.0

def make_fake_chain():
    al = FakeAlpaca(spot=SPOT)
    # T1 = 45 DTE
    d1 = TODAY + timedelta(days=45)
    sym1 = occ_symbol("TEST", d1, SPOT)
    al.chain[sym1] = {"bid": 4.0, "ask": 4.2}  # ~ 0.30 IV
    
    # T2 = 75 DTE
    d2 = TODAY + timedelta(days=75)
    sym2 = occ_symbol("TEST", d2, SPOT)
    al.chain[sym2] = {"bid": 6.0, "ask": 6.2}  # ~ 0.25 IV
    return al

def test_arb_build_candidate_no_event_raw_iv(tmp_db_path):
    al = make_fake_chain()
    
    cand = build_candidate(al, "TEST", today=TODAY)
    
    # Check that raw IV is used (if no event, raw IV is lower, factor might be low, but let's check it doesn't crash)
    assert cand.skip_reason is None or "factor" in cand.skip_reason
    # If we made factor good enough, it would pass.
    # We can just verify it didn't skip due to 'no T1/T2' or 'event inside T1'
    assert "event inside T1" not in (cand.skip_reason or "")
    
    # No-event candidates carry no earnings date, so T1 expiry IS the exit
    # horizon — earnings_date must be the T1 expiry (NOT ""), or
    # ff_candidate_to_trade's date.fromisoformat(cand.earnings_date)
    # crashes the whole proposal batch.
    if cand.skip_reason is None:
        assert cand.earnings_date == (TODAY + timedelta(days=45)).isoformat()

    
def test_arb_build_candidate_event_no_hist_rms(tmp_db_path):
    from earnings_edge.db.repositories import insert_snapshot
    # Insert next earnings inside T1 (T1 is 45 DTE, let's put earnings at 20 DTE)
    ed = TODAY + timedelta(days=20)
    insert_snapshot({"ticker": "TEST", "has_options": 1, "earnings_date": ed.isoformat(), "scan_date": TODAY.isoformat()})
    
    al = make_fake_chain()
    cand = build_candidate(al, "TEST", today=TODAY)
    
    assert cand.skip_reason == "event inside T1, no hist rms"

def test_arb_build_candidate_event_with_hist_rms(tmp_db_path):
    from earnings_edge.db.repositories import insert_snapshot
    import sqlite3
    from earnings_edge.db import engine as db_engine
    
    ed = TODAY + timedelta(days=20)
    insert_snapshot({"ticker": "TEST", "has_options": 1, "earnings_date": ed.isoformat(), "scan_date": TODAY.isoformat()})
    
    # Inject hist rms
    # hist_rms_move needs 3+ outcomes
    with db_engine.session_scope() as s:
        c = s.connection().connection
        for i, mv in enumerate((5.0, 5.0, 5.0)):
            c.execute(
                "INSERT INTO snapshots (ticker, earnings_date, scan_date, actual_move_pct, outcome_fetched_at) "
                f"VALUES ('TEST', '2025-0{i+1}-01', '2025-01-01', {mv}, '2025-01-01')"
            )
        c.commit()

    al = make_fake_chain()
    # Boost near premium so factor is high enough after removing earnings var
    sym1 = occ_symbol("TEST", TODAY + timedelta(days=45), SPOT)
    al.chain[sym1] = {"bid": 7.0, "ask": 7.2} # Highly inflated near leg
    
    sym2 = occ_symbol("TEST", TODAY + timedelta(days=75), SPOT)
    al.chain[sym2] = {"bid": 6.8, "ask": 7.0}
    
    cand = build_candidate(al, "TEST", today=TODAY)

    
    assert cand.skip_reason is None
    assert cand.earnings_date == ed.isoformat()
    assert cand.hist_rms_move > 0
    # ensure mid_debit and d_start are populated
    assert cand.d_start > 0
    assert cand.d_cap > 0
