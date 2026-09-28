"""Forward-factor signal regressions: DST-correct ladder clock, event-aware
arb target debit, and quote validation before IV is solved."""

from __future__ import annotations

import math
import sqlite3
from datetime import UTC, date, datetime, timedelta

import pytest

from earnings_edge import forward_factor_arb as arb
from earnings_edge.fwd_factor import LadderSpec, leg_quote_ok, occ_symbol
from earnings_edge.fwd_factor_ladder import build_candidate as ladder_build_candidate

# ── ladder clock ──────────────────────────────────────────────────────────


def test_ladder_window_tracks_est_in_winter():
    """14:00 ET is 19:00 UTC in December. The fixed UTC-4 'ET' put the first
    rung at 18:00 UTC (13:00 EST) and the last one an hour before 15:45."""
    spec = LadderSpec()
    assert spec.rung_index(datetime(2026, 12, 7, 18, 59, tzinfo=UTC)) is None
    assert spec.rung_index(datetime(2026, 12, 7, 19, 0, tzinfo=UTC)) == 0
    assert spec.rung_index(datetime(2026, 12, 7, 20, 45, tzinfo=UTC)) == 7
    assert spec.rung_index(datetime(2026, 12, 7, 20, 46, tzinfo=UTC)) is None


def test_ladder_window_unchanged_in_summer():
    spec = LadderSpec()
    assert spec.rung_index(datetime(2026, 7, 27, 17, 59, tzinfo=UTC)) is None
    assert spec.rung_index(datetime(2026, 7, 27, 18, 0, tzinfo=UTC)) == 0


# ── arb target debit ──────────────────────────────────────────────────────


def test_required_near_iv_round_trips_through_ex_earnings_factor():
    """Pricing the near leg at the required IV and stripping the event again
    must land exactly on the target factor."""
    fwd, factor, T1, move = 0.30, 0.25, 45 / 365, 0.06
    iv = arb.required_near_iv_for_factor(fwd, factor, T1, move)
    ex = arb.calculate_ex_earnings_iv(iv, T1, move)
    assert ex == pytest.approx(fwd * (1 + factor))
    assert arb.calculate_forward_factor(ex, fwd) == pytest.approx(factor)


def test_event_inside_t1_lowers_the_debit_cap():
    """Without the event variance the near leg was priced ex-earnings (too
    cheap), inflating D* — the ladder would pay up for event-rich calendars."""
    kw = dict(far_price=8.0, spot=100.0, strike=100.0, T1=45 / 365, fwd_vol=0.30, target_factor=0.25)
    no_event = arb.target_debit_for_factor(**kw)
    with_event = arb.target_debit_for_factor(**kw, event_move=0.06)
    assert with_event < no_event
    assert arb.target_debit_for_factor(**kw, event_move=0.0) == no_event


# ── quote validation ──────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "bid,ask,ok",
    [(1.0, 1.2, True), (1.0, 1.0, True), (0.0, 1.2, False), (1.3, 1.2, False), (None, 1.0, False)],
)
def test_leg_quote_ok(bid, ask, ok):
    assert leg_quote_ok(bid, ask) is ok


class _ArbAlpaca:
    def __init__(self, chain):
        self.chain = chain

    def get_stock_latest_trade(self, ticker):
        return 150.0

    def get_options_chain_snapshots(self, ticker):
        return self.chain


def test_arb_rejects_zero_bid_before_solving_iv(tmp_db_path, monkeypatch):
    today = date(2026, 7, 25)
    near = occ_symbol("TEST", today + timedelta(days=45), 150.0)
    far = occ_symbol("TEST", today + timedelta(days=75), 150.0)
    solved = []
    monkeypatch.setattr(arb, "implied_volatility", lambda *a, **k: solved.append(a) or 0.3)
    cand = arb.build_candidate(
        _ArbAlpaca({near: {"bid": 0.0, "ask": 4.2}, far: {"bid": 6.0, "ask": 6.2}}), "TEST", today=today
    )
    assert cand.skip_reason == "invalid quotes"
    assert solved == []


def test_ladder_rejects_zero_bid_leg(tmp_path):
    from earnings_edge.db import engine as db_engine

    db = tmp_path / "ff.db"
    db_engine.configure(db)
    conn = sqlite3.connect(str(db))
    for i, mv in enumerate((3.5, 4.0, 4.5)):
        conn.execute(
            "INSERT INTO snapshots (ticker, earnings_date, scan_date, timing, actual_move_pct, "
            "outcome_fetched_at) VALUES ('TEST', ?, '2026-01-14', 'Post Market', ?, '2026-01-16')",
            (f"2026-0{i + 1}-15", mv),
        )
    conn.commit()
    conn.close()

    today = date(2026, 7, 27)
    earnings = today + timedelta(days=1)
    near = occ_symbol("TEST", today + timedelta(days=45), 100.0)
    far = occ_symbol("TEST", today + timedelta(days=73), 100.0)

    class Alpaca:
        def get_stock_latest_trade(self, symbol):
            return 100.0

        def get_options_chain_snapshots(self, underlying, page_limit=250):
            return {near: {"bid": 0.0, "ask": 5.0}, far: {"bid": 6.0, "ask": 6.1}}

    cand = ladder_build_candidate(Alpaca(), "TEST", earnings, today=today)
    assert cand.skip_reason is not None
    assert cand.skip_reason.startswith("invalid leg quotes")
    assert not math.isnan(cand.near_ask)
