"""Forward-factor signal regressions: DST-correct ladder clock, event-aware
arb target debit, quote validation before IV is solved, market-anchored
ladder pricing, same-strike calendars, liquidity gates, and timing-aware
historical event moves."""

from __future__ import annotations

import math
import sqlite3
from datetime import UTC, date, datetime, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from earnings_edge import forward_factor_arb as arb
from earnings_edge import fwd_factor_ladder as ffl
from earnings_edge.fwd_factor import LadderSpec, leg_quote_ok, occ_symbol
from earnings_edge.fwd_factor_ladder import build_candidate as ladder_build_candidate
from earnings_edge.services.outcome_service import OutcomeService

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


# ── ladder entry pricing ──────────────────────────────────────────────────


def test_ladder_never_bids_above_the_market():
    """IDT 2026-09-25: mid 0.67, D* 2.33 → cap 2.34; the old ladder bid 2.33
    and paper filled 2.40. The limit must start at mid and stay ≤ ask."""
    spec = LadderSpec()
    mid, ask, cap = 0.67, 0.95, 2.34
    limits = [spec.market_limit(r, mid, ask, cap) for r in range(spec.n_rungs)]
    assert limits[0] == 0.67
    assert limits[-1] == 0.95  # final rung concedes to the ask, not the model cap
    assert limits == sorted(limits)
    assert max(limits) <= ask


def test_ladder_cap_still_binds_below_the_ask():
    spec = LadderSpec()
    limits = [spec.market_limit(r, 1.50, 2.10, 1.80) for r in range(spec.n_rungs)]
    assert limits[0] == 1.50 and limits[-1] == 1.80
    assert max(limits) <= 1.80


def test_ladder_rests_at_cap_when_mid_is_above_it():
    assert LadderSpec().market_limit(0, 2.00, 2.20, 1.90) == 1.90


def test_ladder_rung_count_matches_window():
    assert LadderSpec().n_rungs == 8  # 14:00 .. 15:45 every 15 min


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


# ── pair selection ────────────────────────────────────────────────────────

FRI = date(2026, 10, 2)


def _chain(*contracts):
    return {occ_symbol("CCL", exp, k): {"bid": 1.0, "ask": 1.1} for exp, k in contracts}


def test_calendar_legs_share_one_strike():
    """2026-09 CCL: sold the 22.5 near, bought the 22.0 far — a diagonal the
    calendar math doesn't price. Both legs must use a strike both list."""
    near_exp, far_exp = FRI, FRI + timedelta(days=28)
    chain = _chain((near_exp, 22.5), (near_exp, 22.0), (far_exp, 22.0), (far_exp, 23.0))
    t1, t2 = ffl._pick_pair(chain, 22.4, FRI - timedelta(days=7), FRI - timedelta(days=1), "Pre Market")
    assert t1["strike"] == t2["strike"] == 22.0
    assert (t1["expiry"], t2["expiry"]) == (near_exp, far_exp)


def test_no_pair_when_expiries_share_no_strike():
    chain = _chain((FRI, 22.5), (FRI + timedelta(days=28), 23.0))
    assert ffl._pick_pair(chain, 22.4, FRI - timedelta(days=7), FRI, "Pre Market") == (None, None)


def test_after_close_event_needs_an_expiry_after_the_event_date():
    """An expiry ON an after-close event date settles before the move."""
    chain = _chain((FRI, 22.0), (FRI + timedelta(days=7), 22.0), (FRI + timedelta(days=35), 22.0))
    today = FRI - timedelta(days=3)
    bmo, _ = ffl._pick_pair(chain, 22.0, today, FRI, "Pre Market")
    amc, _ = ffl._pick_pair(chain, 22.0, today, FRI, "Post Market")
    unknown, _ = ffl._pick_pair(chain, 22.0, today, FRI, None)
    assert bmo["expiry"] == FRI
    assert amc["expiry"] == unknown["expiry"] == FRI + timedelta(days=7)


def test_combo_liquidity_gate():
    assert ffl.combo_liquidity_reason(4.9, 5.1, 6.4, 6.6) is None  # 0.40 wide on a 1.50 mid
    assert "combo spread" in ffl.combo_liquidity_reason(4.4, 5.4, 6.0, 7.2)  # 2.20 wide on 1.70
    assert "mid debit" in ffl.combo_liquidity_reason(0.30, 0.32, 0.45, 0.47)  # 0.15 mid


# ── timing-aware historical moves ─────────────────────────────────────────


def test_event_timing_from_yahoo_timestamp():
    assert ffl.event_timing_from_timestamp(datetime(2026, 7, 29, 16, 5)) == "Post Market"
    assert ffl.event_timing_from_timestamp(datetime(2026, 7, 29, 6, 30)) == "Pre Market"
    assert ffl.event_timing_from_timestamp(datetime(2026, 7, 29)) is None  # time unknown
    assert ffl.backfill_tag(None) == "Backfill:2d"


def _ms(d: date) -> int:
    return int(datetime(d.year, d.month, d.day).timestamp() * 1000)


def _event_bars(ed: date):
    # quiet before, +1% on the event date (pre-announcement for AMC), +8% the next day
    return [
        {"t": _ms(ed - timedelta(days=1)), "c": 100.0, "h": 100.5, "l": 99.5},
        {"t": _ms(ed), "c": 101.0, "h": 101.5, "l": 100.0},
        {"t": _ms(ed + timedelta(days=1)), "c": 109.08, "h": 110.0, "l": 104.0},
    ]


def test_unknown_timing_window_spans_both_sessions():
    ed = date(2026, 7, 29)
    bmo = OutcomeService.outcome_from_bars(_event_bars(ed), ed)
    both = OutcomeService.outcome_from_bars(_event_bars(ed), ed, span_both=True)
    assert bmo["actual_move_pct"] == pytest.approx(1.0)  # misses an AMC reaction entirely
    assert both["actual_move_pct"] == pytest.approx(9.08)


def test_legacy_backfill_rows_are_re_measured_timing_aware(tmp_path):
    """Legacy rows used the BMO window for every name: an after-close
    reporter's +8% reaction was stored as +1%, understating hist RMS."""
    from earnings_edge.db import engine as db_engine
    from earnings_edge.db.repositories import snapshots_legacy_backfill_rows

    db = tmp_path / "hist.db"
    db_engine.configure(db)
    ed = date(2026, 7, 29)
    conn = sqlite3.connect(str(db))
    conn.execute(
        "INSERT INTO snapshots (ticker, earnings_date, scan_date, timing, actual_move_pct, outcome_fetched_at) "
        "VALUES ('AMCX', ?, ?, 'Backfill', 1.0, '2026-08-01')",
        (ed.isoformat(), ed.isoformat()),
    )
    conn.commit()

    yahoo = MagicMock()
    yahoo.get_earnings_dates.return_value = pd.DataFrame(
        index=pd.DatetimeIndex([datetime(2026, 7, 29, 16, 5)])
    )
    lse = MagicMock()
    lse.daily_bars.return_value = _event_bars(ed)
    ffl._hist_refresh_attempted.discard("AMCX")
    with (
        patch("yfinance.Ticker", return_value=yahoo),
        patch.object(ffl, "_lse_bars_client", return_value=lse),
        patch.object(ffl, "_polygon_bars_client", return_value=None),
    ):
        assert ffl.refresh_legacy_hist_moves("AMCX", today=date(2026, 9, 28)) == 1

    timing, move = conn.execute(
        "SELECT timing, actual_move_pct FROM snapshots WHERE ticker='AMCX'"
    ).fetchone()
    conn.close()
    assert timing == "Backfill:AMC"
    assert move == pytest.approx(8.0)  # 101.00 → 109.08, the reaction session
    assert snapshots_legacy_backfill_rows("AMCX") == []
