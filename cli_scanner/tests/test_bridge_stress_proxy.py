"""Earnings-stress max-loss proxy for naked short premium (SHORT_STRADDLE /
SHORT_STRANGLE without wings) in StrategyBridge._structure_cost.

When the scan layer provides ``features["expected_move_dollars"]`` (the ATM
straddle price = market-implied earnings move in $), the per-unit max-loss
proxy is ``EARNINGS_STRESS_MULTIPLE * expected_move_dollars * 100`` — the
stock moving twice the priced-in move against the position. The crude
``strike * 100 * _NOTIONAL_RISK_FRAC`` notional proxy remains the fallback
when the feature is missing or non-positive.
"""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock

import pytest

from earnings_edge.alpaca_bridge import (
    EARNINGS_STRESS_MULTIPLE,
    BridgeConfig,
    StrategyBridge,
)
from earnings_edge.trading_types import Trade
from framework.risk.manager import RiskManager
from framework.risk.sizing import SizeContext, build_sizer
from earnings_edge.db import engine as db_engine


# ── Helpers ------------------------------------------------------------------

def _client():
    client = MagicMock()
    client.position_symbols.return_value = set()
    client.get_account.return_value = {"equity": "100000", "buying_power": "50000"}
    client.get_positions.return_value = []
    client.get_option_snapshot.return_value = {}
    client.submit_multi_leg_order.return_value = {
        "id": "o1", "status": "accepted", "legs": [],
    }
    return client


def _plain_bridge(client=None):
    """Bridge without the risk layer — direct _structure_cost unit tests."""
    return StrategyBridge(client=client or _client(), config=BridgeConfig())


def _sized_bridge(tmp_path, client=None, risk_pct=0.01):
    """Bridge with risk manager + vol_target sizer wired (end-to-end sizing)."""
    db_engine.configure(tmp_path / "fw.db")
    return StrategyBridge(
        client=client or _client(),
        config=BridgeConfig(),
        risk_manager=RiskManager(),
        sizer_resolver=lambda name: {"name": "vol_target", "risk_pct": risk_pct},
    )


def _straddle(credit=4.0, strike=150.0, expected_move_dollars=None):
    features = {"atm_strike": strike, "expiry": date(2026, 8, 21)}
    if expected_move_dollars is not None:
        features["expected_move_dollars"] = expected_move_dollars
    return Trade(
        ticker="XYZ", earnings_date=date(2026, 7, 29), scan_date=date(2026, 7, 28),
        strategy="short_straddle", side="SHORT_STRADDLE", entry_price=credit,
        features=features, model_score=0.6, ml_decision="TAKE",
    )


def _condor(credit=1.50, expected_move_dollars=None):
    features = {
        "short_put": 180.0, "long_put": 170.0,
        "short_call": 200.0, "long_call": 210.0,
        "expiry": date(2026, 8, 21),
    }
    if expected_move_dollars is not None:
        features["expected_move_dollars"] = expected_move_dollars
    return Trade(
        ticker="XYZ", earnings_date=date(2026, 7, 29), scan_date=date(2026, 7, 28),
        strategy="short_straddle", side="IRON_CONDOR", entry_price=credit,
        features=features, model_score=0.6, ml_decision="TAKE",
    )


def _calendar(entry_price=1.85, expected_move_dollars=None):
    features = {
        "near_strike": 150.0, "far_strike": 150.0,
        "near_expiry": date(2026, 7, 31), "far_expiry": date(2026, 8, 28),
    }
    if expected_move_dollars is not None:
        features["expected_move_dollars"] = expected_move_dollars
    return Trade(
        ticker="XYZ", earnings_date=date(2026, 7, 29), scan_date=date(2026, 7, 28),
        strategy="calendar_call_ml", side="CALENDAR", entry_price=entry_price,
        features=features, model_score=0.7, ml_decision="TAKE",
    )


# ── Unit: _structure_cost ----------------------------------------------------

def test_stress_proxy_used_when_expected_move_present():
    bridge = _plain_bridge()
    trade = _straddle(credit=4.0, strike=150.0, expected_move_dollars=8.0)
    legs = bridge._build_legs(trade)
    cost = bridge._structure_cost(trade, legs, 1)
    # 2.0 × $8.00 × 100 = $1,600 — NOT the old notional proxy (150×100×0.20 = $3,000)
    assert cost == pytest.approx(2.0 * 8.0 * 100)
    assert cost == pytest.approx(1600.0)
    assert cost < 150.0 * 100 * 0.20


def test_stress_proxy_floored_at_premium():
    bridge = _plain_bridge()
    # Tiny expected move: stress proxy ($200) < premium collected ($400)
    trade = _straddle(credit=4.0, strike=150.0, expected_move_dollars=1.0)
    legs = bridge._build_legs(trade)
    assert bridge._structure_cost(trade, legs, 1) == pytest.approx(400.0)


def test_stress_proxy_scales_with_qty():
    bridge = _plain_bridge()
    trade = _straddle(credit=4.0, strike=150.0, expected_move_dollars=8.0)
    legs = bridge._build_legs(trade)
    assert bridge._structure_cost(trade, legs, 3) == pytest.approx(4800.0)


@pytest.mark.parametrize("em", [None, 0.0, -3.5])
def test_notional_fallback_when_expected_move_unusable(em):
    bridge = _plain_bridge()
    trade = _straddle(credit=4.0, strike=150.0, expected_move_dollars=em)
    legs = bridge._build_legs(trade)
    # Exactly the pre-change behavior: max(strike×100×0.20, premium) = $3,000
    assert bridge._structure_cost(trade, legs, 1) == pytest.approx(3000.0)


def test_defined_risk_condor_unchanged_even_with_expected_move():
    bridge = _plain_bridge()
    trade = _condor(credit=1.50, expected_move_dollars=8.0)
    legs = bridge._build_legs(trade)
    # Winged path takes precedence: (10 − 1.5) × 100 = 850, stress proxy ignored
    assert bridge._structure_cost(trade, legs, 1) == pytest.approx(850.0)


def test_debit_structure_unchanged_even_with_expected_move():
    bridge = _plain_bridge()
    trade = _calendar(entry_price=1.85, expected_move_dollars=8.0)
    legs = bridge._build_legs(trade)
    assert bridge._structure_cost(trade, legs, 1) == pytest.approx(185.0)


# ── _exit_by: structural deadline (near-leg expiry for differential-expiry
# structures; None for single-expiry structures) ------------------------------

def test_exit_by_is_the_near_leg_expiry_for_a_calendar():
    bridge = _plain_bridge()
    legs = bridge._build_legs(_calendar())
    assert bridge._exit_by(legs) == date(2026, 7, 31)  # near, not far (8/28)


def test_exit_by_none_for_single_expiry_structure():
    bridge = _plain_bridge()
    legs = bridge._build_legs(_straddle())
    assert bridge._exit_by(legs) is None


def test_execute_trade_sets_exit_by_on_result_for_calendar():
    bridge = _plain_bridge()
    result = bridge.execute_trade(_calendar())
    assert result is not None
    assert result.exit_by == date(2026, 7, 31)


def test_execute_trade_exit_by_none_for_single_expiry_structure():
    bridge = _plain_bridge()
    result = bridge.execute_trade(_straddle())
    assert result is not None
    assert result.exit_by is None


# ── Sizer math (documents the budget boundary) --------------------------------

def test_vol_target_budget_boundary():
    sizer = build_sizer("vol_target", {"risk_pct": 0.01})
    # budget = 1% × $100k = $1,000
    ctx = lambda ml: SizeContext(equity=100_000, buying_power=50_000,
                                 price_per_unit=ml, max_loss_per_unit=ml)
    assert sizer.quantity(ctx(2.0 * 4.0 * 100)) == 1   # EM $4  → $800  → 1
    assert sizer.quantity(ctx(2.0 * 8.0 * 100)) == 0   # EM $8  → $1,600 → veto
    assert sizer.quantity(ctx(2.0 * 15.0 * 100)) == 0  # EM $15 → $3,000 → veto


# ── End-to-end: execute_trade through the vol_target sizer --------------------

def test_sizer_expresses_affordable_name(tmp_path):
    """EM $4 → stress proxy $800 ≤ $1,000 budget → trade submits at qty 1."""
    bridge = _sized_bridge(tmp_path)
    result = bridge.execute_trade(
        _straddle(credit=4.0, strike=50.0, expected_move_dollars=4.0))
    assert result is not None
    assert bridge.skip_reasons["size_veto"] == 0
    assert bridge.client.submit_multi_leg_order.call_count == 1


def test_sizer_still_vetoes_genuinely_risky_name(tmp_path):
    """EM $15 → stress proxy $3,000 > $1,000 budget → qty 0 → size_veto."""
    bridge = _sized_bridge(tmp_path)
    result = bridge.execute_trade(
        _straddle(credit=15.0, strike=200.0, expected_move_dollars=15.0))
    assert result is None
    assert bridge.skip_reasons["size_veto"] == 1
    assert bridge.client.submit_multi_leg_order.call_count == 0


def test_sizer_veto_without_expected_move_feature(tmp_path):
    """Regression: no expected_move_dollars → notional proxy still vetoes
    expensive strikes exactly as before the change."""
    bridge = _sized_bridge(tmp_path)
    result = bridge.execute_trade(
        _straddle(credit=4.0, strike=150.0, expected_move_dollars=None))
    assert result is None
    assert bridge.skip_reasons["size_veto"] == 1
