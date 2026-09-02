"""Tests for the backtester realism layer: fills, commissions, REG-T margin, capacity."""

from __future__ import annotations

import math

import pytest

from earnings_edge.backtest.realism import (
    OptionLeg,
    capacity_cap,
    ibkr_commission,
    realistic_fill,
    regt_margin,
)


# ── ibkr_commission ----------------------------------------------------------

def test_commission_tier_boundaries():
    # worst-case IBKR tiers: <$0.05 -> $0.25, $0.05-$0.10 -> $0.50, >=$0.10 -> $0.65
    assert ibkr_commission(0.049) == pytest.approx(0.25)
    assert ibkr_commission(0.05) == pytest.approx(0.50)
    assert ibkr_commission(0.099) == pytest.approx(0.50)
    assert ibkr_commission(0.10) == pytest.approx(0.65)
    assert ibkr_commission(5.00) == pytest.approx(0.65)


def test_commission_scales_with_contracts():
    assert ibkr_commission(0.049, contracts=4) == pytest.approx(1.00)
    assert ibkr_commission(0.07, contracts=10) == pytest.approx(5.00)
    assert ibkr_commission(1.20, contracts=3) == pytest.approx(1.95)


# ── realistic_fill -----------------------------------------------------------

def test_fill_buy_above_mid_sell_below_mid():
    buy = realistic_fill(mid=1.00, bid=0.90, ask=1.10, side="buy")
    sell = realistic_fill(mid=1.00, bid=0.90, ask=1.10, side="sell")
    assert buy > 1.00
    assert sell < 1.00
    # symmetric book -> symmetric slippage
    assert buy - 1.00 == pytest.approx(1.00 - sell)


def test_fill_default_participation_is_half_spread_fraction():
    # half_spread = 0.10; default participation 0.5 -> deviation 0.05
    fill = realistic_fill(mid=1.00, bid=0.90, ask=1.10, side="buy")
    assert fill == pytest.approx(1.05)


def test_fill_never_worse_than_far_touch():
    # extreme parameters still clamp to the ask (buy) / bid (sell)
    buy = realistic_fill(mid=1.00, bid=0.50, ask=1.50, volume=1, open_interest=1,
                         is_otm=True, side="buy", spread_participation=2.0)
    sell = realistic_fill(mid=1.00, bid=0.50, ask=1.50, volume=1, open_interest=1,
                          is_otm=True, side="sell", spread_participation=2.0)
    assert buy == pytest.approx(1.50)
    assert sell == pytest.approx(0.50)


def test_fill_empty_book_may_cross_the_touch():
    # volume == 0 AND open_interest == 0 -> empty book, model walks past the touch
    fill = realistic_fill(mid=1.00, bid=0.90, ask=1.10, volume=0, open_interest=0,
                          is_otm=True, side="buy")
    assert fill > 1.10


def test_fill_lower_volume_worse_fill():
    liquid = realistic_fill(mid=1.00, bid=0.90, ask=1.10, volume=100_000,
                            open_interest=100_000, side="buy")
    thin = realistic_fill(mid=1.00, bid=0.90, ask=1.10, volume=10,
                          open_interest=10, side="buy")
    assert thin > liquid
    # same for sells: thinner book -> lower fill
    liquid_s = realistic_fill(mid=1.00, bid=0.90, ask=1.10, volume=100_000,
                              open_interest=100_000, side="sell")
    thin_s = realistic_fill(mid=1.00, bid=0.90, ask=1.10, volume=10,
                            open_interest=10, side="sell")
    assert thin_s < liquid_s


def test_fill_otm_penalized_vs_atm():
    atm = realistic_fill(mid=1.00, bid=0.90, ask=1.10, is_otm=False, side="buy")
    otm = realistic_fill(mid=1.00, bid=0.90, ask=1.10, is_otm=True, side="buy")
    assert otm > atm


def test_fill_none_depth_means_no_widening():
    # unknown depth -> base deviation only (documented assumption)
    fill = realistic_fill(mid=1.00, bid=0.90, ask=1.10, side="buy")
    assert fill == pytest.approx(1.05)


def test_fill_zero_spread_returns_mid():
    assert realistic_fill(mid=1.00, bid=1.00, ask=1.00, side="buy") == pytest.approx(1.00)


def test_fill_rejects_crossed_book():
    with pytest.raises(ValueError):
        realistic_fill(mid=1.00, bid=1.10, ask=0.90, side="buy")


# ── regt_margin --------------------------------------------------------------

def test_regt_long_option_paid_in_full():
    legs = [OptionLeg(action="buy", kind="call", strike=100.0,
                      underlying_price=100.0, premium=3.00)]
    assert regt_margin(legs) == pytest.approx(300.0)


def test_regt_long_straddle_sums_premiums():
    legs = [
        OptionLeg(action="buy", kind="call", strike=100.0, underlying_price=100.0, premium=3.00),
        OptionLeg(action="buy", kind="put", strike=100.0, underlying_price=100.0, premium=2.50),
    ]
    assert regt_margin(legs) == pytest.approx(550.0)


def test_regt_short_put_hand_computed():
    # base = strike = 95; OTM amount = 100 - 95 = 5
    # max(0.20*95 - 5, 0.10*95) * 100 + 2.00*100 = max(14, 9.5)*100 + 200 = 1600
    legs = [OptionLeg(action="sell", kind="put", strike=95.0,
                      underlying_price=100.0, premium=2.00)]
    assert regt_margin(legs) == pytest.approx(1600.0)


def test_regt_short_call_hand_computed():
    # base = underlying = 100; OTM amount = 105 - 100 = 5
    # max(0.20*100 - 5, 0.10*100) * 100 + 2.00*100 = max(15, 10)*100 + 200 = 1700
    legs = [OptionLeg(action="sell", kind="call", strike=105.0,
                      underlying_price=100.0, premium=2.00)]
    assert regt_margin(legs) == pytest.approx(1700.0)


def test_regt_short_call_itm_uses_20pct_floor():
    # ITM call: OTM amount = 0 -> max(0.20*100, 0.10*100)*100 + 12*100 = 3200
    legs = [OptionLeg(action="sell", kind="call", strike=90.0,
                      underlying_price=100.0, premium=12.00)]
    assert regt_margin(legs) == pytest.approx(3200.0)


def test_regt_defined_risk_credit_spread():
    # put credit spread: sell 95P @ 2.00, buy 90P @ 1.00 -> credit 1.00, width 5
    # margin = width*100 - credit*100 = 400
    legs = [
        OptionLeg(action="sell", kind="put", strike=95.0, underlying_price=100.0, premium=2.00),
        OptionLeg(action="buy", kind="put", strike=90.0, underlying_price=100.0, premium=1.00),
    ]
    assert regt_margin(legs) == pytest.approx(400.0)


def test_regt_defined_risk_debit_spread():
    # call debit spread: buy 100C @ 3.00, sell 105C @ 1.00 -> debit 2.00
    # margin = debit paid in full = 200
    legs = [
        OptionLeg(action="buy", kind="call", strike=100.0, underlying_price=100.0, premium=3.00),
        OptionLeg(action="sell", kind="call", strike=105.0, underlying_price=100.0, premium=1.00),
    ]
    assert regt_margin(legs) == pytest.approx(200.0)


def test_regt_short_strangle_sums_naked_margins():
    legs = [
        OptionLeg(action="sell", kind="call", strike=105.0, underlying_price=100.0, premium=2.00),
        OptionLeg(action="sell", kind="put", strike=95.0, underlying_price=100.0, premium=2.00),
    ]
    # 1700 (short call) + 1600 (short put)
    assert regt_margin(legs) == pytest.approx(3300.0)


# ── capacity_cap -------------------------------------------------------------

def test_capacity_cap_basic():
    assert capacity_cap(volume=10_000, open_interest=5_000, participation=0.10) == 500


def test_capacity_cap_uses_min_of_volume_and_oi():
    assert capacity_cap(volume=100, open_interest=50_000, participation=0.10) == 10


def test_capacity_cap_none_handling():
    assert capacity_cap(volume=None, open_interest=5_000, participation=0.10) == 500
    assert capacity_cap(volume=5_000, open_interest=None, participation=0.10) == 500
    assert capacity_cap(volume=None, open_interest=None) == math.inf


def test_capacity_cap_zero_liquidity():
    assert capacity_cap(volume=0, open_interest=5_000) == 0
