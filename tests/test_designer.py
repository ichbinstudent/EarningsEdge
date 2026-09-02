"""Tests for the Position Designer analytics (earnings_edge.designer)."""

from __future__ import annotations

import math
from datetime import date

import numpy as np
import pytest
from scipy.stats import norm

from earnings_edge.designer import (
    Leg,
    analyze,
    effective_fill_iv,
    net_premium,
    optimal_delta_hedge,
    pnl_at_date,
    pnl_at_expiry,
    position_greeks,
    rv_scenario,
    tag_structure,
)
from earnings_edge.option_math import (
    black_scholes_gamma,
    black_scholes_price,
    black_scholes_rho,
    black_scholes_theta,
    black_scholes_vega,
)

AS_OF = date(2026, 8, 20)
EXP = date(2026, 9, 18)  # 29 days out
EXP2 = date(2026, 12, 18)
R = 0.045


def call(action: str, strike: float, price: float, iv: float = 0.30,
         qty: int = 1, expiry: date = EXP) -> Leg:
    return Leg(action=action, kind="call", strike=strike, expiry=expiry,
               quantity=qty, price=price, iv=iv)


def put(action: str, strike: float, price: float, iv: float = 0.30,
        qty: int = 1, expiry: date = EXP) -> Leg:
    return Leg(action=action, kind="put", strike=strike, expiry=expiry,
               quantity=qty, price=price, iv=iv)


def stock(action: str, price: float, qty: int = 1) -> Leg:
    return Leg(action=action, kind="stock", strike=0.0, expiry=EXP,
               quantity=qty, price=price, iv=0.0)


# ── Leg basics ---------------------------------------------------------------

def test_leg_signed_quantity():
    assert call("buy", 100, 5.0).signed_quantity == 1
    assert call("sell", 100, 5.0, qty=2).signed_quantity == -2
    assert stock("buy", 100, qty=10).multiplier == 1
    assert call("buy", 100, 5.0).multiplier == 100


def test_leg_validation():
    with pytest.raises(ValueError):
        Leg(action="hold", kind="call", strike=100, expiry=EXP,
            quantity=1, price=5.0, iv=0.3)
    with pytest.raises(ValueError):
        Leg(action="buy", kind="call", strike=100, expiry=EXP,
            quantity=0, price=5.0, iv=0.3)


# ── P&L at expiry (known answers) ---------------------------------------------

def test_long_call_pnl_at_expiry():
    legs = [call("buy", 100, 5.0)]
    grid = np.array([90.0, 100.0, 105.0, 110.0])
    pnl = pnl_at_expiry(legs, grid)
    # per share: max(S-100,0)-5, times 100
    np.testing.assert_allclose(pnl, [-500.0, -500.0, 0.0, 500.0])


def test_short_put_and_stock_pnl_at_expiry():
    legs = [put("sell", 100, 4.0), stock("buy", 95.0, qty=100)]
    pnl = pnl_at_expiry(legs, np.array([90.0, 110.0]))
    # put: (4 - max(100-S,0)) * 100 ; stock: (S-95)*100
    np.testing.assert_allclose(pnl, [(4 - 10) * 100 + (90 - 95) * 100,
                                     4 * 100 + (110 - 95) * 100])


# ── Iron condor analyze (hand-computed) ---------------------------------------

def iron_condor() -> list[Leg]:
    # 90/95/105/110 iron condor for a $2.00 net credit
    return [
        put("buy", 90, 1.00),
        put("sell", 95, 2.00),
        call("sell", 105, 2.00),
        call("buy", 110, 1.00),
    ]


def test_iron_condor_analyze():
    res = analyze(iron_condor(), S=100.0, r=R, as_of=AS_OF)
    assert res["max_profit"] == pytest.approx(200.0, abs=1.0)
    assert res["max_loss"] == pytest.approx(-300.0, abs=1.0)
    assert len(res["breakevens"]) == 2
    assert res["breakevens"][0] == pytest.approx(93.0, abs=0.1)
    assert res["breakevens"][1] == pytest.approx(107.0, abs=0.1)
    assert res["net_premium"] == pytest.approx(-200.0)  # net credit
    assert 0.0 < res["win_rate"] < 1.0
    assert res["structure"]["direction"] == "Neutral"
    assert res["structure"]["risk"] == "Defined"
    assert res["structure"]["vol_exposure"] == "Short"


def test_unbounded_profit_and_loss():
    long_call = analyze([call("buy", 100, 5.0)], S=100.0, r=R, as_of=AS_OF)
    assert long_call["max_profit"] == math.inf
    assert long_call["max_loss"] == pytest.approx(-500.0, abs=1.0)
    short_call = analyze([call("sell", 100, 5.0)], S=100.0, r=R, as_of=AS_OF)
    assert short_call["max_loss"] == -math.inf
    assert short_call["structure"]["risk"] == "Undefined"
    assert short_call["structure"]["direction"] == "Bearish"


# ── Greeks --------------------------------------------------------------------

def test_stock_leg_greeks():
    g = position_greeks([stock("buy", 100.0, qty=10)], S=100.0, r=R, as_of=AS_OF)
    assert g["delta"] == pytest.approx(10.0)
    assert g["gamma"] == pytest.approx(0.0)
    assert g["theta"] == pytest.approx(0.0)
    assert g["vega"] == pytest.approx(0.0)


def test_new_greeks_known_answers():
    S, K, T, r, sigma = 100.0, 100.0, 0.25, 0.05, 0.20
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    expected_gamma = norm.pdf(d1) / (S * sigma * math.sqrt(T))
    expected_vega = S * norm.pdf(d1) * math.sqrt(T)
    assert black_scholes_gamma(S, K, T, r, sigma) == pytest.approx(expected_gamma)
    assert black_scholes_vega(S, K, T, r, sigma) == pytest.approx(expected_vega)
    # same gamma/vega for calls and puts
    assert black_scholes_gamma(S, K, T, r, sigma, "put") == pytest.approx(expected_gamma)
    # ATM call decays, rho positive for calls / negative for puts
    assert black_scholes_theta(S, K, T, r, sigma, "call") < 0
    assert black_scholes_rho(S, K, T, r, sigma, "call") > 0
    assert black_scholes_rho(S, K, T, r, sigma, "put") < 0


def test_position_greeks_long_call():
    g = position_greeks([call("buy", 100, 5.0, iv=0.30)], S=100.0, r=R, as_of=AS_OF)
    assert 40.0 < g["delta"] < 70.0  # ~0.5 delta x 100
    assert g["gamma"] > 0
    assert g["theta"] < 0
    assert g["vega"] > 0


# ── P&L at date / IV shocks ----------------------------------------------------

def test_iv_shock_raises_long_lowers_short():
    T = (EXP - AS_OF).days / 365
    grid = np.array([100.0])
    long_call = [call("buy", 100, 5.0, iv=0.30)]
    short_call = [call("sell", 100, 5.0, iv=0.30)]
    base_long = pnl_at_date(long_call, grid, T, r=R)[0]
    shocked_long = pnl_at_date(long_call, grid, T, r=R, iv_shock=0.05)[0]
    base_short = pnl_at_date(short_call, grid, T, r=R)[0]
    shocked_short = pnl_at_date(short_call, grid, T, r=R, iv_shock=0.05)[0]
    assert shocked_long > base_long
    assert shocked_short < base_short


def test_per_expiry_shock_only_affects_matching_legs():
    legs = [call("buy", 100, 5.0, expiry=EXP), call("buy", 105, 8.0, expiry=EXP2)]
    T = (EXP - AS_OF).days / 365
    grid = np.array([100.0])
    per_expiry = pnl_at_date(legs, grid, T, r=R, iv_shock={EXP: 0.10})[0]
    per_leg = pnl_at_date(legs, grid, T, r=R, iv_shock={0: 0.10})[0]
    other_expiry = pnl_at_date(legs, grid, T, r=R, iv_shock={EXP2: 0.10})[0]
    unshocked = pnl_at_date(legs, grid, T, r=R)[0]
    assert per_expiry == pytest.approx(per_leg)
    assert per_expiry != pytest.approx(other_expiry)
    assert per_expiry > unshocked  # shocked leg is long


# ── RV scenario -----------------------------------------------------------------

def test_rv_scenario_keys_and_determinism():
    legs = iron_condor()
    kwargs = dict(S=100.0, r=R, forecast_rv=0.25, n_sims=2000, seed=42, as_of=AS_OF)
    res1 = rv_scenario(legs, **kwargs)
    res2 = rv_scenario(legs, **kwargs)
    expected = {
        "mean_return", "mean_pnl", "return_std", "win_rate", "kelly_fraction",
        "min_return", "percentile_25", "median_return", "percentile_75",
        "max_return", "min_pnl", "percentile_25_pnl", "median_pnl",
        "percentile_75_pnl", "max_pnl",
    }
    assert expected <= set(res1)
    assert res1 == res2  # deterministic under fixed seed
    assert res1["min_return"] <= res1["percentile_25"] <= res1["median_return"]
    assert res1["median_return"] <= res1["percentile_75"] <= res1["max_return"]


def test_rv_scenario_deep_itm_stock_always_wins():
    # stock bought at $50, spot $100 -> pnl = S_T - 50 > 0 for any S_T > 0
    res = rv_scenario([stock("buy", 50.0)], S=100.0, r=R, forecast_rv=0.25,
                      n_sims=2000, seed=7, as_of=AS_OF)
    assert res["win_rate"] == 1.0
    assert res["mean_pnl"] > 0


def test_rv_scenario_mean_pnl_sign_long_vs_short_call():
    # fills at 40% IV but realized vol forecast only 20%: long loses, short wins
    px = black_scholes_price(100.0, 100.0, (EXP - AS_OF).days / 365, R, 0.40)
    kwargs = dict(S=100.0, r=R, forecast_rv=0.20, n_sims=5000, seed=1, as_of=AS_OF)
    long_res = rv_scenario([call("buy", 100, px, iv=0.40)], **kwargs)
    short_res = rv_scenario([call("sell", 100, px, iv=0.40)], **kwargs)
    assert long_res["mean_pnl"] < 0
    assert short_res["mean_pnl"] > 0


# ── Hedge / fill IV --------------------------------------------------------------

def test_optimal_delta_hedge_zeroes_delta():
    legs = [call("buy", 100, 5.0, iv=0.30), put("buy", 95, 2.0, iv=0.32)]
    hedge = optimal_delta_hedge(legs, S=100.0, r=R, as_of=AS_OF)
    g = position_greeks(legs, S=100.0, r=R, as_of=AS_OF)
    assert g["delta"] + hedge == pytest.approx(0.0)
    assert hedge < 0  # net long delta -> sell shares


def test_effective_fill_iv_cost_weighted():
    legs = [call("buy", 100, 1.00, iv=0.20), call("buy", 105, 3.00, iv=0.40)]
    res = effective_fill_iv(legs, fair_iv=0.30)
    # weights 100 vs 300 -> (0.2*100 + 0.4*300) / 400 = 0.35
    assert res["effective_fill_iv"] == pytest.approx(0.35)
    assert res["fair_iv"] == pytest.approx(0.30)
    assert res["fill_minus_fair"] == pytest.approx(0.05)


def test_net_premium():
    assert net_premium(iron_condor()) == pytest.approx(-200.0)
    assert net_premium([call("buy", 100, 5.0)]) == pytest.approx(500.0)


# ── Multi-expiry (calendar) semantics ---------------------------------------------

def calendar() -> list[Leg]:
    """Long call calendar: sell near, buy far, same strike."""
    return [
        Leg("sell", "call", 100.0, EXP, 1, 3.0, 0.40),
        Leg("buy", "call", 100.0, EXP2, 1, 6.0, 0.32),
    ]


def test_calendar_win_rate_positive_uses_front_expiry():
    # Regression: win-rate was 0.0 because profitability was tested with
    # final-expiry P&L (degenerate for calendars: both legs share a strike,
    # so the position is always worth -debit at final expiry) against
    # front-expiry breakevens. The evaluation horizon must be the front expiry.
    res = analyze(calendar(), S=100.0, r=R, as_of=AS_OF)
    assert len(res["breakevens"]) == 2
    assert 0.0 < res["win_rate"] < 1.0


def test_calendar_win_rate_matches_lognormal_front_horizon():
    legs = calendar()
    res = analyze(legs, S=100.0, r=R, as_of=AS_OF)
    T_front = (EXP - AS_OF).days / 365
    sigma = effective_fill_iv(legs, 0.0)["effective_fill_iv"]
    mu = math.log(100.0) + (R - 0.5 * sigma ** 2) * T_front
    sd = sigma * math.sqrt(T_front)
    lo, hi = res["breakevens"]
    p = float(norm.cdf((math.log(hi) - mu) / sd) - norm.cdf((math.log(lo) - mu) / sd))
    assert res["win_rate"] == pytest.approx(p, abs=0.02)


def test_rv_scenario_calendar_settles_at_front_expiry():
    # Regression: settlement at max expiry made every simulated outcome equal
    # -debit (zero variance, win_rate 0). Settlement must happen at the front
    # expiry with back legs BSM-repriced, so the calendar tent survives.
    res = rv_scenario(calendar(), S=100.0, r=R, forecast_rv=0.30,
                      n_sims=5000, seed=3, as_of=AS_OF)
    assert res["T_years"] == pytest.approx((EXP - AS_OF).days / 365)
    assert res["return_std"] > 0
    assert res["max_pnl"] > 0
    assert 0.0 < res["win_rate"] < 1.0


def test_rv_scenario_single_expiry_unchanged_horizon():
    # Single-expiry positions keep the (only) expiry as the horizon.
    res = rv_scenario(iron_condor(), S=100.0, r=R, forecast_rv=0.25,
                      n_sims=1000, seed=5, as_of=AS_OF)
    assert res["T_years"] == pytest.approx((EXP - AS_OF).days / 365)
