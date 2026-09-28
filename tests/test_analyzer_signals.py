"""Signal-integrity regressions for the analyzer → validator → model-row path.

Each test pins a failure mode where bad or missing market data used to turn
into a *passing* signal (sentinel ratios, NaN comparisons, placeholder IVs).
"""

from __future__ import annotations

import math
from datetime import date, datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from earnings_edge import live_signals
from earnings_edge.analyzer import OptionsAnalyzer
from earnings_edge.bot_scanner import build_calendar_model_feature_row
from earnings_edge.market_data_provider import OptionChainData
from earnings_edge.models import AnalysisResult, EarningsCandidate, TickerReport, ValidationMetrics
from earnings_edge.validator import StockValidator


def _bars(close: float = 100.0, days: int = 65, flat: bool = False) -> pd.DataFrame:
    idx = pd.date_range(end=datetime.today(), periods=days, freq="B").normalize()
    rng = np.random.default_rng(7)
    closes = np.full(days, close) if flat else close * np.exp(np.cumsum(rng.normal(0, 0.015, days)))
    opens = closes if flat else closes * (1 + rng.normal(0, 0.004, days))
    highs = closes if flat else np.maximum(opens, closes) * 1.006
    lows = closes if flat else np.minimum(opens, closes) * 0.994
    return pd.DataFrame(
        {"Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": 3_000_000.0},
        index=pd.Index(idx, name="Date"),
    )


def _row(strike, iv, bid=1.0, ask=1.2, last=1.1, delta=0.5):
    return {
        "contractSymbol": f"X{strike}",
        "strike": strike,
        "bid": bid,
        "ask": ask,
        "lastPrice": last,
        "impliedVolatility": iv,
        "openInterest": 5000,
        "volume": 100.0,
        "delta": delta,
        "inTheMoney": False,
    }


class ChainProvider:
    """Three expiries (7/30/60d) with per-expiry IV and straddle legs."""

    name = "stub"
    max_expiries_hint = None

    def __init__(self, price=100.0, bars=None, legs=None, expiries_days=(7, 30, 60)):
        self.price = price
        self._bars = bars if bars is not None else _bars(price)
        today = date.today()
        self.expiries = [(today + timedelta(days=d)).isoformat() for d in expiries_days]
        default_ivs = [0.80, 0.50, 0.40]
        # legs[expiry] = (call_row_kwargs, put_row_kwargs)
        self.legs = legs or {
            e: ({"iv": default_ivs[i % 3]}, {"iv": default_ivs[i % 3], "delta": -0.5})
            for i, e in enumerate(self.expiries)
        }

    def history(self, ticker, period="1d"):
        if period == "1d":
            return self._bars.tail(1)
        return self._bars

    def options_expiries(self, ticker):
        return list(self.expiries)

    def option_chain(self, ticker, expiry):
        call_kw, put_kw = self.legs[expiry]
        return OptionChainData(
            calls=pd.DataFrame([_row(self.price, **call_kw)]),
            puts=pd.DataFrame([_row(self.price, **put_kw)]),
            oi_available=True,
            source=self.name,
        )


# ── analyzer ──────────────────────────────────────────────────────────────


def test_missing_realized_vol_is_nan_not_9999_sentinel():
    """A NaN in the bar window used to make RV NaN → iv30_rv30 == 9999,
    which passed the validator and the live short-straddle IV/RV gates."""
    bars = _bars()
    bars.iloc[-1, bars.columns.get_loc("Open")] = np.nan
    result = OptionsAnalyzer().compute_recommendation("T", provider=ChainProvider(bars=bars))
    assert result.ok, result.error
    assert math.isnan(result.iv30_rv30)
    assert result.rv30 is None
    assert result.recommendation == "HOLD"


def test_zero_realized_vol_is_nan_not_9999_sentinel():
    result = OptionsAnalyzer().compute_recommendation("T", provider=ChainProvider(bars=_bars(flat=True)))
    assert result.ok, result.error
    assert math.isnan(result.iv30_rv30)


def test_placeholder_iv_on_one_side_does_not_halve_atm_iv():
    """Yahoo reports 1e-5 IV for unquoted strikes; averaging it in halved
    the ATM point and bent the term structure."""
    p = ChainProvider()
    front = p.expiries[0]
    p.legs[front] = ({"iv": 1e-5}, {"iv": 0.80, "delta": -0.5})
    result = OptionsAnalyzer().compute_recommendation("T", provider=p)
    assert result.ok, result.error
    assert result.atm_iv_near == pytest.approx(0.80)
    assert result.atm_call_iv is None
    assert result.atm_put_iv == pytest.approx(0.80)


def test_expected_move_uses_first_expiry_spanning_earnings():
    """The nearest expiry (7d) settles before a 10d-out announcement; the
    straddle / ATM IV must come from the 30d expiry, like the backtest rows."""
    p = ChainProvider()
    e7, e30, _ = p.expiries
    p.legs[e7] = ({"iv": 0.80, "bid": 1.0, "ask": 1.0}, {"iv": 0.80, "bid": 1.0, "ask": 1.0, "delta": -0.5})
    p.legs[e30] = (
        {"iv": 0.50, "bid": 4.0, "ask": 4.0, "delta": 0.53},
        {"iv": 0.50, "bid": 3.0, "ask": 3.0, "delta": -0.47},
    )
    earnings = date.today() + timedelta(days=10)
    result = OptionsAnalyzer().compute_recommendation("T", earnings_date=earnings, provider=p)
    assert result.ok, result.error
    assert result.expected_move == f"{7.0 / result.current_price * 100:.2f}%"
    assert result.atm_iv_near == pytest.approx(0.50)
    assert result.atm_call_delta == pytest.approx(0.53)
    assert result.atm_put_delta == pytest.approx(-0.47)


def test_expected_move_without_earnings_date_uses_nearest_expiry():
    p = ChainProvider()
    e7 = p.expiries[0]
    p.legs[e7] = ({"iv": 0.80, "bid": 1.0, "ask": 1.0}, {"iv": 0.80, "bid": 2.0, "ask": 2.0, "delta": -0.5})
    result = OptionsAnalyzer().compute_recommendation("T", provider=p)
    assert result.expected_move == f"{3.0 / result.current_price * 100:.2f}%"


def test_straddle_falls_back_to_last_price_when_book_is_empty():
    """Yahoo zeroes bid/ask outside RTH; the straddle became 0 and the
    expected move silently turned into N/A."""
    p = ChainProvider()
    e7 = p.expiries[0]
    p.legs[e7] = (
        {"iv": 0.80, "bid": 0.0, "ask": 0.0, "last": 2.5},
        {"iv": 0.80, "bid": 0.0, "ask": 0.0, "last": 1.5, "delta": -0.5},
    )
    result = OptionsAnalyzer().compute_recommendation("T", provider=p)
    assert result.expected_move == f"{4.0 / result.current_price * 100:.2f}%"


def test_single_expiry_fails_instead_of_passing_every_gate():
    """One ATM point can't define a slope; the NaN slope/ratio it produced
    compared False against every hard gate, so the name passed them all."""
    p = ChainProvider(expiries_days=(7,))
    p.legs = {p.expiries[0]: ({"iv": 0.6}, {"iv": 0.6, "delta": -0.5})}
    result = OptionsAnalyzer().compute_recommendation("T", provider=p)
    assert not result.ok
    assert "term structure" in result.error


def test_negative_forward_variance_has_no_fair_iv():
    p = ChainProvider()
    # steep contango: far IV so low the fair short-leg variance goes negative
    p.legs[p.expiries[1]] = ({"iv": 0.05}, {"iv": 0.05, "delta": -0.5})
    p.legs[p.expiries[2]] = ({"iv": 0.05}, {"iv": 0.05, "delta": -0.5})
    result = OptionsAnalyzer().compute_recommendation("T", provider=p)
    assert result.ok, result.error
    assert result.sigma_short_leg_fair is None or math.isfinite(result.sigma_short_leg_fair)
    assert result.actual_to_fair_ratio is None or math.isfinite(result.actual_to_fair_ratio)


# ── validator ─────────────────────────────────────────────────────────────


def _analysis(**over) -> AnalysisResult:
    base = dict(
        ticker="T",
        current_price=100.0,
        recommendation="HOLD",
        iv30_rv30=1.4,
        term_slope=-0.01,
        term_structure_valid=True,
        term_structure_tier2=False,
        expected_move="6.00%",
        avg_volume_pass=True,
        atm_call_delta=0.5,
        atm_put_delta=-0.5,
        atm_iv_near=0.62,
        atm_call_iv=0.63,
        atm_put_iv=0.61,
        rv30=0.30,
    )
    base.update(over)
    return AnalysisResult(**base)


def _validator(analysis: AnalysisResult) -> StockValidator:
    analyzer = OptionsAnalyzer()
    analyzer.compute_recommendation = lambda t, ed=None, provider=None: analysis
    browser = SimpleNamespace(get_win_rate=lambda t: SimpleNamespace(win_rate=0.0, quarters=0))
    return StockValidator(analyzer, browser, provider=ChainProvider())


def _candidate() -> EarningsCandidate:
    return EarningsCandidate(ticker="T", timing="Post Market", earnings_date=date.today())


@pytest.mark.parametrize("ratio", [float("nan"), 0.0, float("inf")])
def test_validator_fails_closed_on_unusable_iv_rv(ratio):
    result = _validator(_analysis(iv30_rv30=ratio)).validate(_candidate())
    assert not result.passed and not result.near_miss
    assert "IV/RV unavailable" in result.reason


def test_validator_nan_expected_move_is_not_recorded():
    result = _validator(_analysis(expected_move="nan%")).validate(_candidate())
    assert result.metrics.expected_move_pct == 0.0
    assert "Expected move" not in result.reason


def test_validator_carries_rv_and_event_ivs_into_metrics():
    result = _validator(_analysis()).validate(_candidate())
    m = result.metrics
    assert (m.rv30, m.atm_iv_near, m.atm_call_iv, m.atm_put_iv) == (0.30, 0.62, 0.63, 0.61)


# ── live model row ────────────────────────────────────────────────────────


def _report(**metric_over) -> TickerReport:
    m = ValidationMetrics(
        price=100.0,
        iv_rv_ratio=2.0,
        sigma_short_leg=0.80,
        expected_move_pct=6.0,
        expected_move_dollars=6.0,
    )
    for k, v in metric_over.items():
        setattr(m, k, v)
    return TickerReport(
        ticker="T", passed=True, tier=1, near_miss=False, reason="", metrics=m, earnings_date=date.today()
    )


def test_feature_row_uses_real_rv_and_event_ivs_like_training_rows():
    row = build_calendar_model_feature_row(
        _report(rv30=0.31, atm_iv_near=0.62, atm_call_iv=0.63, atm_put_iv=0.61), {"strike": 100.0}
    )
    assert row["rv30"] == row["hist_vol_3m"] == 0.31
    assert (row["atm_iv_near"], row["atm_call_iv"], row["atm_put_iv"]) == (0.62, 0.63, 0.61)
    assert row["sigma_short_leg"] == 0.80


def test_feature_row_falls_back_to_estimates_without_carried_metrics():
    row = build_calendar_model_feature_row(_report(), {"strike": 100.0})
    assert row["rv30"] == pytest.approx(0.40)  # 0.80 / 2.0
    assert row["atm_iv_near"] == row["atm_call_iv"] == row["atm_put_iv"] == 0.80


# ── live straddle gates ───────────────────────────────────────────────────


def test_live_straddle_gates_reject_sentinel_iv_rv():
    today = date.today()
    base = {
        "ticker": "T",
        "_earnings": today + timedelta(days=1),
        "scan_timestamp": today.isoformat(),
        "strike": 100.0,
        "price": 100.0,
        "near_expiry": (today + timedelta(days=4)).isoformat(),
        "expected_move_pct": 8.0,
        "expected_move_dollars": 8.0,
    }
    df = pd.DataFrame(
        [
            {**base, "ticker": "SENTINEL", "iv_rv_ratio": 9999.0},
            {**base, "ticker": "NAN", "iv_rv_ratio": float("nan")},
            {**base, "ticker": "GOOD", "iv_rv_ratio": 1.6},
        ]
    )
    for strategy in ("vol_risk_premium", "short_straddle"):
        tickers = [t.ticker for t in live_signals.build_live_trades(df, strategy)]
        assert tickers == ["GOOD"], strategy
