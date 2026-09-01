"""
Core options math: Black-Scholes pricing, implied-volatility solver,
Yang-Zhang realised-volatility estimator, and IV term-structure builder.
"""

import logging
import warnings
from datetime import datetime, timedelta
from typing import Callable, List, Optional

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from .config import get_logger
from .models import AnalysisResult
from .option_math import black_scholes_price, implied_volatility  # noqa: F401  (re-export)

logger = get_logger("analyzer")


def _thin_expiries(dates: List[str], max_count: Optional[int]) -> List[str]:
    """Reduce expiries to *max_count* while keeping near and far anchors."""
    if not max_count or len(dates) <= max_count:
        return dates
    if max_count < 2:
        return dates[:1]
    picks = sorted({
        0,
        len(dates) - 1,
        *{
            round(i * (len(dates) - 1) / (max_count - 1))
            for i in range(1, max_count - 1)
        },
    })
    return [dates[i] for i in picks]


# ── OptionsAnalyzer class ────────────────────────────────────────────

class OptionsAnalyzer:
    """Stateless options analysis helper (volatility, term structure, recommendation)."""

    def __init__(self) -> None:
        self._simple_vol_warned = False

    # -- helpers ----------------------------------------------------------

    @staticmethod
    def filter_dates(dates: List[str], min_dte: int = 45) -> List[str]:
        """Return expiration dates ≥ *min_dte* days out (plus the first one before)."""
        today = datetime.today().date()
        cutoff = today + timedelta(days=min_dte)
        sorted_dates = sorted(datetime.strptime(d, "%Y-%m-%d").date() for d in dates)

        arr: list = []
        for i, d in enumerate(sorted_dates):
            if d >= cutoff:
                arr = [x.strftime("%Y-%m-%d") for x in sorted_dates[: i + 1]]
                break

        if arr:
            if arr[0] == today.strftime("%Y-%m-%d") and len(arr) > 1:
                return arr[1:]
            return arr
        return [d.strftime("%Y-%m-%d") for d in sorted_dates]

    def yang_zhang_volatility(
        self,
        price_data: pd.DataFrame,
        window: int = 30,
        trading_periods: int = 252,
        return_last_only: bool = True,
    ) -> float:
        """Yang-Zhang drift-independent volatility estimator."""
        try:
            log_ho = np.log(price_data["High"] / price_data["Open"])
            log_lo = np.log(price_data["Low"] / price_data["Open"])
            log_co = np.log(price_data["Close"] / price_data["Open"])
            log_oc = np.log(price_data["Open"] / price_data["Close"].shift(1))

            rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)
            close_vol = (log_oc ** 2).rolling(window).sum() / (window - 1.0)
            open_vol = (np.log(price_data["Open"] / price_data["Close"].shift(1)) ** 2).rolling(window).sum() / (window - 1.0)
            window_rs = rs.rolling(window).sum() / (window - 1.0)

            k = 0.34 / (1.34 + (window + 1) / (window - 1))
            result = np.sqrt(open_vol + k * close_vol + (1 - k) * window_rs) * np.sqrt(trading_periods)

            return result.iloc[-1] if return_last_only else result.dropna()
        except Exception as exc:
            if not self._simple_vol_warned:
                warnings.warn(f"Yang-Zhang failed: {exc}. Falling back to simple vol.")
                self._simple_vol_warned = True
            return self._simple_volatility(price_data, window, trading_periods, return_last_only)

    @staticmethod
    def _simple_volatility(
        price_data: pd.DataFrame,
        window: int = 30,
        trading_periods: int = 252,
        return_last_only: bool = True,
    ) -> float:
        try:
            vol = price_data["Close"].pct_change().rolling(window).std() * np.sqrt(trading_periods)
            return vol.iloc[-1] if return_last_only else vol
        except Exception as exc:
            warnings.warn(f"Simple vol failed: {exc}")
            return np.nan

    @staticmethod
    def build_term_structure(days: List[int], ivs: List[float]) -> Callable[[float], float]:
        """Linear interpolation of DTE → ATM IV."""
        try:
            d = np.array(days)
            v = np.array(ivs)
            order = d.argsort()
            d, v = d[order], v[order]
            spline = interp1d(d, v, kind="linear", fill_value="extrapolate")

            def _spline(dte: float) -> float:
                if dte < d[0]:
                    return float(v[0])
                if dte > d[-1]:
                    return float(v[-1])
                return float(spline(dte))

            return _spline
        except Exception as exc:
            warnings.warn(f"Term-structure build failed: {exc}")
            return lambda _: np.nan

    # -- main entry point -------------------------------------------------

    def compute_recommendation(
        self,
        ticker: str,
        earnings_date: Optional[datetime.date] = None,
        provider=None,
    ) -> AnalysisResult:
        """Full options analysis for a single ticker.

        *provider* defaults to the global resilient market-data provider
        (Yahoo primary, Polygon fallback — see market_data_provider.py).
        """
        try:
            if provider is None:
                from .market_data_provider import get_provider
                provider = get_provider()

            ticker = ticker.strip().upper()
            if not ticker:
                return AnalysisResult.fail("", "No symbol provided.")

            expiries = provider.options_expiries(ticker)
            if not expiries:
                return AnalysisResult.fail(ticker, f"No options for {ticker}.")

            exp_dates = _thin_expiries(
                self.filter_dates(expiries), provider.max_expiries_hint
            )
            options_chains = {d: provider.option_chain(ticker, d) for d in exp_dates}

            hist = provider.history(ticker, "1d")
            if hist.empty:
                return AnalysisResult.fail(ticker, "No price data available")
            current_price = hist["Close"].iloc[-1]

            # --- realised vol -------------------------------------------------
            hist_data = provider.history(ticker, "3mo")
            hist_vol = self.yang_zhang_volatility(hist_data)

            today = datetime.today().date()
            atm_ivs: dict[str, float] = {}
            bid_ivs: dict[str, float] = {}
            ask_ivs: dict[str, float] = {}
            straddle = None
            atm_call_delta = None
            atm_put_delta = None
            nearest_call_iv = None
            nearest_put_iv = None

            for i, (exp_date, chain) in enumerate(options_chains.items()):
                calls, puts = chain.calls, chain.puts
                if calls.empty or puts.empty:
                    continue

                call_idx = (calls["strike"] - current_price).abs().idxmin()
                put_idx = (puts["strike"] - current_price).abs().idxmin()

                call_iv = calls.loc[call_idx, "impliedVolatility"]
                put_iv = puts.loc[put_idx, "impliedVolatility"]
                atm_iv = (call_iv + put_iv) / 2.0
                if not np.isfinite(atm_iv):
                    # e.g. Polygon fallback: unsolvable EOD close for this
                    # expiry — drop the point so the term structure stays sane
                    continue
                atm_ivs[exp_date] = atm_iv

                call_bid = calls.loc[call_idx, "bid"]
                call_ask = calls.loc[call_idx, "ask"]
                put_bid = puts.loc[put_idx, "bid"]
                put_ask = puts.loc[put_idx, "ask"]

                T = (datetime.strptime(exp_date, "%Y-%m-%d").date() - today).days / 365
                strike = calls.loc[call_idx, "strike"]

                if call_bid > 0 and call_ask > 0 and put_bid > 0 and put_ask > 0:
                    bid_iv = implied_volatility(call_bid, current_price, strike, T, 0.04, "call")
                    ask_iv = implied_volatility(call_ask, current_price, strike, T, 0.04, "call")
                else:
                    bid_iv = atm_iv
                    ask_iv = atm_iv

                bid_ivs[exp_date] = bid_iv
                ask_ivs[exp_date] = ask_iv

                if i == 0:
                    nearest_call_iv = call_iv
                    nearest_put_iv = put_iv
                    call_mid = (calls.loc[call_idx, "bid"] + calls.loc[call_idx, "ask"]) / 2
                    put_mid = (puts.loc[put_idx, "bid"] + puts.loc[put_idx, "ask"]) / 2
                    straddle = call_mid + put_mid

                    if "delta" in calls.columns:
                        atm_call_delta = calls.loc[call_idx, "delta"]
                    if "delta" in puts.columns:
                        atm_put_delta = puts.loc[put_idx, "delta"]

            if not atm_ivs:
                return AnalysisResult.fail(ticker, "Could not calculate ATM IVs")

            dtes = [(datetime.strptime(e, "%Y-%m-%d").date() - today).days for e in atm_ivs]
            ivs_mid = list(atm_ivs.values())
            ivs_bid = list(bid_ivs.values())
            ivs_ask = list(ask_ivs.values())

            term_spline_mid = self.build_term_structure(dtes, ivs_mid)
            iv30 = term_spline_mid(45)
            slope = (term_spline_mid(45) - term_spline_mid(min(dtes))) / (45 - min(dtes))

            # --- short-leg fair-value calc ------------------------------------
            short_leg_days = 4
            if earnings_date:
                if isinstance(earnings_date, str):
                    earnings_date = datetime.strptime(earnings_date, "%Y-%m-%d").date()
                valid = [datetime.strptime(e, "%Y-%m-%d").date() for e in exp_dates
                         if datetime.strptime(e, "%Y-%m-%d").date() >= earnings_date]
                if valid:
                    short_leg_days = max(1, min((e - today).days for e in valid))
                else:
                    short_leg_days = max(1, min((earnings_date - today).days, 35))

            sigma_baseline_mid = min(ivs_mid) if ivs_mid else None
            sigma_short_leg_fair = None
            sigma_short_leg_bid = None
            actual_to_fair_ratio = None

            if dtes and sigma_baseline_mid is not None:
                idx_long = min(range(len(dtes)), key=lambda i: abs(dtes[i] - 30))
                T_long = dtes[idx_long]
                sigma_long_leg_ask = ivs_ask[idx_long]
                T_short = short_leg_days
                denom = T_short
                if denom:
                    sigma_short_leg_fair = np.sqrt(
                        (sigma_long_leg_ask ** 2 * T_long - sigma_baseline_mid ** 2 * (T_long - T_short)) / T_short
                    )
                idx_short = min(range(len(dtes)), key=lambda i: abs(dtes[i] - short_leg_days))
                sigma_short_leg_bid = ivs_bid[idx_short]
                if sigma_short_leg_fair:
                    actual_to_fair_ratio = ((sigma_short_leg_bid / sigma_short_leg_fair) - 1) * 100

            avg_volume = hist_data["Volume"].rolling(30).mean().dropna().iloc[-1]
            expected_move_str = f"{(straddle / current_price * 100):.2f}%" if straddle else "N/A"

            recommendation = (
                "BUY" if iv30 < hist_vol and avg_volume >= 1_500_000
                else "SELL" if iv30 > hist_vol * 1.2
                else "HOLD"
            )

            return AnalysisResult(
                ticker=ticker,
                current_price=current_price,
                recommendation=recommendation,
                iv30_rv30=iv30 / hist_vol if hist_vol > 0 else 9999,
                term_slope=slope,
                term_structure_valid=slope <= -0.004,
                term_structure_tier2=-0.006 < slope <= -0.004,
                expected_move=expected_move_str,
                avg_volume_pass=avg_volume >= 1_500_000,
                sigma_baseline_1y=sigma_baseline_mid,
                sigma_short_leg_fair=sigma_short_leg_fair if sigma_short_leg_fair is not None and not np.isnan(sigma_short_leg_fair) else None,
                sigma_short_leg=sigma_short_leg_bid,
                actual_to_fair_ratio=actual_to_fair_ratio,
                atm_call_delta=atm_call_delta,
                atm_put_delta=atm_put_delta,
                atm_iv_near=list(atm_ivs.values())[0] if atm_ivs else None,
                atm_call_iv=nearest_call_iv,
                atm_put_iv=nearest_put_iv,
                rv30=hist_vol,
                hist_vol_3m=hist_vol,
            )

        except Exception as exc:
            logger.error(f"Error analyzing {ticker}: {exc}")
            return AnalysisResult.fail(ticker, f"Failed: {exc}")
