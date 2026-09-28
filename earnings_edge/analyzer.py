"""
Core options math: Black-Scholes pricing, implied-volatility solver,
Yang-Zhang realised-volatility estimator, and IV term-structure builder.
"""

import datetime as _dtmod  # module for annotations (datetime.date)
import warnings
from datetime import date as _date_cls  # noqa: F401
from datetime import datetime, timedelta
from typing import Callable

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from .config import get_logger
from .models import AnalysisResult
from .option_math import black_scholes_price, implied_volatility  # noqa: F401  (re-export)

logger = get_logger("analyzer")

# Per-contract IVs outside this band are feed artefacts, not prices: Yahoo
# emits 1e-5 placeholders for unquoted strikes and a failed BSM inversion
# can return absurd values. Averaging one into an ATM IV halves (or blows
# up) that expiry's point and bends the whole term structure.
MIN_SANE_IV = 0.01
MAX_SANE_IV = 5.0

# Term-structure anchor for the slope/"iv30" reading (DTE, days).
TERM_ANCHOR_DTE = 45


def _finite(value) -> float | None:
    """float(value) when finite, else None."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _sane_iv(value) -> float | None:
    iv = _finite(value)
    if iv is None or not (MIN_SANE_IV <= iv <= MAX_SANE_IV):
        return None
    return iv


def _atm_iv(call_iv, put_iv) -> float | None:
    """Mean of the sane call/put IVs; one bad side falls back to the other."""
    vals = [v for v in (_sane_iv(call_iv), _sane_iv(put_iv)) if v is not None]
    return float(np.mean(vals)) if vals else None


def _leg_mid(chain: pd.DataFrame, idx) -> float | None:
    """Bid/ask mid for one contract; lastPrice when the book is empty/crossed.

    Yahoo zeroes bid/ask outside regular hours, which used to turn the
    straddle into 0 and the expected move into "N/A".
    """
    bid = _finite(chain.loc[idx, "bid"]) if "bid" in chain.columns else None
    ask = _finite(chain.loc[idx, "ask"]) if "ask" in chain.columns else None
    if bid is not None and ask is not None and bid >= 0 and ask > 0 and ask >= bid:
        return (bid + ask) / 2.0
    last = _finite(chain.loc[idx, "lastPrice"]) if "lastPrice" in chain.columns else None
    return last if last is not None and last > 0 else None


def _thin_expiries(dates: list[str], max_count: int | None) -> list[str]:
    """Reduce expiries to *max_count* while keeping near and far anchors."""
    if not max_count or len(dates) <= max_count:
        return dates
    if max_count < 2:
        return dates[:1]
    picks = sorted(
        {
            0,
            len(dates) - 1,
            *{round(i * (len(dates) - 1) / (max_count - 1)) for i in range(1, max_count - 1)},
        }
    )
    return [dates[i] for i in picks]


# ── OptionsAnalyzer class ────────────────────────────────────────────


class OptionsAnalyzer:
    """Stateless options analysis helper (volatility, term structure, recommendation)."""

    def __init__(self) -> None:
        self._simple_vol_warned = False

    # -- helpers ----------------------------------------------------------

    @staticmethod
    def filter_dates(dates: list[str], min_dte: int = 45) -> list[str]:
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
            close_vol = (log_oc**2).rolling(window).sum() / (window - 1.0)
            open_vol = (np.log(price_data["Open"] / price_data["Close"].shift(1)) ** 2).rolling(
                window
            ).sum() / (window - 1.0)
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
    def build_term_structure(days: list[int], ivs: list[float]) -> Callable[[float], float]:
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
        earnings_date: _dtmod.date | None = None,
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

            exp_dates = _thin_expiries(self.filter_dates(expiries), provider.max_expiries_hint)
            options_chains = {d: provider.option_chain(ticker, d) for d in exp_dates}

            hist = provider.history(ticker, "1d")
            if hist.empty:
                return AnalysisResult.fail(ticker, "No price data available")
            current_price = hist["Close"].iloc[-1]

            # --- realised vol -------------------------------------------------
            hist_data = provider.history(ticker, "3mo")
            hist_vol = _finite(self.yang_zhang_volatility(hist_data))
            if hist_vol is not None and hist_vol <= 0:
                hist_vol = None

            if isinstance(earnings_date, str):
                earnings_date = datetime.strptime(earnings_date, "%Y-%m-%d").date()

            today = datetime.today().date()
            atm_ivs: dict[str, float] = {}
            bid_ivs: dict[str, float] = {}
            ask_ivs: dict[str, float] = {}
            # per-expiry snapshot of the ATM pair, used to read the event expiry
            atm_rows: dict[str, dict] = {}

            for exp_date, chain in options_chains.items():
                calls, puts = chain.calls, chain.puts
                if calls.empty or puts.empty:
                    continue

                call_idx = (calls["strike"] - current_price).abs().idxmin()
                put_idx = (puts["strike"] - current_price).abs().idxmin()

                call_iv = calls.loc[call_idx, "impliedVolatility"]
                put_iv = puts.loc[put_idx, "impliedVolatility"]
                atm_iv = _atm_iv(call_iv, put_iv)
                if atm_iv is None:
                    # e.g. Polygon fallback: unsolvable EOD close for this
                    # expiry — drop the point so the term structure stays sane
                    continue
                atm_ivs[exp_date] = atm_iv

                call_bid = _finite(calls.loc[call_idx, "bid"])
                call_ask = _finite(calls.loc[call_idx, "ask"])
                put_bid = _finite(puts.loc[put_idx, "bid"])
                put_ask = _finite(puts.loc[put_idx, "ask"])

                T = (datetime.strptime(exp_date, "%Y-%m-%d").date() - today).days / 365
                strike = calls.loc[call_idx, "strike"]

                bid_iv = ask_iv = None
                if all(v is not None and v > 0 for v in (call_bid, call_ask, put_bid, put_ask)) and T > 0:
                    bid_iv = _sane_iv(implied_volatility(call_bid, current_price, strike, T, 0.04, "call"))
                    ask_iv = _sane_iv(implied_volatility(call_ask, current_price, strike, T, 0.04, "call"))
                # an unsolvable side (bid below intrinsic, stale ask) must not
                # leak NaN into the fair-value math — fall back to the mid IV
                bid_ivs[exp_date] = bid_iv if bid_iv is not None else atm_iv
                ask_ivs[exp_date] = ask_iv if ask_iv is not None else atm_iv

                call_mid = _leg_mid(calls, call_idx)
                put_mid = _leg_mid(puts, put_idx)
                atm_rows[exp_date] = {
                    "call_iv": _sane_iv(call_iv),
                    "put_iv": _sane_iv(put_iv),
                    "atm_iv": atm_iv,
                    "straddle": call_mid + put_mid if call_mid is not None and put_mid is not None else None,
                    "call_delta": calls.loc[call_idx, "delta"] if "delta" in calls.columns else None,
                    "put_delta": puts.loc[put_idx, "delta"] if "delta" in puts.columns else None,
                }

            if not atm_ivs:
                return AnalysisResult.fail(ticker, "Could not calculate ATM IVs")

            # The expected move / ATM greeks must come from the first expiry
            # that actually spans the event — matching the backtest rows
            # (polygon_backfill: expiry_gte=earnings_date) and the calendar
            # short leg. The nearest listed expiry can settle before the
            # announcement and understates the move.
            event_expiry = next(iter(atm_rows))
            if earnings_date:
                event_expiry = next(
                    (e for e in atm_rows if datetime.strptime(e, "%Y-%m-%d").date() >= earnings_date),
                    event_expiry,
                )
            event = atm_rows[event_expiry]
            straddle = event["straddle"]

            dtes = [(datetime.strptime(e, "%Y-%m-%d").date() - today).days for e in atm_ivs]
            ivs_mid = list(atm_ivs.values())
            ivs_bid = list(bid_ivs.values())
            ivs_ask = list(ask_ivs.values())

            if len(dtes) < 2 or min(dtes) >= TERM_ANCHOR_DTE:
                return AnalysisResult.fail(
                    ticker,
                    f"Insufficient term structure ({len(dtes)} expiries, nearest {min(dtes)}d)",
                )
            term_spline_mid = self.build_term_structure(dtes, ivs_mid)
            iv30 = _finite(term_spline_mid(TERM_ANCHOR_DTE))
            front_iv = _finite(term_spline_mid(min(dtes)))
            if iv30 is None or front_iv is None:
                return AnalysisResult.fail(ticker, "Could not build IV term structure")
            slope = (iv30 - front_iv) / (TERM_ANCHOR_DTE - min(dtes))

            # --- short-leg fair-value calc ------------------------------------
            short_leg_days = 4
            if earnings_date:
                valid = [
                    datetime.strptime(e, "%Y-%m-%d").date()
                    for e in exp_dates
                    if datetime.strptime(e, "%Y-%m-%d").date() >= earnings_date
                ]
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
                if T_short:
                    fair_var = (
                        sigma_long_leg_ask**2 * T_long - sigma_baseline_mid**2 * (T_long - T_short)
                    ) / T_short
                    # negative forward variance has no real fair IV
                    sigma_short_leg_fair = float(np.sqrt(fair_var)) if fair_var > 0 else None
                idx_short = min(range(len(dtes)), key=lambda i: abs(dtes[i] - short_leg_days))
                sigma_short_leg_bid = ivs_bid[idx_short]
                if sigma_short_leg_fair:
                    actual_to_fair_ratio = ((sigma_short_leg_bid / sigma_short_leg_fair) - 1) * 100

            avg_volume = hist_data["Volume"].rolling(30).mean().dropna().iloc[-1]
            expected_move_str = (
                f"{(straddle / current_price * 100):.2f}%" if straddle and straddle > 0 else "N/A"
            )

            # No realized vol (thin/NaN bars) means IV/RV is unknown. It used
            # to be reported as 9999, which sailed through every IV/RV gate —
            # including the short-straddle ones. NaN fails closed downstream.
            iv_rv = iv30 / hist_vol if hist_vol is not None else float("nan")
            recommendation = (
                "HOLD"
                if hist_vol is None
                else "BUY"
                if iv30 < hist_vol and avg_volume >= 1_500_000
                else "SELL"
                if iv30 > hist_vol * 1.2
                else "HOLD"
            )

            return AnalysisResult(
                ticker=ticker,
                current_price=current_price,
                recommendation=recommendation,
                iv30_rv30=iv_rv,
                term_slope=slope,
                term_structure_valid=slope <= -0.004,
                term_structure_tier2=-0.006 < slope <= -0.004,
                expected_move=expected_move_str,
                avg_volume_pass=avg_volume >= 1_500_000,
                sigma_baseline_1y=sigma_baseline_mid,
                sigma_short_leg_fair=sigma_short_leg_fair,
                sigma_short_leg=sigma_short_leg_bid,
                actual_to_fair_ratio=actual_to_fair_ratio,
                atm_call_delta=_finite(event["call_delta"]),
                atm_put_delta=_finite(event["put_delta"]),
                atm_iv_near=event["atm_iv"],
                atm_call_iv=event["call_iv"],
                atm_put_iv=event["put_iv"],
                rv30=hist_vol,
                hist_vol_3m=hist_vol,
            )

        except Exception as exc:
            logger.error(f"Error analyzing {ticker}: {exc}")
            return AnalysisResult.fail(ticker, f"Failed: {exc}")
