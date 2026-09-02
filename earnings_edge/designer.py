"""Position Designer — multi-leg options position analytics.

Replicates oquants.com's Position Designer (`/dashboard/designer/[symbol]`):
leg table with fill price/IV, structure tagging (Direction x Risk x Vol
Exposure), P&L at expiry and at any pre-expiry date with IV scenario shocks
(single / per-expiry / per-leg — the "Link sliders" / "Per Expiry" / "Per
Position" modes), summary metrics (max profit/loss, breakevens, greeks,
win rate), RV forecast-simulation and the optimal delta hedge.

Conventions:
- All P&L and greeks are position-level dollars: option legs use the 100x
  equity multiplier, stock legs are 1x.
- Vega is per 1.00 of vol (0.01 = one vol point), theta per year,
  rho per 1.00 of rate — matching option_math.py's BSM convention.
- Time is ACT/365 years. `as_of` defaults to today; pass it explicitly in
  tests for determinism.

Win-rate method (`analyze`): terminal spot is lognormal with blended sigma
(position-cost-weighted average of leg fill IVs), drift r, horizon = max
leg expiry. Breakevens partition [0, inf) into intervals; the win rate is
the summed probability of the profitable intervals.

RV scenario (`rv_scenario`): terminal spot is simulated lognormal with
sigma = forecast_rv over the position's max expiry (drift r), each leg
settled at its expiry payoff on the simulated terminal spot. Returns are
P&L divided by the risk basis: abs(net premium) when nonzero, else
abs(max loss) when finite, else spot. Kelly fraction uses the continuous
approximation f* = mean_return / return variance.

All functions are pure — no I/O — so the whole module is unit-testable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Optional, Union

import numpy as np
from scipy.stats import norm

from .option_math import (
    black_scholes_delta,
    black_scholes_gamma,
    black_scholes_price,
    black_scholes_theta,
    black_scholes_vega,
)

RISK_FREE_RATE = 0.045  # match fwd_factor / polygon_backfill convention

OPTION_MULTIPLIER = 100
DAYS_PER_YEAR = 365
MIN_IV = 1e-4  # floor for shocked IVs so BSM stays defined

# iv_shock: single additive shock, per-expiry dict, or per-leg-index dict
IvShock = Union[float, dict, None]


# ── Legs and positions ---------------------------------------------------------

@dataclass(frozen=True)
class Leg:
    """One position leg. `strike` is 0 for stock, `price`/`iv` are fill values."""

    action: str        # "buy" | "sell"
    kind: str          # "call" | "put" | "stock"
    strike: float      # 0 for stock
    expiry: date
    quantity: int
    price: float       # fill price per share
    iv: float          # fill IV, decimal (unused for stock)

    def __post_init__(self) -> None:
        if self.action not in ("buy", "sell"):
            raise ValueError(f"action must be 'buy' or 'sell', got {self.action!r}")
        if self.kind not in ("call", "put", "stock"):
            raise ValueError(f"kind must be 'call', 'put' or 'stock', got {self.kind!r}")
        if self.quantity <= 0:
            raise ValueError("quantity must be positive (sign comes from action)")
        if self.kind == "stock" and self.strike != 0:
            raise ValueError("stock legs must have strike 0")

    @property
    def signed_quantity(self) -> int:
        """+qty for buys, -qty for sells."""
        return self.quantity if self.action == "buy" else -self.quantity

    @property
    def is_option(self) -> bool:
        return self.kind in ("call", "put")

    @property
    def multiplier(self) -> int:
        return OPTION_MULTIPLIER if self.is_option else 1


Position = list[Leg]


def years_to_expiry(leg: Leg, as_of: date) -> float:
    """ACT/365 years from *as_of* to the leg's expiry (floored at 0)."""
    return max((leg.expiry - as_of).days, 0) / DAYS_PER_YEAR


def net_premium(legs: Position) -> float:
    """Net premium paid in dollars (positive = net debit, negative = credit)."""
    return float(sum(l.signed_quantity * l.price * l.multiplier for l in legs))


# ── Structure tagging ----------------------------------------------------------

def tag_structure(legs: Position, S: Optional[float] = None, r: float = RISK_FREE_RATE,
                  as_of: Optional[date] = None) -> dict:
    """Classify Direction x Risk x Vol Exposure like oquants' structure tags.

    Direction: signed leg count weighted by directional sign (+1 calls/stock,
    -1 puts) — or net position delta when S is given. Risk: Undefined when
    any expiry has more short than long contracts of a kind (naked shorts).
    Vol Exposure: sign of position vega when S is given, else net signed
    option quantity, ties broken by net option premium (paid -> Long).
    """
    if S is not None:
        g = position_greeks(legs, S, r, as_of=as_of)
        dir_score = g["delta"]
        vol_score = g["vega"]
    else:
        dir_score = sum(l.signed_quantity * (-1 if l.kind == "put" else 1) for l in legs)
        vol_score = sum(l.signed_quantity for l in legs if l.is_option)
        if vol_score == 0:
            option_premium = sum(l.signed_quantity * l.price for l in legs if l.is_option)
            vol_score = option_premium  # paid for options -> long vol

    risk = "Defined"
    by_kind_expiry: dict[tuple[str, date], int] = {}
    for leg in legs:
        if leg.is_option:
            key = (leg.kind, leg.expiry)
            by_kind_expiry[key] = by_kind_expiry.get(key, 0) + leg.signed_quantity
    if any(net < 0 for net in by_kind_expiry.values()):
        risk = "Undefined"

    return {
        "direction": "Bullish" if dir_score > 0 else "Bearish" if dir_score < 0 else "Neutral",
        "risk": risk,
        "vol_exposure": "Long" if vol_score > 0 else "Short" if vol_score < 0 else "Neutral",
    }


# ── Payoffs and repricing -------------------------------------------------------

def _expiry_value_per_share(leg: Leg, S: np.ndarray) -> np.ndarray:
    """Terminal value per share at spot S (vectorized)."""
    if leg.kind == "call":
        return np.maximum(S - leg.strike, 0.0)
    if leg.kind == "put":
        return np.maximum(leg.strike - S, 0.0)
    return np.asarray(S, dtype=float)


def _value_per_share(leg: Leg, S: np.ndarray, T: float, r: float, sigma: float) -> np.ndarray:
    """BSM value per share at T years remaining (intrinsic at/below T=0)."""
    if not leg.is_option:
        return np.asarray(S, dtype=float)
    if T <= 0:
        return _expiry_value_per_share(leg, S)
    price = black_scholes_price(float(S), leg.strike, T, r, sigma, leg.kind) if S.ndim == 0 else None
    if price is not None:
        return np.asarray(price)
    return np.array([black_scholes_price(float(s), leg.strike, T, r, sigma, leg.kind)
                     for s in np.nditer(S)])


def _position_value(legs: Position, S_grid, value_fn) -> np.ndarray:
    """Sum over legs of signed_qty * multiplier * (value - fill price)."""
    S = np.asarray(S_grid, dtype=float)
    pnl = np.zeros_like(S)
    for leg in legs:
        pnl = pnl + leg.signed_quantity * leg.multiplier * (value_fn(leg, S) - leg.price)
    return pnl


def pnl_at_expiry(legs: Position, S_grid) -> np.ndarray:
    """P&L at expiry in dollars: terminal payoff minus net premium paid."""
    return _position_value(legs, S_grid, _expiry_value_per_share)


def pnl_at_front_expiry(legs: Position, S_grid, r: float, as_of: date) -> np.ndarray:
    """P&L at the earliest option expiry in the position (evaluating back-month options via BSM)."""
    options = [l for l in legs if l.is_option]
    if not options:
        return pnl_at_expiry(legs, S_grid)
        
    front_expiry = min(l.expiry for l in options)
    
    def value_fn(leg: Leg, S: np.ndarray) -> np.ndarray:
        if not leg.is_option:
            return np.asarray(S, dtype=float)
        
        # Time remaining for this leg when we reach the front expiry
        T = max((leg.expiry - front_expiry).days, 0) / DAYS_PER_YEAR
        
        if T <= 0:
            return _expiry_value_per_share(leg, S)
            
        sigma = max(leg.iv, MIN_IV)
        return _value_per_share(leg, S, T, r, sigma)

    return _position_value(legs, S_grid, value_fn)


def _resolve_shock(leg: Leg, idx: int, iv_shock: IvShock) -> float:
    """Additive vol shock for one leg: single float, per-expiry or per-leg dict."""
    if iv_shock is None:
        return 0.0
    if isinstance(iv_shock, (int, float)):
        return float(iv_shock)
    if leg.expiry in iv_shock:  # "Per Expiry" mode
        return float(iv_shock[leg.expiry])
    if idx in iv_shock:  # "Per Position" (per-leg) mode
        return float(iv_shock[idx])
    return 0.0


def pnl_at_date(legs: Position, S_grid, as_of_T: float, r: float = RISK_FREE_RATE,
                iv_shock: IvShock = None) -> np.ndarray:
    """Pre-expiry P&L in dollars: BSM-reprice every leg at *as_of_T* years
    remaining, using each leg's fill IV plus its resolved scenario shock."""
    def value_fn(leg: Leg, S: np.ndarray) -> np.ndarray:
        sigma = max(leg.iv + _resolve_shock(leg, legs.index(leg) if leg in legs else 0, iv_shock), MIN_IV)
        return _value_per_share(leg, S, as_of_T, r, sigma)

    # resolve per-leg shocks by index without relying on identity lookup
    def value_fn_idx(pair) -> np.ndarray:
        idx, leg = pair
        sigma = max(leg.iv + _resolve_shock(leg, idx, iv_shock), MIN_IV)
        return _value_per_share(leg, S_holder[0], as_of_T, r, sigma)

    S = np.asarray(S_grid, dtype=float)
    pnl = np.zeros_like(S)
    S_holder = [S]
    for idx, leg in enumerate(legs):
        pnl = pnl + leg.signed_quantity * leg.multiplier * (value_fn_idx((idx, leg)) - leg.price)
    return pnl


# ── Greeks ----------------------------------------------------------------------

def _leg_times(legs: Position, T: Optional[float], as_of: Optional[date]) -> list[float]:
    """Per-leg years to expiry: shared *T* if given, else from *as_of*."""
    if T is not None:
        return [T] * len(legs)
    ref = as_of or date.today()
    return [years_to_expiry(leg, ref) for leg in legs]


def position_greeks(legs: Position, S: float, r: float = RISK_FREE_RATE,
                    T: Optional[float] = None, as_of: Optional[date] = None) -> dict:
    """Net position greeks at (S, r): delta/gamma/theta/vega in dollar terms.

    Options use BSM at each leg's fill IV (x100 multiplier); stock legs have
    delta 1 and zero gamma/theta/vega. Expired legs contribute only their
    intrinsic delta.
    """
    totals = {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0}
    for leg, T_leg in zip(legs, _leg_times(legs, T, as_of)):
        sign_mult = leg.signed_quantity * leg.multiplier
        if not leg.is_option:
            totals["delta"] += leg.signed_quantity * 1.0
            continue
        if T_leg <= 0 or leg.iv <= 0:
            if leg.kind == "call":
                totals["delta"] += sign_mult * (1.0 if S > leg.strike else 0.0)
            else:
                totals["delta"] += sign_mult * (-1.0 if S < leg.strike else 0.0)
            continue
        totals["delta"] += sign_mult * black_scholes_delta(S, leg.strike, T_leg, r, leg.iv, leg.kind)
        totals["gamma"] += sign_mult * black_scholes_gamma(S, leg.strike, T_leg, r, leg.iv, leg.kind)
        totals["theta"] += sign_mult * black_scholes_theta(S, leg.strike, T_leg, r, leg.iv, leg.kind) / 365.0
        totals["vega"] += sign_mult * black_scholes_vega(S, leg.strike, T_leg, r, leg.iv, leg.kind)
    return totals


def optimal_delta_hedge(legs: Position, S: float, r: float = RISK_FREE_RATE,
                        as_of: Optional[date] = None) -> float:
    """Signed number of underlying shares that zeroes position delta."""
    return -position_greeks(legs, S, r, as_of=as_of)["delta"]


def effective_fill_iv(legs: Position, fair_iv: float) -> dict:
    """Position-cost-weighted average fill IV vs a *fair_iv* reference.

    Weights are abs(signed_qty * fill price * multiplier) over option legs —
    the dollars of option premium each leg contributes.
    """
    weights = [abs(l.signed_quantity * l.price * l.multiplier) for l in legs if l.is_option]
    ivs = [l.iv for l in legs if l.is_option]
    total = sum(weights)
    eff = sum(w * iv for w, iv in zip(weights, ivs)) / total if total > 0 else math.nan
    return {
        "effective_fill_iv": eff,
        "fair_iv": fair_iv,
        "fill_minus_fair": eff - fair_iv,
    }


# ── Summary analysis -------------------------------------------------------------

def _right_tail_slope(legs: Position) -> float:
    """dP&L/dS as S -> infinity (per $1 of spot)."""
    return float(sum(l.signed_quantity * l.multiplier
                     for l in legs if l.kind in ("call", "stock")))


def _breakevens(legs: Position, grid: np.ndarray, pnl: np.ndarray) -> list[float]:
    """Sign changes on the grid, linearly interpolated."""
    bes: list[float] = []
    sign = np.sign(pnl)
    for i in range(len(grid) - 1):
        if sign[i] == 0:
            bes.append(float(grid[i]))
        elif sign[i] * sign[i + 1] < 0:
            frac = pnl[i] / (pnl[i] - pnl[i + 1])
            bes.append(float(grid[i] + frac * (grid[i + 1] - grid[i])))
    # dedupe roots that sit on grid points
    out: list[float] = []
    for be in bes:
        if not out or abs(be - out[-1]) > 1e-9:
            out.append(be)
    return out


def _front_expiry(legs: Position) -> Optional[date]:
    """Earliest option expiry in the position, or None for stock-only."""
    expiries = [l.expiry for l in legs if l.is_option]
    return min(expiries) if expiries else None


def _evaluation_horizon(legs: Position, as_of: date) -> float:
    """Years to the evaluation horizon: the front (earliest) option expiry.

    Multi-expiry structures (calendars, diagonals) must be evaluated at the
    front expiry — at the final expiry all legs settle at intrinsic and a
    same-strike calendar degenerates to a constant -debit. Stock-only
    positions fall back to the max leg expiry.
    """
    front = _front_expiry(legs)
    if front is not None:
        return max((front - as_of).days, 0) / DAYS_PER_YEAR
    return max(years_to_expiry(l, as_of) for l in legs)


def _win_rate_at_expiry(legs: Position, S: float, r: float, breakevens: list[float],
                        as_of: date, grid_hi: float) -> float:
    """Probability of profit at the evaluation horizon under a lognormal spot.

    Evaluated at the front expiry with back legs BSM-repriced (same curve the
    breakevens come from). Blended sigma = position-cost-weighted average leg
    fill IV, drift r. See module docstring for the full method.
    """
    option_legs = [l for l in legs if l.is_option]
    if not option_legs:
        return math.nan
    T_win = _evaluation_horizon(legs, as_of)
    if T_win <= 0:
        pnl_now = float(pnl_at_expiry(legs, np.array([S]))[0])
        return 1.0 if pnl_now > 0 else 0.0
    eff = effective_fill_iv(legs, fair_iv=0.0)["effective_fill_iv"]
    sigma = max(eff, MIN_IV)

    def value(s_arr: np.ndarray) -> np.ndarray:
        return pnl_at_front_expiry(legs, s_arr, r, as_of)

    bounds = [0.0] + breakevens + [math.inf]
    mu = math.log(S) + (r - 0.5 * sigma ** 2) * T_win
    sd = sigma * math.sqrt(T_win)

    def prob(a: float, b: float) -> float:
        lo = 0.0 if a <= 0 else float(norm.cdf((math.log(a) - mu) / sd))
        hi = 1.0 if math.isinf(b) else float(norm.cdf((math.log(b) - mu) / sd))
        return hi - lo

    win = 0.0
    for a, b in zip(bounds, bounds[1:]):
        if math.isinf(b):
            profitable = _right_tail_slope(legs) > 0
            if not profitable and b == bounds[-1] and a > 0:
                test = value(np.array([max(grid_hi, a * 1.5)]))[0]
                profitable = test > 0
        else:
            test_s = a + (b - a) / 2 if a > 0 else b / 2
            profitable = value(np.array([test_s]))[0] > 0
        if profitable:
            win += prob(a, b)
    return float(win)


def analyze(legs: Position, S: float, r: float = RISK_FREE_RATE,
            as_of: Optional[date] = None) -> dict:
    """Summary metrics: max profit/loss, breakevens, greeks, premium, win rate.

    Max profit/loss and breakevens come from a fine front-expiry P&L grid
    (0 to 2x the highest strike/spot; back legs BSM-repriced); unbounded
    values are reported as +/-math.inf from the right-tail payoff slope.
    Win rate is the probability of profit at that same horizon.
    """
    ref = as_of or date.today()
    strikes = [l.strike for l in legs if l.is_option] + [S]
    lo = max(S * 1e-4, 1e-6)
    hi = 2.0 * max(strikes)
    grid = np.linspace(lo, hi, 4001)
    pnl = pnl_at_front_expiry(legs, grid, r, ref)

    slope = _right_tail_slope(legs)
    max_profit = math.inf if slope > 1e-12 else float(np.max(pnl))
    max_loss = -math.inf if slope < -1e-12 else float(np.min(pnl))
    breakevens = _breakevens(legs, grid, pnl)

    return {
        "max_profit": max_profit,
        "max_loss": max_loss,
        "breakevens": breakevens,
        "greeks": position_greeks(legs, S, r, as_of=ref),
        "net_premium": net_premium(legs),
        "win_rate": _win_rate_at_expiry(legs, S, r, breakevens, ref, hi),
        "structure": tag_structure(legs),
    }


# ── RV forecast simulation --------------------------------------------------------

def rv_scenario(legs: Position, S: float, r: float, forecast_rv: float,
                n_sims: int = 20000, seed: Optional[int] = None,
                as_of: Optional[date] = None) -> dict:
    """Monte-Carlo terminal P&L distribution at a forecast realized vol.

    Terminal spot is lognormal with sigma = forecast_rv (drift r) over the
    position's evaluation horizon — the front (earliest) option expiry.
    Positions are settled on that horizon: expiring legs at intrinsic, back
    legs BSM-repriced at their fill IV (same valuation as
    ``pnl_at_front_expiry``). Settling everything at the max expiry would
    make same-strike calendars degenerate (constant -debit, zero variance).
    Return basis = abs(net premium), else abs(max loss) when finite, else
    spot. Deterministic under a fixed seed.
    """
    ref = as_of or date.today()
    T = _evaluation_horizon(legs, ref)
    if T <= 0:
        raise ValueError("all legs expired — nothing to simulate")
    if forecast_rv <= 0:
        raise ValueError("forecast_rv must be positive")

    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_sims)
    s_T = S * np.exp((r - 0.5 * forecast_rv ** 2) * T + forecast_rv * math.sqrt(T) * z)
    pnl = pnl_at_front_expiry(legs, s_T, r, ref)

    premium = net_premium(legs)
    if premium != 0:
        basis = abs(premium)
    else:
        summary = analyze(legs, S, r, as_of=ref)
        basis = abs(summary["max_loss"]) if math.isfinite(summary["max_loss"]) else S
    rets = pnl / basis

    mean_return = float(np.mean(rets))
    var_return = float(np.var(rets))
    kelly = mean_return / var_return if var_return > 0 else 0.0

    return {
        "mean_return": mean_return,
        "mean_pnl": float(np.mean(pnl)),
        "return_std": float(np.std(rets)),
        "win_rate": float(np.mean(pnl > 0)),
        "kelly_fraction": float(kelly),
        "min_return": float(np.min(rets)),
        "percentile_25": float(np.percentile(rets, 25)),
        "median_return": float(np.percentile(rets, 50)),
        "percentile_75": float(np.percentile(rets, 75)),
        "max_return": float(np.max(rets)),
        "min_pnl": float(np.min(pnl)),
        "percentile_25_pnl": float(np.percentile(pnl, 25)),
        "median_pnl": float(np.percentile(pnl, 50)),
        "percentile_75_pnl": float(np.percentile(pnl, 75)),
        "max_pnl": float(np.max(pnl)),
        "basis": float(basis),
        "n_sims": n_sims,
        "forecast_rv": forecast_rv,
        "T_years": T,
    }
