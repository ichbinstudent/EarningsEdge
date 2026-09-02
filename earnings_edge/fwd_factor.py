"""Forward-factor calendar: target-price math and limit-ladder logic.

Strategy (David, 2026-07-25): buy the earnings call-calendar as cheaply as
possible. The entry condition is expressed in vol space — the implied
earnings-event move must exceed the ticker's RMS realized event move by a
premium p (start p=25%, concede to p=20%) — and INVERTED into a max debit:

    sigma_1*^2 * T1 = ((1+p) * m_hist)^2 + sigma_fwd^2 * (T1 - tau)
    c1*   = BS(sigma_1*, S, K, T1)
    D*(p) = far_price - c1*          (max debit consistent with premium p)

Cheaper calendar = richer near leg = higher implied event premium, so
"buy below D*(20%)" IS the forward-factor signal. The ladder starts at
D*(25%) (cheapest) and concedes one tick every 15 minutes from 14:00 ET,
hard-capped at D*(20%). Day orders; unfilled ladders die at the close.

All functions are pure — no I/O — so the whole module is unit-testable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from typing import Optional

from .option_math import black_scholes_price, implied_volatility

RISK_FREE_RATE = 0.045  # match polygon_backfill convention

ET = timezone(timedelta(hours=-4), name="EDT")  # fixed EDT; close enough for the 14:00-16:00 window


# ── OCC symbol helpers (pure string ops) ─────────────────────────────

def occ_symbol(ticker: str, expiry: date, strike: float, option_type: str = "call") -> str:
    """Build the 21-char OCC option symbol (matches alpaca_bridge convention)."""
    root = ticker.upper()
    date_code = expiry.strftime("%y%m%d")
    type_code = "C" if option_type.lower().startswith("c") else "P"
    return f"{root}{date_code}{type_code}{int(round(strike * 1000)):08d}"


def occ_parse(symbol: str) -> dict:
    """Parse an OCC option symbol into root/expiry/type/strike."""
    root = symbol[:-15]
    yy, mm, dd = symbol[-15:-13], symbol[-13:-11], symbol[-11:-9]
    return {
        "root": root,
        "expiry": date(2000 + int(yy), int(mm), int(dd)),
        "option_type": "call" if symbol[-9] == "C" else "put",
        "strike": int(symbol[-8:]) / 1000.0,
    }


# ── Target price math ────────────────────────────────────────────────

def required_near_iv(
    sigma_fwd: float,
    T1: float,
    tau: float,
    hist_rms_move: float,
    premium: float,
) -> Optional[float]:
    """Near-leg IV at which the implied event move is exactly (1+premium) x hist RMS.

    sigma_fwd, T1, tau in years/annualized units; hist_rms_move as a fraction
    (0.06 = 6%). Returns None when the decomposition has no real solution.
    """
    if T1 <= 0 or tau < 0 or tau > T1 or sigma_fwd <= 0 or hist_rms_move <= 0:
        return None
    target_event_move = (1.0 + premium) * hist_rms_move
    total_var = target_event_move ** 2 + sigma_fwd ** 2 * (T1 - tau)
    sigma1_sq = total_var / T1
    if sigma1_sq <= 0:
        return None
    return math.sqrt(sigma1_sq)


def target_debit(
    far_price: float,
    spot: float,
    strike: float,
    T1: float,
    sigma_fwd: float,
    tau: float,
    hist_rms_move: float,
    premium: float,
    r: float = RISK_FREE_RATE,
) -> Optional[float]:
    """Max calendar debit consistent with >= premium event richness.

    far_price: current far-leg price (mid for display, ask for executable).
    Returns None if inputs are degenerate. Can be negative (near leg worth
    more than far leg) — caller should treat <= 0 as untradeable.
    """
    iv_star = required_near_iv(sigma_fwd, T1, tau, hist_rms_move, premium)
    if iv_star is None:
        return None
    near_star = black_scholes_price(spot, strike, T1, r, iv_star, "call")
    if not math.isfinite(near_star):
        return None
    return far_price - near_star


def forward_iv(iv_near: float, T1: float, iv_far: float, T2: float) -> Optional[float]:
    """Event-free forward vol between T1 and T2 (variance decomposition)."""
    if T2 <= T1 or iv_near <= 0 or iv_far <= 0:
        return None
    var = iv_far ** 2 * T2 - iv_near ** 2 * T1
    if var <= 0:
        return None
    return math.sqrt(var / (T2 - T1))


def combo_debit(near_bid: float, near_ask: float, far_bid: float, far_ask: float,
                executable: bool = False) -> Optional[float]:
    """Calendar debit from leg quotes.

    executable=False -> combo mid (far_mid - near_mid), for display/distance.
    executable=True  -> combo ask (far_ask - near_bid), the realistic fill cost.
    """
    if executable:
        d = far_ask - near_bid
    else:
        d = (far_bid + far_ask) / 2.0 - (near_bid + near_ask) / 2.0
    if not all(math.isfinite(x) for x in (near_bid, near_ask, far_bid, far_ask)):
        return None
    return d


# ── Distance filter ──────────────────────────────────────────────────

def within_fill_range(mid_debit: float, cap_debit: float, f: float = 0.15) -> bool:
    """Track a candidate only if the current mid is within f of the cap price.

    mid <= cap means the premium threshold is already met (marketable).
    mid > cap*(1+f) means the market is too far off — don't litter the book
    with hopeless resting orders.
    """
    if cap_debit <= 0 or mid_debit <= 0:
        return False
    return mid_debit <= cap_debit * (1.0 + f)


# ── Limit ladder ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class LadderSpec:
    """Price ladder: start cheap at the high-premium target, concede to the cap."""

    start_premium: float = 0.25   # cheapest target (richest required near leg)
    floor_premium: float = 0.20   # hard cap price (minimum acceptable premium)
    start_et: time = time(14, 0)  # first rung 14:00 ET
    step_minutes: int = 15
    tick: float = 0.01
    last_rung_et: time = time(15, 45)  # final reprice; order works till close

    def rung_index(self, now: datetime) -> Optional[int]:
        """Rung number for *now* (0-based), None if outside the ladder window."""
        now_et = now.astimezone(ET)
        start = datetime.combine(now_et.date(), self.start_et, tzinfo=ET)
        last = datetime.combine(now_et.date(), self.last_rung_et, tzinfo=ET)
        if now_et < start:
            return None
        if now_et > last:
            return None
        return int((now_et - start).total_seconds() // (self.step_minutes * 60))

    def limit_at(self, rung: int, debit_start: float, debit_cap: float) -> float:
        """Limit price at *rung*: start + rung*tick, never above the cap."""
        return round(min(debit_start + rung * self.tick, debit_cap), 2)

    def current_limit(self, now: datetime, debit_start: float, debit_cap: float) -> Optional[float]:
        """Convenience: limit price for the current time, or None outside window."""
        rung = self.rung_index(now)
        if rung is None:
            return None
        return self.limit_at(rung, debit_start, debit_cap)
