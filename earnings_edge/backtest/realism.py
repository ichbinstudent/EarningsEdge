"""Backtester realism layer: fill slippage, commissions, REG-T margin, capacity.

Replicates the oquants.com backtester realism model (docs/platform):
  - realistic fills deviate adversely from mid as a fraction of the half-spread,
    widened for thin volume/OI and penalized for OTM options;
  - commissions follow worst-case IBKR per-contract tiers;
  - margin follows REG-T initial requirements (IBKR standard), enabling
    return-on-margin metrics;
  - a capacity cap estimates the max contracts tradeable without price impact.

All functions are pure — no I/O — so the whole module is unit-testable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

CONTRACT_MULTIPLIER = 100  # shares per US equity option contract


# ── Realistic fills ──────────────────────────────────────────────────

def realistic_fill(
    mid: float,
    bid: float,
    ask: float,
    *,
    volume: Optional[float] = None,
    open_interest: Optional[float] = None,
    is_otm: bool = False,
    side: str = "buy",
    spread_participation: float = 0.5,
    otm_penalty: float = 0.10,
    low_liquidity_ref: float = 1000.0,
) -> float:
    """Fill price deviating adversely from mid, in half-spread units.

    Deviation model (fraction of the half-spread, always adverse):

        deviation = spread_participation * liquidity_multiplier
                    + (otm_penalty if is_otm else 0)

    where ``liquidity_multiplier = 1 + max(0, 1 - depth / low_liquidity_ref)``
    and ``depth`` is the smaller of the provided volume / open_interest. A deep
    book (depth >= low_liquidity_ref) leaves the multiplier at 1.0; an empty
    book doubles the base deviation. Unknown depth (both None) is treated as
    liquid — no widening — since we refuse to fabricate slippage from nothing.

    Buys fill at ``mid + deviation * half_spread``, sells at
    ``mid - deviation * half_spread``. The fill is clamped to the far touch
    (ask for buys, bid for sells) — you cannot pay more than the displayed
    offer — UNLESS the book is explicitly empty (volume == 0 AND
    open_interest == 0), in which case the model is allowed to walk past the
    touch, reflecting fills against a book with no displayed size.

    Deterministic: same inputs always produce the same fill.
    """
    if ask < bid:
        raise ValueError(f"crossed/locked book: bid {bid} > ask {ask}")
    if side not in ("buy", "sell"):
        raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")

    half_spread = (ask - bid) / 2.0
    if half_spread <= 0:
        return mid

    depths = [d for d in (volume, open_interest) if d is not None]
    depth = min(depths) if depths else None
    liquidity_multiplier = (
        1.0 + max(0.0, 1.0 - depth / low_liquidity_ref) if depth is not None else 1.0
    )

    deviation = spread_participation * liquidity_multiplier
    if is_otm:
        deviation += otm_penalty

    empty_book = volume == 0 and open_interest == 0
    if side == "buy":
        fill = mid + deviation * half_spread
        return fill if empty_book else min(fill, ask)
    fill = mid - deviation * half_spread
    return fill if empty_book else max(fill, bid)


# ── Commissions ──────────────────────────────────────────────────────

def ibkr_commission(premium_per_contract: float, contracts: int = 1) -> float:
    """Worst-case IBKR per-contract commission tier, in dollars.

        premium < $0.05          -> $0.25 / contract
        $0.05 <= premium < $0.10 -> $0.50 / contract
        premium >= $0.10         -> $0.65 / contract

    "Worst case" = the fixed/tiered ceiling a small retail account actually
    pays; volume discounts are ignored. ``premium_per_contract`` is the option
    price (per share, as quoted), not the dollar value of the contract.
    """
    if premium_per_contract < 0.05:
        rate = 0.25
    elif premium_per_contract < 0.10:
        rate = 0.50
    else:
        rate = 0.65
    return rate * contracts


# ── REG-T margin ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class OptionLeg:
    """One option leg for margin purposes (quantity fixed at 1 contract)."""

    action: str            # "buy" | "sell"
    kind: str              # "call" | "put"
    strike: float
    underlying_price: float
    premium: float         # quoted price per share


def _naked_short_margin(leg: OptionLeg) -> float:
    """REG-T initial margin for one uncovered short option.

        margin = max(20% * base - OTM_amount, 10% * base) * 100 + premium * 100

    Per the oquants convention the base is the underlying price for calls and
    the strike for puts. OTM_amount is how far the option is out of the money
    (0 when ITM): calls ``max(0, strike - underlying)``, puts
    ``max(0, underlying - strike)``.
    """
    if leg.kind == "call":
        base = leg.underlying_price
        otm_amount = max(0.0, leg.strike - leg.underlying_price)
    else:
        base = leg.strike
        otm_amount = max(0.0, leg.underlying_price - leg.strike)
    return (
        max(0.20 * base - otm_amount, 0.10 * base) + leg.premium
    ) * CONTRACT_MULTIPLIER


def _vertical_spread_margin(legs: Sequence[OptionLeg]) -> Optional[float]:
    """Margin for a 2-leg defined-risk vertical, or None if *legs* isn't one.

    A defined-risk vertical is exactly two legs of the same kind (both calls or
    both puts), one bought and one sold, at different strikes. Same expiry is
    assumed — callers must not mix expirations into a single regt_margin call
    unless the structure is a vertical. The short strike is typically further
    OTM for debit spreads and closer to the money for credit spreads; either
    way a 1x1 vertical is defined-risk, so the margin is its max loss:

        credit spread: margin = spread_width * 100 - credit * 100
        debit spread:  margin = debit * 100   (max loss = debit paid)
    """
    if len(legs) != 2:
        return None
    kinds = {leg.kind for leg in legs}
    actions = {leg.action for leg in legs}
    if len(kinds) != 1 or actions != {"buy", "sell"}:
        return None
    long_leg = next(leg for leg in legs if leg.action == "buy")
    short_leg = next(leg for leg in legs if leg.action == "sell")
    width = abs(long_leg.strike - short_leg.strike)
    if width <= 0:
        return None
    net_credit = short_leg.premium - long_leg.premium
    if net_credit >= 0:
        return (width - net_credit) * CONTRACT_MULTIPLIER
    return (-net_credit) * CONTRACT_MULTIPLIER


def regt_margin(legs: Sequence[OptionLeg]) -> float:
    """Total REG-T initial margin (dollars) for a list of option legs.

    Rules:
      - long options are paid in full: margin = premium * 100;
      - naked short options use the standard REG-T formula (see
        ``_naked_short_margin``);
      - a 2-leg defined-risk vertical (same kind, short further OTM than the
        long) is margined at its max loss instead of the naked formula;
      - anything else sums the per-leg requirements (e.g. a short strangle
        sums both naked legs — a simplification of portfolio margining).

    Assumes 1 contract per leg and a single shared expiration for spread
    detection; multiply the result for larger size.
    """
    spread = _vertical_spread_margin(legs)
    if spread is not None:
        return spread
    total = 0.0
    for leg in legs:
        if leg.action == "buy":
            total += leg.premium * CONTRACT_MULTIPLIER
        else:
            total += _naked_short_margin(leg)
    return total


# ── Capacity ─────────────────────────────────────────────────────────

def capacity_cap(
    volume: Optional[float],
    open_interest: Optional[float],
    participation: float = 0.10,
) -> float:
    """Max contracts tradeable without price impact.

        cap = floor(participation * min(volume, open_interest))

    A participant is assumed to be at most ``participation`` (default 10%) of
    the day's liquidity, measured as the smaller of volume and open interest.
    A None input means "unknown" and imposes no constraint: with one side
    unknown the other governs; with both unknown the cap is ``math.inf``.
    """
    depth = min(
        (d for d in (volume, open_interest) if d is not None),
        default=math.inf,
    )
    cap = participation * depth
    return math.inf if math.isinf(cap) else int(math.floor(cap))
