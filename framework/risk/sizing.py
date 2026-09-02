"""Position sizing: strategy-declared, shared between live and backtest.

A ``Sizer`` converts a trade's economics into a contract quantity. Returning
0 vetoes the trade. The same sizer instance must drive live order submission
and the backtest engine so simulated sizing matches production.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class SizeContext:
    """Inputs a sizer may use. All dollar values are per-contract totals
    (i.e. premium x multiplier already applied)."""
    equity: float
    buying_power: float
    price_per_unit: float            # debit paid (or credit received) per contract
    max_loss_per_unit: Optional[float] = None  # defined-risk trades; None = price


class Sizer(ABC):
    """Base class for position sizers."""

    name: str = "base"

    @abstractmethod
    def quantity(self, ctx: SizeContext) -> int:
        """Contracts to trade (0 = veto)."""


class FixedDollarSizer(Sizer):
    """Fixed dollar budget per trade, clamped to ``max_pct_of_equity``.

    A $2,000 budget on a $7k account is 28% of equity — the clamp keeps
    the TOML dollar figure from oversized small live accounts.
    """

    name = "fixed_dollar"

    def __init__(self, budget: float, max_pct_of_equity: float = 0.05):
        if budget <= 0:
            raise ValueError("budget must be positive")
        if not 0 < max_pct_of_equity <= 1:
            raise ValueError("max_pct_of_equity must be in (0, 1]")
        self.budget = budget
        self.max_pct_of_equity = max_pct_of_equity

    def quantity(self, ctx: SizeContext) -> int:
        if ctx.price_per_unit <= 0:
            return 0
        budget = self.budget
        if ctx.equity > 0:
            budget = min(budget, self.max_pct_of_equity * ctx.equity)
        return max(int(budget // ctx.price_per_unit), 0)


class PercentOfPortfolioSizer(Sizer):
    """Cap notional cost at ``pct`` of equity."""

    name = "pct_portfolio"

    def __init__(self, pct: float):
        if not 0 < pct <= 1:
            raise ValueError("pct must be in (0, 1]")
        self.pct = pct

    def quantity(self, ctx: SizeContext) -> int:
        if ctx.price_per_unit <= 0 or ctx.equity <= 0:
            return 0
        budget = self.pct * ctx.equity
        return max(int(budget // ctx.price_per_unit), 0)


class VolTargetSizer(Sizer):
    """Risk a fixed fraction of equity per trade, sized off max loss.

    Falls back to the entry price when no defined max loss is available
    (naked/undefined-risk structures — conservative).
    """

    name = "vol_target"

    def __init__(self, risk_pct: float):
        if not 0 < risk_pct <= 1:
            raise ValueError("risk_pct must be in (0, 1]")
        self.risk_pct = risk_pct

    def quantity(self, ctx: SizeContext) -> int:
        if ctx.equity <= 0:
            return 0
        risk_unit = ctx.max_loss_per_unit
        if risk_unit is None or risk_unit <= 0:
            risk_unit = ctx.price_per_unit
        if risk_unit <= 0:
            return 0
        budget = self.risk_pct * ctx.equity
        return max(int(budget // risk_unit), 0)


_SIZERS = {s.name: s for s in (FixedDollarSizer, PercentOfPortfolioSizer, VolTargetSizer)}


def build_sizer(name: str, params: dict) -> Sizer:
    """Construct a sizer from a strategy-config ``[risk.sizer]`` section."""
    cls = _SIZERS.get(name)
    if cls is None:
        raise ValueError(f"unknown sizer {name!r} (have: {sorted(_SIZERS)})")
    return cls(**params)
