"""Exit rules: pure evaluation of open position groups against strategy config.

Rules come from each strategy's TOML ``[[exits]]`` section and are evaluated
by the ``ExitManager`` against a ``MarketView`` (injectable — tests never
touch a broker). Two execution classes:

- ``ProfitTargetExit`` / ``StopLossExit`` → *auto* (close immediately on breach)
- ``TimeExit`` → *approval* (human confirms; exits can wait for a decision)

P&L convention (per-share, matching ``managed_positions.entry_price``):
- debit structure: paid ``entry`` to open; pnl = (value_now − entry) / entry
- credit structure: received ``entry`` to open; value_now is the buyback
  liability (negative); pnl = (entry + value_now) / entry
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional

# Structure sides that collect premium at entry (everything else is debit).
CREDIT_SIDES = {"SHORT_STRADDLE", "SHORT_STRANGLE", "IRON_CONDOR"}


@dataclass
class LegPos:
    symbol: str
    side: str            # "buy" (long) | "sell" (short)
    qty: float = 1.0
    option_type: str = ""
    strike: float = 0.0
    expiry: Optional[date] = None


@dataclass
class PositionGroup:
    group_id: str
    strategy: str
    legs: list[LegPos]
    entry_price: float          # net per-share premium at entry (positive)
    opened_at: str              # ISO ts
    credit: bool = False        # True = premium received at entry
    event_date: Optional[date] = None   # e.g. earnings date
    qty: int = 1
    exit_by: Optional[date] = None   # structural deadline computed at entry
                                      # (e.g. a calendar's near-leg expiry) —
                                      # None when the structure has no
                                      # differential-expiry deadline

    @property
    def ticker(self) -> str:
        if not self.legs:
            return ""
        from .guards import occ_underlying
        return occ_underlying(self.legs[0].symbol) or ""


@dataclass
class MarketView:
    """What rules see: structure value per share, date/time context."""
    value_now: Optional[float]   # net mid: +mid long legs, −mid short legs
    today: date
    sessions_since_open: int
    sessions_until_event: Optional[int] = None
    minutes_to_close: Optional[int] = None   # None when unknown (clock fetch failed)


@dataclass
class ExitSignal:
    rule: str
    reason: str
    auto: bool                   # True → close immediately; False → approval card
    pnl_pct: Optional[float] = None


# ── P&L ---------------------------------------------------------------------

def pnl_pct(group: PositionGroup, value_now: float) -> Optional[float]:
    if group.entry_price <= 0:
        return None
    if group.credit:
        return (group.entry_price + value_now) / group.entry_price
    return (value_now - group.entry_price) / group.entry_price


def structure_value(legs: list[LegPos], snaps: dict[str, dict]) -> Optional[float]:
    """Net mid value per share: +mid for long legs, −mid for short legs.

    None when any leg lacks a usable quote (conservative: no signal).
    """
    total = 0.0
    for leg in legs:
        mid = _leg_mid(leg, snaps)
        if mid is None:
            return None
        total += (1.0 if leg.side == "buy" else -1.0) * mid * leg.qty
    return total


def _leg_mid(leg: LegPos, snaps: dict[str, dict]) -> Optional[float]:
    snap = snaps.get(leg.symbol) or {}
    q = snap.get("latestQuote") or {}
    bid, ask = q.get("bp"), q.get("ap")
    if bid is None or ask is None:
        return None
    return (float(bid) + float(ask)) / 2.0


def leg_mid(leg: LegPos, snaps: dict[str, dict]) -> Optional[float]:
    """Per-share mid for one leg. Limit prices are per-share; do not scale by qty."""
    return _leg_mid(leg, snaps)


def unit_structure_value(legs: list[LegPos], snaps: dict[str, dict]) -> Optional[float]:
    """Net mid per 1x ratio — ignore stored contract qty so a 9-lot calendar
    is priced at the combo mid, not mid×9 (which would never fill)."""
    unit = [
        LegPos(leg.symbol, leg.side, 1.0, leg.option_type, leg.strike, leg.expiry)
        for leg in legs
    ]
    return structure_value(unit, snaps)


def remaining_close_plan(
    legs: list[LegPos],
    snaps: dict[str, dict],
    today: date,
) -> dict:
    """How to flatten a group when some (or all) legs have no quote.

    Returns ``{"mode", "close_legs", "drop_legs"}``:
    - ``combo``: every leg is quoted — close the structure as one order.
    - ``remaining``: some legs unquoted (typically an expired near) — close
      the still-quoted legs individually and drop the rest locally.
    - ``expired``: nothing quotable and at least one leg's expiry is on/past
      ``today`` — mark the group closed; no broker order.
    - ``no_quote``: nothing quotable and no expiry evidence — retry later.
    """
    quoted: list[LegPos] = []
    missing: list[LegPos] = []
    for leg in legs:
        if _leg_mid(leg, snaps) is not None:
            quoted.append(leg)
        else:
            missing.append(leg)
    if not missing:
        return {"mode": "combo", "close_legs": list(legs), "drop_legs": []}
    if quoted:
        return {"mode": "remaining", "close_legs": quoted, "drop_legs": missing}
    expired = any(leg.expiry is not None and leg.expiry <= today for leg in legs)
    return {
        "mode": "expired" if expired else "no_quote",
        "close_legs": [],
        "drop_legs": list(legs),
    }


# ── Rules ---------------------------------------------------------------------

class ExitRule(ABC):
    name: str = "base"
    auto: bool = False

    @abstractmethod
    def evaluate(self, group: PositionGroup, market: MarketView) -> Optional[ExitSignal]:
        ...


class ProfitTargetExit(ExitRule):
    name = "profit_target"
    auto = True

    def __init__(self, pct: float):
        self.pct = pct

    def evaluate(self, group: PositionGroup, market: MarketView) -> Optional[ExitSignal]:
        if market.value_now is None:
            return None
        pnl = pnl_pct(group, market.value_now)
        if pnl is not None and pnl >= self.pct:
            return ExitSignal(
                rule=self.name, auto=True, pnl_pct=pnl,
                reason=f"pnl {pnl:+.0%} ≥ target {self.pct:+.0%}",
            )
        return None


class StopLossExit(ExitRule):
    name = "stop_loss"
    auto = True

    def __init__(self, pct: float):
        self.pct = pct

    def evaluate(self, group: PositionGroup, market: MarketView) -> Optional[ExitSignal]:
        if market.value_now is None:
            return None
        pnl = pnl_pct(group, market.value_now)
        if pnl is not None and pnl <= -self.pct:
            return ExitSignal(
                rule=self.name, auto=True, pnl_pct=pnl,
                reason=f"pnl {pnl:+.0%} ≤ stop {-self.pct:+.0%}",
            )
        return None


class TimeExit(ExitRule):
    name = "time"
    auto = False  # day-count-from-entry / T-N are approval cards

    def __init__(self, days_after_entry: Optional[int] = None,
                 days_before_event: Optional[int] = None,
                 days_after_event: Optional[int] = None):
        self.days_after_entry = days_after_entry
        self.days_before_event = days_before_event
        self.days_after_event = days_after_event

    def evaluate(self, group: PositionGroup, market: MarketView) -> Optional[ExitSignal]:
        # Post-event deadline: event has arrived (sessions_until_event == 0
        # on event day and every session after). Auto — the vol-crush window
        # is the point of the trade; waiting for a card abandoned fills.
        if self.days_after_event is not None and market.sessions_until_event is not None \
                and market.sessions_until_event <= 0:
            return ExitSignal(
                rule=self.name, auto=True,
                reason=f"event day/past (sessions_until_event="
                       f"{market.sessions_until_event}, days_after_event="
                       f"{self.days_after_event})",
            )
        if self.days_after_entry is not None \
                and market.sessions_since_open >= self.days_after_entry:
            return ExitSignal(
                rule=self.name, auto=False,
                reason=f"{market.sessions_since_open} sessions ≥ {self.days_after_entry} after entry",
            )
        if self.days_before_event is not None and market.sessions_until_event is not None \
                and market.sessions_until_event <= self.days_before_event:
            return ExitSignal(
                rule=self.name, auto=False,
                reason=f"{market.sessions_until_event} sessions to event "
                       f"(exit at T-{self.days_before_event})",
            )
        return None


class ScheduledExit(ExitRule):
    """Structural, entry-computed hard deadline — e.g. a calendar spread's
    near leg expiring. Unlike TimeExit (a TOML day-offset reconstructed at
    evaluation time, always approval-only), this reads a deadline the
    strategy already knew and stored the moment the position was opened
    (``PositionGroup.exit_by``), so there is nothing to misconfigure.

    Fires ``auto=True`` once we're on/past ``exit_by`` AND within
    ``minutes_before_close`` of the session close — a differential-expiry
    deadline is a structural necessity (the near leg is about to vanish),
    the same class of urgency as profit-target/stop-loss, not a discretionary
    "maybe take profit here" judgment call that should wait for a human.
    """
    name = "scheduled"
    auto = True

    def __init__(self, minutes_before_close: int = 90):
        self.minutes_before_close = minutes_before_close

    def evaluate(self, group: PositionGroup, market: MarketView) -> Optional[ExitSignal]:
        if group.exit_by is None or market.minutes_to_close is None:
            return None
        if market.today >= group.exit_by and market.minutes_to_close <= self.minutes_before_close:
            return ExitSignal(
                rule=self.name, auto=True,
                reason=f"exit_by {group.exit_by.isoformat()} reached, "
                       f"{market.minutes_to_close}min to close",
            )
        return None


def build_exit_rules(exits_cfg: list[dict]) -> list[ExitRule]:
    """Build rule objects from a strategy config's [[exits]] array."""
    rules: list[ExitRule] = []
    for e in exits_cfg:
        kind = e.get("rule")
        if kind == "time":
            rules.append(TimeExit(
                days_after_entry=e.get("days_after_entry"),
                days_before_event=e.get("days_before_event"),
                days_after_event=e.get("days_after_event"),
            ))
        elif kind == "profit_target":
            rules.append(ProfitTargetExit(float(e["pct"])))
        elif kind == "stop_loss":
            rules.append(StopLossExit(float(e["pct"])))
        elif kind == "scheduled":
            kwargs = {}
            if "minutes_before_close" in e:
                kwargs["minutes_before_close"] = int(e["minutes_before_close"])
            rules.append(ScheduledExit(**kwargs))
    return rules
