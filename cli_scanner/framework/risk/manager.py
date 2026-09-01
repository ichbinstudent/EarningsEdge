"""Pre-trade risk manager: the single chokepoint for ALL order submission.

Both the human-approval path and any auto-execution path must call
``RiskManager.check_trade`` before an order reaches the broker. This closes
the historical gap where portfolio caps were only enforced in the legacy
auto-trade path.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Optional

from earnings_edge.db import risk_events_list

from .equity import daily_pnl, day_start_equity
from .killswitch import KillSwitch, record_event

logger = logging.getLogger("framework.risk.manager")


def is_gcd_reject(detail: str) -> bool:
    """Alpaca 422 'ratio_qty must be relatively prime' is a client bug, not a broker outage."""
    d = (detail or "").lower()
    return "relatively prime" in d or "gcd[" in d or "gcd " in d


@dataclass(frozen=True)
class RiskLimits:
    """Portfolio-level limits (global defaults; strategy TOML may tighten)."""
    max_pct_per_trade: float = 0.10        # of buying power, per order
    max_pct_per_underlying: float = 0.25   # of equity, aggregate per ticker
    max_pct_per_strategy_day: float = 0.30 # of equity, new spend per strategy/day
    daily_loss_limit_pct: float = 0.05     # of day-start equity → trips kill switch
    min_buying_power: float = 10_000.0     # halt new entries below this
    probation_size_mult: float = 0.5       # size multiplier while on probation
    max_consecutive_rejections: int = 3    # broker rejections → trips kill switch


@dataclass
class RiskDecision:
    approved: bool
    reason: str = ""
    qty_multiplier: float = 1.0
    vetoes: list[str] = field(default_factory=list)

    def veto(self, reason: str) -> None:
        self.approved = False
        self.vetoes.append(reason)
        self.reason = "; ".join(self.vetoes)


class RiskManager:
    """Evaluates trades against limits and records spend/fills.

    Stateless apart from the DB: spend and exposure are derived
    from ``risk_events``/``managed_positions`` so limits survive restarts.
    """

    def __init__(self, limits: Optional[RiskLimits] = None):
        self.limits = limits or RiskLimits()
        self.killswitch = KillSwitch()

    # -- main gate -------------------------------------------------------------

    def check_trade(
        self,
        strategy: str,
        ticker: str,
        est_cost: float,
        qty: int = 1,
        equity: float = 0.0,
        buying_power: float = 0.0,
        underlying_exposure: float = 0.0,
        lifecycle: str = "live",
        limits: Optional[RiskLimits] = None,
        live_broker: Optional[bool] = None,
    ) -> RiskDecision:
        """Approve or veto a proposed order.

        ``est_cost`` is the total dollars at risk for the order (debit x qty,
        or max loss for defined-risk credit structures). ``underlying_exposure``
        is the current aggregate dollars held in ``ticker`` across strategies.
        ``live_broker``: when True, ``lifecycle=paper`` is a veto and a
        missing day-start equity snapshot fail-closes new entries.
        """
        lim = limits or self.limits
        d = RiskDecision(approved=True)

        if self.killswitch.is_halted():
            d.veto(f"kill switch halted ({self.killswitch.status().get('reason')})")

        if live_broker is None:
            try:
                from earnings_edge.alpaca_mode import alpaca_live_enabled
                live_broker = alpaca_live_enabled()
            except Exception:
                live_broker = False

        if lifecycle == "probation":
            d.qty_multiplier = lim.probation_size_mult
        elif lifecycle == "paper" and live_broker:
            d.veto("lifecycle 'paper' cannot submit to a live broker")
        elif lifecycle not in ("live", "paper", "probation"):
            d.veto(f"strategy lifecycle '{lifecycle}' does not permit execution")

        if live_broker:
            start = day_start_equity()
            if start is None:
                d.veto("no day-start equity snapshot — live entries fail closed")

        if buying_power and buying_power < lim.min_buying_power:
            d.veto(f"buying power {buying_power:.0f} < min {lim.min_buying_power:.0f}")

        if est_cost > 0:
            if buying_power and est_cost > lim.max_pct_per_trade * buying_power:
                d.veto(
                    f"cost {est_cost:.0f} > {lim.max_pct_per_trade:.0%} of buying power "
                    f"({lim.max_pct_per_trade * buying_power:.0f})"
                )
            if equity:
                if underlying_exposure + est_cost > lim.max_pct_per_underlying * equity:
                    d.veto(
                        f"underlying exposure {underlying_exposure + est_cost:.0f} > "
                        f"{lim.max_pct_per_underlying:.0%} of equity"
                    )
                day_spend = self._strategy_spend_today(strategy)
                if day_spend + est_cost > lim.max_pct_per_strategy_day * equity:
                    d.veto(
                        f"strategy daily spend {day_spend + est_cost:.0f} > "
                        f"{lim.max_pct_per_strategy_day:.0%} of equity"
                    )

        if not d.approved:
            logger.warning("RISK VETO %s %s: %s", strategy, ticker, d.reason)
            record_event("veto", d.reason, strategy=strategy)
        return d

    # -- bookkeeping -------------------------------------------------------------

    def record_entry(self, strategy: str, ticker: str, cost: float, detail: str = "") -> None:
        """Record committed dollars so daily/strategy budgets are enforced."""
        record_event(
            "entry",
            f"{ticker} cost={cost:.2f} {detail}".strip(), strategy=strategy,
        )

    def check_daily_loss(self, equity_now: float, on: Optional[date] = None) -> bool:
        """Trip the kill switch when the daily loss limit is breached."""
        pnl = daily_pnl(equity_now, on)
        start = day_start_equity(on)
        if pnl is None or start is None or start <= 0:
            return False
        if pnl <= -self.limits.daily_loss_limit_pct * start:
            if not self.killswitch.is_halted():
                self.killswitch.trip(
                    f"daily loss {pnl:.0f} breached {self.limits.daily_loss_limit_pct:.0%} "
                    f"of day-start equity {start:.0f}",
                    by="risk_manager",
                )
            return True
        return False

    def record_broker_rejection(self, strategy: str, detail: str) -> int:
        """Track consecutive rejections; trips the kill switch at the cap.

        GCD / relatively-prime 422s are our qty-encoding bug, not a broker
        outage — they are logged as ``gcd_reject`` and do not increment the
        kill-switch streak.
        """
        if is_gcd_reject(detail):
            record_event("gcd_reject", detail, strategy=strategy)
            return 0
        record_event("rejection", detail, strategy=strategy)
        rows = risk_events_list(limit=self.limits.max_consecutive_rejections)
        streak = 0
        for r in rows:
            if r["event_type"] == "rejection":
                streak += 1
            else:
                break
        if streak >= self.limits.max_consecutive_rejections and not self.killswitch.is_halted():
            self.killswitch.trip(
                f"{streak} consecutive broker rejections", by="risk_manager",
            )
        return streak

    # -- internals ----------------------------------------------------------------

    def strategy_spend_today(self, strategy: str) -> float:
        """Public accessor: dollars committed by ``strategy`` today (UTC)."""
        return self._strategy_spend_today(strategy)

    def _strategy_spend_today(self, strategy: str) -> float:
        today = datetime.now(timezone.utc).date().isoformat()
        rows = risk_events_list(
            event_type="entry", strategy=strategy, since=today, newest_first=False,
        )
        total = 0.0
        for r in rows:
            # detail format: "TICKER cost=123.45 ..."
            for token in (r["detail"] or "").split():
                if token.startswith("cost="):
                    try:
                        total += float(token.split("=", 1)[1])
                    except ValueError:
                        pass
        return total
