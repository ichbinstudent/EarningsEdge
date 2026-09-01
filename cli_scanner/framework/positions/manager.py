"""Exit manager: evaluates exit rules over open position groups and acts.

- Profit-target / stop-loss breaches → *auto-close* via the OrderManager's
  limit walk (paper), notified after the fact.
- Time exits → approval cards in ``exit_proposals`` (deduped per group).

The kill switch is deliberately NOT consulted: it gates entries only.
Closing risk is always allowed.

Fill bookkeeping: positions marked closed with the exit price, trade_events
written, realized PnL recorded — which also feeds lifecycle promotion stats.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from typing import Callable, Optional

from earnings_edge import cards
from earnings_edge.db import (
    exit_proposals_get,
    exit_proposals_insert,
    exit_proposals_list_pending,
    exit_proposals_mark,
    trade_events_insert,
)

from ..core.calendar import get_calendar
from ..core.registry import StrategyRegistry, get_registry
from ..execution.managed import close_positions, open_groups
from ..execution.order_manager import LimitWalkPolicy, ManagedOrder, OrderManager
from .exits import (
    ExitSignal, MarketView, PositionGroup, build_exit_rules, leg_mid,
    remaining_close_plan, unit_structure_value,
)

logger = logging.getLogger("framework.positions.manager")


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class ExitManager:
    def __init__(
        self,
        client,
        registry: Optional[StrategyRegistry] = None,
        order_manager: Optional[OrderManager] = None,
        today: Optional[date] = None,
    ):
        self.client = client
        self.registry = registry or get_registry()
        self.order_manager = order_manager or OrderManager(client)
        self._today = today

    # ── evaluation ---------------------------------------------------------

    def evaluate_all(self) -> dict:
        """One pass over all open groups. Returns stats + messages to push."""
        out: dict = {"groups": 0, "auto_closed": [], "proposed": [], "held": 0, "errors": []}
        groups = open_groups()
        out["groups"] = len(groups)
        if not groups:
            return out

        symbols = sorted({leg.symbol for g in groups for leg in g.legs})
        try:
            snaps = self.client.get_option_snapshots_bulk(*symbols) or {}
        except Exception as exc:
            # Still evaluate time/scheduled rules; close_group will flatten
            # remaining quoted legs (or mark expired ones closed).
            out["errors"].append(f"snapshot fetch failed: {exc}")
            logger.warning("exit eval: snapshots failed: %s", exc)
            snaps = {}

        cal = get_calendar()
        today = self._today or datetime.now(timezone.utc).date()
        minutes_to_close = self._minutes_to_close()

        for group in groups:
            try:
                self._evaluate_group(group, snaps, cal, today, out, minutes_to_close)
            except Exception as exc:
                logger.exception("exit eval failed for %s", group.group_id)
                out["errors"].append(f"{group.group_id}: {exc}")
        return out

    def _minutes_to_close(self) -> Optional[int]:
        """Minutes until the session closes, or None when the market is
        shut or the clock can't be read — ScheduledExit only fires with a
        real number here, so an unreadable clock fails safe (no auto-close
        attempted rather than guessing)."""
        try:
            clock = self.client.get_clock()
        except Exception as exc:
            logger.warning("exit eval: clock fetch failed (%s) — scheduled exits skipped this pass", exc)
            return None
        if not clock.get("is_open"):
            return None
        try:
            now = datetime.fromisoformat(clock["timestamp"])
            close = datetime.fromisoformat(clock["next_close"])
            return max(int((close - now).total_seconds() // 60), 0)
        except (KeyError, ValueError, TypeError) as exc:
            logger.warning("exit eval: could not parse clock (%s)", exc)
            return None

    def _evaluate_group(self, group: PositionGroup, snaps: dict, cal,
                        today: date, out: dict, minutes_to_close: Optional[int] = None) -> None:
        cfg = self.registry.get(group.strategy)
        rules = build_exit_rules(cfg.exits) if cfg else []
        if not rules:
            return

        opened_date = group.opened_at[:10]
        try:
            sessions_since = max(len(cal.sessions_between(
                datetime.strptime(opened_date, "%Y-%m-%d").date(), today)) - 1, 0)
        except ValueError:
            sessions_since = 0
        sessions_until_event = None
        if group.event_date is not None:
            if group.event_date < today:
                sessions_until_event = 0  # event day/past → time exits fire
            else:
                sessions_until_event = max(
                    len(cal.sessions_between(today, group.event_date)) - 1, 0)

        value = unit_structure_value(group.legs, snaps)
        market = MarketView(
            value_now=value, today=today,
            sessions_since_open=sessions_since,
            sessions_until_event=sessions_until_event,
            minutes_to_close=minutes_to_close,
        )

        from ..core.control import effective_execution_mode
        toml_mode = cfg.execution_mode if cfg else "approval"
        is_auto = effective_execution_mode(group.strategy, toml_mode) == "auto"

        signal = next((s for r in rules if (s := r.evaluate(group, market))), None)
        if signal is None:
            out["held"] += 1
            return

        if signal.auto or is_auto:
            mo = self.close_group(group, reason=signal.reason)
            if mo.state in ("filled", "partial"):
                out["auto_closed"].append(
                    f"🔒 {cards.bold('AUTO-EXIT')} {cards.esc(group.strategy)} "
                    f"{cards.bold(group.ticker)} ({cards.esc(signal.rule)}: "
                    f"{cards.esc(signal.reason)}) @ {mo.filled_avg_price}")
            else:
                out["errors"].append(
                    f"{group.ticker}: exit order {mo.state} ({mo.detail}) — retrying next cycle")
        else:
            row = self.propose_exit(group, signal)
            if row is not None:
                out["proposed"].append(row)

    # ── closing --------------------------------------------------------------

    def close_group(self, group: PositionGroup, reason: str = "") -> ManagedOrder:
        """Work a closing order; fall back to remaining-leg closes when the
        combo quote is gone (expired near). Marks the group closed on fill
        or when every remaining unquoted leg is past expiry."""
        today = self._today or datetime.now(timezone.utc).date()
        try:
            snaps = self.client.get_option_snapshots_bulk(
                *[leg.symbol for leg in group.legs]) or {}
        except Exception:
            snaps = {}
        plan = remaining_close_plan(group.legs, snaps, today)

        if plan["mode"] == "expired":
            n = close_positions(group.group_id)
            mo = ManagedOrder(
                client_order_id=f"exit_{group.group_id}_expired",
                side="sell", qty=group.qty, policy="mark",
                state="filled", detail="marked closed: expired unquoted legs",
            )
            self._event("exit_filled", group,
                        detail=f"{reason} | expired unquoted legs closed locally n={n}")
            logger.info("exit marked expired %s %s (%s)", group.strategy, group.ticker, reason)
            return mo

        if plan["mode"] == "no_quote":
            mo = ManagedOrder(
                client_order_id=f"exit_{group.group_id}_noquote",
                side="sell", qty=group.qty, policy="none",
                state="error", detail="no quote",
            )
            self._event("exit_order", group, detail=f"{reason} | order error: no quote")
            logger.warning("exit order error for %s: no quote", group.group_id)
            return mo

        close_legs = plan["close_legs"]
        if plan["mode"] == "combo":
            inverted = [
                {"symbol": leg.symbol,
                 "side": "sell" if leg.side == "buy" else "buy",
                 "ratio_qty": round(float(leg.qty) / float(max(group.qty, 1.0)))}
                for leg in close_legs
            ]
            side = "buy" if group.credit else "sell"
            qty = max(int(group.qty), 1)

            def quote_fn() -> Optional[float]:
                try:
                    now_snaps = self.client.get_option_snapshots_bulk(
                        *[leg.symbol for leg in close_legs]) or {}
                except Exception:
                    return None
                value = unit_structure_value(close_legs, now_snaps)
                return abs(value) if value else None

            mo = self.order_manager.execute(
                inverted, qty, LimitWalkPolicy(steps=3), quote_fn,
                side=side,
                client_order_id=f"exit_{group.group_id}_{int(datetime.now(timezone.utc).timestamp())}",
            )
        else:
            # Remaining-leg path: one single-leg order per still-quoted leg.
            last = None
            filled_any = False
            last_px = None
            for leg in close_legs:
                inverted = [{
                    "symbol": leg.symbol,
                    "side": "sell" if leg.side == "buy" else "buy",
                    "ratio_qty": round(float(leg.qty) / float(max(group.qty, 1.0))),
                }]
                side = inverted[0]["side"]
                qty = max(int(leg.qty), 1)

                def quote_fn(leg=leg) -> Optional[float]:
                    try:
                        now_snaps = self.client.get_option_snapshots_bulk(leg.symbol) or {}
                    except Exception:
                        return None
                    mid = leg_mid(leg, now_snaps)
                    return abs(mid) if mid else None

                last = self.order_manager.execute(
                    inverted, qty, LimitWalkPolicy(steps=3), quote_fn,
                    side=side,
                    client_order_id=(
                        f"exit_{group.group_id}_{leg.symbol}_"
                        f"{int(datetime.now(timezone.utc).timestamp())}"
                    ),
                )
                if last.state in ("filled", "partial"):
                    filled_any = True
                    last_px = last.filled_avg_price
            if last is None:
                last = ManagedOrder(
                    client_order_id=f"exit_{group.group_id}_empty",
                    side="sell", qty=group.qty, policy="none",
                    state="error", detail="no quote",
                )
            if filled_any:
                last.state = "filled" if last.state != "partial" else last.state
                last.filled_avg_price = last_px
            # Do NOT mark the group closed just because the near expired.
            # An exhausted/error remaining-leg leave would orphan the far.
            mo = last
            if mo.state == "exhausted":
                from framework.alerts import DEDUPER
                DEDUPER.emit(
                    "remaining_leg_exhaust",
                    f"⚠️ Remaining-leg close exhausted for {group.ticker} "
                    f"({group.group_id}) — far still open.",
                )

        if mo.state in ("filled", "partial"):
            n = close_positions(group.group_id, exit_price=mo.filled_avg_price)
            realized = None
            if mo.filled_avg_price is not None and group.entry_price > 0:
                sign = -1.0 if group.credit else 1.0
                realized = sign * (mo.filled_avg_price - group.entry_price) * 100 * group.qty
            self._event("exit_filled", group, price=mo.filled_avg_price,
                        detail=f"{reason} | legs closed={n} mode={plan['mode']} realized_pnl={realized}")
            logger.info("exit filled %s %s @ %s (%s)", group.strategy, group.ticker,
                        mo.filled_avg_price, reason)
        else:
            self._event("exit_order", group,
                        detail=f"{reason} | order {mo.state}: {mo.detail}")
            logger.warning("exit order %s for %s: %s", mo.state, group.group_id, mo.detail)
        return mo

    # ── proposals --------------------------------------------------------------

    def propose_exit(self, group: PositionGroup, signal: ExitSignal) -> Optional[dict]:
        """Insert a deduped approval card for a time-based exit."""
        legs_txt = " / ".join(
            f"{'SELL' if leg.side == 'sell' else 'BUY'} {cards.code(leg.symbol)}"
            for leg in group.legs
        )
        subtitle = cards.esc(f"rule: {signal.rule} — {signal.reason}")
        body = [
            cards.esc(f"entry ${group.entry_price:.2f} | opened {group.opened_at[:10]}"),
            f"legs: {legs_txt}",
        ]
        card = cards.card_frame(cards.EXIT_EMOJI, f"EXIT? [{group.strategy}] {group.ticker}",
                                subtitle, body)
        pid = exit_proposals_insert(
            group_id=group.group_id,
            strategy=group.strategy,
            ticker=group.ticker,
            rule=signal.rule,
            reason=signal.reason,
            card_text=card,
        )
        if pid is None:
            return None
        self._event("exit_signal", group, detail=f"{signal.rule}: {signal.reason}")
        return exit_proposals_get(pid)

    def decide_exit(self, proposal_id: int, close: bool, decided_by: Optional[int] = None) -> dict:
        """Handle an approval-card decision. close=True executes immediately."""
        row = exit_proposals_get(proposal_id)
        if row is None:
            return {"ok": False, "error": f"exit proposal #{proposal_id} not found"}
        if row["status"] != "pending":
            return {"ok": False, "error": f"exit proposal #{proposal_id} already {row['status']}"}
        now = _utcnow()
        if not close:
            exit_proposals_mark(
                proposal_id, "snoozed",
                snoozed_until=date.today().isoformat(),
                decided_by=decided_by,
                decided_at=now,
            )
            return {"ok": True, "status": "snoozed"}

        groups = {g.group_id: g for g in open_groups()}
        group = groups.get(row["group_id"])
        if group is None:
            exit_proposals_mark(proposal_id, "expired", decided_at=now)
            return {"ok": False, "error": "position no longer open"}
        mo = self.close_group(group, reason=f"approved exit ({row['rule']})")
        status = "closed" if mo.state in ("filled", "partial") else "pending"
        if status == "closed":
            exit_proposals_mark(
                proposal_id, "closed",
                decided_by=decided_by, decided_at=now,
            )
        return {"ok": status == "closed", "order_state": mo.state, "detail": mo.detail,
                "filled_avg_price": mo.filled_avg_price}

    def pending_exit_proposals(self) -> list[dict]:
        return exit_proposals_list_pending()

    # ── internals ----------------------------------------------------------------

    def _event(self, event_type: str, group: PositionGroup,
               price: Optional[float] = None, detail: str = "") -> None:
        trade_events_insert(
            event_type,
            symbol=group.ticker,
            strategy=group.strategy,
            qty=group.qty,
            price=price,
            detail=detail,
        )

