"""Exit manager: evaluates exit rules over open position groups and acts.

- Profit-target / stop-loss breaches → *auto-close* via the OrderManager's
  limit walk (paper), notified after the fact.
- Time exits → approval cards in ``exit_proposals`` (deduped per group);
  post-event exits (``days_after_event``) and the near-expiry
  ``ScheduledExit`` are auto.
- Escalation: when a combo close stalls on the near-leg expiry day (or a
  stop/post-event exit in the last hour), the group is legged out with
  single-leg orders walked to the natural price, short legs first.
- Assignment: short shares under an open managed debit group (an assigned
  short call) are bought back and the rest of the group closed.

The kill switch is deliberately NOT consulted: it gates entries only.
Closing risk is always allowed.

Fill bookkeeping: positions marked closed with the exit price, trade_events
written, realized PnL recorded — which also feeds lifecycle promotion stats.
"""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime

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
from ..execution.book_lock import book_lock
from ..execution.managed import backfill_exit_by, close_positions, open_groups
from ..execution.order_manager import LimitWalkPolicy, ManagedOrder, NaturalWalkPolicy, OrderManager
from .exits import (
    ExitSignal,
    LegPos,
    MarketView,
    PositionGroup,
    build_exit_rules,
    dedupe_legs,
    leg_mid,
    realized_pnl_dollars,
    remaining_close_plan,
    unit_structure_value,
)

logger = logging.getLogger("framework.positions.manager")

# A triggered stop/post-event exit whose combo close fails inside the last
# hour legs out instead of being carried overnight.
LEG_OUT_LAST_MINUTES = 60
# Shares per equity-option contract (assignment delivers this many).
CONTRACT_MULTIPLIER = 100


def _num(value) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _leg_book(leg: LegPos, snaps: dict) -> tuple[float, float] | None:
    """(bid, ask) for one leg when it has a usable ask; bid may be 0."""
    q = (snaps.get(leg.symbol) or {}).get("latestQuote") or {}
    bid, ask = _num(q.get("bp")), _num(q.get("ap"))
    if ask is None or ask <= 0:
        return None
    return (bid if bid is not None and 0 <= bid <= ask else 0.0), ask


def _utcnow() -> str:
    return datetime.now(UTC).isoformat()


def _sessions_after_reaction(cal, event_date: date, timing: str | None, today: date) -> int | None:
    """Sessions since the first session whose prices contain the announcement.

    Before-open reporters react on the event date itself; after-close — and
    unknown timing, the conservative reading — on the next session. Returns
    0 on the reaction session, >0 after it, -1 before it; None when the
    calendar can't place the dates.
    """
    t = (timing or "").lower()
    before_open = "pre" in t or "bmo" in t or "before" in t
    try:
        reaction = cal.next_session(event_date) if before_open else cal.next_session_after(event_date)
        if today < reaction:
            return -1
        return max(len(cal.sessions_between(reaction, today)) - 1, 0)
    except (IndexError, ValueError, KeyError) as exc:
        logger.warning("exit eval: cannot place event %s on the calendar: %s", event_date, exc)
        return None


class ExitManager:
    def __init__(
        self,
        client,
        registry: StrategyRegistry | None = None,
        order_manager: OrderManager | None = None,
        today: date | None = None,
    ):
        self.client = client
        self.registry = registry or get_registry()
        self.order_manager = order_manager or OrderManager(client)
        self._today = today

    # ── evaluation ---------------------------------------------------------

    def evaluate_all(self) -> dict:
        """One pass over all open groups. Returns stats + messages to push."""
        with book_lock():
            return self._evaluate_all_locked()

    def _evaluate_all_locked(self) -> dict:
        out: dict = {"groups": 0, "auto_closed": [], "proposed": [], "held": 0, "errors": []}
        # Groups booked without a structural deadline (every FF ladder fill
        # before 2026-09-29) get one from their leg expiries — otherwise
        # ScheduledExit is a no-op and the short leg rides into assignment.
        try:
            n = backfill_exit_by()
            if n:
                logger.info("exit eval: backfilled exit_by on %d group(s)", n)
        except Exception as exc:
            logger.warning("exit eval: exit_by backfill failed: %s", exc)
            out["errors"].append(f"exit_by backfill failed: {exc}")
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
        today = self._today or datetime.now(UTC).date()
        minutes_to_close = self._minutes_to_close()
        minutes_since_open = None
        if minutes_to_close is not None:
            try:
                opened = cal.session_open(today)
                minutes_since_open = max(int((datetime.now(UTC) - opened).total_seconds() // 60), 0)
            except (KeyError, ValueError, IndexError) as exc:
                logger.info("exit eval: session open unknown for %s (%s)", today, exc)

        handled = self._flatten_assignments(groups, out, minutes_to_close)
        for group in groups:
            if group.group_id in handled:
                continue
            try:
                self._evaluate_group(group, snaps, cal, today, out, minutes_to_close, minutes_since_open)
            except Exception as exc:
                logger.exception("exit eval failed for %s", group.group_id)
                out["errors"].append(f"{group.group_id}: {exc}")
        return out

    def _minutes_to_close(self) -> int | None:
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

    def _evaluate_group(
        self,
        group: PositionGroup,
        snaps: dict,
        cal,
        today: date,
        out: dict,
        minutes_to_close: int | None = None,
        minutes_since_open: int | None = None,
    ) -> None:
        cfg = self.registry.get(group.strategy)
        rules = build_exit_rules(cfg.exits) if cfg else []
        if not rules:
            return

        opened_date = group.opened_at[:10]
        try:
            sessions_since = max(
                len(cal.sessions_between(datetime.strptime(opened_date, "%Y-%m-%d").date(), today)) - 1, 0
            )
        except ValueError:
            sessions_since = 0
        sessions_until_event = None
        sessions_after_event = None
        if group.event_date is not None:
            if group.event_date < today:
                sessions_until_event = 0  # event day/past → time exits fire
            else:
                sessions_until_event = max(len(cal.sessions_between(today, group.event_date)) - 1, 0)
            sessions_after_event = _sessions_after_reaction(cal, group.event_date, group.timing, today)

        value = unit_structure_value(group.legs, snaps)
        market = MarketView(
            value_now=value,
            today=today,
            sessions_since_open=sessions_since,
            sessions_until_event=sessions_until_event,
            minutes_to_close=minutes_to_close,
            sessions_after_event=sessions_after_event,
            minutes_since_open=minutes_since_open,
        )

        from ..core.control import effective_execution_mode

        toml_mode = cfg.execution_mode if cfg else "approval"
        is_auto = effective_execution_mode(group.strategy, toml_mode) == "auto"

        signal = next((s for r in rules if (s := r.evaluate(group, market))), None)
        if signal is None:
            out["held"] += 1
            return

        if signal.auto or is_auto:
            leg_out = self._leg_out_due(group, signal, today, minutes_to_close)
            mo = self.close_group(group, reason=signal.reason, leg_out=leg_out)
            if mo.state in ("filled", "partial"):
                out["auto_closed"].append(
                    f"🔒 {cards.bold('AUTO-EXIT')} {cards.esc(group.strategy)} "
                    f"{cards.bold(group.ticker)} ({cards.esc(signal.rule)}: "
                    f"{cards.esc(signal.reason)}) @ {mo.filled_avg_price}"
                )
            else:
                out["errors"].append(
                    f"{group.ticker}: exit order {mo.state} ({mo.detail}) — retrying next cycle"
                )
        else:
            row = self.propose_exit(group, signal)
            if row is not None:
                out["proposed"].append(row)

    # ── closing --------------------------------------------------------------

    @staticmethod
    def _leg_out_due(
        group: PositionGroup, signal: ExitSignal, today: date, minutes_to_close: int | None
    ) -> bool:
        """Whether a failed combo close should escalate to single-leg closes.

        - on/after the structural deadline (near-leg expiry day): the short
          leg is about to expire into assignment (2026-09-26 COST);
        - a triggered stop/post-event exit in the session's last hour: don't
          carry it overnight because the combo book was too wide to fill.
        """
        if group.exit_by is not None and today >= group.exit_by:
            return True
        return (
            signal.rule in ("stop_loss", "time")
            and minutes_to_close is not None
            and minutes_to_close <= LEG_OUT_LAST_MINUTES
        )

    def close_group(self, group: PositionGroup, reason: str = "", leg_out: bool = False) -> ManagedOrder:
        """Work a closing order; fall back to remaining-leg closes when the
        combo quote is gone (expired near). Marks the group closed on fill
        or when every remaining unquoted leg is past expiry.

        ``leg_out``: when the combo close does not fill, close the legs one
        by one (short legs first) walking each to its natural price."""
        with book_lock():
            return self._close_group_locked(group, reason, leg_out)

    def _close_group_locked(
        self, group: PositionGroup, reason: str = "", leg_out: bool = False
    ) -> ManagedOrder:
        today = self._today or datetime.now(UTC).date()
        try:
            snaps = self.client.get_option_snapshots_bulk(*[leg.symbol for leg in group.legs]) or {}
        except Exception:
            snaps = {}
        plan = remaining_close_plan(group.legs, snaps, today)

        if plan["mode"] == "expired":
            n = close_positions(group.group_id)
            mo = ManagedOrder(
                client_order_id=f"exit_{group.group_id}_expired",
                side="sell",
                qty=group.qty,
                policy="mark",
                state="filled",
                detail="marked closed: expired unquoted legs",
            )
            self._event("exit_filled", group, detail=f"{reason} | expired unquoted legs closed locally n={n}")
            logger.info("exit marked expired %s %s (%s)", group.strategy, group.ticker, reason)
            return mo

        if plan["mode"] == "no_quote":
            mo = ManagedOrder(
                client_order_id=f"exit_{group.group_id}_noquote",
                side="sell",
                qty=group.qty,
                policy="none",
                state="error",
                detail="no quote",
            )
            self._event("exit_order", group, detail=f"{reason} | order error: no quote")
            logger.warning("exit order error for %s: no quote", group.group_id)
            return mo

        close_legs = plan["close_legs"]
        gid = (group.group_id or "g").replace("-", "")[:12]
        ts = int(datetime.now(UTC).timestamp())
        if plan["mode"] == "combo":
            inverted = [
                {
                    "symbol": leg.symbol,
                    "side": "sell" if leg.side == "buy" else "buy",
                    "ratio_qty": round(float(leg.qty) / float(max(group.qty, 1.0))),
                }
                for leg in close_legs
            ]
            side = "buy" if group.credit else "sell"
            qty = max(int(group.qty), 1)

            def quote_fn() -> float | None:
                try:
                    now_snaps = (
                        self.client.get_option_snapshots_bulk(*[leg.symbol for leg in close_legs]) or {}
                    )
                except Exception:
                    return None
                value = unit_structure_value(close_legs, now_snaps)
                return abs(value) if value else None

            mo = self.order_manager.execute(
                inverted,
                qty,
                LimitWalkPolicy(steps=3),
                quote_fn,
                side=side,
                client_order_id=f"x{gid}{ts}",
            )
            if leg_out and mo.state not in ("filled", "partial"):
                logger.warning(
                    "combo close %s for %s (%s) — legging out", mo.state, group.ticker, group.group_id
                )
                return self._leg_out(group, reason, today, snaps)
        else:
            # Remaining-leg path: one single-leg order per still-quoted leg.
            last = None
            filled_any = False
            last_px = None
            for leg in close_legs:
                inverted = [
                    {
                        "symbol": leg.symbol,
                        "side": "sell" if leg.side == "buy" else "buy",
                        "ratio_qty": round(float(leg.qty) / float(max(group.qty, 1.0))),
                    }
                ]
                side = inverted[0]["side"]
                qty = max(int(leg.qty), 1)

                def quote_fn(leg=leg) -> float | None:
                    try:
                        now_snaps = self.client.get_option_snapshots_bulk(leg.symbol) or {}
                    except Exception:
                        return None
                    mid = leg_mid(leg, now_snaps)
                    return abs(mid) if mid else None

                # urgent: walk to the natural (marketable) instead of 1% off mid
                book = _leg_book(leg, snaps) if leg_out else None
                policy = NaturalWalkPolicy(*book, steps=3) if book else LimitWalkPolicy(steps=3)
                last = self.order_manager.execute(
                    inverted,
                    qty,
                    policy,
                    quote_fn,
                    side=side,
                    client_order_id=f"x{gid}{leg.symbol[-6:]}{ts}",
                )
                if last.state in ("filled", "partial"):
                    filled_any = True
                    last_px = last.filled_avg_price
            if last is None:
                last = ManagedOrder(
                    client_order_id=f"exit_{group.group_id}_empty",
                    side="sell",
                    qty=group.qty,
                    policy="none",
                    state="error",
                    detail="no quote",
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
            fill = abs(mo.filled_avg_price) if mo.filled_avg_price is not None else None
            n = close_positions(group.group_id, exit_price=fill)
            realized = realized_pnl_dollars(group, fill) if fill is not None else None
            self._event(
                "exit_filled",
                group,
                price=fill,
                detail=f"{reason} | legs closed={n} mode={plan['mode']} realized_pnl={realized}",
            )
            logger.info(
                "exit filled %s %s @ %s (%s)", group.strategy, group.ticker, mo.filled_avg_price, reason
            )
        else:
            self._event("exit_order", group, detail=f"{reason} | order {mo.state}: {mo.detail}")
            logger.warning("exit order %s for %s: %s", mo.state, group.group_id, mo.detail)
        return mo

    def _leg_out(self, group: PositionGroup, reason: str, today: date, snaps: dict) -> ManagedOrder:
        """Close the group leg by leg — short legs first — walking each
        single-leg order to its natural price.

        Short first: it carries the assignment risk. If a short can't be
        bought back, the long legs are left on as its hedge (selling them
        would leave a naked short call). Filled legs are closed on the book
        one by one, so a partial leg-out resumes where it stopped next cycle.
        """
        from earnings_edge.db import managed_positions_close_symbol

        held = self._broker_symbols()
        if held is not None:
            group = self._drop_unheld_legs(group, held)
            if not group.legs:
                return ManagedOrder(
                    client_order_id=f"legout_{group.group_id}",
                    side="close",
                    qty=group.qty,
                    policy=NaturalWalkPolicy.name,
                    state="filled",
                    detail="no legs left at the broker",
                )
        legs = sorted(dedupe_legs(group.legs), key=lambda leg: leg.side != "sell")
        gid = (group.group_id or "g").replace("-", "")[:12]
        ts = int(datetime.now(UTC).timestamp())
        closed = 0
        detail = ""
        last: ManagedOrder | None = None
        for leg in legs:
            close_side = "sell" if leg.side == "buy" else "buy"
            book = _leg_book(leg, snaps)
            if book is None:
                if leg.expiry is not None and leg.expiry < today:
                    managed_positions_close_symbol(group.group_id, leg.symbol)
                    closed += 1
                    self._event("exit_filled", group, detail=f"{reason} | leg-out: {leg.symbol} expired")
                    continue
                detail = f"no quote for {leg.symbol}"
                if leg.side == "sell":
                    break
                continue
            bid, ask = book
            if close_side == "sell" and bid <= 0:
                detail = f"no bid for long {leg.symbol}"  # nothing to sell into yet
                continue
            last = self.order_manager.execute(
                [{"symbol": leg.symbol, "side": close_side, "ratio_qty": 1}],
                max(int(leg.qty), 1),
                NaturalWalkPolicy(bid, ask, steps=3),
                lambda bid=bid, ask=ask: (bid + ask) / 2.0,
                side=close_side,
                client_order_id=f"l{gid}{leg.symbol[-8:]}{ts}",
            )
            if last.state in ("filled", "partial"):
                managed_positions_close_symbol(group.group_id, leg.symbol, exit_price=last.filled_avg_price)
                closed += 1
                self._event(
                    "exit_filled",
                    group,
                    price=last.filled_avg_price,
                    detail=f"{reason} | leg-out {close_side} {leg.symbol} @ {last.filled_avg_price}",
                )
                continue
            detail = f"{leg.symbol} {last.state}: {last.detail}"
            if leg.side == "sell":
                break  # keep the hedge while the short is still open

        state = "filled" if closed == len(legs) else "partial" if closed else "exhausted"
        mo = ManagedOrder(
            client_order_id=f"legout_{group.group_id}",
            side="close",
            qty=group.qty,
            policy=NaturalWalkPolicy.name,
            state=state,
            filled_avg_price=last.filled_avg_price if last is not None else None,
            detail=f"leg-out closed {closed}/{len(legs)} legs" + (f"; {detail}" if detail else ""),
        )
        if state != "filled":
            from framework.alerts import DEDUPER

            DEDUPER.emit(
                f"legout_{group.group_id}",
                f"🚨 {group.ticker} ({group.strategy}): urgent close incomplete — {mo.detail}",
            )
        logger.warning("leg-out %s %s: %s", group.ticker, state, mo.detail)
        return mo

    def _flatten_assignments(self, groups: list[PositionGroup], out: dict, minutes_to_close) -> set[str]:
        """Close shares delivered by an assigned short call, then the rest of
        that calendar. Returns the group ids handled this pass.

        Detection is limited to SHORT stock under the ticker of an open
        managed debit group, capped at 100 x the group's contracts — the
        paper account is shared with another system's stock book, so
        anything else is not ours to touch.
        """
        handled: set[str] = set()
        if minutes_to_close is None:
            return handled  # market closed: no stock order possible
        candidates = [g for g in groups if not g.credit and g.ticker]
        getter = getattr(self.client, "get_positions", None)
        if not candidates or getter is None:
            return handled
        try:
            positions = [p for p in (getter() or []) if isinstance(p, dict)]
        except Exception as exc:
            out["errors"].append(f"assignment check: positions fetch failed: {exc}")
            return handled
        from .guards import parse_occ

        stock = {p.get("symbol"): p for p in positions if p.get("symbol") and parse_occ(p["symbol"]) is None}
        for group in candidates:
            pos = stock.get(group.ticker)
            shares = _num((pos or {}).get("qty"))
            if pos is None or shares is None or shares >= 0:
                continue
            cover = int(min(abs(shares), CONTRACT_MULTIPLIER * max(group.qty, 1)))
            from framework.alerts import DEDUPER

            DEDUPER.emit(
                f"assign_{group.group_id}",
                f"🚨 Assignment: {group.ticker} short call assigned → {shares:+.0f} shares "
                f"({group.strategy}). Buying back {cover} and closing the calendar.",
            )
            self._event("assignment_flatten", group, detail=f"broker shares {shares:+.0f}; covering {cover}")
            handled.add(group.group_id)

            def last_trade(ticker=group.ticker) -> float | None:
                try:
                    return self.client.get_stock_latest_trade(ticker)
                except Exception:
                    return None

            mo = self.order_manager.execute(
                [{"symbol": group.ticker, "side": "buy", "ratio_qty": 1}],
                cover,
                LimitWalkPolicy(steps=3),  # stock: final rung 1% through the last trade
                last_trade,
                side="buy",
                client_order_id=f"asg{group.ticker[:6]}{int(datetime.now(UTC).timestamp())}",
            )
            if mo.state not in ("filled", "partial"):
                out["errors"].append(f"{group.ticker}: assigned shares not covered ({mo.state}: {mo.detail})")
                continue
            out["auto_closed"].append(
                f"🔒 {cards.bold('ASSIGNMENT')} {cards.esc(group.strategy)} {cards.bold(group.ticker)}: "
                f"covered {cover} shares @ {mo.filled_avg_price}"
            )
            # The assigned short is gone at the broker; "buying it back" would
            # OPEN a long call. Close only what the broker still holds.
            group = self._drop_unheld_legs(group, {p.get("symbol") for p in positions})
            if group.legs:
                rest = self.close_group(group, reason="assignment cleanup", leg_out=True)
                if rest.state not in ("filled", "partial"):
                    out["errors"].append(
                        f"{group.ticker}: post-assignment close {rest.state} ({rest.detail})"
                    )
        return handled

    def _broker_symbols(self) -> set[str] | None:
        """Symbols currently held at the broker; None when unavailable."""
        getter = getattr(self.client, "get_positions", None)
        if getter is None:
            return None
        try:
            syms = {p.get("symbol") for p in (getter() or []) if isinstance(p, dict) and p.get("symbol")}
        except Exception as exc:
            logger.warning("exit: broker positions unavailable (%s) — trusting the local book", exc)
            return None
        # An empty answer while we hold managed legs is far likelier a glitch
        # than a flat book (reconcile owns "gone at broker"): trust the book.
        return syms or None

    def _drop_unheld_legs(self, group: PositionGroup, held: set[str]) -> PositionGroup:
        """Close book legs the broker no longer holds; return the group without them."""
        from dataclasses import replace

        from earnings_edge.db import managed_positions_close_symbol

        gone = [leg for leg in group.legs if leg.symbol not in held]
        for leg in gone:
            managed_positions_close_symbol(group.group_id, leg.symbol)
            self._event(
                "close_detected", group, detail=f"{leg.symbol} no longer at broker (expired/assigned)"
            )
        return replace(group, legs=[leg for leg in group.legs if leg.symbol in held]) if gone else group

    # ── proposals --------------------------------------------------------------

    def propose_exit(self, group: PositionGroup, signal: ExitSignal) -> dict | None:
        """Insert a deduped approval card for a time-based exit."""
        legs_txt = " / ".join(
            f"{'SELL' if leg.side == 'sell' else 'BUY'} {cards.code(leg.symbol)}" for leg in group.legs
        )
        subtitle = cards.esc(f"rule: {signal.rule} — {signal.reason}")
        body = [
            cards.esc(f"entry ${group.entry_price:.2f} | opened {group.opened_at[:10]}"),
            f"legs: {legs_txt}",
        ]
        card = cards.card_frame(cards.EXIT_EMOJI, f"EXIT? [{group.strategy}] {group.ticker}", subtitle, body)
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

    def decide_exit(self, proposal_id: int, close: bool, decided_by: int | None = None) -> dict:
        """Handle an approval-card decision. close=True executes immediately."""
        row = exit_proposals_get(proposal_id)
        if row is None:
            return {"ok": False, "error": f"exit proposal #{proposal_id} not found"}
        if row["status"] != "pending":
            return {"ok": False, "error": f"exit proposal #{proposal_id} already {row['status']}"}
        now = _utcnow()
        if not close:
            exit_proposals_mark(
                proposal_id,
                "snoozed",
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
                proposal_id,
                "closed",
                decided_by=decided_by,
                decided_at=now,
            )
        return {
            "ok": status == "closed",
            "order_state": mo.state,
            "detail": mo.detail,
            "filled_avg_price": mo.filled_avg_price,
        }

    def pending_exit_proposals(self) -> list[dict]:
        return exit_proposals_list_pending()

    # ── internals ----------------------------------------------------------------

    def _event(
        self, event_type: str, group: PositionGroup, price: float | None = None, detail: str = ""
    ) -> None:
        trade_events_insert(
            event_type,
            symbol=group.ticker,
            strategy=group.strategy,
            qty=group.qty,
            price=price,
            detail=detail,
        )
