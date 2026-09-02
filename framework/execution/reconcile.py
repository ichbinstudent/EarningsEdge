"""Broker reconciliation: local managed positions vs. actual broker state.

Runs at bot startup and periodically. Diffs ``client.get_positions()`` against
the ``managed_positions`` table so the bot notices fills that happened while
it was down, manual broker-side trades, and positions closed externally.
Populates the ``alpaca_positions`` snapshot table and ``trade_events`` — the
dashboard panels for both already exist but had no writer until now.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from earnings_edge.db import (
    adopted_positions_insert,
    adopted_positions_symbols,
    alpaca_positions_insert,
    ff_ladders_recent,
    managed_positions_close_by_id,
    managed_positions_list,
    risk_events_insert,
    table_exists,
    trade_events_insert,
)

logger = logging.getLogger("framework.execution.reconcile")


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ReconcileReport:
    run_at: str = ""
    broker_count: int = 0
    matched: int = 0
    orphans: list[str] = field(default_factory=list)       # at broker, unknown locally
    closed_externally: list[str] = field(default_factory=list)  # local open, gone at broker
    assignments: list[str] = field(default_factory=list)   # stock under a short-call ticker
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"broker={self.broker_count} matched={self.matched} "
            f"orphans={len(self.orphans)} closed={len(self.closed_externally)} "
            f"assignments={len(self.assignments)} errors={len(self.errors)}"
        )


class Reconciler:
    def __init__(self, client):
        self.client = client

    def run(self) -> ReconcileReport:
        report = ReconcileReport(run_at=_utcnow())
        
        # 0. Cancel any hanging/orphaned limit orders on startup
        try:
            open_orders = self.client.get_orders(status="open")
            for o in open_orders:
                oid = o.get("id")
                try:
                    self.client.cancel_order(oid)
                    logger.info("reconcile: cancelled hanging order %s", oid)
                except Exception as cancel_exc:
                    logger.warning("reconcile: failed to cancel hanging order %s: %s", oid, cancel_exc)
        except Exception as exc:
            report.errors.append(f"cancel_hanging_orders failed: {exc}")

        try:
            broker_positions = self.client.get_positions()
        except Exception as exc:
            report.errors.append(f"get_positions failed: {exc}")
            logger.error("reconcile: get_positions failed: %s", exc)
            return report
        report.broker_count = len(broker_positions)
        broker_by_symbol = {p.get("symbol"): p for p in broker_positions if p.get("symbol")}

        local_open = managed_positions_list()
        local_by_symbol = {r["symbol"]: r for r in local_open}

        # One-time baseline: on the very first reconcile (no managed positions
        # and no adoption history), adopt all current broker positions as
        # pre-existing 'unmanaged' — no orphan alerts for what was already there.
        adopted = adopted_positions_symbols()
        if not adopted and not local_by_symbol and broker_by_symbol:
            ts0 = report.run_at
            for sym in broker_by_symbol:
                adopted_positions_insert(sym, ts0)
            risk_events_insert(
                "baseline_adopted",
                f"{len(broker_by_symbol)} pre-existing broker positions adopted as unmanaged",
                ts=ts0,
            )
            adopted = set(broker_by_symbol)
            logger.info("reconcile: baseline adopted %d pre-existing positions", len(adopted))

        ts = report.run_at
        # 1. Snapshot every broker position (dashboard panel + attribution).
        for sym, pos in broker_by_symbol.items():
            local = local_by_symbol.get(sym)
            strategy = local["strategy"] if local else "unmanaged"
            alpaca_positions_insert(
                ts=ts, symbol=sym,
                qty=_f(pos.get("qty")), side=pos.get("side"),
                avg_entry_price=_f(pos.get("avg_entry_price")),
                current_price=_f(pos.get("current_price")),
                market_value=_f(pos.get("market_value")),
                unrealized_pl=_f(pos.get("unrealized_pl")),
                strategy=strategy, managed=1 if local else 0,
            )

        # 2. Matched positions.
        for sym in broker_by_symbol.keys() & local_by_symbol.keys():
            report.matched += 1

        # 3. Orphans: at the broker but neither tracked locally nor adopted at
        #    baseline (manual trade or fill while the bot was down).
        raw_orphans = list(broker_by_symbol.keys() - local_by_symbol.keys() - adopted)
        still_orphans = self._adopt_ff_ladder_orphans(raw_orphans, broker_by_symbol)
        for sym in still_orphans:
            report.orphans.append(sym)
            pos = broker_by_symbol[sym]
            self._event("orphan_found", sym, "unmanaged",
                        qty=_f(pos.get("qty")), price=_f(pos.get("current_price")),
                        detail="position at broker with no local record")
            logger.warning("reconcile: orphan position %s (qty=%s)", sym, pos.get("qty"))
        if report.orphans:
            from framework.alerts import DEDUPER
            shown = ", ".join(report.orphans[:8])
            extra = f" (+{len(report.orphans) - 8} more)" if len(report.orphans) > 8 else ""
            DEDUPER.emit(
                "orphan",
                f"⚠️ {len(report.orphans)} orphan position(s) at broker: {shown}{extra}",
            )

        # 4. Closed externally: tracked open locally but gone at the broker.
        for sym in local_by_symbol.keys() - broker_by_symbol.keys():
            row = local_by_symbol[sym]
            report.closed_externally.append(sym)
            managed_positions_close_by_id(row["id"], closed_at=ts)
            self._event("close_detected", sym, row["strategy"],
                        qty=row["qty"], detail="position no longer at broker; marked closed")
            logger.info("reconcile: %s closed externally (strategy=%s)", sym, row["strategy"])
        if report.closed_externally:
            from framework.alerts import DEDUPER
            shown = ", ".join(report.closed_externally[:8])
            extra = (f" (+{len(report.closed_externally) - 8} more)"
                     if len(report.closed_externally) > 8 else "")
            DEDUPER.emit(
                "missing",
                f"⚠️ {len(report.closed_externally)} local position(s) missing at broker: "
                f"{shown}{extra}",
            )

        # 5. Assignment: stock (no OCC parse) at broker under a ticker that
        #    still has a short call on the local book (expired near → shares).
        assigned = classify_assignments(broker_positions, local_open)
        for sym in assigned:
            report.assignments.append(sym)
            self._event("assignment_detected", sym, "unmanaged",
                        detail="stock position under a short-call ticker")
            logger.warning("reconcile: assignment %s", sym)

        logger.info("reconcile: %s", report.summary())
        return report

    def _adopt_ff_ladder_orphans(self, orphans: list[str],
                                 broker_by_symbol: dict) -> list[str]:
        """Bind broker legs that match a recent ff_ladders row into managed_positions.

        Expired/disarmed ladders that filled after we stopped polling used to
        sit as orphans (ATLO/BA) and were invisible to ExitManager.
        """
        if not orphans:
            return []
        if not table_exists("ff_ladders"):
            return list(orphans)
        import json as _json
        try:
            rows = ff_ladders_recent(80)
        except Exception:
            return list(orphans)
        remaining = set(orphans)
        from .managed import record_open_positions
        for row in rows:
            try:
                cand = _json.loads(row["candidate_json"] or "{}")
            except Exception:
                continue
            near, far = cand.get("near_symbol"), cand.get("far_symbol")
            hit = [s for s in (near, far) if s in remaining]
            if not hit:
                continue
            legs = []
            for sym, side in ((near, "sell"), (far, "buy")):
                if not sym or sym not in remaining:
                    continue
                pos = broker_by_symbol.get(sym) or {}
                legs.append({
                    "symbol": sym, "side": side,
                    "ratio_qty": abs(_f(pos.get("qty")) or 1),
                    "option_type": "call",
                    "strike": cand.get("strike"),
                    "expiry": cand.get("near_expiry") if sym == near else cand.get("far_expiry"),
                })
            if not legs:
                continue
            try:
                record_open_positions(
                    legs, "ff_ladder",
                    group_id=str(row["order_id"] or row["id"]),
                    order_id=row["order_id"],
                    entry_price=_f((broker_by_symbol.get(hit[0]) or {}).get("avg_entry_price")),
                    metadata={"side": "CALENDAR", "credit": False,
                              "earnings_date": cand.get("earnings_date"),
                              "adopted_from": "ff_ladders"},
                )
            except Exception as exc:
                logger.warning("reconcile: ladder adopt failed %s: %s", hit, exc)
                continue
            for sym in hit:
                remaining.discard(sym)
                self._event("orphan_adopted", sym, "ff_ladder",
                            detail=f"auto-adopted from ff_ladders id={row['id']}")
                logger.info("reconcile: adopted ladder orphan %s (ff_ladders %s)",
                            sym, row["id"])
        return list(remaining)

    def _event(self, event_type: str, symbol: str, strategy: Optional[str],
               qty: Optional[float] = None, price: Optional[float] = None,
               detail: str = "") -> None:
        trade_events_insert(
            event_type, symbol=symbol, strategy=strategy,
            qty=qty, price=price, detail=detail,
        )


def classify_assignments(broker_positions: list, local_open: list) -> list[str]:
    """Stock symbols at the broker whose underlying still has a short call locally."""
    import json as _json
    from ..positions.guards import occ_underlying, parse_occ

    short_call_tickers: set[str] = set()
    for row in local_open:
        meta = {}
        try:
            meta = _json.loads(row["metadata"] or "{}") if isinstance(row, dict) or hasattr(row, "keys") else {}
            if not isinstance(meta, dict):
                meta = {}
        except Exception:
            meta = {}
        symbol = row["symbol"] if not isinstance(row, dict) else row.get("symbol")
        side = (meta.get("leg_side") or "").lower()
        otype = (meta.get("option_type") or "").lower()
        if side == "sell" and otype == "call" and symbol:
            t = occ_underlying(symbol)
            if t:
                short_call_tickers.add(t)
    found: list[str] = []
    for p in broker_positions:
        if not isinstance(p, dict):
            continue
        sym = p.get("symbol") or ""
        if parse_occ(sym) is not None:
            continue
        if sym in short_call_tickers:
            found.append(sym)
    return found


def _f(value) -> Optional[float]:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
