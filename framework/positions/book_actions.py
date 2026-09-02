"""Operator book actions: close / adopt / ignore / mark-local-closed.

All writes audit ``trade_events``. Broker mutations go through
``client.close_position`` (one symbol). Local group close uses
``mark_group_closed`` only after the broker side is gone or the close filled.
"""
from __future__ import annotations

import logging
from datetime import date
from typing import Optional

from earnings_edge.db import adopted_positions_insert, trade_events_insert

from ..execution.managed import mark_group_closed, open_groups, record_open_positions
from .book import ticker_of
from .guards import parse_occ

logger = logging.getLogger("framework.positions.book_actions")


def close_symbol(client, symbol: str, *, by: str = "operator") -> dict:
    """Close one broker symbol, then drop matching local open rows if gone."""
    try:
        resp = client.close_position(symbol)
    except Exception as exc:
        _event("close_failed", symbol, None, detail=f"{by}: {exc}")
        return {"ok": False, "error": str(exc)}
    # If the local book still has this symbol, leave it — reconcile/next
    # close pass will mark missing. If it was the last live leg of a group
    # and the broker accepted, we still wait for reconcile unless the
    # caller closes the group explicitly.
    _event("close_submitted", symbol, None,
           detail=f"{by}: broker accepted {getattr(resp, 'get', lambda k, d=None: d)('id') if isinstance(resp, dict) else ''}")
    if isinstance(resp, dict):
        return {"ok": True, "order_id": resp.get("id"), "status": resp.get("status")}
    return {"ok": True, "order_id": None, "status": "submitted"}


def close_group_at_broker(client, group_id: str, *, by: str = "operator") -> dict:
    """Close every local leg that is still at the broker; mark group closed
    only if every close succeeded or the symbol was already gone."""
    groups = {g.group_id: g for g in open_groups()}
    group = groups.get(group_id)
    if group is None:
        return {"ok": False, "error": f"group {group_id} not open"}
    errors = []
    closed = []
    broker_syms = set()
    try:
        broker_syms = {p.get("symbol") for p in client.get_positions() if p.get("symbol")}
    except Exception as exc:
        return {"ok": False, "error": f"get_positions failed: {exc}"}
    for leg in group.legs:
        if leg.symbol not in broker_syms:
            continue
        result = close_symbol(client, leg.symbol, by=by)
        if result.get("ok"):
            closed.append(leg.symbol)
        else:
            errors.append(f"{leg.symbol}: {result.get('error')}")
    if errors:
        return {"ok": False, "error": "; ".join(errors), "closed": closed}
    n = mark_group_closed(group_id, f"{by}: closed at broker",
                          ticker=group.ticker, strategy=group.strategy)
    return {"ok": True, "closed": closed, "local_rows": n}


def adopt_orphan(broker_pos: dict, *, strategy: str = "unmanaged",
                 by: str = "operator") -> dict:
    """Book an orphan broker position into managed_positions."""
    symbol = broker_pos.get("symbol")
    if not symbol:
        return {"ok": False, "error": "no symbol"}
    parsed = parse_occ(symbol)
    side = "sell" if (broker_pos.get("side") or "").lower() == "short" else "buy"
    qty = abs(float(broker_pos.get("qty") or 1))
    expiry = parsed.expiry if parsed else None
    legs = [{
        "symbol": symbol,
        "side": side,
        "ratio_qty": qty,
        "option_type": parsed.option_type if parsed else "",
        "strike": parsed.strike if parsed else 0.0,
        "expiry": expiry,
    }]
    exit_by = expiry
    gid = f"adopt-{symbol}"
    record_open_positions(
        legs, strategy, group_id=gid,
        entry_price=_f(broker_pos.get("avg_entry_price")),
        exit_by=exit_by if isinstance(exit_by, date) else None,
        metadata={"side": "ORPHAN", "adopted_by": by, "earnings_date": None},
    )
    _event("adopted", symbol, strategy, detail=f"{by}: adopted orphan")
    return {"ok": True, "group_id": gid}


def ignore_orphan(symbol: str, *, by: str = "operator") -> dict:
    """Treat as baseline (DAL-style) so reconcile stops alerting."""
    adopted_positions_insert(symbol)
    _event("ignored", symbol, None, detail=f"{by}: ignored orphan")
    return {"ok": True}


def mark_missing_closed(group_id: str, *, by: str = "operator") -> dict:
    groups = {g.group_id: g for g in open_groups()}
    group = groups.get(group_id)
    if group is None:
        return {"ok": False, "error": "not open"}
    n = mark_group_closed(
        group_id, f"{by}: broker already flat",
        ticker=group.ticker, strategy=group.strategy,
    )
    return {"ok": True, "local_rows": n}


def _event(event_type: str, symbol: Optional[str], strategy: Optional[str],
           detail: str = "") -> None:
    trade_events_insert(
        event_type, symbol=symbol, strategy=strategy, detail=detail,
    )


def _f(value):
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def find_broker_pos(positions: list[dict], symbol: str) -> Optional[dict]:
    for p in positions:
        if p.get("symbol") == symbol:
            return p
    return None


def find_group_id(symbol: str) -> Optional[str]:
    for g in open_groups():
        if any(leg.symbol == symbol for leg in g.legs):
            return g.group_id
    return None
