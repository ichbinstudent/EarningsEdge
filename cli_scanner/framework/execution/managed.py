"""managed_positions read/write helpers.

Writers: the approval execution path (entries) and the reconciler (external
closes). Readers: reconciliation, assignment guards, exit evaluation.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from typing import Optional

from earnings_edge.db import (
    managed_positions_close as _mp_close,
    managed_positions_list,
    managed_positions_open,
    managed_positions_set_exit_by,
    trade_events_insert,
)


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def record_open_positions(
    legs: list[dict],
    strategy: str,
    group_id: str,
    order_id: Optional[str] = None,
    entry_price: Optional[float] = None,
    metadata: Optional[dict] = None,
    exit_by: Optional[date] = None,
) -> int:
    """Insert one open row per leg for a submitted structure. Returns count.

    ``exit_by``: structural deadline computed once at entry (e.g. a
    calendar's near-leg expiry) — see ScheduledExit in
    framework.positions.exits. Same value stored on every leg row in the
    group; None for structures with no differential-expiry deadline.
    """
    return managed_positions_open(
        legs, strategy, group_id,
        order_id=order_id, entry_price=entry_price,
        metadata=metadata, exit_by=exit_by,
    )


def open_positions(strategy: Optional[str] = None) -> list[dict]:
    return managed_positions_list(strategy=strategy)


def open_groups() -> list:
    """Open rows collapsed into PositionGroups (keyed by group_id)."""
    from ..positions.exits import CREDIT_SIDES, LegPos, PositionGroup

    groups: dict[str, PositionGroup] = {}
    for row in open_positions():
        gid = row["group_id"] or f"__row_{row['id']}"
        meta = json.loads(row.get("metadata") or "{}")
        g = groups.get(gid)
        if g is None:
            event_date = None
            if meta.get("earnings_date"):
                try:
                    event_date = datetime.strptime(meta["earnings_date"], "%Y-%m-%d").date()
                except ValueError:
                    pass
            side = meta.get("side") or ""
            exit_by = None
            if row.get("exit_by"):
                try:
                    exit_by = datetime.strptime(row["exit_by"], "%Y-%m-%d").date()
                except ValueError:
                    pass
            g = PositionGroup(
                group_id=gid,
                strategy=row["strategy"],
                legs=[],
                entry_price=float(row.get("entry_price") or 0),
                opened_at=row["opened_at"],
                credit=bool(meta.get("credit", side in CREDIT_SIDES)),
                event_date=event_date,
                qty=int(row.get("qty") or 1),
                exit_by=exit_by,
            )
            groups[gid] = g
        expiry = None
        if meta.get("expiry"):
            try:
                expiry = datetime.strptime(meta["expiry"], "%Y-%m-%d").date()
            except ValueError:
                pass
        g.legs.append(LegPos(
            symbol=row["symbol"],
            side=meta.get("leg_side") or "buy",
            qty=float(row.get("qty") or 1),
            option_type=meta.get("option_type") or "",
            strike=float(meta.get("strike") or 0),
            expiry=expiry,
        ))
    return list(groups.values())


def close_positions(
    group_id: str,
    exit_price: Optional[float] = None,
    closed_at: Optional[str] = None,
) -> int:
    """Mark all open rows in a group closed. Returns rows updated."""
    return _mp_close(group_id, exit_price=exit_price, closed_at=closed_at)


def backfill_exit_by() -> int:
    """Set ``exit_by`` on open multi-expiry groups from the earliest leg expiry."""
    n = 0
    for group in open_groups():
        if group.exit_by is not None:
            continue
        expiries = {leg.expiry for leg in group.legs if leg.expiry is not None}
        if len(expiries) < 2:
            continue
        exit_by = min(expiries).isoformat()
        managed_positions_set_exit_by(group.group_id, exit_by)
        n += 1
    return n


def mark_group_closed(
    group_id: str,
    reason: str,
    ticker: str = "",
    strategy: str = "",
) -> int:
    """Local flatten: close the group and write a trade_event with ``reason``."""
    n = close_positions(group_id)
    trade_events_insert(
        "exit_filled",
        symbol=ticker or None,
        strategy=strategy or None,
        detail=f"local flatten: {reason}",
    )
    return n
