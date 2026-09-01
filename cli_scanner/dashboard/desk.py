"""Mini App desk: broker-truth book + inbox snapshot and write actions.

Same functions the Telegram bot calls — no second implementation.
"""
from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Optional

from earnings_edge.bot_views import book_action_banner, pending_exits
from earnings_edge.db import (
    job_runs_failed,
    pending_trades_brief,
    table_exists,
    trade_events_list,
)
from earnings_edge.db.engine import configure
from earnings_edge.inbox import assemble_inbox
from framework.execution.managed import open_groups
from framework.positions.book import classify_book
from framework.risk.killswitch import KillSwitch

from earnings_edge.db import adopted_positions_symbols, risk_state_get

ROOT = Path(__file__).resolve().parent.parent


def db_path() -> Path:
    return Path(os.environ.get("DASH_DB", ROOT / "data" / "earnings_ml.db"))


def _kill_status_ro() -> dict:
    """Read kill-switch state without INSERT (desk GETs must stay read-only)."""
    try:
        row = risk_state_get(ensure=False)
    except Exception:
        return {"halted": False, "reason": None, "tripped_by": None}
    if not row:
        return {"halted": False, "reason": None, "tripped_by": None}
    return {
        "halted": bool(row.get("halted")),
        "reason": row.get("reason"),
        "tripped_by": row.get("tripped_by"),
    }


def _ignored_symbols() -> set[str]:
    try:
        return adopted_positions_symbols()
    except Exception:
        return set()


def _item(it) -> dict:
    d = asdict(it)
    for key in ("event_date", "expiry"):
        if d.get(key) is not None:
            d[key] = str(d[key])
    return d


def load_desk(*, get_positions: Optional[Callable] = None) -> dict:
    """Book + inbox + kill switch for the Mini App desk."""
    configure(db_path())
    broker, err = [], None
    if get_positions is not None:
        try:
            broker = list(get_positions() or [])
        except Exception as exc:
            err = str(exc)[:160]
            broker = []
    book = classify_book(open_groups(), broker, ignored=_ignored_symbols())
    ks = _kill_status_ro()
    try:
        entries = pending_trades_brief(30) if table_exists("pending_trades") else []
        exits = pending_exits()
        raw_orphans = trade_events_list(event_type="orphan_found", limit=20) if table_exists("trade_events") else []
        raw_assignments = (
            trade_events_list(event_type="assignment_detected", limit=20)
            if table_exists("trade_events") else []
        )
        jobs = job_runs_failed(10) if table_exists("job_runs") else []
        
        # Filter out stale events that are no longer true
        broker_syms = {p.get("symbol") for p in broker if p.get("symbol")}
        managed_syms = {leg.symbol for g in open_groups() for leg in g.legs}
        ignored_syms = _ignored_symbols()

        def _filter_events(events):
            valid = []
            seen = set()
            for row in events:
                sym = row.get("symbol")
                if not sym or sym in seen:
                    continue
                seen.add(sym)
                if sym not in broker_syms:
                    continue  # Gone from broker
                if sym in managed_syms:
                    continue  # We adopted it
                if sym in ignored_syms:
                    continue  # We explicitly ignored it
                valid.append(row)
            return valid

        orphans = _filter_events(raw_orphans)
        assignments = _filter_events(raw_assignments)
    except Exception:
        entries, exits, orphans, assignments, jobs = [], [], [], [], []
    inbox = assemble_inbox(
        entries=entries, exits=exits, orphans=orphans,
        assignments=assignments, jobs=jobs)
    return {
        "kill": {
            "halted": bool(ks.get("halted")),
            "reason": ks.get("reason"),
            "tripped_by": ks.get("tripped_by"),
        },
        "broker_error": err,
        "broker_count": len(broker) if get_positions is not None and err is None else None,
        "book": {
            "managed": [_item(i) for i in book.managed],
            "orphan": [_item(i) for i in book.orphan],
            "missing": [_item(i) for i in book.missing],
        },
        "inbox": [
            {
                "kind": i.kind,
                "item_id": i.item_id,
                "ticker": i.ticker,
                "detail": i.detail,
                "strategy": i.strategy,
                "expired": i.expired,
                "actions": list(i.actions),
            }
            for i in inbox.items
        ],
    }


def run_desk_action(
    op: str,
    payload: dict,
    *,
    by: str,
    client=None,
    store=None,
) -> dict:
    """Dispatch one operator action. Returns ``{ok, banner, ...}``."""
    from framework.positions import book_actions as ba

    configure(db_path())
    op = (op or "").strip()
    if op == "halt":
        KillSwitch().trip("manual halt via mini app", by)
        return {"ok": True, "banner": "🛑 Kill switch tripped."}
    if op == "resume":
        KillSwitch().resume(by)
        return {"ok": True, "banner": "✅ Kill switch released."}

    if op == "adopt":
        symbol = str(payload.get("symbol") or "")
        if client is None:
            return {"ok": False, "banner": "⚠️ no broker client"}
        pos = ba.find_broker_pos(client.get_positions(), symbol)
        if not pos:
            return {"ok": False, "banner": book_action_banner("adopt", {"ok": False, "error": "symbol not at broker"}, symbol)}
        result = ba.adopt_orphan(pos, by=by)
        return {"ok": result.get("ok"), "banner": book_action_banner("adopt", result, symbol), **result}
    if op == "ignore":
        symbol = str(payload.get("symbol") or "")
        result = ba.ignore_orphan(symbol, by=by)
        return {"ok": result.get("ok"), "banner": book_action_banner("ignore", result, symbol), **result}
    if op == "close":
        symbol = str(payload.get("symbol") or "")
        if client is None:
            return {"ok": False, "banner": "⚠️ no broker client"}
        result = ba.close_symbol(client, symbol, by=by)
        return {"ok": result.get("ok"), "banner": book_action_banner("close", result, symbol), **result}
    if op == "mark_closed":
        gid = str(payload.get("group_id") or "")
        result = ba.mark_missing_closed(gid, by=by)
        return {"ok": result.get("ok"), "banner": book_action_banner("mark_closed", result, gid), **result}

    if op in ("exec", "skip"):
        if store is None:
            return {"ok": False, "banner": "⚠️ no proposal store"}
        pid = int(payload.get("id") or 0)
        from earnings_edge.trade_approval import execute_proposal, reject_proposal
        if op == "exec":
            result = execute_proposal(store, pid, decided_by=_uid(by))
        else:
            result = reject_proposal(store, pid, decided_by=_uid(by))
        ok = bool(result.get("ok"))
        banner = ("✅ Executed." if op == "exec" and ok else
                  "❌ Skipped." if op == "skip" and ok else
                  f"⚠️ {result.get('error') or result}")
        return {"ok": ok, "banner": banner, **result}

    if op in ("exit_close", "exit_snooze"):
        pid = int(payload.get("id") or 0)
        from framework.positions.manager import ExitManager
        if client is None:
            return {"ok": False, "banner": "⚠️ no broker client"}
        result = ExitManager(client).decide_exit(
            pid, op == "exit_close", decided_by=_uid(by))
        ok = bool(result.get("ok"))
        banner = ("🔒 Exit filled." if op == "exit_close" and ok else
                  "⏰ Exit snoozed." if op == "exit_snooze" and ok else
                  f"⚠️ {result.get('error') or result}")
        return {"ok": ok, "banner": banner, **result}

    return {"ok": False, "banner": f"⚠️ unknown op {op}"}


def _uid(by: str) -> Optional[int]:
    try:
        if by.startswith("webapp:"):
            return int(by.split(":", 1)[1])
        return int(by)
    except (TypeError, ValueError):
        return None


def open_write_conn():
    """Configure the engine on the dashboard DB path (no raw connection)."""
    return configure(db_path())
