"""Live dashboard + Telegram Mini App desk.

Standalone Starlette service (separate process from the Telegram bot) that
reads the shared SQLite DB in WAL mode and pushes panel diffs over a
websocket. Write actions (adopt / execute / halt) go through the same
functions as Telegram and require verified Mini App initData + operator lock.

Run:  .venv/bin/python3.12 -m uvicorn dashboard.server:app --host 127.0.0.1 --port 8503
Public Mini App: set TELEGRAM_WEBAPP_URL=https://… and put HTTPS in front.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocket, WebSocketDisconnect

from dashboard.desk import load_desk, run_desk_action
from dashboard.tg_auth import InitDataError, require_operator
from earnings_edge.db import (
    alpaca_positions_list,
    calendar_call_trades_ml_frame,
    calendar_call_trades_stats,
    ff_backfill_progress,
    ff_ladders_recent,
    pending_trades_recent,
    scan_runs_recent,
    scanner_scan_outputs_latest,
    snapshots_max_scan_date,
    table_exists,
    trade_events_list,
)
from earnings_edge.db.engine import configure

logger = logging.getLogger("earnings_edge.dashboard")

ROOT = Path(__file__).resolve().parent.parent
BOT_HEALTH_URL = os.environ.get("DASH_BOT_HEALTH", "http://127.0.0.1:8502/health")
POLL_SECONDS = float(os.environ.get("DASH_POLL_SECONDS", "3"))

STATIC = Path(__file__).resolve().parent / "static"


def _db_path() -> Path:
    from dashboard.desk import db_path
    return db_path()


# ── panel registry ──────────────────────────────────────────────────────

@dataclass
class Panel:
    id: str
    title: str
    kind: str  # 'stats' | 'table' | 'text'
    provider: Callable[[], Any]


def p_bot_health() -> dict:
    try:
        r = httpx.get(BOT_HEALTH_URL, timeout=2.0)
        h = r.json()
        if h.get("ready") is True:
            bot = "🟢 ready"
        elif "ready" in h:
            bot = "🟡 " + ", ".join(h.get("reasons") or ["not ready"])
        else:
            bot = "🟢 running" if h.get("status") == "ok" else "🟡 degraded"
        return {"stats": {
            "bot": bot,
            "uptime": f"{h.get('uptime_secs', 0) / 3600:.1f} h",
        }}
    except Exception:
        return {"stats": {"bot": "🔴 unreachable", "uptime": "—"}}


def p_ff_backfill() -> dict:
    if not table_exists("ff_snapshots"):
        return {"stats": {"status": "table not created"}}
    prog = ff_backfill_progress()
    total_pairs = prog["total_pairs"]
    done, ok, skipped = prog["done"], prog["ok"], prog["skipped"]
    pct = (done / total_pairs * 100) if total_pairs else 0
    stats = {
        "progress": f"{done} / {total_pairs} ({pct:.0f}%)",
        "ok": ok, "skipped": skipped,
        "last write": prog["last_at"] or "—",
    }
    if done >= total_pairs and total_pairs:
        avg_pr = prog.get("avg_pr")
        stats.update({
            "status": "✅ complete",
            "avg premium_ratio": f"{avg_pr:.2f}" if avg_pr else "—",
            "premium ≥ 1.2": f"{prog.get('rich', 0)} / {prog.get('premium_n', 0)}",
        })
    return {"stats": stats}


def p_latest_scans() -> dict:
    if not table_exists("scan_runs"):
        return {"empty": "no scans yet"}
    rows = scan_runs_recent(10)
    return {"columns": ["scan_timestamp", "scanner_name", "trigger_type",
                        "candidate_count", "take_count", "secs", "success"],
            "rows": rows}


def p_scan_outputs() -> dict:
    if not table_exists("scanner_scan_outputs"):
        return {"empty": "no scan outputs yet"}
    latest, rows = scanner_scan_outputs_latest()
    if not latest:
        return {"empty": "no scan outputs yet"}
    return {"title_suffix": latest, "columns": ["ticker", "earnings_date", "tier",
                                                "display_status", "price"],
            "rows": rows}


def p_proposals() -> dict:
    if not table_exists("pending_trades"):
        return {"empty": "no proposals yet"}
    rows = pending_trades_recent(20)
    return {"columns": ["id", "created_at", "strategy", "ticker", "status",
                        "score", "decided_at"], "rows": rows}


def p_ff_ladders() -> dict:
    if not table_exists("ff_ladders"):
        return {"empty": "no ladders yet (table created on first arm)"}
    rows = ff_ladders_recent(20)
    shown = [{k: r.get(k) for k in ("id", "ticker", "status", "rung", "order_id", "updated_at")}
             for r in rows]
    return {"columns": ["id", "ticker", "status", "rung", "order_id", "updated_at"],
            "rows": shown}


def p_positions() -> dict:
    if not table_exists("alpaca_positions"):
        return {"empty": "no positions yet (table created by trading service)"}
    rows = alpaca_positions_list(20)
    cols = list(rows[0].keys()) if rows else []
    return {"columns": cols, "rows": rows}


def p_events() -> dict:
    if not table_exists("trade_events"):
        return {"empty": "no trade events yet"}
    rows = trade_events_list(limit=20)
    cols = list(rows[0].keys()) if rows else []
    return {"columns": cols, "rows": rows}


def p_calendar_stats() -> dict:
    if not table_exists("calendar_call_trades"):
        return {"empty": "no calendar trades yet"}
    r = calendar_call_trades_stats()
    return {"stats": {"trades": r["n"], "avg debit": r["avg_debit"] or "—",
                      "closed": r["closed"], "avg P&L": r["avg_pnl"] or "—"}}


def p_picks() -> dict:
    if not table_exists("snapshots"):
        return {"empty": "no snapshots yet"}

    latest_str = snapshots_max_scan_date()
    if not latest_str:
        return {"empty": "no snapshots yet"}

    from datetime import datetime
    import pandas as pd
    from earnings_edge.picks import generate_picks

    as_of = datetime.strptime(latest_str[:10], "%Y-%m-%d").date()
    picks = generate_picks(as_of)
    
    frames = []
    for name, df in picks.items():
        if df.empty:
            continue
        df = df.head(5).copy()
        df.insert(0, "strategy", name)
        frames.append(df)
        
    if not frames:
        return {"empty": "no picks found"}
        
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.fillna("—")
    rows = combined.to_dict(orient="records")
    return {"title_suffix": str(as_of), "columns": list(combined.columns), "rows": rows}


def p_backtest_summary() -> dict:
    if not table_exists("calendar_call_trades"):
        return {"empty": "no calendar trades yet"}

    from earnings_edge.backtest.realism import ibkr_commission
    from earnings_edge.backtest.stats import trade_stats, portfolio_metrics, cross_sectional_test

    df = calendar_call_trades_ml_frame()
    if df.empty:
        return {"empty": "no calendar trades yet"}

    returns = df["return_on_debit"].dropna()
    gross = trade_stats(returns)
    
    commissions = (
        df["near_entry"].apply(ibkr_commission)
        + df["far_entry"].apply(ibkr_commission)
        + df["near_exit"].apply(ibkr_commission)
        + df["far_exit"].apply(ibkr_commission)
    )
    net_pnl = df["pnl_dollars"] - commissions
    debit_dollars = df["net_debit"] * 100.0
    net_returns = (net_pnl / debit_dollars).where(debit_dollars > 0).dropna()
    net = trade_stats(net_returns)
    
    init_cap = 100000.0
    by_date = df.groupby("scan_date")["pnl_dollars"].sum().sort_index()
    equity = [init_cap] + list(init_cap + by_date.cumsum())
    pm = portfolio_metrics(equity, periods_per_year=252)
    
    stats = {
        "trades": len(df),
        "gross mean": f"{gross.mean:+.4f}",
        "net mean": f"{net.mean:+.4f}",
        "net win%": f"{net.win_rate*100:.1f}%",
        "cagr": f"{pm.cagr:+.2%}",
        "sharpe": f"{pm.sharpe:.2f}",
        "max dd": f"{pm.max_drawdown:.2%}",
    }
    
    if "model_score" in df and df["model_score"].notna().sum() >= 3:
        cs = cross_sectional_test(df["model_score"], df["return_on_debit"], n_buckets=5)
        stats["spearman rho"] = f"{cs.spearman_rho:+.3f}"
        
    return {"stats": stats}


PANELS: list[Panel] = [
    Panel("bot_health", "Bot Status", "stats", p_bot_health),
    Panel("ff_backfill", "FF Backfill (30/+30)", "stats", p_ff_backfill),
    Panel("picks", "Top Picks", "table", p_picks),
    Panel("backtest_summary", "Backtest Summary", "stats", p_backtest_summary),
    Panel("latest_scans", "Latest Scans", "table", p_latest_scans),
    Panel("scan_outputs", "Scan Output", "table", p_scan_outputs),
    Panel("proposals", "Trade Proposals", "table", p_proposals),
    Panel("ff_ladders", "FF Ladders", "table", p_ff_ladders),
    Panel("positions", "Alpaca Positions", "table", p_positions),
    Panel("events", "Trade Events", "table", p_events),
    Panel("calendar_stats", "Calendar Strategy", "stats", p_calendar_stats),
]


# ── state building + broadcast ───────────────────────────────────────────

def build_panels() -> dict[str, dict]:
    out: dict[str, dict] = {}
    configure(_db_path())
    for p in PANELS:
        try:
            payload = p.provider()
        except Exception as exc:
            payload = {"empty": f"provider error: {exc}"}
        out[p.id] = {"id": p.id, "title": p.title, "kind": p.kind, "payload": payload}
    return out


class Hub:
    def __init__(self) -> None:
        self.clients: set[WebSocket] = set()
        self.hashes: dict[str, str] = {}
        self.latest: dict[str, dict] = {}
        self._task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        self._task = asyncio.create_task(self._poll_loop())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _poll_loop(self) -> None:
        while True:
            try:
                await self._tick()
            except Exception as exc:
                logger.error("dashboard poll failed: %s", exc)
            await asyncio.sleep(POLL_SECONDS)

    async def _tick(self) -> None:
        if not self.clients:
            return  # nothing connected — skip DB work entirely
        panels = await asyncio.to_thread(build_panels)
        self.latest = panels
        for pid, state in panels.items():
            digest = hashlib.sha256(
                json.dumps(state, sort_keys=True, default=str).encode()
            ).hexdigest()
            if self.hashes.get(pid) != digest:
                self.hashes[pid] = digest
                await self._broadcast({"type": "panel", **state})

    async def _broadcast(self, msg: dict) -> None:
        dead = []
        for ws in list(self.clients):
            try:
                await ws.send_text(json.dumps(msg, default=str))
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.clients.discard(ws)

    async def join(self, ws: WebSocket, *, already_accepted: bool = False) -> None:
        if not already_accepted:
            await ws.accept()
        self.clients.add(ws)
        await ws.send_text(json.dumps({
            "type": "hello",
            "panels": [{"id": p.id, "title": p.title, "kind": p.kind} for p in PANELS],
        }))
        # send current state immediately (build if nothing cached yet)
        if not self.latest:
            self.latest = await asyncio.to_thread(build_panels)
            self.hashes = {
                pid: hashlib.sha256(
                    json.dumps(st, sort_keys=True, default=str).encode()
                ).hexdigest()
                for pid, st in self.latest.items()
            }
        for st in self.latest.values():
            await ws.send_text(json.dumps({"type": "panel", **st}, default=str))


hub = Hub()


# ── app ──────────────────────────────────────────────────────────────────

def _client_host(request: Request) -> str:
    return (request.client.host if request.client else "") or ""


def _loopback(request: Request) -> bool:
    return _client_host(request) in {"127.0.0.1", "::1", "testclient"}


def _init_data_of(request: Request, body: Optional[dict] = None) -> str:
    """Pull initData from header, Authorization, query, or JSON body.

    Tailscale Funnel and some Telegram WebViews drop custom headers;
    the query/body copies are the reliable path.
    """
    auth = (request.headers.get("Authorization") or "").strip()
    if auth.lower().startswith("tma "):
        auth = auth[4:].strip()
    else:
        auth = ""
    return (
        (request.headers.get("X-Telegram-Init-Data") or "").strip()
        or auth
        or (request.query_params.get("initData") or "").strip()
        or (str((body or {}).get("initData") or "").strip())
    )


def _operator_or_error(request: Request, body: Optional[dict] = None):
    try:
        return require_operator(_init_data_of(request, body)), None
    except InitDataError as exc:
        logger.warning("miniapp auth: %s", exc)
        return None, JSONResponse({"ok": False, "error": str(exc)}, status_code=403)


async def index(request):
    return HTMLResponse((STATIC / "index.html").read_text())


async def api_state(request):
    if not _loopback(request):
        _, err = _operator_or_error(request)
        if err is not None:
            return err
    return JSONResponse(await asyncio.to_thread(build_panels), media_type="application/json")


def _desk_sync():
    from earnings_edge.alpaca_trading import create_client
    try:
        client = create_client()
        getter = client.get_positions
    except Exception:
        getter = None
    return load_desk(get_positions=getter)


async def api_desk(request):
    uid, err = _operator_or_error(request)
    if err is not None:
        return err
    try:
        data = await asyncio.to_thread(_desk_sync)
    except Exception as exc:
        logger.exception("desk load failed")
        return JSONResponse({"ok": False, "error": f"desk busy: {exc}"}, status_code=503)
    data["user_id"] = uid
    return JSONResponse(data)


async def api_me(request):
    uid, err = _operator_or_error(request)
    if err is not None:
        return err
    return JSONResponse({"ok": True, "user_id": uid, "operator": True})


def _action_sync(op: str, payload: dict, uid: int) -> dict:
    from earnings_edge.alpaca_trading import create_client
    from earnings_edge.trade_approval import PendingTradeStore
    try:
        client = create_client()
    except Exception:
        client = None
    store = PendingTradeStore(str(_db_path()))
    return run_desk_action(
        op, payload, by=f"webapp:{uid}", client=client, store=store)


async def api_action(request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"ok": False, "error": "bad json"}, status_code=400)
    if not isinstance(body, dict):
        return JSONResponse({"ok": False, "error": "bad json"}, status_code=400)
    uid, err = _operator_or_error(request, body)
    if err is not None:
        return err
    op = str(body.get("op") or "")
    result = await asyncio.to_thread(_action_sync, op, body, uid)
    return JSONResponse(result)


async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    if not _loopback(ws):
        init = (ws.query_params.get("initData") or "").strip()
        if not init:
            try:
                raw = await asyncio.wait_for(ws.receive_text(), timeout=8)
                try:
                    msg = json.loads(raw)
                    init = str(msg.get("initData") or "").strip()
                except Exception:
                    init = raw.strip()
            except Exception:
                init = ""
        try:
            require_operator(init)
        except InitDataError:
            await ws.close(code=4403)
            return
    await hub.join(ws, already_accepted=True)
    try:
        while True:
            await ws.receive_text()  # client pings/acks — content ignored
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        hub.clients.discard(ws)


@asynccontextmanager
async def lifespan(app):
    await hub.start()
    yield
    await hub.stop()


app = Starlette(
    routes=[
        Route("/", index),
        Route("/api/state", api_state),
        Route("/api/desk", api_desk),
        Route("/api/me", api_me),
        Route("/api/action", api_action, methods=["POST"]),
        WebSocketRoute("/ws", ws_endpoint),
    ],
    lifespan=lifespan,
)
