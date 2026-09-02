#!/usr/bin/env python3
"""
Unified Telegram bot for the Earnings Edge trading scanners.

US earnings-options Telegram bot (scan, proposals, FF ladder, risk).
German crash alerts run in crash_alert.py — not here.
"""

import asyncio
import json
import logging
import os
import sys
import threading
import time
from datetime import datetime, timezone
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Dict, Optional, Set

from dotenv import load_dotenv

# Load .env BEFORE importing earnings_edge: several modules (e.g. config.py)
# snapshot get_settings() at import time, so dotenv must run first.
load_dotenv()

# Ensure project root on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from earnings_edge import cards
from earnings_edge.base import BaseScanner
from earnings_edge.bot_live import ProgressMessage
from earnings_edge.bot_scanner import EarningsCalendarScanner
from earnings_edge.ops_auth import auth_message, is_operator, operator_chat_ids, operators_configured
from earnings_edge.trade_approval import (
    PendingTradeStore,
    build_ff_proposals,
    build_proposals,
    execute_proposal,
    reject_proposal,
)
from earnings_edge.alpaca_trading import create_client
from earnings_edge.collectors.earnings_calendar import EarningsCalendarCollector
from earnings_edge.db import (
    exit_proposals_get,
    managed_positions_list,
    persist_picks,
    scan_runs_latest_success,
    snapshots_max_scan_date,
)
from earnings_edge.earnings import scan_dates
from earnings_edge.fwd_factor_ladder import LadderRunner, build_candidate
from earnings_edge.subscriptions import (
    SIGNAL_STRATEGIES,
    StrategySubscriptions,
    funnel_line,
    partition_by_mode,
    route_proposals,
)
from framework.core.config import load_strategy_configs
from framework.ops import InstanceLock, install_secret_redaction

import pytz
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from telegram import (
    Update, InlineKeyboardButton, InlineKeyboardMarkup,
    MenuButtonWebApp, ReplyKeyboardMarkup, ReplyKeyboardRemove, WebAppInfo,
)
from telegram.constants import ParseMode
from telegram.ext import (
    Application, CallbackQueryHandler, CommandHandler,
    ContextTypes, MessageHandler, filters,
)

HTML = ParseMode.HTML

logging.basicConfig(
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("trading_bot")

# ── Keyboard layouts ──────────────────────────────────────────────────────

MAIN_KB = [
    ["🖥 Status", "💼 Positions"],
    ["📡 Signals", "📥 Pending"],
    ["🎯 Picks", "📐 Designer"],
    ["⚙️ Jobs", "🛠 Settings"],
]
TRADING_KB = [
    ["🖥 Status", "💼 Positions"],
    ["📥 Pending", "⚙️ Jobs", "🛠 Settings"],
    ["⬅️ Back to Main"],
]
RUN_KB = [["⬅️ Back to Main"]]


def _main_reply_kb() -> ReplyKeyboardMarkup:
    """MAIN_KB plus Open desk. The key is plain text: reply-keyboard
    web_app buttons open a WebView without initData on several clients.
    Tapping it sends the working inline Mini App button instead.
    """
    from dashboard.tg_auth import webapp_url
    rows = [list(r) for r in MAIN_KB]
    if webapp_url():
        rows.append(["🖥 Open desk"])
    return ReplyKeyboardMarkup(rows, resize_keyboard=True)


def _desk_webapp_markup() -> Optional[InlineKeyboardMarkup]:
    from dashboard.tg_auth import webapp_url
    url = webapp_url()
    if not url:
        return None
    return InlineKeyboardMarkup([[
        InlineKeyboardButton("🖥 Open desk", web_app=WebAppInfo(url=url)),
    ]])


# ── Health endpoint ──────────────────────────────────────────────────────

class _HealthHandler(BaseHTTPRequestHandler):
    """Ready/not-ready JSON. ``facts_fn`` is swapped in by TradingBot.run."""
    facts_fn = staticmethod(lambda: {"ready": False, "reasons": ["uninitialized"]})
    _started_at = time.monotonic()

    def do_GET(self):
        if self.path != "/health":
            self.send_response(404)
            self.end_headers()
            return
        from framework.health import health_ready
        facts = type(self).facts_fn()
        facts = dict(facts)
        broker = facts.pop("broker", None)
        result = health_ready(**facts) if "lock_held" in facts else facts
        result = dict(result)
        if broker:
            result["broker"] = broker
        result["uptime_secs"] = round(time.monotonic() - self._started_at, 1)
        body = json.dumps(result).encode()
        self.send_response(200 if result.get("ready") else 503)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass  # suppress request logging


def _start_health_server(port: int = 8502):
    """Run the health HTTP server in a daemon thread."""
    try:
        server = HTTPServer(("127.0.0.1", port), _HealthHandler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        logger.info("Health endpoint started on http://127.0.0.1:%d/health", port)
    except OSError as exc:
        logger.warning("Could not start health endpoint on port %d: %s", port, exc)


def _start_dashboard_server(port: int = 8503):
    """Run the mini app dashboard (uvicorn) in a daemon thread."""
    import uvicorn
    import asyncio
    
    def run_dashboard():
        try:
            from dashboard.server import app
            config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
            server = uvicorn.Server(config)
            asyncio.run(server.serve())
        except Exception as exc:
            logger.error("Dashboard server failed: %s", exc)

    threading.Thread(target=run_dashboard, daemon=True).start()
    logger.info("Dashboard server started on http://127.0.0.1:%d", port)


# ── Bot ───────────────────────────────────────────────────────────────────

class TradingBot:
    def __init__(self, token: str):
        self.token = token
        self.subscribers_file = os.path.join(os.path.dirname(__file__), "data", "subscribers.json")
        self.scanners: Dict[str, BaseScanner] = {}
        self.subscribers: Dict[str, Set[int]] = {}
        self.application = None
        self._instance_lock = None
        self._monitors: dict[int, asyncio.Task] = {}  # chat_id -> live monitor task

        self._load_subscribers()
        # Per-strategy signal subscriptions (opt-out model; /signals).
        self.strategy_subs = StrategySubscriptions(
            os.path.join(os.path.dirname(__file__), "data", "strategy_subscribers.json")
        )
        self._register(EarningsCalendarScanner())

        # Human-in-the-loop trade approval: daily proposals -> operator
        # confirms via inline button -> Alpaca execution. Never auto-executes.
        self.approval_store = PendingTradeStore()

        # Forward-factor ladder: 13:45 ET proposals, operator arms via the
        # standard proposal card, bot walks a limit ladder 14:00-15:45 ET
        # (25% -> 20% premium target price).
        self._ff_runner: LadderRunner | None = None

        self.scheduler = BlockingScheduler()
        self._scheduler_thread = None
        self._main_loop: asyncio.AbstractEventLoop | None = None

        # Framework layer: per-strategy TOML configs (risk limits, lifecycle,
        # execution mode). Framework tables are created on first connection.
        self.strategy_configs = load_strategy_configs()
        try:
            from framework.core.registry import StrategyRegistry
            StrategyRegistry(self.strategy_configs).sync_lifecycle()
        except Exception as exc:
            logger.warning("lifecycle sync from configs failed: %s", exc)

    def _health_facts(self) -> dict:
        """Facts for ``health_ready`` — used by the HTTP handler."""
        from framework.core.calendar import get_calendar
        from framework.risk.equity import latest_equity
        lock_held = bool(getattr(self, "_instance_lock", None) and self._instance_lock._fh)
        last_eq = last_scan = None
        clock_ok = False
        market_open = False
        skip = False
        try:
            clock = create_client().get_clock()
            clock_ok = True
            market_open = bool(clock.get("is_open"))
        except Exception as e:
            logger.error("Job failed: %s", e)
        try:
            eq = latest_equity()
            last_eq = eq["ts"] if eq else None
            last_scan = scan_runs_latest_success()
            from earnings_edge.db import job_runs_latest
            skip_row = job_runs_latest("equity_snapshot")
            if skip_row and skip_row.get("stats_json") and "market closed" in skip_row["stats_json"]:
                skip = True
        except Exception as e:
            logger.error("Job failed: %s", e)
        from earnings_edge.alpaca_mode import broker_label
        return {
            "lock_held": lock_held,
            "last_equity_ts": last_eq,
            "last_scan_ts": last_scan,
            "clock_ok": clock_ok,
            "market_open": market_open,
            "equity_skipped_closed": skip,
            "broker": broker_label(),
        }

    async def _capture_loop(self, application) -> None:
        """post_init hook: runs INSIDE the application's event loop before
        polling starts. The scheduler thread must dispatch coroutines onto
        THIS loop — creating a separate loop and touching application.bot
        from it fails with 'Event loop is closed' (broken pushes)."""
        self._main_loop = asyncio.get_running_loop()
        # Catch up on anything that happened at the broker while we were down.
        application.create_task(self._reconcile())
        from dashboard.tg_auth import webapp_url
        url = webapp_url()
        if url:
            try:
                await application.bot.set_chat_menu_button(
                    menu_button=MenuButtonWebApp(
                        text="Desk", web_app=WebAppInfo(url=url)))
            except Exception as exc:
                logger.warning("Mini App menu button not set: %s", exc)

    def _dispatch(self, coro, timeout: float = 900.0) -> None:
        """Run a coroutine on the application's loop from the scheduler thread."""
        if self._main_loop is None:
            logger.error("main loop not captured yet — dropping scheduled task")
            return

        async def _wrapped():
            try:
                return await coro
            finally:
                await self._flush_alerts()

        future = asyncio.run_coroutine_threadsafe(_wrapped(), self._main_loop)
        try:
            future.result(timeout=timeout)
        except TimeoutError as exc:
            logger.exception("scheduled task timed out after %ss: %s", timeout, exc)
            future.cancel()
        except Exception as exc:
            logger.exception("scheduled task failed: %s", exc)

    async def _flush_alerts(self):
        """Push any DEDUPER outbox messages to the approval chat."""
        from framework.alerts import DEDUPER
        for msg in DEDUPER.drain():
            await self._push_risk_alert(msg)

    def _desk_facts_sync(self) -> dict:
        """Broker + DB facts shared by /status and /monitor."""
        from earnings_edge.bot_views import collect_desk_facts
        from framework.alerts import emit_broker_failure, emit_clock_failure
        client = create_client()
        facts = collect_desk_facts(
            get_clock=client.get_clock, get_positions=client.get_positions)
        if facts.get("clock_exc") is not None:
            emit_clock_failure(facts["clock_exc"])
        if facts.get("positions_exc") is not None:
            emit_broker_failure(facts["positions_exc"])
        return facts

    def _approval_override(self) -> Optional[int]:
        """TELEGRAM_APPROVAL_CHAT_ID override chat, if configured."""
        env = os.environ.get("TELEGRAM_APPROVAL_CHAT_ID", "").strip()
        if env:
            try:
                return int(env)
            except ValueError:
                logger.warning("TELEGRAM_APPROVAL_CHAT_ID %r is not an int, ignoring", env)
        return None

    def _approval_chats(self) -> Set[int]:
        """Who receives trade proposals/signal notifications.

        If the operator allow-list is configured it is the only destination.
        Otherwise fall back to subscribers (notifications only — execute/close
        still fail closed via ``_risk_authorized``).
        """
        ops = operator_chat_ids()
        if ops:
            return set(ops)
        override = self._approval_override()
        if override is not None:
            return {override}
        chats: Set[int] = set()
        for uids in self.subscribers.values():
            chats |= uids
        chats |= self.strategy_subs.known_users()
        return chats

    # ── Registration / persistence ─────────────────────────────────────

    def _register(self, scanner: BaseScanner):
        self.scanners[scanner.name] = scanner
        self.subscribers.setdefault(scanner.name, set())
        logger.info("Registered scanner: %s (schedule: %s)", scanner.name, scanner.schedule)

    def _load_subscribers(self):
        if os.path.exists(self.subscribers_file):
            try:
                with open(self.subscribers_file) as f:
                    self.subscribers = {k: set(v) for k, v in json.load(f).items()}
            except Exception as exc:
                logger.error("Failed to load subscribers: %s", exc)

    def _save_subscribers(self):
        os.makedirs(os.path.dirname(self.subscribers_file), exist_ok=True)
        with open(self.subscribers_file, "w") as f:
            json.dump({k: list(v) for k, v in self.subscribers.items()}, f, indent=2)

    def _subscribe(self, name: str, uid: int) -> bool:
        if name in self.scanners:
            self.subscribers[name].add(uid)
            self._save_subscribers()
            return True
        return False

    def _unsubscribe(self, name: str, uid: int) -> bool:
        if name in self.subscribers:
            self.subscribers[name].discard(uid)
            self._save_subscribers()
            return True
        return False

    def _user_subs(self, uid: int):
        return [n for n, uids in self.subscribers.items() if uid in uids]

    # ── Format helpers ─────────────────────────────────────────────────

    @staticmethod
    def _chunk_text(text: str, limit: int = 3800) -> list[str]:
        """Split text into Telegram-safe chunks."""
        if len(text) <= limit:
            return [text]
        chunks = []
        buf = ""
        for para in text.split("\n\n"):
            candidate = f"{buf}\n\n{para}" if buf else para
            if len(candidate) <= limit:
                buf = candidate
                continue
            if buf:
                chunks.append(buf)
                buf = ""
            if len(para) <= limit:
                buf = para
            else:
                start = 0
                while start < len(para):
                    chunks.append(para[start:start + limit])
                    start += limit
        if buf:
            chunks.append(buf)
        return chunks

    async def _send_panel(self, update: Update, text: str, reply_markup=None, parse_mode=None):
        """Send a message and delete the previous panel to reduce chat clutter."""
        chat_id = update.effective_chat.id
        old_id = getattr(self, "_last_panel_ids", {}).get(chat_id)
        if old_id:
            try:
                await self.application.bot.delete_message(chat_id, old_id)
            except Exception as e:
                logger.debug("Could not delete old panel %s: %s", old_id, e)

        msg = await update.message.reply_text(text, reply_markup=reply_markup, parse_mode=parse_mode)
        
        # Do not track messages that carry the main ReplyKeyboardMarkup,
        # otherwise deleting them later will hide the user's keyboard.
        from telegram import ReplyKeyboardMarkup
        if not isinstance(reply_markup, ReplyKeyboardMarkup):
            if not hasattr(self, "_last_panel_ids"):
                self._last_panel_ids = {}
            self._last_panel_ids[chat_id] = msg.message_id
        return msg

    async def _edit_panel(self, query, text: str, markup=None, parse_mode=None) -> None:
        """Edit the tapped message in place. Falls back to a new message."""
        try:
            await query.edit_message_text(text, reply_markup=markup, parse_mode=parse_mode)
            return
        except Exception as exc:
            logger.debug("edit_message_text skipped: %s", exc)
        try:
            await query.message.reply_text(text, reply_markup=markup, parse_mode=parse_mode)
        except Exception as exc:
            logger.error("panel send failed: %s", exc)

    def _positions_panel_sync(self, banner: Optional[str] = None):
        from earnings_edge.bot_views import build_positions_panel
        broker, err = None, None
        try:
            broker = create_client().get_positions()
        except Exception as exc:
            err = str(exc)[:120]
        return build_positions_panel(
            broker_positions=broker, broker_error=err, banner=banner)

    async def _refresh_positions_query(self, query, banner: Optional[str] = None) -> None:
        text, rows = await asyncio.to_thread(self._positions_panel_sync, banner)
        await self._flush_alerts()
        markup = InlineKeyboardMarkup(rows) if rows else None
        await self._edit_panel(query, text, markup)

    # ── Command handlers ───────────────────────────────────────────────

    async def _cmd_start(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        kb = _main_reply_kb()
        from dashboard.tg_auth import webapp_url
        url = webapp_url()
        desk = ("• Open desk — Mini App book / inbox / halt\n" if url else "")
        await self._send_panel(update, 
            "🚀 <b>Trading desk</b>\n\n"
            "Six keys stay put. Open a panel, tap a button — the same "
            "message updates (no need to reopen Positions after Adopt).\n\n"
            "• Status — scan / equity / broker / orphans\n"
            "• Positions — live book: adopt / ignore / close\n"
            "• Signals — mute + approval vs auto\n"
            "• Pending — entry / exit / orphan inbox\n"
            "• Picks & Designer — strategy picks and scenario modeling\n"
            "• Jobs — scheduled run history\n"
            "• Settings — halt / resume / restart\n"
            f"{desk}",
            reply_markup=kb, parse_mode=HTML,
        )
        ikb = _desk_webapp_markup()
        if ikb:
            await update.message.reply_text(
                "Open the web desk from this button (not by pasting the URL):",
                reply_markup=ikb,
            )

    async def _cmd_help(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        kb = _main_reply_kb()
        lines = [
            "🆘 <b>Trading desk help</b>\n",
            "Keyboard: Status · Positions · Signals · Pending · Jobs · Settings\n",
            "• /start — main menu",
            "• /run — scan on demand then build cards",
            "• /pending — inbox (entries, exits, orphans, failed jobs)",
            "• /propose — build proposals from the latest scan",
            "• /signals — notifications + execution mode",
            "• /setups — what each strategy trades",
            "• /status — dashboard (scan/equity/reconcile/broker)",
            "• /monitor — live ops panel",
            "• /positions — broker-truth book + actions",
            "• /jobs — scheduled job history",
            "• /picks — top picks from the latest scan",
            "• /designer <ticker> <legs> — position designer & rv scenarios",
            "• /settings — halt / resume / lifecycle / restart",
            "• /scanners and /subscriptions alias /signals\n",
            "⏰ Schedules:",
        ]
        from dashboard.tg_auth import webapp_url
        if webapp_url():
            lines.insert(-1, "• Open desk — Mini App book / inbox / halt")
        for name, sc in self.scanners.items():
            lines.append(f"• {name}: {sc.schedule}")
        lines += [
            "\n📱 Use the keyboard at the bottom for quick access!",
        ]
        await self._send_panel(update, 
            "\n".join(lines), reply_markup=kb, parse_mode=HTML,
        )

    async def _cmd_scanners(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        # Deprecated surface — folded into /signals (unified signal surface).
        await self._cmd_signals(update, ctx)

    async def _cmd_subscriptions(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        # Deprecated surface — folded into /signals.
        await self._cmd_signals(update, ctx)

    # ── Signal subscriptions (/signals) + trade setups (/setups) ───────

    def _effective_modes_sync(self) -> dict:
        """strategy -> effective execution mode (TOML default + DB override)."""
        from framework.core.control import effective_execution_mode
        out = {}
        for name in SIGNAL_STRATEGIES:
            cfg = self.strategy_configs.get(name)
            toml = cfg.execution_mode if cfg else "approval"
            out[name] = effective_execution_mode(name, toml)
        from earnings_edge.alpaca_mode import force_approval_on_live
        if force_approval_on_live():
            for k, v in list(out.items()):
                if v == "auto":
                    out[k] = "approval"
        return out

    def _signals_kb(self, uid: int, modes: dict) -> InlineKeyboardMarkup:
        rows = []
        for name in SIGNAL_STRATEGIES:
            on = self.strategy_subs.is_subscribed(name, uid)
            mark = "✅" if on else "❌"
            cb = f"sig_off_{name}" if on else f"sig_on_{name}"
            mode_label = "⚡ auto" if modes.get(name) == "auto" else "👤 approval"
            rows.append([
                InlineKeyboardButton(f"{mark} {name}", callback_data=cb),
                InlineKeyboardButton(mode_label, callback_data=f"sig_mode_{name}"),
            ])
        return InlineKeyboardMarkup(rows)

    @staticmethod
    def _signals_intro() -> str:
        return (
            "📡 <b>Signals</b> — one surface for everything the bot pushes.\n\n"
            "<b>Left button</b> — notifications: ✅ you receive this strategy's "
            "cards, ❌ muted.\n"
            "<b>Right button</b> — execution mode (GLOBAL, same for everyone):\n"
            "  👤 <code>approval</code> — card pushed, a human clicks Execute (trade by hand)\n"
            "  ⚡ <code>auto</code> — bot executes immediately within risk limits and notifies\n\n"
            "What each strategy trades: /setups · pause/resume + spend: /strategies\n"
            "Refresh the scan and build fresh signals on demand: /run"
        )

    async def _cmd_signals(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        uid = update.effective_user.id
        modes = await asyncio.to_thread(self._effective_modes_sync)
        await self._send_panel(update, 
            self._signals_intro(), reply_markup=self._signals_kb(uid, modes), parse_mode=HTML)

    async def _cmd_setups(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        from earnings_edge.bot_views import SETUP_STRATEGIES, setup_menu_text
        ikb = [[InlineKeyboardButton(name, callback_data=f"setup_{name}")]
               for name in SETUP_STRATEGIES]
        await self._send_panel(update, 
            setup_menu_text(), reply_markup=InlineKeyboardMarkup(ikb), parse_mode=HTML)

    def _run_panel(self) -> tuple[str, InlineKeyboardMarkup]:
        msg = ("🔄 <b>Run scanner</b>\n\n"
               "Scans on demand, then builds and pushes signal cards "
               "from that data. Tap a scanner — this message updates.\n\n")
        ikb = []
        for name, sc in self.scanners.items():
            msg += f"📈 {cards.bold(name)}\n  ⏰ {cards.esc(sc.get_schedule_description())}\n\n"
            ikb.append([InlineKeyboardButton(f"🚀 Run {name}", callback_data=f"run_{name}")])
        return msg, InlineKeyboardMarkup(ikb)

    async def _cmd_run(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        msg, ikb = self._run_panel()
        await self._send_panel(update, msg, reply_markup=ikb, parse_mode=HTML)

    # ── Callback handler ───────────────────────────────────────────────

    async def _handle_callback(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        await query.answer()
        uid = query.from_user.id
        data = query.data

        if data.startswith("bk_"):
            await self._handle_book_callback(query, uid, data)

        elif data.startswith("desk_"):
            await self._handle_desk_refresh(query, uid, data)

        elif data.startswith("in_"):
            await self._handle_inbox_callback(query, uid, data)

        elif data.startswith("pt_"):
            await self._handle_approval_callback(update, data)

        elif data.startswith("set_"):
            await self._handle_settings_callback(query, uid, data)

        elif data.startswith("sig_on_") or data.startswith("sig_off_"):
            # plain text: strategy names contain underscores
            enable = data.startswith("sig_on_")
            name = data[len("sig_on_"):] if enable else data[len("sig_off_"):]
            try:
                self.strategy_subs.set_subscribed(name, uid, enable)
            except KeyError:
                await query.edit_message_text(f"Unknown strategy: {name}")
                return
            state = "✅ subscribed" if enable else "❌ muted"
            modes = await asyncio.to_thread(self._effective_modes_sync)
            await self._edit_panel(
                query,
                f"{self._signals_intro()}\n\n→ {name}: {state}",
                self._signals_kb(uid, modes),
            )

        elif data == "sig_noop":
            pass  # section header row — no action

        elif data == "mon_stop":
            await self._stop_monitor(query.message.chat_id)

        elif data.startswith("sig_mode_"):
            name = data[len("sig_mode_"):]
            if name not in SIGNAL_STRATEGIES:
                await query.edit_message_text(f"Unknown strategy: {name}")
                return
            if not self._risk_authorized(uid):
                await query.edit_message_text(auth_message())
                return

            def flip_mode():
                from framework.core.control import (
                    effective_execution_mode, set_execution_mode)
                cfg = self.strategy_configs.get(name)
                toml = cfg.execution_mode if cfg else "approval"
                cur = effective_execution_mode(name, toml)
                new = "auto" if cur != "auto" else "approval"
                set_execution_mode(name, new, by=str(uid))
                return cur, new

            cur, new = await asyncio.to_thread(flip_mode)
            modes = await asyncio.to_thread(self._effective_modes_sync)
            note = ("⚡ bot now executes this strategy's proposals immediately "
                    "(risk-gated) and notifies" if new == "auto" else
                    "👤 proposals now require a human click on the card")
            await self._edit_panel(
                query,
                f"{self._signals_intro()}\n\n→ {name}: {cur} → {new} (GLOBAL). {note}",
                self._signals_kb(uid, modes),
            )

        elif data == "setup_back":
            from earnings_edge.bot_views import SETUP_STRATEGIES, setup_menu_text
            ikb = [[InlineKeyboardButton(name, callback_data=f"setup_{name}")]
                   for name in SETUP_STRATEGIES]
            await query.edit_message_text(
                setup_menu_text(), reply_markup=InlineKeyboardMarkup(ikb))

        elif data.startswith("setup_"):
            from earnings_edge.bot_views import setup_card
            name = data[len("setup_"):]
            kb = InlineKeyboardMarkup(
                [[InlineKeyboardButton("⬅️ All setups", callback_data="setup_back")]])
            await query.edit_message_text(setup_card(name), reply_markup=kb)

        elif data.startswith("st_on_") or data.startswith("st_off_"):
            if not self._risk_authorized(uid):
                await query.edit_message_text(auth_message())
                return
            enable = data.startswith("st_on_")
            name = data.split("_", 2)[2]

            def flip():
                from framework.core.control import set_enabled
                set_enabled(name, enable, by=str(uid))
            await asyncio.to_thread(flip)
            note = (f"▶️ {name} resumed — proposals will include it again."
                    if enable else
                    f"⏸ {name} paused — no new proposals; open positions still exit.")
            text, ikb = await asyncio.to_thread(self._strategies_panel_sync)
            await self._edit_panel(query, f"{note}\n\n{text}", InlineKeyboardMarkup(ikb))

        elif data == "ex_close_grp":
            if not self._risk_authorized(uid):
                await query.edit_message_text(auth_message())
                return
            await self._decide_group_bulk(query, uid, "exit")

        elif data.startswith("ex_close_"):
            if not self._risk_authorized(uid):
                await query.edit_message_text(auth_message())
                return
            pid = int(data[len("ex_close_"):])
            await self._decide_in_group(query, uid, "exit", pid, "exec")

        elif data.startswith("ex_skip_"):
            pid = int(data[len("ex_skip_"):])
            await self._decide_in_group(query, uid, "exit", pid, "skip")


        elif data.startswith("sub_"):
            name = data[4:]
            if self._subscribe(name, uid):
                await query.edit_message_text(f"✅ Subscribed to {name}!")
            else:
                await query.edit_message_text(f"❌ Failed to subscribe to <b>{name}</b>.", parse_mode=HTML)

        elif data.startswith("unsub_"):
            name = data[5:]
            if self._unsubscribe(name, uid):
                await query.edit_message_text(f"✅ Unsubscribed from {name}.")
            else:
                await query.edit_message_text(f"❌ Failed to unsubscribe from {name}.")

        elif data.startswith("run_"):
            name = data[4:]
            if name not in self.scanners:
                await query.edit_message_text(f"❌ Unknown scanner: {name}")
                return
            again = InlineKeyboardMarkup(
                [[InlineKeyboardButton("🔄 Run again", callback_data="desk_run")]])
            pm = ProgressMessage(self.application.bot, query.message.chat_id,
                                 f"Scanning {name}")
            await pm.attach(query.message)
            try:
                await pm.set_stage("fetching data + analysing (30–60 min on heavy days)")
                # NB: never scan() on the event loop — it starves every
                # framework job dispatched via _dispatch (postmortem 2026-07-31)
                result = await asyncio.to_thread(self.scanners[name].scan)
            except Exception as exc:
                await pm.finish(f"❌ {name} error: {exc}", reply_markup=again)
                return
            if not result.get("success"):
                await pm.finish(
                    f"❌ {name} failed: {result.get('error', 'Unknown')}",
                    reply_markup=again)
                return
            # Chain straight into signal building — a scan on its own is
            # not actionable, and manual runs used to show a raw report
            # dump here instead of the actual trade signals it produces.
            try:
                market_open = bool(
                    (await asyncio.to_thread(create_client().get_clock)).get("is_open"))
            except Exception:
                market_open = False
            if not market_open:
                await pm.finish(
                    f"✅ {name} scan complete.\n"
                    "⏰ Market is closed — signal build only runs 09:30–16:00 ET "
                    "(quotes would be stale outside that window). Tap Run again "
                    "during market hours to turn this data into proposal cards.",
                    reply_markup=again)
                return
            await pm.set_stage("building signals from fresh scan data")
            try:
                await self._propose_and_push()
            except Exception as exc:
                logger.exception("manual signal build after %s failed", name)
                await pm.finish(
                    f"⚠️ {name} scan complete, but signal build failed: {exc}",
                    reply_markup=again)
                return
            from earnings_edge import trade_approval
            funnel = funnel_line(trade_approval.LAST_FUNNEL)
            tail = f"\n{funnel}" if funnel else ""
            await pm.finish(
                f"✅ {name} scan complete — signals built and pushed.{tail}",
                reply_markup=again)

    # ── Keyboard handler ───────────────────────────────────────────────

    async def _handle_keyboard(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        text = update.message.text
        if text == "📊 Scanners" or text == "📊 List Scanners":
            await self._cmd_scanners(update, ctx)
        elif text == "📋 My Subscriptions":
            await self._cmd_subscriptions(update, ctx)
        elif text == "🔄 Run Scanner":
            await self._cmd_run(update, ctx)
        elif text == "🖥 Status":
            await self._cmd_status(update, ctx)
        elif text == "📡 Monitor":
            await self._cmd_monitor(update, ctx)
        elif text == "💼 Positions":
            await self._cmd_positions(update, ctx)
        elif text == "🧾 Orders":
            await self._cmd_orders(update, ctx)
        elif text == "💰 Equity":
            await self._cmd_equity(update, ctx)
        elif text == "⚙️ Strategies":
            await self._cmd_strategies(update, ctx)
        elif text == "🛠 Settings":
            await self._cmd_settings(update, ctx)
        elif text == "🖥 Open desk":
            ikb = _desk_webapp_markup()
            if ikb:
                await self._send_panel(update, 
                    "Open the web desk from this button (reply-keyboard Mini Apps "
                    "do not get a login token on this client):",
                    reply_markup=ikb,
                )
            else:
                await self._send_panel(update, "Mini App URL is not configured.")
        elif text == "📥 Pending":
            await self._cmd_pending(update, ctx)
        elif text == "⚙️ Jobs":
            await self._cmd_jobs(update, ctx)
        elif text == "📡 Signals":
            await self._cmd_signals(update, ctx)
        elif text == "📖 Trade Setups":
            await self._cmd_setups(update, ctx)
        elif text == "❓ Help":
            await self._cmd_help(update, ctx)
        elif text == "🎯 Picks":
            await self._cmd_picks(update, ctx)
        elif text == "📐 Designer":
            # Just show usage since designer needs args
            await self._send_panel(update, "Usage: /designer <ticker> <legs...>\nExample: /designer AAPL buy call 190 2026-10-16 1 5.0 0.3")
        elif text == "🚪 Close Keyboard":
            await self._send_panel(update, 
                "⌨️ Keyboard closed. /start to bring it back.",
                reply_markup=ReplyKeyboardRemove(),
            )
        elif text == "⬅️ Back to Main":
            kb = _main_reply_kb()
            await self._send_panel(update, "🏠 Main Menu", reply_markup=kb)

    # ── Trade approval flow ────────────────────────────────────────────

    @staticmethod
    def _row_kb(row: dict, kind: str) -> InlineKeyboardMarkup:
        """Execute/Skip (entry) or Close/Snooze (exit) buttons for one row —
        used both as a standalone single-card keyboard and, via _grouped_kb,
        as one row inside a batched multi-ticker message."""
        if kind == "entry":
            return InlineKeyboardMarkup([[
                InlineKeyboardButton("✅ Execute", callback_data=f"pt_exec_{row['id']}"),
                InlineKeyboardButton("❌ Skip", callback_data=f"pt_skip_{row['id']}"),
            ]])
        return InlineKeyboardMarkup([[
            InlineKeyboardButton("🔒 Close now", callback_data=f"ex_close_{row['id']}"),
            InlineKeyboardButton("⏰ Snooze", callback_data=f"ex_skip_{row['id']}"),
        ]])

    @staticmethod
    def _grouped_kb(rows: list[dict], kind: str) -> InlineKeyboardMarkup:
        """Keyboard for a per-strategy batch message: one Execute/Skip (or
        Close/Snooze) row per ticker, plus a bulk row when there's more than
        one. The bulk/rebuild logic reads proposal ids back off the clicked
        message's own keyboard (_kb_ids) rather than threading extra state,
        so this needs no bookkeeping beyond the callback_data itself."""
        exec_prefix, skip_prefix = ("pt_exec_", "pt_skip_") if kind == "entry" else ("ex_close_", "ex_skip_")
        exec_emoji, skip_emoji = ("✅", "❌") if kind == "entry" else ("🔒", "⏰")
        kb_rows = [
            [InlineKeyboardButton(f"{exec_emoji} {row['ticker']}", callback_data=f"{exec_prefix}{row['id']}"),
             InlineKeyboardButton(f"{skip_emoji} {row['ticker']}", callback_data=f"{skip_prefix}{row['id']}")]
            for row in rows
        ]
        if len(rows) > 1:
            label = f"✅ Execute ALL {len(rows)}" if kind == "entry" else f"🔒 Close ALL {len(rows)}"
            action = "pt_exec_grp" if kind == "entry" else "ex_close_grp"
            kb_rows.append([InlineKeyboardButton(label, callback_data=action)])
        return InlineKeyboardMarkup(kb_rows)

    @staticmethod
    def _kb_ids(query, prefixes: tuple) -> list[int]:
        """Proposal ids referenced by the clicked message's own keyboard.

        Rebuilding a decided batch from this (rather than a fresh
        strategy-wide DB query) means a decision only ever touches the exact
        set of rows that were shown in THIS message, even if a later push
        cycle already sent a newer batch for the same strategy.
        """
        markup = query.message.reply_markup if query.message else None
        if not markup:
            return []
        ids: set[int] = set()
        for kb_row in markup.inline_keyboard:
            for btn in kb_row:
                cd = btn.callback_data or ""
                for prefix in prefixes:
                    if cd.startswith(prefix):
                        tail = cd[len(prefix):]
                        if tail.isdigit():
                            ids.add(int(tail))
        return sorted(ids)

    async def _push_batches(self, uid: int, rows: list[dict], kind: str) -> None:
        """Push proposal/exit rows to one chat, grouped by strategy.

        A strategy with a single row still gets the classic single-card
        layout (full card_text + a 2-button keyboard). A strategy with
        several rows collapses into ONE bold-headed batch message with a
        compact per-ticker line and a keyboard row per ticker — this is
        what keeps a heavy signal day from flooding the chat with a wall of
        near-identical messages.
        """
        for strategy, srows in cards.group_by_strategy(rows).items():
            if len(srows) == 1:
                row = srows[0]
                try:
                    await self.application.bot.send_message(
                        chat_id=uid, text=row["card_text"],
                        reply_markup=self._row_kb(row, kind), parse_mode=HTML)
                except Exception as exc:
                    logger.error("%s card push to %d failed: %s", kind, uid, exc)
            else:
                text = cards.group_message(strategy, srows, kind)
                try:
                    await self.application.bot.send_message(
                        chat_id=uid, text=text,
                        reply_markup=self._grouped_kb(srows, kind), parse_mode=HTML)
                except Exception as exc:
                    logger.error("%s batch push to %d failed: %s", kind, uid, exc)

    async def _propose_and_push(self):
        """Build today's trade proposals and push them for confirmation.

        Routing: TELEGRAM_APPROVAL_CHAT_ID (if set) receives every strategy;
        otherwise each proposal goes to scanner subscribers minus that
        strategy's /signals opt-outs. Strategies with zero recipients still
        build and persist (audit) but are not pushed.
        """
        override = self._approval_override()
        universe = self._approval_chats()
        if not universe:
            logger.warning("trade proposals: no approval chats (no subscribers, TELEGRAM_APPROVAL_CHAT_ID unset)")
            return
        # Never build/push proposals when the US market is closed: quotes in
        # the scan frame are stale after hours and any confirm would submit
        # outside the session (the exact "trading after close" failure).
        try:
            clock = await asyncio.to_thread(create_client().get_clock)
            if not clock.get("is_open"):
                logger.info("trade proposals: US market closed — skipping build")
                return
        except Exception as exc:
            logger.error("trade proposals: clock check failed (%s) — skipping build", exc)
            from framework.alerts import emit_clock_failure
            emit_clock_failure(exc)
            await self._flush_alerts()
            return
        try:
            rows = await asyncio.to_thread(build_proposals, self.approval_store)
        except Exception as exc:
            logger.exception("proposal build failed: %s", exc)
            return
        from earnings_edge import trade_approval
        funnel = funnel_line(trade_approval.LAST_FUNNEL)
        if not rows:
            logger.info("no trade proposals today")
            if funnel:
                logger.info("proposal %s", funnel)
            return
        # Execution-mode split: 'auto' strategies execute immediately (still
        # risk-gated inside execute_proposal — kill switch, portfolio caps,
        # lifecycle all apply), then push a notification. 'approval' rows go
        # out as cards for a human click, as before.
        modes = await asyncio.to_thread(self._effective_modes_sync)
        approval_rows, auto_rows = partition_by_mode(
            rows, lambda s: modes.get(s, "approval"))
        auto_notices = []
        if auto_rows:
            from earnings_edge.trade_approval import execute_proposal
            for row in auto_rows:
                try:
                    res = await asyncio.to_thread(
                        execute_proposal, self.approval_store, row["id"])
                except Exception as exc:
                    logger.exception("auto execution of proposal %d failed", row["id"])
                    res = {"ok": False, "error": f"exception: {exc}"}
                if res.get("ok"):
                    text = (f"⚡ AUTO-EXECUTED\n{row['card_text']}\n\n"
                            f"→ order {cards.esc(res.get('order_id'))} · status {cards.esc(res.get('status'))}"
                            f" · avg {cards.esc(res.get('filled_avg_price'))}")
                else:
                    text = (f"⚠️ AUTO {cards.esc(row['strategy'])} — execution failed/vetoed\n"
                            f"{row['card_text']}\n\n→ {cards.esc(res.get('error'))}")
                auto_notices.append({"strategy": row["strategy"], "text": text})
            logger.info("auto mode: executed %d proposal(s): %s",
                        len(auto_rows),
                        {r["strategy"]: r["id"] for r in auto_rows})
        routed = route_proposals(
            approval_rows, universe=universe, subs=self.strategy_subs, override_chat=override,
        )
        notice_routed = route_proposals(
            auto_notices, universe=universe, subs=self.strategy_subs, override_chat=override,
        )
        not_pushed = sorted({r["strategy"] for r in approval_rows} - {r["strategy"] for rs in routed.values() for r in rs})
        if not_pushed:
            logger.info("proposals built but not pushed (no subscribers): %s", not_pushed)
        for uid, urows in notice_routed.items():
            for strategy, snotices in cards.group_by_strategy(urows).items():
                if len(snotices) == 1:
                    text = snotices[0]["text"]
                else:
                    body = "\n\n".join(n["text"] for n in snotices)
                    text = f"{cards.header(cards.AUTO_EMOJI, f'{strategy} — {len(snotices)} auto')}\n\n{body}"
                try:
                    await self.application.bot.send_message(chat_id=uid, text=text, parse_mode=HTML)
                except Exception as exc:
                    logger.error("auto notice push to %d failed: %s", uid, exc)
        rows = approval_rows
        for uid, user_rows in routed.items():
            await self._push_batches(uid, user_rows, "entry")
            if len(user_rows) > 1:
                counts = {s: len(r) for s, r in cards.group_by_strategy(user_rows).items()}
                extra = "⏰ Confirm before 16:00 ET for same-day fills."
                if not_pushed:
                    extra += f" (not pushed, no subscribers: {', '.join(not_pushed)})"
                summary = cards.batch_overview(counts, "entry", extra=extra)
                if funnel:
                    summary += f"\n{cards.esc(funnel)}"
                try:
                    await self.application.bot.send_message(chat_id=uid, text=summary, parse_mode=HTML)
                except Exception as exc:
                    logger.error("batch summary push to %d failed: %s", uid, exc)
            elif funnel:
                try:
                    await self.application.bot.send_message(chat_id=uid, text=funnel)
                except Exception as exc:
                    logger.error("funnel push to %d failed: %s", uid, exc)

    def _propose_sync(self):
        self._dispatch(self._propose_and_push())

    # ── Framework jobs: equity snapshots, reconcile, guards ───────────

    async def _equity_snapshot(self):
        from earnings_edge.jobs import equity_snapshot_job
        await equity_snapshot_job(self)

    async def _reconcile(self):
        from earnings_edge.jobs import reconcile_job
        await reconcile_job(self)

    async def _guard_eval(self):
        from earnings_edge.jobs import guard_eval_job
        await guard_eval_job(self)

    async def _push_risk_alert(self, text: str):
        for uid in self._approval_chats():
            try:
                await self.application.bot.send_message(chat_id=uid, text=text, parse_mode=HTML)
            except Exception as exc:
                logger.error("risk alert push to %d failed: %s", uid, exc)

    def _equity_snapshot_sync(self):
        self._dispatch(self._equity_snapshot())

    def _reconcile_sync(self):
        self._dispatch(self._reconcile())

    def _guard_eval_sync(self):
        self._dispatch(self._guard_eval())

    # ── Exit engine ─────────────────────────────────────────────────────

    @staticmethod
    def _decide_exit(proposal_id: int, close: bool, uid: int) -> dict:
        from framework.positions.manager import ExitManager
        return ExitManager(create_client()).decide_exit(
            proposal_id, close, decided_by=uid)

    async def _exit_eval(self):
        from earnings_edge.jobs import exit_eval_job
        await exit_eval_job(self)

    def _exit_eval_sync(self):
        self._dispatch(self._exit_eval())

    # ── Forward-factor ladder ──────────────────────────────────────────

    @property
    def ff(self) -> LadderRunner:
        if self._ff_runner is None:
            self._ff_runner = LadderRunner(create_client())
        return self._ff_runner

    def _build_ff_candidates(self) -> list:
        eastern = pytz.timezone("US/Eastern")
        post_date, pre_date = scan_dates(None, eastern)
        alpaca = create_client()
        out = []
        collector = EarningsCalendarCollector()
        for target in dict.fromkeys([post_date, pre_date]):
            try:
                earnings = collector.fetch(target)
            except Exception as exc:
                logger.error("FF: earnings fetch for %s failed: %s", target, exc)
                continue
            for cand in earnings:
                try:
                    c = build_candidate(alpaca, cand.ticker, target)
                except Exception as exc:
                    logger.info("FF: candidate %s failed: %s", cand.ticker, exc)
                    continue
                if c.skip_reason:
                    logger.info("FF: skip %s — %s", c.ticker, c.skip_reason)
                    continue
                out.append(c)
        # best first: smallest distance of mid below the cap
        out.sort(key=lambda c: c.mid_debit / c.d_cap)
        return out[:10]

    async def _ff_propose_and_push(self):
        """13:45 ET: build FF ladder candidates and push proposal cards.

        Same internal process as every other strategy: candidates persist as
        pending_trades proposals, cards carry the standard Execute/Skip
        keyboard, and confirming runs execute_proposal() which arms the
        ladder instead of submitting a combo order.
        """
        from earnings_edge.trade_approval import build_ff_proposals
        from earnings_edge.fwd_factor_ladder import build_candidate as build_ff_candidate
        from earnings_edge.forward_factor_arb import build_candidate as build_arb_candidate
        from earnings_edge.db import snapshots_earnings_on_date
        from earnings_edge.models import EarningsCandidate

        alpaca = create_client()
        today = datetime.now(timezone.utc).date()
        target = today + timedelta(days=1)
        
        # Pull earnings for the target day from already-collected snapshots
        # (chain_cache populates this hourly) rather than re-running a scan.
        earnings = [
            EarningsCandidate(ticker=t, timing=timing or "Unknown", earnings_date=target, source="db")
            for t, timing in await asyncio.to_thread(snapshots_earnings_on_date, earnings_date=target.isoformat())
        ]

        # Build candidates for ff_ladder
        ff_candidates = []
        if earnings:
            for cand in earnings:
                try:
                    c = build_ff_candidate(alpaca, cand.ticker, target)
                    if not c.skip_reason:
                        c.strategy_override = "ff_ladder"
                        ff_candidates.append(c)
                except Exception as exc:
                    logger.info("FF: candidate %s failed: %s", cand.ticker, exc)
                    
        ff_candidates.sort(key=lambda c: c.mid_debit / c.d_cap)
        ff_candidates = ff_candidates[:10]

        # Build candidates for forward_factor_arb
        arb_candidates = []
        if earnings:
            for cand in earnings:
                try:
                    c = build_arb_candidate(alpaca, cand.ticker, target)
                    if not c.skip_reason:
                        c.strategy_override = "forward_factor_arb"
                        arb_candidates.append(c)
                except Exception as exc:
                    logger.info("ARB: candidate %s failed: %s", cand.ticker, exc)
                    
        arb_candidates.sort(key=lambda c: c.mid_debit / c.d_cap)
        arb_candidates = arb_candidates[:10]

        all_candidates = ff_candidates + arb_candidates
        if not all_candidates:
            logger.info("FF/ARB: no ladder candidates today")
            return

        rows = await asyncio.to_thread(
            build_ff_proposals, self.approval_store, all_candidates)
        if not rows:
            logger.info("FF/ARB: %d candidates, all already pending", len(all_candidates))
            return
        
        chats = self._approval_chats()
        if not chats:
            logger.warning("FF proposals: no approval chats")
            return
        if not await asyncio.to_thread(self._strategy_enabled, "ff_ladder"):
            logger.info("FF proposals: ff_ladder disabled by operator — skipping")
            return
        try:
            clock = await asyncio.to_thread(create_client().get_clock)
            if not clock.get("is_open"):
                logger.info("FF proposals: market closed today (holiday/weekend) — skipping")
                return
        except Exception as exc:
            logger.error("FF proposals: clock check failed (%s) — aborting", exc)
            from framework.alerts import emit_clock_failure
            emit_clock_failure(exc)
            await self._flush_alerts()
            return
        try:
            candidates = await asyncio.to_thread(self._build_ff_candidates)
        except Exception as exc:
            logger.exception("FF candidate build failed: %s", exc)
            return
        if not candidates:
            logger.info("FF: no ladder candidates today")
            return
        rows = await asyncio.to_thread(
            build_ff_proposals, self.approval_store, candidates)
        if not rows:
            logger.info("FF: %d candidates, all already pending", len(candidates))
            return
        # Auto mode: confirm immediately through the same execution path
        # (risk-gated inside LadderRunner.arm) and notify instead of cards.
        modes = await asyncio.to_thread(self._effective_modes_sync)
        if modes.get("ff_ladder") == "auto":
            notices = []
            armed = 0
            for row in rows:
                res = await asyncio.to_thread(
                    execute_proposal, self.approval_store, row["id"])
                if res.get("ok"):
                    armed += 1
                    notices.append({
                        "strategy": "ff_ladder",
                        "text": (f"⚡ AUTO-ARMED\n{row['card_text']}\n\n"
                                 f"→ ladder #{cards.esc(res.get('ladder_id'))} armed — "
                                 f"steps 14:00–15:45 ET"),
                    })
                else:
                    notices.append({
                        "strategy": "ff_ladder",
                        "text": (f"⚠️ AUTO ff_ladder — arm failed/vetoed\n"
                                 f"{row['card_text']}\n\n→ {cards.esc(res.get('error'))}"),
                    })
            logger.info("FF auto mode: armed %d/%d proposals", armed, len(rows))
            routed = route_proposals(
                notices, universe=chats, subs=self.strategy_subs,
                override_chat=self._approval_override())
            for uid, urows in routed.items():
                body = "\n\n".join(n["text"] for n in urows)
                text = (body if len(urows) == 1 else
                        f"{cards.header(cards.AUTO_EMOJI, f'ff_ladder — {len(urows)} auto')}\n\n{body}")
                try:
                    await self.application.bot.send_message(chat_id=uid, text=text, parse_mode=HTML)
                except Exception as exc:
                    logger.error("FF auto notice to %d failed: %s", uid, exc)
            return
        # same opt-out routing as scan-chained proposals (/signals)
        routed = route_proposals(
            rows, universe=chats, subs=self.strategy_subs,
            override_chat=self._approval_override())
        if not routed:
            logger.info("FF: %d proposals, zero recipients after /signals opt-outs", len(rows))
            return
        for uid, urows in routed.items():
            await self._push_batches(uid, urows, "entry")

    async def _ff_step_and_report(self):
        """Every 15 min 14:00–15:45 ET: reprice armed ladders, push events."""
        try:
            clock = await asyncio.to_thread(create_client().get_clock)
            if not clock.get("is_open"):
                return  # market closed — quotes would be stale anyway
        except Exception as exc:
            logger.error("FF step: clock check failed (%s) — skipping step", exc)
            from framework.alerts import emit_clock_failure
            emit_clock_failure(exc)
            await self._flush_alerts()
            return
        try:
            await asyncio.to_thread(self.ff.step, datetime.now(timezone.utc))
        except Exception as exc:
            logger.exception("FF ladder step failed: %s", exc)
            return
        events = self.ff.drain_events()
        if not events:
            return
        routed = route_proposals(
            [{"strategy": "ff_ladder", "text": ev} for ev in events],
            universe=self._approval_chats(), subs=self.strategy_subs,
            override_chat=self._approval_override())
        for uid, urows in routed.items():
            for row in urows:
                try:
                    await self.application.bot.send_message(chat_id=uid, text=row["text"])
                except Exception as exc:
                    logger.error("FF event push to %d failed: %s", uid, exc)

    def _ff_propose_sync(self):
        self._dispatch(self._ff_propose_and_push())

    def _ff_step_sync(self):
        self._dispatch(self._ff_step_and_report())

    @staticmethod
    async def _edit_card_keep(query, footer: str) -> None:
        """Append an outcome footer to the card instead of replacing it.

        The signal itself (ticker, legs, strikes, prices) must stay visible
        so it can still be traded manually; only the inline keyboard is
        dropped. A repeat click replaces the previous footer."""
        base = (query.message.text or "").split("\n→ ")[0]
        try:
            await query.edit_message_text(f"{base}\n→ {footer}")
        except Exception:
            await query.message.reply_text(f"Proposal outcome: {footer}")

    # ── Decision outcomes: (ticker, footer_text) for one proposal ───────
    # Shared by both the classic single-card path and the grouped-batch
    # rebuild path (_decide_in_group / _decide_group_bulk) so wording stays
    # identical regardless of how many rows a message happened to carry.

    async def _entry_exec_outcome(self, pid: int, uid: int) -> tuple[str, str]:
        row = self.approval_store.get(pid)
        ticker = row["ticker"] if row else str(pid)
        result = await asyncio.to_thread(
            execute_proposal, self.approval_store, pid, decided_by=uid)
        if result.get("ok"):
            if result.get("ladder_id") is not None:
                footer = (f"✅ FF ladder #{result['ladder_id']} armed — "
                          f"steps 14:00–15:45 ET")
            else:
                footer = (f"✅ Executed — order {result['order_id']} "
                          f"({result['status']})")
        else:
            footer = f"⚠️ NOT executed: {result.get('error')}"
        return ticker, footer

    async def _entry_skip_outcome(self, pid: int, uid: int) -> tuple[str, str]:
        row = self.approval_store.get(pid)
        ticker = row["ticker"] if row else str(pid)
        result = await asyncio.to_thread(reject_proposal, self.approval_store, pid, decided_by=uid)
        footer = "❌ Skipped." if result.get("ok") else f"⚠️ {result.get('error')}"
        return ticker, footer

    def _exit_row(self, proposal_id: int) -> Optional[dict]:
        return exit_proposals_get(proposal_id)

    async def _exit_close_outcome(self, pid: int, uid: int) -> tuple[str, str]:
        row = self._exit_row(pid)
        ticker = row["ticker"] if row else str(pid)
        result = await asyncio.to_thread(self._decide_exit, pid, True, uid)
        if result.get("ok"):
            footer = f"🔒 Exit #{pid} filled @ {result.get('filled_avg_price')}"
        else:
            footer = f"⚠️ Exit #{pid}: {result.get('error') or result.get('order_state') or result.get('detail')}"
        return ticker, footer

    async def _exit_snooze_outcome(self, pid: int, uid: int) -> tuple[str, str]:
        row = self._exit_row(pid)
        ticker = row["ticker"] if row else str(pid)
        result = await asyncio.to_thread(self._decide_exit, pid, False, uid)
        footer = (f"⏰ Exit #{pid} snoozed until tomorrow." if result.get("ok")
                 else f"⚠️ Exit #{pid}: {result.get('error')}")
        return ticker, footer

    async def _decide_in_group(self, query, uid: int, kind: str, pid: int, action: str) -> None:
        """Decide one proposal that may be part of a grouped batch message.

        A single-item message keeps the classic in-place footer
        (_edit_card_keep). A multi-item message rebuilds around whichever
        ids are still on the CLICKED message's own keyboard (_kb_ids) —
        never a fresh strategy-wide query — so a later, unrelated batch for
        the same strategy can never bleed into this one.
        """
        prefixes = ("pt_exec_", "pt_skip_") if kind == "entry" else ("ex_close_", "ex_skip_")
        ids = self._kb_ids(query, prefixes)
        grouped = len(ids) > 1

        if action == "exec" and not grouped:
            # Keep the card visible; only the footer shows in-flight state.
            verb = "Executing" if kind == "entry" else "Closing position"
            await self._edit_card_keep(query, f"⏳ {verb} #{pid}…")

        if kind == "entry":
            ticker, footer = await (self._entry_exec_outcome(pid, uid) if action == "exec"
                                    else self._entry_skip_outcome(pid, uid))
            fetch = self.approval_store.get
        else:
            ticker, footer = await (self._exit_close_outcome(pid, uid) if action == "exec"
                                    else self._exit_snooze_outcome(pid, uid))
            fetch = self._exit_row

        if not grouped:
            await self._edit_card_keep(query, footer)
            return

        remaining = [r for rid in ids if rid != pid
                    and (r := fetch(rid)) and r["status"] == "pending"]
        outcome_line = f"{cards.esc(ticker)}: {cards.esc(footer)}"
        if not remaining:
            base = (query.message.text or "").split("\n\n→ ")[0]
            try:
                await query.edit_message_text(
                    f"{base}\n\n→ {outcome_line}\nAll decided.", parse_mode=HTML)
            except Exception as exc:
                logger.error("group finish edit failed: %s", exc)
            return
        strategy = remaining[0]["strategy"]
        text = f"{cards.group_message(strategy, remaining, kind)}\n\n→ {outcome_line}"
        try:
            await query.edit_message_text(
                text, reply_markup=self._grouped_kb(remaining, kind), parse_mode=HTML)
        except Exception as exc:
            logger.error("group rebuild edit failed: %s", exc)

    async def _decide_group_bulk(self, query, uid: int, kind: str) -> None:
        """'Execute/Close ALL' inside one grouped batch message — acts on
        exactly the ids still on the clicked message's keyboard."""
        prefixes = ("pt_exec_", "pt_skip_") if kind == "entry" else ("ex_close_", "ex_skip_")
        ids = self._kb_ids(query, prefixes)
        if not ids:
            await query.edit_message_text("Nothing pending.")
            return
        lines = []
        for pid in ids:
            if kind == "entry":
                ticker, footer = await self._entry_exec_outcome(pid, uid)
            else:
                ticker, footer = await self._exit_close_outcome(pid, uid)
            lines.append(f"{cards.esc(ticker)}: {cards.esc(footer)}")
        base = (query.message.text or "").split("\n\n→ ")[0]
        try:
            await query.edit_message_text(
                f"{base}\n\n→ Batch decided:\n" + "\n".join(lines), parse_mode=HTML)
        except Exception as exc:
            logger.error("group bulk edit failed: %s", exc)

    async def _handle_book_callback(self, query, uid: int, data: str) -> None:
        """Close / adopt / ignore / mark-closed — operator-only, confirm on close.

        Every outcome re-renders this same Positions message so the book
        updates without tapping the reply keyboard again.
        """
        if data == "bk_rf":
            await self._refresh_positions_query(query, "🔄 Refreshed.")
            return
        if not self._risk_authorized(uid):
            await query.edit_message_text(auth_message())
            return
        if data.startswith("bk_xs_"):
            symbol = data[len("bk_xs_"):]
            kb = InlineKeyboardMarkup([[
                InlineKeyboardButton("✅ Confirm close", callback_data=f"bk_oks_{symbol}"),
                InlineKeyboardButton("❌ Cancel", callback_data="bk_nop"),
            ]])
            await self._edit_panel(
                query, f"Close {symbol} at the broker?", kb)
            return
        if data.startswith("bk_xg_"):
            gid = data[len("bk_xg_"):]
            kb = InlineKeyboardMarkup([[
                InlineKeyboardButton("✅ Confirm close group", callback_data=f"bk_okg_{gid}"),
                InlineKeyboardButton("❌ Cancel", callback_data="bk_nop"),
            ]])
            await self._edit_panel(
                query, f"Close group {gid[:12]}… at the broker?", kb)
            return
        if data == "bk_nop":
            await self._refresh_positions_query(query, "Cancelled.")
            return

        from earnings_edge.bot_views import book_action_banner

        def act():
            from framework.positions import book_actions as ba
            client = create_client()
            if data.startswith("bk_oks_"):
                return "close", data[len("bk_oks_"):], ba.close_symbol(
                    client, data[len("bk_oks_"):], by=f"telegram:{uid}")
            if data.startswith("bk_okg_"):
                return "close_group", data[len("bk_okg_"):], ba.close_group_at_broker(
                    client, data[len("bk_okg_"):], by=f"telegram:{uid}")
            if data.startswith("bk_ad_"):
                symbol = data[len("bk_ad_"):]
                pos = ba.find_broker_pos(client.get_positions(), symbol)
                if not pos:
                    return "adopt", symbol, {"ok": False, "error": "symbol not at broker"}
                return "adopt", symbol, ba.adopt_orphan(pos, by=f"telegram:{uid}")
            if data.startswith("bk_ig_"):
                symbol = data[len("bk_ig_"):]
                return "ignore", symbol, ba.ignore_orphan(symbol, by=f"telegram:{uid}")
            if data.startswith("bk_ml_"):
                gid = data[len("bk_ml_"):]
                return "mark_closed", gid, ba.mark_missing_closed(
                    gid, by=f"telegram:{uid}")
            return "unknown", "", {"ok": False, "error": "unknown book action"}

        kind, target, result = await asyncio.to_thread(act)
        banner = book_action_banner(kind, result, target)
        await self._refresh_positions_query(query, banner)

    async def _handle_approval_callback(self, update: Update, data: str) -> None:
        query = update.callback_query
        uid = query.from_user.id
        acting = data.startswith("pt_exec") or data == "pt_exec_all" or data == "pt_exec_grp"
        if acting and not self._risk_authorized(uid):
            await query.edit_message_text(auth_message())
            return
        if data == "pt_exec_all":
            pending = await asyncio.to_thread(self.approval_store.list_pending)
            if not pending:
                await query.edit_message_text("No pending proposals.")
                return
            base = (query.message.text or "").split("\n\n→ ")[0]
            await self._edit_card_keep(query, f"⏳ Executing {len(pending)} proposals…")
            lines = []
            for row in pending:
                result = await asyncio.to_thread(
                    execute_proposal, self.approval_store, row["id"], decided_by=uid)
                if result.get("ok"):
                    lines.append(f"✅ #{row['id']} {row['ticker']} {row['side']} — "
                                 f"order {result['order_id']} ({result['status']})")
                else:
                    lines.append(f"⚠️ #{row['id']} {row['ticker']} {row['side']} — "
                                 f"{result.get('error')}")
            try:
                await query.edit_message_text(
                    f"{base}\n\n→ " + "\n".join(lines))
            except Exception as exc:
                logger.error("pt_exec_all edit failed: %s", exc)
        elif data == "pt_exec_grp":
            await self._decide_group_bulk(query, uid, "entry")
        elif data.startswith("pt_exec_"):
            pid = int(data[len("pt_exec_"):])
            await self._decide_in_group(query, uid, "entry", pid, "exec")
        elif data.startswith("pt_skip_"):
            pid = int(data[len("pt_skip_"):])
            await self._decide_in_group(query, uid, "entry", pid, "skip")
        elif data.startswith(("ff_arm_", "ff_skip_")):
            # Legacy arm-cards from before FF ladders moved onto the standard
            # proposal store — the in-memory candidate map no longer exists.
            await self._edit_card_keep(
                query, "⚠️ stale card — FF ladders now arrive as standard "
                       "Execute/Skip proposals in the 13:45 ET batch.")

    # ── Risk/lifecycle sync helpers ──────────────────────────────────────
    # Pure DB-facing work shared by the slash commands and the /settings
    # inline menu, so both surfaces stay in lockstep with one implementation.

    def _halt_sync(self, by: str) -> None:
        from framework.risk.killswitch import KillSwitch
        KillSwitch().trip("manual halt via bot", by)

    def _resume_sync(self, by: str) -> None:
        from framework.risk.killswitch import KillSwitch
        KillSwitch().resume(by)

    def _risk_status_sync(self) -> str:
        from framework.execution.lifecycle import LifecycleManager
        from framework.risk.equity import latest_equity
        from framework.risk.killswitch import KillSwitch
        ks = KillSwitch().status()
        eq = latest_equity()
        states = LifecycleManager().all_states()
        lines = ["<b>Risk status</b>"]
        if ks.get("halted"):
            lines.append(f"🛑 <b>HALTED:</b> {cards.esc(ks.get('reason'))} (by {cards.esc(ks.get('tripped_by'))})")
        else:
            lines.append("🟢 <b>Kill switch:</b> armed (not halted)")
        if eq:
            lines.append(f"💰 <b>Equity:</b> ${eq['equity']:,.0f} | <b>BP:</b> ${eq['buying_power']:,.0f} "
                         f"(as of {eq['ts'][:16]})")
        else:
            lines.append("💰 <b>No equity snapshots yet</b>")
        if states:
            lines.append("📊 <b>Lifecycle:</b> " + ", ".join(f"<code>{cards.esc(k)}</code>={v}" for k, v in states.items()))
        return "\n".join(lines)

    def _lifecycle_state_sync(self, name: str) -> str:
        from framework.execution.lifecycle import LifecycleManager
        return LifecycleManager().state(name)

    def _lifecycle_step_sync(self, name: str, promote: bool, by: str) -> tuple[str, str]:
        from framework.execution.lifecycle import LIFECYCLES, LifecycleManager
        lm = LifecycleManager()
        cur = lm.state(name)
        idx = LIFECYCLES.index(cur) + (1 if promote else -1)
        idx = max(0, min(idx, len(LIFECYCLES) - 1))
        lm.set_state(name, LIFECYCLES[idx], by=by)
        return cur, LIFECYCLES[idx]

    def _lifecycle_menu_data_sync(self) -> tuple[list[str], dict]:
        from framework.core.registry import get_registry
        from framework.execution.lifecycle import LifecycleManager
        registry = get_registry()
        states = LifecycleManager().all_states()
        return sorted(set(registry.configs) | set(states)), states

    async def _cmd_halt(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not self._risk_authorized(update.effective_chat.id):
            await update.message.reply_text(auth_message())
            return
        await asyncio.to_thread(self._halt_sync, "operator")
        await self._flush_alerts()
        await update.message.reply_text("🛑 KILL SWITCH TRIPPED — no orders will submit until /resume.")

    async def _cmd_resume(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not self._risk_authorized(update.effective_chat.id):
            await update.message.reply_text(auth_message())
            return
        await asyncio.to_thread(self._resume_sync, "operator")
        await update.message.reply_text("✅ Kill switch released — order submission re-enabled.")

    async def _cmd_risk(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        text = await asyncio.to_thread(self._risk_status_sync)
        await update.message.reply_text(text, parse_mode=HTML)

    async def _cmd_promote(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        await self._cmd_set_lifecycle(update, promote=True)

    async def _cmd_demote(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        await self._cmd_set_lifecycle(update, promote=False)

    async def _cmd_set_lifecycle(self, update: Update, promote: bool):
        if not self._risk_authorized(update.effective_chat.id):
            await update.message.reply_text(auth_message())
            return
        args = (update.message.text or "").split()
        if len(args) != 2:
            await update.message.reply_text("Usage: /promote <strategy> | /demote <strategy>")
            return
        cur, new = await asyncio.to_thread(
            self._lifecycle_step_sync, args[1], promote, "telegram")
        await update.message.reply_text(f"📊 <code>{cards.esc(args[1])}</code>: {cur} → {new}", parse_mode=HTML)

    # ── Settings menu ─────────────────────────────────────────────────
    # Consolidates the risk/admin controls (halt, resume, risk status,
    # strategy lifecycle, restart) that previously existed only as
    # slash commands with no button/menu discoverability.

    @staticmethod
    def _settings_kb() -> InlineKeyboardMarkup:
        return InlineKeyboardMarkup([
            [InlineKeyboardButton("🛡 Risk Status", callback_data="set_risk")],
            [InlineKeyboardButton("🛑 Halt Trading", callback_data="set_halt"),
             InlineKeyboardButton("✅ Resume Trading", callback_data="set_resume")],
            [InlineKeyboardButton("📊 Strategy Lifecycle", callback_data="set_lifecycle")],
            [InlineKeyboardButton("🔄 Restart Bot", callback_data="set_restart")],
            [InlineKeyboardButton("🏠 Back to Main Menu", callback_data="set_home")],
        ])

    async def _cmd_settings(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        await self._send_panel(update, "🛠 <b>Settings</b>", parse_mode=HTML,
                                        reply_markup=self._settings_kb())

    @staticmethod
    def _lifecycle_kb(names: list[str], states: dict) -> InlineKeyboardMarkup:
        rows = [
            [InlineKeyboardButton(f"{name} ({states.get(name, 'paper')})", callback_data="sig_noop")]
            + [InlineKeyboardButton("⬆️ Promote", callback_data=f"set_promote_{name}"),
               InlineKeyboardButton("⬇️ Demote", callback_data=f"set_demote_{name}")]
            for name in names
        ]
        rows.append([InlineKeyboardButton("⬅️ Back to Settings", callback_data="set_back")])
        return InlineKeyboardMarkup(rows)

    async def _lifecycle_menu_text_kb(self) -> tuple[str, InlineKeyboardMarkup]:
        names, states = await asyncio.to_thread(self._lifecycle_menu_data_sync)
        text = "📊 <b>Strategy Lifecycle</b>\nTap Promote/Demote to step paper → probation → live."
        return text, self._lifecycle_kb(names, states)

    async def _handle_settings_callback(self, query, uid: int, data: str) -> None:
        if data == "set_back":
            await query.edit_message_text("🛠 <b>Settings</b>", parse_mode=HTML,
                                           reply_markup=self._settings_kb())
            return
        if data == "set_home":
            # The persistent reply keyboard is a separate UI element from
            # this inline menu — Telegram can't restore it via an edit, only
            # by sending a fresh message with reply_markup set.
            kb = _main_reply_kb()
            await query.message.reply_text("🏠 Main Menu", reply_markup=kb)
            return
        if data == "set_risk":
            text = await asyncio.to_thread(self._risk_status_sync)
            kb = InlineKeyboardMarkup(
                [[InlineKeyboardButton("⬅️ Back to Settings", callback_data="set_back")]])
            await query.edit_message_text(text, reply_markup=kb)  # no parse_mode: see _cmd_risk
            return
        if data in ("set_halt", "set_resume"):
            if not self._risk_authorized(uid):
                await query.edit_message_text(auth_message())
                return
            if data == "set_halt":
                await asyncio.to_thread(self._halt_sync, f"telegram:{uid}")
                await self._flush_alerts()
                note = "🛑 KILL SWITCH TRIPPED — no orders will submit until resumed."
            else:
                await asyncio.to_thread(self._resume_sync, f"telegram:{uid}")
                note = "✅ Kill switch released — order submission re-enabled."
            await query.edit_message_text(
                f"{note}\n\n🛠 <b>Settings</b>", parse_mode=HTML,
                reply_markup=self._settings_kb())
            return
        if data == "set_lifecycle":
            text, kb = await self._lifecycle_menu_text_kb()
            await query.edit_message_text(text, parse_mode=HTML, reply_markup=kb)
            return
        if data.startswith(("set_promote_", "set_demote_")):
            if not self._risk_authorized(uid):
                await query.edit_message_text(auth_message())
                return
            promote = data.startswith("set_promote_")
            name = data[len("set_promote_"):] if promote else data[len("set_demote_"):]
            if promote:
                cur = await asyncio.to_thread(self._lifecycle_state_sync, name)
                if cur == "probation":
                    kb = InlineKeyboardMarkup([[
                        InlineKeyboardButton("✅ Type-confirm LIVE", callback_data=f"set_live_{name}"),
                        InlineKeyboardButton("❌ Cancel", callback_data="set_lifecycle"),
                    ]])
                    await query.edit_message_text(
                        f"⚠️ Promote {cards.esc(name)} from probation to <b>LIVE</b>?\n"
                        "This allows real (or paper-live) execution. Tap confirm.",
                        parse_mode=HTML, reply_markup=kb)
                    return
            cur, new = await asyncio.to_thread(
                self._lifecycle_step_sync, name, promote, f"telegram:{uid}")
            text, kb = await self._lifecycle_menu_text_kb()
            text = f"📊 {cards.esc(name)}: {cards.esc(cur)} → {cards.esc(new)}\n\n{text}"
            await query.edit_message_text(text, parse_mode=HTML, reply_markup=kb)
            return
        if data.startswith("set_live_"):
            if not self._risk_authorized(uid):
                await query.edit_message_text(auth_message())
                return
            name = data[len("set_live_"):]
            cur, new = await asyncio.to_thread(
                self._lifecycle_step_sync, name, True, f"telegram:{uid}")
            text, kb = await self._lifecycle_menu_text_kb()
            text = f"📊 {cards.esc(name)}: {cards.esc(cur)} → {cards.esc(new)}\n\n{text}"
            await query.edit_message_text(text, parse_mode=HTML, reply_markup=kb)
            return
        if data == "set_restart":
            if not self._risk_authorized(uid):
                await query.edit_message_text(auth_message())
                return
            kb = InlineKeyboardMarkup([[
                InlineKeyboardButton("✅ Confirm restart", callback_data="set_restart_confirm"),
                InlineKeyboardButton("❌ Cancel", callback_data="set_back"),
            ]])
            await query.edit_message_text(
                "⚠️ Restart the bot now?\n"
                "Picks up any deployed code changes. Back online in ~10-15s; "
                "live /monitor sessions and in-flight scans will be interrupted.",
                reply_markup=kb)
            return
        if data == "set_restart_confirm":
            if not self._risk_authorized(uid):
                await query.edit_message_text(auth_message())
                return
            await query.edit_message_text(
                "🔄 Restarting — back online in ~10-15s (picking up any code changes)…")
            logger.warning("Bot restart requested via Telegram settings (uid=%s)", uid)
            await self._restart_bot()
            return

    async def _restart_bot(self) -> None:
        """Exit the process so the supervisor relaunches a fresh interpreter
        that re-imports every module from disk — the same effect as
        `systemctl restart trading-bot.service`, without needing shell/sudo
        access from inside the bot.

        The exit code MUST be non-zero: the systemd unit uses
        Restart=on-failure, which does not relaunch on a clean exit(0), and
        run-bot.sh's supervisor loop restarts on any exit code — so a
        non-zero code is the one choice that works under both.
        """
        try:
            await asyncio.wait_for(
                asyncio.to_thread(self.scheduler.shutdown, wait=False), timeout=3.0)
        except Exception as exc:
            logger.warning("scheduler shutdown before restart: %s", exc)
        os._exit(1)

    def _risk_authorized(self, chat_id: int) -> bool:
        """Fail-closed: only chats on TELEGRAM_APPROVAL_CHAT_ID may act."""
        return is_operator(chat_id)

    def _next_events_sync(self, limit: int = 4) -> list:
        """Next scheduler fire times, formatted in ET (empty on any failure)."""
        try:
            jobs = [j for j in self.scheduler.get_jobs() if j.next_run_time]
        except Exception:
            return []
        jobs.sort(key=lambda j: j.next_run_time)
        et = pytz.timezone("US/Eastern")
        today = datetime.now(et).date()
        out = []
        for j in jobs[:limit]:
            t = j.next_run_time.astimezone(et)
            day = "" if t.date() == today else t.strftime("%a ")
            out.append(f"{j.name} {day}{t.strftime('%H:%M')} ET")
        return out

    # ── Live monitor (/monitor) ────────────────────────────────────────

    async def _cmd_monitor(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        chat_id = update.effective_chat.id
        await self._stop_monitor(chat_id)
        self._monitors[chat_id] = asyncio.create_task(self._monitor_loop(chat_id))

    async def _stop_monitor(self, chat_id: int) -> None:
        task = self._monitors.pop(chat_id, None)
        if task is not None:
            task.cancel()
            try:
                await task
            except Exception:
                pass

    def _monitor_text_sync(self, tick: int) -> str:
        from earnings_edge import trade_approval
        from earnings_edge.bot_views import desk_view_kwargs, monitor_view, pending_exits
        facts = self._desk_facts_sync()
        return monitor_view(
            tick=tick,
            pending_proposals=len(self.approval_store.list_pending()),
            pending_exits=len(pending_exits()),
            next_events=self._next_events_sync(),
            funnel=funnel_line(trade_approval.LAST_FUNNEL),
            **desk_view_kwargs(facts))

    async def _monitor_loop(self, chat_id: int) -> None:
        """Self-updating ops panel: every 30s, capped at 15 min."""
        msg = None
        text = ""
        kb = InlineKeyboardMarkup(
            [[InlineKeyboardButton("⏹ Stop monitor", callback_data="mon_stop")]])
        try:
            for tick in range(30):
                try:
                    text = await asyncio.to_thread(self._monitor_text_sync, tick)
                    await self._flush_alerts()
                except Exception as exc:
                    text = f"📡 monitor error: {exc}"
                if msg is None:
                    msg = await self.application.bot.send_message(
                        chat_id=chat_id, text=text, reply_markup=kb)
                else:
                    try:
                        await msg.edit_text(text, reply_markup=kb)
                    except Exception as exc:
                        logger.debug("monitor edit skipped: %s", exc)
                if tick < 29:
                    await asyncio.sleep(30)
            try:
                await msg.edit_text(text + "\n\n⏹ monitor expired — /monitor to restart")
            except Exception:
                pass
        except asyncio.CancelledError:
            if msg is not None:
                try:
                    await msg.edit_text("⏹ Monitor stopped.")
                except Exception:
                    pass
            raise
        finally:
            self._monitors.pop(chat_id, None)

    async def _cmd_status(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        text = await asyncio.to_thread(self._status_text_sync)
        await self._flush_alerts()
        await self._send_panel(update, 
            text, reply_markup=self._desk_refresh_kb("st"), parse_mode=HTML)

    def _status_text_sync(self) -> str:
        from earnings_edge import trade_approval
        from earnings_edge.bot_views import desk_view_kwargs, pending_exits, status_view
        facts = self._desk_facts_sync()
        return status_view(
            pending_proposals=len(self.approval_store.list_pending()),
            pending_exits=len(pending_exits()),
            next_events=self._next_events_sync(),
            funnel=funnel_line(trade_approval.LAST_FUNNEL),
            **desk_view_kwargs(facts),
        )

    @staticmethod
    def _desk_refresh_kb(which: str) -> InlineKeyboardMarkup:
        return InlineKeyboardMarkup(
            [[InlineKeyboardButton("🔄 Refresh", callback_data=f"desk_{which}")]])

    async def _handle_desk_refresh(self, query, uid: int, data: str) -> None:
        if data == "desk_st":
            text = await asyncio.to_thread(self._status_text_sync)
            await self._flush_alerts()
            await self._edit_panel(query, text, self._desk_refresh_kb("st"), parse_mode=HTML)
        elif data == "desk_jb":
            text = await asyncio.to_thread(self._jobs_text_sync)
            await self._edit_panel(query, text, self._desk_refresh_kb("jb"), parse_mode=HTML)
        elif data == "desk_pd":
            await self._refresh_pending_query(query)
        elif data == "desk_or":
            text = await asyncio.to_thread(self._orders_text_sync)
            await self._edit_panel(query, text, self._desk_refresh_kb("or"), parse_mode=HTML)
        elif data == "desk_eq":
            text = await asyncio.to_thread(self._equity_text_sync)
            await self._edit_panel(query, text, self._desk_refresh_kb("eq"), parse_mode=HTML)
        elif data == "desk_run":
            msg, ikb = self._run_panel()
            await self._edit_panel(query, msg, ikb, parse_mode=HTML)
        else:
            await query.answer()

    async def _cmd_positions(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        text, rows = await asyncio.to_thread(self._positions_panel_sync)
        await self._flush_alerts()
        markup = InlineKeyboardMarkup(rows) if rows else None
        # One message: book + actions. Reply keyboard stays MAIN_KB.
        await self._send_panel(update, text, reply_markup=markup, parse_mode=HTML)

    def _orders_text_sync(self) -> str:
        from earnings_edge.bot_views import orders_view
        return orders_view()

    async def _cmd_orders(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        text = await asyncio.to_thread(self._orders_text_sync)
        await self._send_panel(update, text, reply_markup=self._desk_refresh_kb("or"), parse_mode=HTML)

    def _jobs_text_sync(self) -> str:
        from earnings_edge.bot_views import jobs_view
        return jobs_view()

    async def _cmd_jobs(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        text = await asyncio.to_thread(self._jobs_text_sync)
        await self._send_panel(update, text, reply_markup=self._desk_refresh_kb("jb"), parse_mode=HTML)

    def _equity_text_sync(self) -> str:
        from earnings_edge.bot_views import equity_view
        return equity_view()

    async def _cmd_equity(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        text = await asyncio.to_thread(self._equity_text_sync)
        await self._send_panel(update, text, reply_markup=self._desk_refresh_kb("eq"), parse_mode=HTML)

    def _strategies_panel_sync(self):
        from earnings_edge.bot_views import strategies_view
        text, buttons = strategies_view()
        ikb = [[InlineKeyboardButton(
            ("⏸ Pause " if b["enabled"] else "▶️ Resume ") + b["name"],
            callback_data=("st_off_" if b["enabled"] else "st_on_") + b["name"],
        )] for b in buttons]
        return text, ikb

    async def _cmd_strategies(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        text, ikb = await asyncio.to_thread(self._strategies_panel_sync)
        await self._send_panel(update, text, reply_markup=InlineKeyboardMarkup(ikb))

    async def _cmd_exits(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        def gather():
            from earnings_edge.bot_views import pending_exits
            return pending_exits()
        rows = await asyncio.to_thread(gather)
        if not rows:
            await update.message.reply_text("🚪 No pending exit proposals.")
            return
        counts = {s: len(r) for s, r in cards.group_by_strategy(rows).items()}
        await update.message.reply_text(
            cards.batch_overview(counts, "exit"), parse_mode=HTML)
        await self._push_batches(update.effective_chat.id, rows, "exit")

    def _strategy_enabled(self, name: str) -> bool:
        """Effective on/off for a strategy: TOML flag + operator override."""
        from framework.core.control import effective_enabled
        from framework.core.registry import get_registry
        return effective_enabled(name, get_registry().is_enabled(name))

    def _pending_panel_sync(self, banner: Optional[str] = None):
        from earnings_edge.inbox import assemble_inbox, inbox_keyboard, render_inbox
        from earnings_edge.bot_views import pending_exits
        from earnings_edge.db import job_runs_failed, trade_events_list
        from earnings_edge.db.repositories import adopted_positions_symbols
        from framework.execution.managed import open_positions
        from earnings_edge.alpaca_bridge import create_client

        entries = self.approval_store.list_pending()
        exits = pending_exits()
        jobs = job_runs_failed(10)
        
        # Pull latest events
        raw_orphans = trade_events_list(event_type="orphan_found", limit=50)
        raw_assigns = trade_events_list(event_type="assignment_detected", limit=50)

        # Filter out stale events that are no longer true
        try:
            broker_syms = {p.get("symbol") for p in create_client().get_positions() if p.get("symbol")}
        except Exception:
            broker_syms = set()

        managed_syms = {row.get("symbol") for row in open_positions() if row.get("symbol")}
        ignored_syms = adopted_positions_symbols()

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
        assignments = _filter_events(raw_assigns)

        inbox = assemble_inbox(
            entries=entries, exits=exits, orphans=orphans,
            assignments=assignments, jobs=jobs)
        text = render_inbox(inbox)
        if banner:
            text = f"{cards.esc(banner)}\n\n{text}"
        return text, inbox_keyboard(inbox)

    async def _refresh_pending_query(self, query, banner: Optional[str] = None) -> None:
        text, rows = await asyncio.to_thread(self._pending_panel_sync, banner)
        await self._edit_panel(query, text, InlineKeyboardMarkup(rows), parse_mode=HTML)

    async def _handle_inbox_callback(self, query, uid: int, data: str) -> None:
        """Act on a Pending-inbox row and rewrite that same message."""
        acting = data.startswith(("in_ex_", "in_cl_", "in_ad_", "in_xs_"))
        if acting and not self._risk_authorized(uid):
            await query.edit_message_text(auth_message())
            return
        banner = "🔄 Refreshed."
        if data.startswith("in_ex_"):
            _, banner = await self._entry_exec_outcome(int(data[len("in_ex_"):]), uid)
        elif data.startswith("in_sk_"):
            _, banner = await self._entry_skip_outcome(int(data[len("in_sk_"):]), uid)
        elif data.startswith("in_cl_"):
            _, banner = await self._exit_close_outcome(int(data[len("in_cl_"):]), uid)
        elif data.startswith("in_sn_"):
            _, banner = await self._exit_snooze_outcome(int(data[len("in_sn_"):]), uid)
        elif data.startswith("in_ad_") or data.startswith("in_ig_") or data.startswith("in_xs_"):
            from earnings_edge.bot_views import book_action_banner
            from framework.positions import book_actions as ba

            def act():
                client = create_client()
                if data.startswith("in_ad_"):
                    symbol = data[len("in_ad_"):]
                    pos = ba.find_broker_pos(client.get_positions(), symbol)
                    if not pos:
                        return "adopt", symbol, {"ok": False, "error": "symbol not at broker"}
                    return "adopt", symbol, ba.adopt_orphan(pos, by=f"telegram:{uid}")
                if data.startswith("in_ig_"):
                    symbol = data[len("in_ig_"):]
                    return "ignore", symbol, ba.ignore_orphan(
                        symbol, by=f"telegram:{uid}")
                symbol = data[len("in_xs_"):]
                return "close", symbol, ba.close_symbol(
                    client, symbol, by=f"telegram:{uid}")

            kind, target, result = await asyncio.to_thread(act)
            banner = book_action_banner(kind, result, target)
        await self._refresh_pending_query(query, banner)

    async def _cmd_pending(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        text, rows = await asyncio.to_thread(self._pending_panel_sync)
        await self._send_panel(update, 
            text, parse_mode=HTML, reply_markup=InlineKeyboardMarkup(rows))

    async def _cmd_propose(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        pm = ProgressMessage(self.application.bot, update.effective_chat.id,
                             "Building trade proposals")
        await pm.start()
        try:
            await pm.set_stage("live signals → legs → risk gates → persist")
            await self._propose_and_push()
            from earnings_edge import trade_approval
            funnel = funnel_line(trade_approval.LAST_FUNNEL)
            tail = f"\n{funnel}" if funnel else ""
            await pm.finish(f"✅ Proposals built and pushed.{tail}")
        except Exception as exc:
            logger.exception("manual proposal build failed")
            await pm.finish(f"❌ Proposal build failed: {exc}")

    async def _send_rich_table_message(self, chat_id: int, as_of: str, picks: dict, format_picks_df) -> bool:
        import httpx
        
        html = f"<h3>OQuants Picks (As of {as_of})</h3>\n"
        for name, df in picks.items():
            html += f"<h4>{name.upper()}</h4>\n"
            if df.empty:
                if name == "momentum_skew":
                    html += "<p><i>No picks (momentum/skew inputs unpopulated in DB)</i></p>\n"
                else:
                    html += "<p><i>No picks found.</i></p>\n"
            else:
                formatted_df = format_picks_df(name, df)
                display_df = formatted_df.head(50)  # Relaxed to 50 rows due to larger rich message limit
                
                html += "<table bordered striped compact>\n"
                html += "<tr>"
                for col in display_df.columns:
                    html += f"<th>{col}</th>"
                html += "</tr>\n"
                
                for _, row in display_df.iterrows():
                    html += "<tr>"
                    for val in row:
                        html += f"<td>{val}</td>"
                    html += "</tr>\n"
                html += "</table>\n"
                
                if len(formatted_df) > 50:
                    html += f"<p><i>... and {len(formatted_df) - 50} more rows.</i></p>\n"

        payload = {
            "chat_id": chat_id,
            "rich_message": {
                "html": html
            }
        }
        
        url = f"{self.application.bot.base_url}/sendRichMessage"
        
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(url, json=payload, timeout=15.0)
                if resp.status_code != 200:
                    logger.error(f"sendRichMessage failed with {resp.status_code}: {resp.text}")
                    return False
                return True
        except Exception as e:
            logger.error(f"sendRichMessage exception: {e}")
            return False

    async def _cmd_picks(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        import pandas as pd
        from earnings_edge.picks import generate_picks
        from datetime import datetime

        def format_picks_df(name: str, df: pd.DataFrame) -> pd.DataFrame:
            if df.empty: return df
            res = df.copy()
            def fmt_dt(d): return str(d)[5:10] if not pd.isna(d) else ""
            def fmt_f(v): return f"{v:.1f}" if not pd.isna(v) else ""
            def fmt_i(v): return f"{v:.0f}" if not pd.isna(v) else ""

            if name == "earnings":
                if "implied_vs_avg_realized" in res.columns and res["implied_vs_avg_realized"].notna().any():
                    res = res.sort_values("implied_vs_avg_realized", ascending=False)
                elif "implied_move" in res.columns:
                    res = res.sort_values("implied_move", ascending=False)
                cols = [c for c in ["ticker", "announcement_date", "announcement_time", "implied_move", "implied_vs_avg_realized", "historical_events_count"] if c in res.columns]
                res = res[cols].rename(columns={"ticker": "tckr", "announcement_date": "date", "announcement_time": "tm", "implied_move": "impl%", "implied_vs_avg_realized": "im_v_rl", "historical_events_count": "hist"})
                if "date" in res.columns: res["date"] = res["date"].apply(fmt_dt)
                if "impl%" in res.columns: res["impl%"] = res["impl%"].apply(fmt_f)
                if "im_v_rl" in res.columns: res["im_v_rl"] = res["im_v_rl"].apply(fmt_f)
                if "hist" in res.columns: res["hist"] = res["hist"].apply(fmt_i)
            elif name == "momentum_skew":
                cols = [c for c in ["ticker", "direction", "skew_zscore", "cs_momentum", "next_earnings_date"] if c in res.columns]
                res = res[cols].rename(columns={"ticker": "tckr", "direction": "dir", "skew_zscore": "zscr", "cs_momentum": "mom", "next_earnings_date": "date"})
                if "zscr" in res.columns: res["zscr"] = res["zscr"].apply(fmt_f)
                if "mom" in res.columns: res["mom"] = res["mom"].apply(fmt_i)
                if "date" in res.columns: res["date"] = res["date"].apply(fmt_dt)
            elif name == "forward_factor":
                if "forward_factor" in res.columns: res = res.sort_values("forward_factor", ascending=False)
                cols = [c for c in ["ticker", "next_earnings_date", "forward_factor"] if c in res.columns]
                res = res[cols].rename(columns={"ticker": "tckr", "next_earnings_date": "date", "forward_factor": "fwd_fct"})
                if "fwd_fct" in res.columns: res["fwd_fct"] = res["fwd_fct"].apply(fmt_f)
                if "date" in res.columns: res["date"] = res["date"].apply(fmt_dt)
            elif name == "vrp":
                iv_col = "iv_pctl_1y" if "iv_pctl_1y" in res.columns and res["iv_pctl_1y"].notna().any() else "iv_rv"
                ret_cols = [c for c in res.columns if "return" in c or "win_rate" in c]
                ret_cols = [c for c in ret_cols if res[c].notna().any()][:2]
                cols = [c for c in ["ticker", iv_col, "next_earnings_date"] + ret_cols if c in res.columns]
                rename_map = {"ticker": "tckr", "iv_pctl_1y": "iv_pct", "iv_rv": "iv_rv", "next_earnings_date": "date"}
                for c in ret_cols: rename_map[c] = "".join([w[0] for w in c.split("_")[:2]]) + "_" + c.split("_")[-1]
                res = res[cols].rename(columns=rename_map)
                for c in res.columns:
                    if c not in ["tckr", "date"]: res[c] = res[c].apply(fmt_f)
                if "date" in res.columns: res["date"] = res["date"].apply(fmt_dt)
            return res

        latest_str = snapshots_max_scan_date()
        if not latest_str:
            await update.message.reply_text("No data in snapshots table.")
            return
        as_of = datetime.strptime(str(latest_str)[:10], "%Y-%m-%d").date()
        picks = await asyncio.to_thread(generate_picks, as_of)

        chat_id = update.message.chat_id
        if await self._send_rich_table_message(chat_id, as_of, picks, format_picks_df):
            return

        msg = f"🎯 <b>OQuants Picks (As of {as_of})</b>\n\n"
        for name, df in picks.items():
            msg += f"<b>{name.upper()}</b>\n"
            if df.empty:
                if name == "momentum_skew":
                    msg += "<i>No picks (momentum/skew inputs unpopulated in DB)</i>\n\n"
                else:
                    msg += "<i>No picks found.</i>\n\n"
            else:
                formatted_df = format_picks_df(name, df)
                display_df = formatted_df.head(10)
                msg += f"<pre>{display_df.to_string(index=False)}</pre>\n"
                if len(formatted_df) > 10:
                    msg += f"<i>... and {len(formatted_df) - 10} more rows.</i>\n"
                msg += "\n"

        if len(msg) > 3800:
            msg = msg[:3750] + "\n... (truncated)"
        await self._send_panel(update, msg, parse_mode=HTML)

    async def _cmd_designer(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not ctx.args or len(ctx.args) < 2:
            await update.message.reply_text("Usage: /designer <ticker> <legs...>\nExample: /designer AAPL buy call 190 2026-10-16 1 5.0 0.3")
            return
        
        ticker = ctx.args[0].upper()
        raw_legs = [a.strip('"\'') for a in ctx.args[1:]]
        
        # In case legs are passed as a single string wrapped in quotes due to CLI testing habits
        if len(raw_legs) == 1:
            raw_legs = raw_legs[0].split()

        from earnings_edge.designer import Leg, analyze, rv_scenario
        from datetime import datetime

        try:
            from earnings_edge.alpaca_trading import create_client
            client = create_client()
            px = client.get_stock_latest_trade(ticker)
            if px is None:
                await self._send_panel(update, f"Could not fetch spot price for {ticker}")
                return
            spot = float(px)
        except Exception as e:
            await update.message.reply_text(f"Price fetch failed: {e}")
            return

        legs = []
        i = 0
        try:
            from earnings_edge.alpaca_trading import create_client
            client = create_client()

            while i < len(raw_legs):
                if raw_legs[i].lower() not in ('buy', 'sell'):
                    raise ValueError(f"Expected buy/sell but got {raw_legs[i]}")

                # Find the start of the next leg
                j = i + 1
                while j < len(raw_legs) and raw_legs[j].lower() not in ('buy', 'sell'):
                    j += 1

                leg_len = j - i
                if leg_len not in (4, 5, 7):
                    raise ValueError(f"Invalid leg length {leg_len}. Must be 4 (auto), 5 (w/ qty), or 7 (w/ price & iv) parts.")

                action = raw_legs[i].lower()
                kind = raw_legs[i+1].lower()
                strike = float(raw_legs[i+2])
                expiry = datetime.strptime(raw_legs[i+3], "%Y-%m-%d").date()

                if leg_len == 7:
                    qty = int(raw_legs[i+4])
                    price = float(raw_legs[i+5])
                    iv = float(raw_legs[i+6])
                else:
                    qty = int(raw_legs[i+4]) if leg_len == 5 else 1
                    c = client.find_option_contract(ticker, kind, strike, expiry)
                    if not c:
                        raise ValueError(f"Could not find contract {ticker} {strike} {kind} {expiry}")

                    snap_res = client.get_option_snapshots_bulk(c['symbol'])
                    snap = snap_res.get(c['symbol'], {})
                    iv = snap.get('impliedVolatility')
                    quote = snap.get('latestQuote', {})
                    ap, bp = quote.get('ap', 0), quote.get('bp', 0)
                    if ap and bp:
                        price = (ap + bp) / 2
                    else:
                        price = snap.get('latestTrade', {}).get('p') or 0.0
                    if not iv:
                        # Alpaca's data tier carries no greeks: solve IV from
                        # the mid instead of assuming a flat 30%.
                        from datetime import date as _date
                        from earnings_edge.option_math import implied_volatility
                        T = max((expiry - _date.today()).days, 1) / 365.0
                        iv = (implied_volatility(price, spot, strike, T, 0.045, kind)
                              if price > 0 else None)
                        if not iv or iv != iv:  # None or NaN
                            raise ValueError(
                                f"Could not solve IV for {ticker} {strike} {kind} {expiry} "
                                f"(no greeks from broker, unusable quote)")

                legs.append(Leg(action, kind, strike, expiry, qty, price, iv))
                i = j
        except Exception as e:
            await self._send_panel(update, f"Error parsing legs: {e}")
            return

        summary = analyze(legs, spot, 0.045)
        greeks = summary.get("greeks", {})
        structure = summary.get("structure", {})
        
        msg = f"<b>Designer: {ticker}</b>\nSpot: ${spot:.2f}\n\n"
        msg += "<b>Structure</b>\n"
        msg += f"Direction: {structure.get('direction', '?')} | Risk: {structure.get('risk', '?')} | Vol Exposure: {structure.get('vol_exposure', '?')}\n"
        msg += f"Net Premium: ${summary.get('net_premium', 0):.2f}\n"
        msg += f"Max Profit: {summary.get('max_profit', '?')}\n"
        msg += f"Max Loss: {summary.get('max_loss', '?')}\n"
        msg += f"Breakevens: {', '.join(map(str, summary.get('breakevens', [])))}\n"
        win_rate = summary.get('win_rate')
        if win_rate is not None:
            msg += f"Win Rate: {win_rate*100:.2f}%\n"
        
        if greeks:
            msg += "\n<b>Greeks</b>\n"
            msg += f"Δ: {greeks.get('delta', 0):.2f} | Γ: {greeks.get('gamma', 0):.2f} | Θ: {greeks.get('theta', 0):.2f} | ν: {greeks.get('vega', 0):.2f}\n"

        await self._send_panel(update, msg, parse_mode=HTML)

    # ── Scheduled scanner run ──────────────────────────────────────────

    async def _run_and_push(self, scanner_name: str):
        if scanner_name not in self.scanners:
            return
        sc = self.scanners[scanner_name]
        logger.info("Scheduled run: %s", scanner_name)
        try:
            # Scan in a worker thread: scans take 30-60 min and would
            # otherwise block the event loop, starving every dispatched
            # framework job (equity snapshots, reconcile, FF steps) until
            # they hit the 900s dispatch timeout.
            result = await asyncio.to_thread(sc.scan)
            from framework.scan_retry import should_chain_proposals, should_retry_scan
            if should_retry_scan(result):
                logger.error("Scanner %s failed or empty: %s", scanner_name, result.get("error"))
                from framework.alerts import DEDUPER
                DEDUPER.emit("scan_fail", f"scan {scanner_name} failed: {result.get('error')}")
                await self._flush_alerts()
                self._schedule_one_scan_retry(scanner_name)
                return
            if not should_chain_proposals(result):
                return
            if scanner_name == "Earnings Calendar" and datetime.now(pytz.utc).weekday() < 5:
                logger.info("Chaining trade proposals after Earnings Calendar scan")
                await self._propose_and_push()
        except Exception as exc:
            logger.exception("Scheduled run error for %s", scanner_name)

    def _schedule_one_scan_retry(self, scanner_name: str) -> dict:
        """Exactly one 12-minute follow-up; does not stack."""
        from framework.scan_retry import record_retry
        rec = record_retry()
        try:
            if self.scheduler.get_job("scan_retry"):
                return rec
            when = datetime.fromisoformat(rec["next_run"])
            self.scheduler.add_job(
                self._run_sync, "date", run_date=when,
                args=[scanner_name], id="scan_retry",
            )
        except Exception as exc:
            logger.warning("could not schedule scan retry: %s", exc)
        return rec

    def _run_sync(self, scanner_name: str):
        self._dispatch(self._run_and_push(scanner_name))

    def _setup_scheduler(self):
        tz = pytz.timezone("Europe/Berlin")
        for name, sc in self.scanners.items():
            try:
                trigger = CronTrigger.from_crontab(sc.schedule, timezone=tz)
                self.scheduler.add_job(
                    self._run_sync, trigger=trigger, args=[name],
                    id=f"scanner_{name}", name=f"Run {name}",
                )
                logger.info("Scheduled %s: %s (Berlin TZ)", name, sc.schedule)
            except Exception as exc:
                logger.error("Failed to schedule %s: %s", name, exc)
        # NOTE: no separate proposal cron — trade proposals are chained to the
        # Earnings Calendar scan (14:00 ET weekdays) in _run_and_push so they
        # are built from fresh scan data and confirmed during US market hours.

        # Forward-factor ladder: 13:45 ET proposals (19:45 Berlin), stepping
        # every 15 min 14:00-15:45 ET (20:00-21:45 Berlin). Berlin TZ tracks
        # ET correctly across DST for these slots.
        try:
            self.scheduler.add_job(
                self._ff_propose_sync,
                trigger=CronTrigger.from_crontab("45 19 * * mon-fri", timezone=tz),
                id="ff_ladder_propose", name="FF ladder proposals",
            )
            self.scheduler.add_job(
                self._ff_step_sync,
                trigger=CronTrigger.from_crontab("0,15,30,45 20-21 * * mon-fri", timezone=tz),
                id="ff_ladder_step", name="FF ladder step",
            )
            logger.info("Scheduled FF ladder: proposals 13:45 ET, steps 14:00-15:45 ET")
        except Exception as exc:
            logger.error("Failed to schedule FF ladder jobs: %s", exc)

        # Framework jobs (Berlin TZ tracks ET across DST for these slots):
        # equity snapshots every 15 min 09:30-16:30 ET, reconcile every 30 min,
        # assignment-guard evaluation 15:45 ET daily.
        try:
            self.scheduler.add_job(
                self._equity_snapshot_sync,
                trigger=CronTrigger.from_crontab("*/15 15-22 * * mon-fri", timezone=tz),
                id="equity_snapshot", name="Equity snapshot + loss check",
            )
            self.scheduler.add_job(
                self._reconcile_sync,
                trigger=CronTrigger.from_crontab("*/30 15-22 * * mon-fri", timezone=tz),
                id="reconcile", name="Broker reconciliation",
            )
            self.scheduler.add_job(
                self._guard_eval_sync,
                trigger=CronTrigger.from_crontab("45 21 * * mon-fri", timezone=tz),
                id="assignment_guard", name="Assignment guard eval",
            )
            self.scheduler.add_job(
                self._exit_eval_sync,
                trigger=CronTrigger.from_crontab("*/15 15-22 * * mon-fri", timezone=tz),
                id="exit_eval", name="Exit rule evaluation",
            )
            self.scheduler.add_job(
                self._backup_sync,
                trigger=CronTrigger.from_crontab("15 6 * * *", timezone=tz),
                id="db_backup", name="SQLite backup",
            )
            # Hourly integrity check so corruption is caught within ~1h instead
            # of surfacing a day later via a failed backup or missed scan.
            self.scheduler.add_job(
                self._db_health_sync,
                trigger=CronTrigger.from_crontab("5 * * * *", timezone=tz),
                id="db_health_check", name="SQLite integrity check",
            )
            # Pre-market picks pipeline: 07:00 ET (13:00 Berlin) weekdays —
            # refresh chains + signals, generate and persist today's picks.
            self.scheduler.add_job(
                self._picks_sync,
                trigger=CronTrigger.from_crontab("0 13 * * mon-fri", timezone=tz),
                id="daily_picks", name="Daily picks pipeline",
            )
            # Hourly Alpaca options-chain cache 09:05–16:05 ET (Berlin 15–22).
            self.scheduler.add_job(
                self._chain_cache_sync,
                trigger=CronTrigger.from_crontab("5 15-22 * * mon-fri", timezone=tz),
                id="chain_cache", name="Hourly Alpaca chain cache",
            )
            logger.info("Scheduled framework jobs: equity */15, reconcile */30, guard 15:45 ET, exits */15, backup 06:15, chain cache hourly")
        except Exception as exc:
            logger.error("Failed to schedule framework jobs: %s", exc)

    def _backup_sync(self):
        from earnings_edge.jobs import db_backup_job
        db_backup_job()

    def _db_health_sync(self):
        self._dispatch(self._db_health_check())

    async def _db_health_check(self):
        from earnings_edge.jobs import db_health_check_job
        await db_health_check_job(self)

    def _picks_sync(self):
        self._dispatch(self._picks_pipeline())

    def _chain_cache_sync(self):
        self._dispatch(self._chain_cache())

    async def _chain_cache(self):
        from earnings_edge.jobs import chain_cache_job
        await chain_cache_job(self)

    async def _picks_pipeline(self):
        from earnings_edge.jobs import picks_pipeline_job
        await picks_pipeline_job(self)

    # ── Main entry ─────────────────────────────────────────────────────

    def run(self):
        install_secret_redaction()
        try:
            self._instance_lock = InstanceLock().acquire()
        except RuntimeError as exc:
            logger.error("%s", exc)
            sys.exit(1)
        if not operators_configured():
            logger.warning("TELEGRAM_APPROVAL_CHAT_ID unset — execute/halt/close/promote/restart disabled")
        _HealthHandler.facts_fn = self._health_facts
        self.application = (
            Application.builder()
            .token(self.token)
            .post_init(self._capture_loop)
            .build()
        )

        self.application.add_handler(CommandHandler("start", self._cmd_start))
        self.application.add_handler(CommandHandler("help", self._cmd_help))
        self.application.add_handler(CommandHandler("scanners", self._cmd_scanners))
        self.application.add_handler(CommandHandler("subscriptions", self._cmd_subscriptions))
        self.application.add_handler(CommandHandler("run", self._cmd_run))
        self.application.add_handler(CommandHandler("pending", self._cmd_pending))
        self.application.add_handler(CommandHandler("propose", self._cmd_propose))
        self.application.add_handler(CommandHandler("signals", self._cmd_signals))
        self.application.add_handler(CommandHandler("setups", self._cmd_setups))
        self.application.add_handler(CommandHandler("exits", self._cmd_exits))
        self.application.add_handler(CommandHandler("status", self._cmd_status))
        self.application.add_handler(CommandHandler("monitor", self._cmd_monitor))
        self.application.add_handler(CommandHandler("positions", self._cmd_positions))
        self.application.add_handler(CommandHandler("orders", self._cmd_orders))
        self.application.add_handler(CommandHandler("jobs", self._cmd_jobs))
        self.application.add_handler(CommandHandler("equity", self._cmd_equity))
        self.application.add_handler(CommandHandler("strategies", self._cmd_strategies))
        self.application.add_handler(CommandHandler("halt", self._cmd_halt))
        self.application.add_handler(CommandHandler("resume", self._cmd_resume))
        self.application.add_handler(CommandHandler("risk", self._cmd_risk))
        self.application.add_handler(CommandHandler("promote", self._cmd_promote))
        self.application.add_handler(CommandHandler("demote", self._cmd_demote))
        self.application.add_handler(CommandHandler("settings", self._cmd_settings))
        self.application.add_handler(CommandHandler("picks", self._cmd_picks))
        self.application.add_handler(CommandHandler("designer", self._cmd_designer))
        self.application.add_handler(CallbackQueryHandler(self._handle_callback))
        self.application.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_keyboard)
        )

        self._setup_scheduler()
        self._scheduler_thread = threading.Thread(target=self.scheduler.start, daemon=True)
        self._scheduler_thread.start()

        _start_health_server()
        _start_dashboard_server()

        logger.info("Trading bot starting (polling mode)...")
        try:
            self.application.run_polling()
        except KeyboardInterrupt:
            pass
        finally:
            if self.scheduler.running:
                self.scheduler.shutdown(wait=True)
            logger.info("Bot shut down.")


def main():
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        logger.error("TELEGRAM_BOT_TOKEN not set in .env")
        sys.exit(1)
    bot = TradingBot(token)
    bot.run()


if __name__ == "__main__":
    main()
