"""Text views for the bot's operational commands (/status, /positions, ...).

Pure functions over a framework DB connection (+ optional live account dict):
no Telegram imports, fully unit-testable. All output is PLAIN TEXT — strategy
names and OCC symbols contain underscores, which silently break Telegram
Markdown parsing (see bot.py _cmd_risk note).
"""

from __future__ import annotations

import html
import json
from datetime import date, datetime, timezone
from typing import Optional

from earnings_edge.db import (
    adopted_positions_symbols,
    equity_snapshots_daily_avg,
    equity_snapshots_equities,
    exit_proposals_list_pending,
    ff_ladders_count_armed,
    job_runs_failed,
    job_runs_latest,
    job_runs_list,
    scan_runs_latest_success,
    strategy_state_list,
    table_exists,
    trade_events_list,
)


def _ts_short(ts: Optional[str]) -> str:
    return (ts or "")[:16].replace("T", " ")


def _age(ts: Optional[str]) -> str:
    if not ts:
        return "?"
    try:
        t = datetime.fromisoformat(ts)
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        mins = int((datetime.now(timezone.utc) - t).total_seconds() // 60)
        if mins < 60:
            return f"{mins}m ago"
        return f"{mins // 60}h{mins % 60:02d} ago"
    except ValueError:
        return "?"


# ── /status ─────────────────────────────────────────────────────────────

def _equity_curve(points: int = 16) -> str:
    """Sparkline of the last `points` equity snapshots ('' when unavailable)."""
    from earnings_edge.bot_live import sparkline
    try:
        vals = equity_snapshots_equities(points)
    except Exception:
        return ""
    return sparkline(vals)


def status_view(*, market_open: Optional[bool] = None,
                pending_proposals: int = 0, pending_exits: int = 0,
                next_events: Optional[list] = None,
                funnel: Optional[str] = None,
                last_scan_ts: Optional[str] = None,
                last_equity_ts: Optional[str] = None,
                reconcile_summary: Optional[str] = None,
                broker_ok: Optional[bool] = None,
                broker_count: Optional[int] = None,
                orphan_count: Optional[int] = None,
                sha: Optional[str] = None,
                started_at: Optional[str] = None) -> str:
    from framework.execution.managed import open_groups
    from framework.risk.equity import daily_pnl, latest_equity
    from framework.risk.killswitch import KillSwitch

    lines = ["🖥 <b>SYSTEM STATUS</b>", ""]
    if market_open is not None:
        lines.append("<b>Market:</b> " + ("🟢 open" if market_open else "⚫ closed"))

    ks = KillSwitch().status()
    if ks.get("halted"):
        lines.append(f"<b>Kill switch:</b> 🛑 HALTED — {html.escape(str(ks.get('reason')))} (by {html.escape(str(ks.get('tripped_by')))})")
    else:
        lines.append("<b>Kill switch:</b> 🟢 armed")

    eq = latest_equity()
    if eq:
        pnl = daily_pnl( eq["equity"])
        pnl_txt = f" | day PnL ${pnl:+,.0f}" if pnl is not None else ""
        lines.append(f"<b>Equity:</b> ${eq['equity']:,.0f} | <b>BP</b> ${eq['buying_power']:,.0f}"
                     f"{pnl_txt} ({_age(eq['ts'])})")
        curve = _equity_curve()
        if curve:
            lines.append(f"  {curve}")
    else:
        lines.append("<b>Equity:</b> no snapshots yet")

    try:
        states = strategy_state_list()
    except Exception:
        states = []
    if states:
        parts = []
        for s in states:
            tag = s["lifecycle"]
            if s["enabled"] == 0:
                tag += "·OFF"
            parts.append(f"{s['name']}={tag}")
        lines.append("<b>Strategies:</b> " + ", ".join(parts))

    groups = open_groups()
    lines.append(f"<b>Open position groups:</b> {len(groups)}"
                 + (f" vs broker {broker_count}" if broker_count is not None else ""))
    if orphan_count is not None:
        lines.append(f"<b>Orphans: {orphan_count}</b>")
    lines.append(f"<b>Pending proposals: {pending_proposals}</b> | <b>Pending exit cards: {pending_exits}</b>")
    if last_scan_ts:
        lines.append(f"<b>Last scan:</b> {_age(last_scan_ts)} ({_ts_short(last_scan_ts)})")
    elif last_scan_ts == "":
        lines.append("<b>Last scan:</b> never")
    eq_ts = last_equity_ts or (eq["ts"] if eq else None)
    if last_equity_ts:
        lines.append(f"<b>Last equity snapshot:</b> {_age(last_equity_ts)}")
    if reconcile_summary:
        lines.append(f"<b>Last reconcile:</b> {html.escape(str(reconcile_summary))}")
    if broker_ok is not None:
        lines.append("<b>Broker:</b> " + ("reachable" if broker_ok else "unreachable"))
    if sha:
        lines.append(f"<b>Rev:</b> <code>{sha}</code>")
    if started_at:
        lines.append(f"<b>Started:</b> {_ts_short(started_at)}")

    if next_events:
        lines.append("")
        lines.append("<b>Next:</b> " + " · ".join(next_events))
    if funnel:
        lines.append(funnel)

    try:
        fails = job_runs_failed(3)
    except Exception:
        fails = []
    if fails:
        lines.append("")
        lines.append("<b>Recent job failures:</b>")
        for f in fails:
            lines.append(f"  ✗ {f['job_name']} ({_ts_short(f['finished_at'])}): "
                         f"{(f['error'] or '')[:80]}")
    return "\n".join(lines)


# Shared kwargs /status and /monitor pass into their views.
DESK_VIEW_KEYS = (
    "market_open", "last_scan_ts", "last_equity_ts", "reconcile_summary",
    "broker_ok", "broker_count", "orphan_count", "sha", "started_at",
)


def desk_view_kwargs(facts: dict) -> dict:
    return {k: facts[k] for k in DESK_VIEW_KEYS if k in facts}


def collect_desk_facts(*,
                       get_clock=None, get_positions=None) -> dict:
    """Broker + DB facts for /status and the live /monitor.

    ``clock_exc`` / ``positions_exc`` stay on the dict so the bot can emit
    alpaca_401 / clock_dns; strip them with ``desk_view_kwargs`` before
    calling a view.
    """
    from framework.execution.managed import open_groups
    from framework.positions.book import classify_book
    from framework.revision import code_sha, started_at_iso
    from framework.risk.equity import latest_equity

    market_open = None
    clock_exc = None
    if get_clock is not None:
        try:
            market_open = bool(get_clock().get("is_open"))
        except Exception as exc:
            clock_exc = exc

    last_scan_ts = ""
    try:
        last_scan_ts = scan_runs_latest_success() or ""
    except Exception:
        last_scan_ts = ""

    recon = None
    try:
        last_recon = job_runs_latest("reconcile", success=1)
        if last_recon and last_recon.get("stats_json"):
            try:
                recon = json.loads(last_recon["stats_json"]).get("summary")
            except Exception:
                recon = last_recon["stats_json"][:80]
    except Exception:
        recon = None

    eq = latest_equity()
    last_equity_ts = eq["ts"] if eq else None

    broker_ok, broker_n, orphans = None, None, None
    positions_exc = None
    if get_positions is not None:
        try:
            broker = get_positions() or []
            broker_ok, broker_n = True, len(broker)
            orphans = len(classify_book(
                open_groups(), broker, ignored=_ignored_symbols()).orphan)
        except Exception as exc:
            broker_ok = False
            positions_exc = exc

    return {
        "market_open": market_open,
        "last_scan_ts": last_scan_ts,
        "last_equity_ts": last_equity_ts,
        "reconcile_summary": recon,
        "broker_ok": broker_ok,
        "broker_count": broker_n,
        "orphan_count": orphans,
        "sha": code_sha(),
        "started_at": started_at_iso(),
        "clock_exc": clock_exc,
        "positions_exc": positions_exc,
    }


# ── /monitor (live-updating ops panel) ───────────────────────────────────

def monitor_view(*, tick: int,
                 pending_proposals: int = 0, pending_exits: int = 0,
                 next_events: Optional[list] = None,
                 funnel: Optional[str] = None,
                 last_scan_ts: Optional[str] = None,
                 last_equity_ts: Optional[str] = None,
                 reconcile_summary: Optional[str] = None,
                 broker_ok: Optional[bool] = None,
                 broker_count: Optional[int] = None,
                 orphan_count: Optional[int] = None,
                 sha: Optional[str] = None,
                 started_at: Optional[str] = None,
                 market_open: Optional[bool] = None) -> str:
    """Compact ops panel rendered every 30s by the bot's monitor loop."""
    from earnings_edge.bot_live import spinner_frame
    from framework.execution.managed import open_groups
    from framework.risk.equity import daily_pnl, latest_equity
    from framework.risk.killswitch import KillSwitch

    lines = [f"{spinner_frame(tick)} <b>LIVE MONITOR</b> — refreshes every 30s", ""]
    if market_open is not None:
        lines.append("<b>Market:</b> " + ("🟢 open" if market_open else "⚫ closed"))
    ks = KillSwitch().status()
    lines.append("<b>Kill switch:</b> " + (f"🛑 HALTED — {html.escape(str(ks.get('reason')))}"
                                    if ks.get("halted") else "🟢 armed"))
    eq = latest_equity()
    if eq:
        pnl = daily_pnl( eq["equity"])
        pnl_txt = f" | day ${pnl:+,.0f}" if pnl is not None else ""
        lines.append(f"<b>Equity</b> ${eq['equity']:,.0f}{pnl_txt} ({_age(eq['ts'])})")
        curve = _equity_curve()
        if curve:
            lines.append(f"  {curve}")
    try:
        ladders = ff_ladders_count_armed()
    except Exception:
        ladders = 0
    n_groups = len(open_groups())
    extra = f" vs broker {broker_count}" if broker_count is not None else ""
    orph = f" | orphans {orphan_count}" if orphan_count is not None else ""
    lines.append(f"<b>Positions:</b> {n_groups} groups{extra}{orph} | <b>FF ladders armed:</b> {ladders}")
    lines.append(f"<b>Pending:</b> {pending_proposals} proposals | {pending_exits} exit cards")
    if last_scan_ts:
        lines.append(f"<b>Last scan:</b> {_age(last_scan_ts)}")
    elif last_scan_ts == "":
        lines.append("<b>Last scan:</b> never")
    eq_ts = last_equity_ts or (eq["ts"] if eq else None)
    if last_equity_ts:
        lines.append(f"<b>Last equity snapshot:</b> {_age(last_equity_ts)}")
    elif eq_ts:
        lines.append(f"<b>Last equity snapshot:</b> {_age(eq_ts)}")
    if reconcile_summary:
        lines.append(f"<b>Last reconcile:</b> {html.escape(str(reconcile_summary))}")
    if broker_ok is not None:
        lines.append("<b>Broker:</b> " + ("reachable" if broker_ok else "unreachable"))
    if sha:
        lines.append(f"<b>Rev:</b> <code>{sha}</code>")
    if started_at:
        lines.append(f"<b>Started:</b> {_ts_short(started_at)}")
    if next_events:
        lines.append("<b>Next:</b> " + " · ".join(next_events))
    if funnel:
        lines.append(funnel)
    return "\n".join(lines)


# ── /positions ──────────────────────────────────────────────────────────

def _ignored_symbols() -> set[str]:
    try:
        return adopted_positions_symbols()
    except Exception:
        return set()


def positions_view(broker_positions: Optional[list] = None,
                   broker_error: Optional[str] = None) -> str:
    """Render the book. With ``broker_positions``, show managed / orphan /
    missing. Without, fall back to local groups only and say so."""
    from framework.execution.managed import open_groups
    from framework.positions.book import classify_book

    groups = open_groups()
    if broker_positions is None:
        lines = []
        if broker_error:
            lines.append(f"⚠️ Broker unavailable: {broker_error}")
            lines.append("")
        if not groups:
            lines.append("💼 <b>No open managed positions.</b>")
            return "\n".join(lines)
        lines.append(f"💼 <b>OPEN POSITIONS</b> ({len(groups)} groups) — local book only")
        lines.append("")
        for g in groups:
            kind = "credit" if g.credit else "debit"
            header = f"[{g.strategy}] {g.ticker} x{g.qty} ({kind})"
            if g.event_date:
                header += f" — event {g.event_date.isoformat()}"
            lines.append(header)
            lines.append(f"  entry ${g.entry_price:.2f} | opened {_ts_short(g.opened_at)}")
            for leg in g.legs:
                side = "SELL" if leg.side == "sell" else "BUY"
                exp = leg.expiry.isoformat() if leg.expiry else "?"
                lines.append(f"  {side} {leg.qty:g} {leg.symbol} ({leg.option_type} "
                             f"{leg.strike:g} {exp})")
            lines.append("")
        return "\n".join(lines).rstrip()

    book = classify_book(groups, broker_positions, ignored=_ignored_symbols())
    if not book.managed and not book.orphan and not book.missing:
        return "💼 No positions at broker or locally."
    lines = [
        f"💼 BOOK  broker={book.broker_count}  local={book.local_count}  "
        f"orphans={len(book.orphan)}  missing={len(book.missing)}",
        "",
    ]
    if book.managed:
        lines.append("— MANAGED (matched) —")
        lines.extend(_render_items(book.managed))
        lines.append("")
    if book.orphan:
        lines.append("— ORPHAN (at broker, not local) —")
        lines.extend(_render_items(book.orphan))
        lines.append("")
    if book.missing:
        lines.append("— MISSING (local open, not at broker) —")
        lines.extend(_render_items(book.missing))
        lines.append("")
    return "\n".join(lines).rstrip()


def _render_items(items) -> list[str]:
    lines = []
    for it in items:
        tag = f"[{it.strategy}] " if it.strategy else ""
        upl = f" uPL ${it.upl:+,.0f}" if it.upl is not None else ""
        px = f" @ {it.current_price:g}" if it.current_price is not None else ""
        ev = f" event {it.event_date.isoformat()}" if it.event_date else ""
        exp = f" exp {it.expiry.isoformat()}" if it.expiry else ""
        lines.append(
            f"{tag}{it.ticker} {it.side} {it.qty:g} {it.symbol}{px}{upl}{ev}{exp}"
        )
    return lines


def book_action_banner(kind: str, result: dict, target: str = "") -> str:
    """One-line outcome for the live book panel (plain text)."""
    tgt = target or result.get("symbol") or result.get("group_id") or ""
    if not result.get("ok"):
        return f"⚠️ {result.get('error') or 'action failed'}"
    if kind == "adopt":
        return f"✅ Adopted {tgt} — now on the managed book."
    if kind == "ignore":
        return f"✅ Ignoring {tgt} — reconcile will stop alerting."
    if kind == "mark_closed":
        return f"✅ Marked {tgt} closed locally."
    if kind == "close":
        return f"✅ Close submitted for {tgt}."
    if kind == "close_group":
        n = len(result.get("closed") or [])
        return f"✅ Closed group {tgt} ({n} legs at broker)."
    return "✅ Done."


def build_positions_panel(broker_positions: Optional[list] = None,
                          broker_error: Optional[str] = None,
                          banner: Optional[str] = None) -> tuple[str, list]:
    """Book text + inline rows for one Telegram message that can be edited."""
    text = positions_view(broker_positions=broker_positions,
                          broker_error=broker_error)
    if banner:
        text = f"{banner}\n\n{text}"
    rows = positions_keyboard_for(broker_positions)
    return text, rows


def positions_keyboard_for(broker_positions: Optional[list]) -> list[list]:
    from telegram import InlineKeyboardButton
    if broker_positions is None:
        return [[InlineKeyboardButton("🔄 Refresh", callback_data="bk_rf")]]
    from framework.execution.managed import open_groups
    from framework.positions.book import classify_book
    book = classify_book(open_groups(), broker_positions,
                         ignored=_ignored_symbols())
    return positions_keyboard(book)


def positions_keyboard(book) -> list[list]:
    """Inline button rows for a classified Book (callback_data ≤ 64 bytes)."""
    from telegram import InlineKeyboardButton
    rows = []
    seen_groups: set[str] = set()
    for it in book.managed:
        if it.group_id and it.group_id not in seen_groups:
            seen_groups.add(it.group_id)
            rows.append([
                InlineKeyboardButton(
                    f"🔒 Close {it.ticker}",
                    callback_data=f"bk_xg_{it.group_id}"),
            ])
        rows.append([
            InlineKeyboardButton(
                f"Close {it.symbol[-12:]}",
                callback_data=f"bk_xs_{it.symbol}"),
        ])
    for it in book.orphan:
        rows.append([
            InlineKeyboardButton(f"Adopt {it.symbol[-12:]}", callback_data=f"bk_ad_{it.symbol}"),
            InlineKeyboardButton(f"Close {it.symbol[-12:]}", callback_data=f"bk_xs_{it.symbol}"),
            InlineKeyboardButton("Ignore", callback_data=f"bk_ig_{it.symbol}"),
        ])
    seen_missing: set[str] = set()
    for it in book.missing:
        if it.group_id and it.group_id not in seen_missing:
            seen_missing.add(it.group_id)
            rows.append([
                InlineKeyboardButton(
                    f"Mark closed {it.ticker}",
                    callback_data=f"bk_ml_{it.group_id}"),
            ])
    # Telegram cap 100 buttons; keep Refresh even on a large book.
    if len(rows) > 98:
        rows = rows[:98]
    rows.append([InlineKeyboardButton("🔄 Refresh", callback_data="bk_rf")])
    return rows


# ── /orders ─────────────────────────────────────────────────────────────

def orders_view(limit: int = 12) -> str:
    rows = trade_events_list(limit=limit)
    if not rows:
        return "🧾 <b>No trade events yet.</b>"
    lines = [f"🧾 <b>RECENT TRADE EVENTS</b> ({len(rows)})", ""]
    for r in rows:
        bits = [f"<code>{_ts_short(r['ts'])}</code>  <b>{html.escape(str(r['event_type']))}</b>"]
        if r["strategy"]:
            bits.append(f"[<code>{html.escape(str(r['strategy']))}</code>]")
        if r["symbol"]:
            bits.append(f"<code>{html.escape(str(r['symbol']))}</code>")
        if r["price"] is not None:
            bits.append(f"@ {r['price']}")
        lines.append(" ".join(bits))
        if r["detail"]:
            lines.append(f"  <i>{html.escape(str(r['detail'])[:110])}</i>")
    return "\n".join(lines)


# ── /jobs ───────────────────────────────────────────────────────────────

def jobs_view(limit: int = 12) -> str:
    rows = job_runs_list(limit=limit)
    if not rows:
        return "⚙️ <b>No job runs recorded yet.</b>"
    lines = [f"⚙️ <b>JOB RUNS</b> ({len(rows)} most recent)", ""]
    for r in rows:
        mark = "✓" if r["success"] else "✗"
        line = f"{mark} <b>{html.escape(str(r['job_name']))}</b> — started <code>{_ts_short(r['started_at'])}</code>"
        lines.append(line)
        if r["success"]:
            stats = r["stats_json"] or "{}"
            try:
                parsed = json.loads(stats)
                summary = ", ".join(f"{k}={v}" for k, v in list(parsed.items())[:4])
            except (ValueError, TypeError):
                summary = ""
            if summary:
                lines.append(f"  {summary[:110]}")
        elif r["error"]:
            lines.append(f"  error: {r['error'][:110]}")
    return "\n".join(lines)


# ── /equity ─────────────────────────────────────────────────────────────

def equity_view(days: int = 7) -> str:
    from framework.risk.equity import daily_pnl, day_start_equity, latest_equity

    eq = latest_equity()
    if not eq:
        return "💰 <b>No equity snapshots yet</b> (snapshots run during market hours)."
    lines = ["💰 <b>EQUITY</b>", ""]
    lines.append(f"<b>Latest:</b> ${eq['equity']:,.0f} | <b>BP</b> ${eq['buying_power']:,.0f} "
                 f"({ _age(eq['ts'])})")
    start = day_start_equity()
    pnl = daily_pnl( eq["equity"])
    if start:
        pct = (pnl / start * 100) if pnl is not None else 0.0
        lines.append(f"<b>Day start:</b> ${start:,.0f} | <b>Day PnL:</b> ${pnl:+,.0f} ({pct:+.2f}%)")

    rows = equity_snapshots_daily_avg(days)
    if len(rows) > 1:
        lines.append("")
        lines.append("Daily closes (avg):")
        for r in reversed(rows):
            lines.append(f"  {r['d']}  ${r['e']:,.0f}")
    return "\n".join(lines)


# ── /strategies ─────────────────────────────────────────────────────────

def strategies_view(registry=None) -> tuple[str, list[dict]]:
    """Per-strategy status lines + button specs for the toggle keyboard.

    Returns (text, buttons) where buttons = [{"name", "enabled"}] in display
    order — the bot turns these into st_on_/st_off_ callbacks.
    """
    from framework.core.control import effective_enabled, effective_execution_mode
    from framework.core.registry import get_registry
    from framework.execution.lifecycle import LifecycleManager
    from framework.risk.manager import RiskManager

    registry = registry or get_registry()
    lm = LifecycleManager()
    states = lm.all_states()
    names = sorted(set(registry.configs) | set(states))
    rm = RiskManager()

    from earnings_edge.alpaca_mode import broker_label
    broker = broker_label()
    live_mark = "🔴 LIVE BROKER" if broker == "live" else "paper broker"
    lines = ["⚙️ <b>STRATEGIES</b>", live_mark, ""]
    buttons: list[dict] = []
    for name in names:
        toml_on = registry.is_enabled(name)
        on = effective_enabled(name, toml_on)
        lifecycle = states.get(name, "paper")
        cfg = registry.get(name)
        toml_mode = cfg.execution_mode if cfg else "approval"
        mode = effective_execution_mode(name, toml_mode)
        mode_src = " (override)" if mode != toml_mode else ""
        spend = rm.strategy_spend_today(name)
        mark = "🟢" if on else "⏸"
        src = "" if on == toml_on else (" (TOML off)" if not toml_on else " (override)")
        lines.append(f"{mark} <b>{html.escape(str(name))}</b> — {lifecycle} | {mode}{mode_src} | today ${spend:,.0f}{src}")
        if cfg and cfg.sizer:
            params = {k: v for k, v in cfg.sizer.items() if k != "name"}
            lines.append(f"    sizer {cfg.sizer.get('name')} {params}")
        buttons.append({"name": name, "enabled": on})
    lines.append("")
    lines.append("Tap a button to pause/resume. Paused strategies stop producing "
                 "proposals; open positions keep being managed to exit. "
                 "Execution mode (approval/auto) toggles live in /signals.")
    return "\n".join(lines), buttons


# ── /exits ──────────────────────────────────────────────────────────────

def pending_exits() -> list[dict]:
    return exit_proposals_list_pending()


# ── /setups (trade-setup help cards) ─────────────────────────────────────

SETUP_STRATEGIES = [
    "calendar_call_ml",
    "vol_risk_premium",
    "short_straddle",
    "ff_ladder",
    "forward_factor_arb",
]

_SETUP_BODY = {
    "calendar_call_ml": (
        "Structure: LONG call calendar into earnings.\n"
        "  SELL 1x <near expiry> ATM call  (first expiry after earnings)\n"
        "  BUY  1x <far expiry> ATM call   (~+28 days)\n"
        "Entry rule: ridge ML filter (return_on_debit regression) must score "
        "the candidate TAKE; you pay the combo ASK debit (conservative fill "
        "assumption).\n"
        "Sizing: pct_portfolio — 5% of equity per trade.\n"
        "Thesis: the near leg's IV collapses harder than the far leg's after "
        "the report (IV crush differential). The backtest exits on the first "
        "option close after earnings — this is not a hold-to-expiry trade."
    ),
    "vol_risk_premium": (
        "Structure: SHORT straddle into earnings (UNDEFINED RISK).\n"
        "  SELL 1x <near expiry> ATM call\n"
        "  SELL 1x <near expiry> ATM put\n"
        "Entry rule: IV/RV >= 1.4 AND expected move >= 6% — systematic "
        "harvest of the earnings vol premium (implied moves exceed realized "
        "by ~6pp on average in the backtest). You collect the straddle "
        "credit.\n"
        "Sizing: vol_target — risk 1% of equity vs a 2x-expected-move stress "
        "proxy; names whose stress loss exceeds the budget are vetoed.\n"
        "Risk note: naked short premium. A move beyond ~2x the priced move "
        "loses more than the stress proxy. Backtest: 94.8% hit, +$4.4k, max "
        "DD -$13 — but the tail is real and unhedged."
    ),
    "short_straddle": (
        "Structure: SHORT straddle into earnings (UNDEFINED RISK), same legs "
        "as vol_risk_premium.\n"
        "  SELL 1x <near expiry> ATM call + SELL 1x ATM put\n"
        "Entry rule: IV/RV >= 1.2 AND expected move >= 6% (looser IV/RV gate "
        "than vol_risk_premium; the magnitude-model gate runs filter-only "
        "live, mirroring the backtest fallback).\n"
        "Sizing: vol_target — 1% of equity vs the 2x-expected-move stress "
        "proxy.\n"
        "Risk note: same undefined-risk profile as vol_risk_premium."
    ),
    "ff_ladder": (
        "Structure: LONG call calendar, 30-60 DTE (NOT the front expiry).\n"
        "  SELL 1x <T1 ~30 DTE> ATM call   (expiry containing earnings)\n"
        "  BUY  1x <T2 ~+30d> ATM call     (event-free)\n"
        "Entry rule: implied event move >= 1.2x the ticker's RMS realized "
        "earnings move, expressed as a max debit D*. Entry is worked as a "
        "patient MLEG limit ladder 14:00-15:45 ET: start at D* x 1.25, step "
        "+$0.01 every 15 min toward D* x 1.20 hard cap, day orders only.\n"
        "Sizing: fixed_dollar — $2,000 budget per trade.\n"
        "Note: proposals arrive 13:45 ET; arming the ladder is the approval. "
        "Exits are managed by the exit engine per the TOML below."
    ),
    "forward_factor_arb": (
        "Structure: LONG call calendar on a rich front vs cheap forward vol.\n"
        "  SELL 1x near ATM call\n"
        "  BUY  1x far ATM call (~+30 days)\n"
        "Entry rule: forward factor (front IV vs forward vol) above 1.1; worked "
        "as a limit ladder from 1.5 down to 1.25.\n"
        "Sizing: fixed_dollar — $2,000 budget per trade.\n"
        "Thesis: term-structure mispricing, not an earnings-gap bet."
    ),
}


def _toml_exits(name: str, strategies_dir: Optional[str] = None) -> list[dict]:
    """[[exits]] rules from strategies/<name>.toml; [] when unreadable."""
    import tomllib
    from pathlib import Path

    base = Path(strategies_dir) if strategies_dir else Path(__file__).resolve().parent.parent / "strategies"
    try:
        with open(base / f"{name}.toml", "rb") as f:
            cfg = tomllib.load(f)
        return list(cfg.get("exits") or [])
    except Exception:
        return []


def _toml_meta(name: str, strategies_dir: Optional[str] = None) -> str:
    """One-line config summary: mode, lifecycle, sizer, limits from the TOML."""
    import tomllib
    from pathlib import Path

    base = Path(strategies_dir) if strategies_dir else Path(__file__).resolve().parent.parent / "strategies"
    try:
        with open(base / f"{name}.toml", "rb") as f:
            cfg = tomllib.load(f)
    except Exception:
        return ""
    sec = cfg.get("strategy") or {}
    risk = cfg.get("risk") or {}
    sizer = risk.get("sizer") or {}
    sizer_s = f"{sizer.get('name')} { {k: v for k, v in sizer.items() if k != 'name'} }" if sizer else "—"
    limits = risk.get("limits") or {}
    limits_s = str(limits) if limits else "framework defaults"
    return (
        f"Config (strategies/{name}.toml): execution_mode={sec.get('execution_mode', 'approval')}, "
        f"lifecycle={sec.get('lifecycle', 'paper')}, sizer={sizer_s}, limits={limits_s}. "
        "Edit the TOML + bot restart to change exits/sizing/limits; execution mode "
        "and pause/resume switch live from /signals and /strategies."
    )


def _format_exits(exits: list[dict]) -> str:
    if not exits:
        return "Exits: none configured."
    parts = []
    for e in exits:
        rule = e.get("rule")
        if rule == "profit_target":
            parts.append(f"profit target +{e.get('pct', 0) * 100:.0f}%")
        elif rule == "stop_loss":
            parts.append(f"stop loss -{e.get('pct', 0) * 100:.0f}%")
        elif rule == "time" and "days_after_entry" in e:
            parts.append(f"time exit {e['days_after_entry']}d after entry")
        elif rule == "time" and "days_before_event" in e:
            parts.append(f"time exit {e['days_before_event']}d before event")
        elif rule == "time" and "days_after_event" in e:
            n = int(e["days_after_event"])
            parts.append("time exit on event day" if n == 0
                         else f"time exit {n}d after event")
        elif rule == "scheduled":
            mins = e.get("minutes_before_close", 90)
            parts.append(f"auto-close ~{mins}min before close on/after the "
                         f"structural deadline (e.g. near-leg expiry)")
        else:
            parts.append(str(e))
    return "Exits (from TOML): " + "; ".join(parts) + "."


def setup_menu_text() -> str:
    return (
        "<b>TRADE SETUPS</b> — pick a strategy to see exactly which options the bot "
        "buys/sells, the entry rule, sizing, and exits.\n\n"
        + "\n".join(f"  /{i + 1}  <code>{html.escape(n)}</code>" for i, n in enumerate(SETUP_STRATEGIES))
    )


def setup_card(name: str, strategies_dir: Optional[str] = None) -> str:
    """Plain-text setup card for one strategy (exits cited from its TOML)."""
    body = _SETUP_BODY.get(name)
    if body is None:
        return f"No setup card for <code>{html.escape(str(name))}</code>."
    header = f"<b>SETUP:</b> <code>{html.escape(str(name))}</code>"
    meta = _toml_meta(name, strategies_dir)
    tail = f"{_format_exits(_toml_exits(name, strategies_dir))}\n{meta}" if meta else _format_exits(_toml_exits(name, strategies_dir))
    return f"{header}\n\n{html.escape(body)}\n{html.escape(tail)}"
