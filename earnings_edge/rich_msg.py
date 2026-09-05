import html
import logging
from typing import Any

import httpx

logger = logging.getLogger("rich_msg")


async def send_rich_html(bot: Any, chat_id: int, html_str: str, reply_markup: Any = None) -> bool:
    """Send a rich HTML message via the raw /sendRichMessage endpoint."""
    url = f"{bot.base_url}/sendRichMessage"
    payload = {"chat_id": chat_id, "rich_message": {"html": html_str}}
    if reply_markup is not None:
        payload["reply_markup"] = reply_markup.to_dict() if hasattr(reply_markup, "to_dict") else reply_markup

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


async def edit_rich_html(
    bot: Any, chat_id: int, message_id: int, html_str: str, reply_markup: Any = None
) -> bool:
    """Edit a message with rich HTML via the raw /editMessageText endpoint."""
    url = f"{bot.base_url}/editMessageText"
    payload = {"chat_id": chat_id, "message_id": message_id, "rich_message": {"html": html_str}}
    if reply_markup is not None:
        payload["reply_markup"] = reply_markup.to_dict() if hasattr(reply_markup, "to_dict") else reply_markup

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(url, json=payload, timeout=15.0)
            if resp.status_code != 200:
                logger.error(f"editMessageText (rich) failed with {resp.status_code}: {resp.text}")
                return False
            return True
    except Exception as e:
        logger.error(f"editMessageText (rich) exception: {e}")
        return False


def orders_rich_view(limit: int = 12) -> str:
    from earnings_edge.db import trade_events_list

    events = trade_events_list(limit=limit)
    out = ["<h3>ORDERS</h3>"]
    if not events:
        out.append("<p><i>No trade events yet.</i></p>")
        return "\n".join(out)

    out.append("<table bordered striped compact>")
    out.append("<tr><th>Time</th><th>Event</th><th>Strategy</th><th>Symbol</th><th>Price</th></tr>")

    details = []
    for row in events:
        ts_short = str(row["ts"])[:16][5:].replace("T", " ")  # MM-DD HH:MM
        ev = str(row["event_type"] or "")
        st = str(row["strategy"] or "")
        sym = str(row["symbol"] or "")

        price_val = row["price"]
        price = f"{price_val:.2f}" if price_val is not None else ""

        out.append(
            f"<tr>"
            f"<td>{html.escape(ts_short)}</td>"
            f"<td>{html.escape(ev)}</td>"
            f"<td>{html.escape(st)}</td>"
            f"<td>{html.escape(sym)}</td>"
            f"<td>{html.escape(price)}</td>"
            f"</tr>"
        )

        detail_val = str(row["detail"] or "").strip()
        if detail_val:
            if len(detail_val) > 110:
                detail_val = detail_val[:107] + "..."
            details.append(
                f"<p><i>{html.escape(ts_short)} {html.escape(sym)}</i> — {html.escape(detail_val)}</p>"
            )
    out.append("</table>")
    out.extend(details)
    return "\n".join(out)


def jobs_rich_view(limit: int = 12) -> str:
    from earnings_edge.db import job_runs_list

    runs = job_runs_list(limit=limit)
    out = ["<h3>JOB RUNS</h3>"]
    if not runs:
        out.append("<p><i>No job runs found.</i></p>")
        return "\n".join(out)

    out.append("<table bordered striped compact>")
    out.append("<tr><th>Status</th><th>Job</th><th>Started</th><th>Summary</th></tr>")
    for r in runs:
        ok = bool(r["success"])
        status = "✓" if ok else "✗"
        job = str(r["job_name"])
        ts_str = r["started_at"][:19].replace("T", " ")

        err = r["error"] or ""
        if err:
            summary_val = str(err)
        else:
            stats = r["stats_json"]
            if stats:
                import json

                try:
                    s_dict = json.loads(stats)
                    items = [f"{k}={v}" for k, v in list(s_dict.items())[:4]]
                    summary_val = " ".join(items)
                except Exception:
                    summary_val = str(stats)
            else:
                summary_val = ""

        summary = (summary_val[:97] + "...") if len(summary_val) > 100 else summary_val

        out.append(
            f"<tr>"
            f"<td>{html.escape(status)}</td>"
            f"<td>{html.escape(job)}</td>"
            f"<td>{html.escape(ts_str)}</td>"
            f"<td>{html.escape(summary)}</td>"
            f"</tr>"
        )
    out.append("</table>")
    return "\n".join(out)


def equity_rich_view(days: int = 7) -> str:
    from earnings_edge.db import equity_snapshots_daily_avg
    from framework.risk.equity import daily_pnl, day_start_equity, latest_equity

    current = latest_equity()
    if not current:
        return "<h3>EQUITY</h3>\n<p><i>No equity data available.</i></p>"

    val = current["equity"]
    bp = current["buying_power"]
    start_val = day_start_equity()
    pnl = daily_pnl(val)
    pnl = pnl if pnl is not None else 0.0

    out = ["<h3>EQUITY</h3>"]
    if start_val is not None:
        out.append(
            f"<p>Latest: ${val:,.2f} | BP: ${bp:,.2f} | Day-start: ${start_val:,.2f} | Day-PnL: ${pnl:,.2f}</p>"
        )
    else:
        out.append(f"<p>Latest: ${val:,.2f} | BP: ${bp:,.2f} | Day-PnL: ${pnl:,.2f}</p>")

    history = equity_snapshots_daily_avg(days=days)
    if not history:
        out.append("<p><i>No daily history.</i></p>")
        return "\n".join(out)

    out.append("<table bordered striped compact>")
    out.append("<tr><th>Date</th><th>Equity</th></tr>")
    for r in history:
        date = str(r["d"])
        eq = r["e"]
        eq_str = f"${eq:,.2f}" if eq is not None else ""
        out.append(f"<tr><td>{html.escape(date)}</td><td>{html.escape(eq_str)}</td></tr>")
    out.append("</table>")

    return "\n".join(out)


def positions_rich_view(
    broker_positions: list | None = None, broker_error: str | None = None, banner: str | None = None
) -> str:
    from earnings_edge.bot_views import _ignored_symbols
    from framework.execution.managed import open_groups
    from framework.positions.book import classify_book

    groups = open_groups()
    out = []

    if banner:
        out.append(f"<h3>{html.escape(banner)}</h3>")

    if broker_positions is None:
        if broker_error:
            out.append(f"<p>⚠️ Broker unavailable: {html.escape(broker_error)}</p>")
        if not groups:
            out.append("<h3>No open managed positions.</h3>")
            return "\n".join(out)
        out.append(f"<h3>OPEN POSITIONS ({len(groups)} groups) — local book only</h3>")

        out.append("<table bordered striped compact>")
        out.append("<tr><th>Strategy</th><th>Ticker</th><th>Type</th><th>Entry</th><th>Opened</th></tr>")
        for g in groups:
            kind = "credit" if g.credit else "debit"
            out.append(
                f"<tr><td>{html.escape(str(g.strategy))}</td>"
                f"<td>{html.escape(str(g.ticker))}</td>"
                f"<td>{html.escape(kind)}</td>"
                f"<td>${g.entry_price:.2f}</td>"
                f"<td>{html.escape(str(g.opened_at)[:16].replace('T', ' '))}</td></tr>"
            )
        out.append("</table>")

        for g in groups:
            out.append(f"<p><b>{html.escape(g.ticker)} legs:</b></p><ul>")
            for leg in g.legs:
                side = "SELL" if leg.side == "sell" else "BUY"
                exp = leg.expiry.isoformat() if leg.expiry else "?"
                out.append(
                    f"<li>{side} {leg.qty:g} {html.escape(leg.symbol)} ({leg.option_type} {leg.strike:g} {exp})</li>"
                )
            out.append("</ul>")
        return "\n".join(out)

    book = classify_book(groups, broker_positions, ignored=_ignored_symbols())
    if not book.managed and not book.orphan and not book.missing:
        out.append("<h3>No positions at broker or locally.</h3>")
        return "\n".join(out)

    out.append(
        f"<h3>BOOK: broker={book.broker_count} local={book.local_count} orphans={len(book.orphan)} missing={len(book.missing)}</h3>"
    )

    def render_table(items, title):
        if not items:
            return ""
        lines = [f"<h4>{title}</h4>", "<table bordered striped compact>"]
        lines.append("<tr><th>Ticker</th><th>Strategy</th><th>Qty</th><th>uPL</th><th>Price</th></tr>")
        details = []
        for it in items:
            upl = f"${it.upl:+,.0f}" if it.upl is not None else ""
            px = f"{it.current_price:g}" if it.current_price is not None else ""
            lines.append(
                f"<tr><td>{html.escape(str(it.ticker))}</td>"
                f"<td>{html.escape(str(it.strategy))}</td>"
                f"<td>{it.qty:g}</td>"
                f"<td>{html.escape(upl)}</td>"
                f"<td>{html.escape(px)}</td></tr>"
            )
            ev = f" event {it.event_date.isoformat()}" if it.event_date else ""
            exp = f" exp {it.expiry.isoformat()}" if it.expiry else ""
            details.append(
                f"<p><i>{html.escape(str(it.ticker))}</i>: {html.escape(str(it.symbol))}{html.escape(ev)}{html.escape(exp)}</p>"
            )
        lines.append("</table>")
        lines.extend(details)
        return "\n".join(lines)

    if book.managed:
        out.append(render_table(book.managed, "MANAGED (matched)"))
    if book.orphan:
        out.append(render_table(book.orphan, "ORPHAN (at broker, not local)"))
    if book.missing:
        out.append(render_table(book.missing, "MISSING (local open, not at broker)"))

    return "\n".join(out)


def pending_rich_view(inbox: Any, banner: str | None = None) -> str:
    out = []
    if banner:
        out.append(f"<h3>{html.escape(banner)}</h3>")

    groups = inbox.grouped()
    if not groups:
        out.append("<h3>📥 Pending inbox</h3><p>Nothing pending.</p>")
        return "\n".join(out)

    out.append("<h3>📥 Pending inbox</h3>")
    labels = {
        "entry": "Entries",
        "exit": "Exits",
        "orphan": "Orphans",
        "assignment": "Assignments",
        "job": "Failed jobs",
        "expired": "Expired (stale)",
    }

    for kind, items in groups.items():
        out.append(f"<h4>{html.escape(labels.get(kind, kind))}</h4>")
        out.append("<table bordered striped compact>")
        out.append("<tr><th>Ticker</th><th>Kind</th><th>Strategy</th><th>Created</th><th>Detail</th></tr>")

        for it in items:
            expired = " (EXPIRED)" if it.expired else ""
            ts = str(it.created_at)[:16].replace("T", " ") if it.created_at else ""
            out.append(
                f"<tr><td><b>{html.escape(str(it.ticker))}</b></td>"
                f"<td>{html.escape(it.kind)}{expired}</td>"
                f"<td>{html.escape(str(it.strategy))}</td>"
                f"<td>{html.escape(ts)}</td>"
                f"<td>{html.escape(str(it.detail))}</td></tr>"
            )
        out.append("</table>")

    return "\n".join(out)


def strategies_rich_view(registry: Any = None) -> tuple[str, list[dict]]:
    from earnings_edge.alpaca_mode import broker_label
    from framework.core.control import effective_enabled, effective_execution_mode
    from framework.core.registry import get_registry
    from framework.execution.lifecycle import LifecycleManager
    from framework.risk.manager import RiskManager

    registry = registry or get_registry()
    lm = LifecycleManager()
    states = lm.all_states()
    names = sorted(set(registry.configs) | set(states))
    rm = RiskManager()

    broker = broker_label()
    live_mark = "🔴 LIVE BROKER" if broker == "live" else "paper broker"

    out = [f"<h3>⚙️ STRATEGIES ({html.escape(live_mark)})</h3>"]
    out.append("<table bordered striped compact>")
    out.append("<tr><th>Status</th><th>Name</th><th>Lifecycle</th><th>Mode</th><th>Spend Today</th></tr>")

    buttons = []
    details = []

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

        out.append(
            f"<tr><td>{mark}</td>"
            f"<td><b>{html.escape(str(name))}</b></td>"
            f"<td>{html.escape(lifecycle)}</td>"
            f"<td>{html.escape(mode)}{html.escape(mode_src)}</td>"
            f"<td>${spend:,.0f}{html.escape(src)}</td></tr>"
        )
        if cfg and cfg.sizer:
            params = {k: v for k, v in cfg.sizer.items() if k != "name"}
            details.append(
                f"<p><i>{html.escape(str(name))} sizer:</i> {html.escape(cfg.sizer.get('name', ''))} {html.escape(str(params))}</p>"
            )

        buttons.append({"name": name, "enabled": on})

    out.append("</table>")
    out.extend(details)
    out.append(
        "<p><i>Tap a button to pause/resume. Paused strategies stop producing proposals; open positions keep being managed to exit. Execution mode (approval/auto) toggles live in /signals.</i></p>"
    )

    return "\n".join(out), buttons


def status_rich_view(
    *,
    market_open: bool | None = None,
    pending_proposals: int = 0,
    pending_exits: int = 0,
    next_events: list | None = None,
    funnel: str | None = None,
    last_scan_ts: str | None = None,
    last_equity_ts: str | None = None,
    reconcile_summary: str | None = None,
    broker_ok: bool | None = None,
    broker_count: int | None = None,
    orphan_count: int | None = None,
    sha: str | None = None,
    started_at: str | None = None,
) -> str:
    from earnings_edge.bot_views import _age, _equity_curve, _ts_short
    from earnings_edge.db import job_runs_failed, strategy_state_list
    from framework.execution.managed import open_groups
    from framework.risk.equity import daily_pnl, latest_equity
    from framework.risk.killswitch import KillSwitch

    out = ["<h3>🖥 SYSTEM STATUS</h3>"]

    # SYSTEM SECTION
    out.append("<h4>System</h4>")
    out.append("<ul>")
    if market_open is not None:
        out.append("<li><b>Market:</b> " + ("🟢 open" if market_open else "⚫ closed") + "</li>")
    if broker_ok is not None:
        out.append("<li><b>Broker:</b> " + ("reachable" if broker_ok else "unreachable") + "</li>")
    if sha:
        out.append(f"<li><b>Rev:</b> <code>{html.escape(sha)}</code></li>")
    if started_at:
        out.append(f"<li><b>Started:</b> {html.escape(_ts_short(started_at))}</li>")
    out.append("</ul>")

    # KILL SWITCH SECTION
    out.append("<h4>Kill Switch</h4>")
    ks = KillSwitch().status()
    if ks.get("halted"):
        out.append(
            f"<p>🛑 HALTED — {html.escape(str(ks.get('reason')))} (by {html.escape(str(ks.get('tripped_by')))})</p>"
        )
    else:
        out.append("<p>🟢 armed</p>")

    # EQUITY SECTION
    out.append("<h4>Equity</h4>")
    eq = latest_equity()
    if eq:
        pnl = daily_pnl(eq["equity"])
        pnl_txt = f" | day PnL ${pnl:+,.0f}" if pnl is not None else ""
        out.append(
            f"<p><b>${eq['equity']:,.0f}</b> | BP ${eq['buying_power']:,.0f}"
            f"{pnl_txt} (<i>{html.escape(_age(eq['ts']))}</i>)</p>"
        )
        curve = _equity_curve()
        if curve:
            out.append(f"<p><code>{html.escape(curve)}</code></p>")
    else:
        out.append("<p>no snapshots yet</p>")

    # STRATEGIES SECTION
    out.append("<h4>Strategies</h4>")
    try:
        states = strategy_state_list()
    except Exception:
        states = []
    if states:
        out.append("<table bordered striped compact>")
        out.append("<tr><th>Strategy</th><th>Lifecycle</th></tr>")
        for s in states:
            tag = s["lifecycle"]
            if s["enabled"] == 0:
                tag += "·OFF"
            out.append(f"<tr><td>{html.escape(str(s['name']))}</td><td>{html.escape(str(tag))}</td></tr>")
        out.append("</table>")
    else:
        out.append("<p>None active.</p>")

    # POSITIONS COUNTS
    out.append("<h4>Counts</h4>")
    groups = open_groups()
    out.append("<ul>")
    out.append(
        f"<li><b>Open groups:</b> {len(groups)}"
        + (f" vs broker {broker_count}" if broker_count is not None else "")
        + "</li>"
    )
    if orphan_count is not None:
        out.append(f"<li><b>Orphans:</b> {orphan_count}</li>")
    out.append(f"<li><b>Pending:</b> {pending_proposals} proposals, {pending_exits} exit cards</li>")
    out.append("</ul>")

    # TIMESTAMPS & LOGS
    out.append("<h4>Activity</h4>")
    out.append("<ul>")
    if last_scan_ts:
        out.append(
            f"<li><b>Last scan:</b> {html.escape(_age(last_scan_ts))} ({html.escape(_ts_short(last_scan_ts))})</li>"
        )
    elif last_scan_ts == "":
        out.append("<li><b>Last scan:</b> never</li>")
    if last_equity_ts:
        out.append(f"<li><b>Last equity snapshot:</b> {html.escape(_age(last_equity_ts))}</li>")
    if reconcile_summary:
        out.append(f"<li><b>Last reconcile:</b> {html.escape(str(reconcile_summary))}</li>")
    if next_events:
        out.append("<li><b>Next events:</b> " + " · ".join(html.escape(e) for e in next_events) + "</li>")
    out.append("</ul>")

    if funnel:
        out.append(f"<p>{html.escape(funnel)}</p>")

    try:
        fails = job_runs_failed(3)
    except Exception:
        fails = []
    if fails:
        out.append("<h4>Recent job failures</h4><ul>")
        for f in fails:
            err = (f["error"] or "")[:80]
            out.append(
                f"<li>✗ {html.escape(f['job_name'])} ({html.escape(_ts_short(f['finished_at']))}): {html.escape(err)}</li>"
            )
        out.append("</ul>")

    return "\n".join(out)
