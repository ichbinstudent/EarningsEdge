import html
import logging
import httpx

logger = logging.getLogger("rich_msg")

async def send_rich_html(bot, chat_id: int, html_str: str, reply_markup=None) -> bool:
    """Send a rich HTML message via the raw /sendRichMessage endpoint."""
    url = f"{bot.base_url}/sendRichMessage"
    payload = {
        "chat_id": chat_id,
        "rich_message": {
            "html": html_str
        }
    }
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

async def edit_rich_html(bot, chat_id: int, message_id: int, html_str: str, reply_markup=None) -> bool:
    """Edit a message with rich HTML via the raw /editMessageText endpoint."""
    url = f"{bot.base_url}/editMessageText"
    payload = {
        "chat_id": chat_id,
        "message_id": message_id,
        "rich_message": {
            "html": html_str
        }
    }
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

def orders_rich_view(limit=12) -> str:
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
        ts_short = str(row.get("ts", ""))[:16][5:].replace("T", " ")  # MM-DD HH:MM
        ev = str(row.get("event_type") or "")
        st = str(row.get("strategy") or "")
        sym = str(row.get("symbol") or "")

        price_val = row.get("price")
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

        detail_val = str(row.get("detail") or "").strip()
        if detail_val:
            if len(detail_val) > 110:
                detail_val = detail_val[:107] + "..."
            details.append(
                f"<p><i>{html.escape(ts_short)} {html.escape(sym)}</i> — {html.escape(detail_val)}</p>"
            )
    out.append("</table>")
    out.extend(details)
    return "\n".join(out)

def jobs_rich_view(limit=12) -> str:
    from earnings_edge.db import job_runs_list
    runs = job_runs_list(limit=limit)
    out = ["<h3>JOB RUNS</h3>"]
    if not runs:
        out.append("<p><i>No job runs found.</i></p>")
        return "\n".join(out)
        
    out.append("<table bordered striped compact>")
    out.append("<tr><th>Status</th><th>Job</th><th>Started</th><th>Summary</th></tr>")
    for r in runs:
        ok = r.get("success", False)
        status = "✓" if ok else "✗"
        job = str(r.get("job_name", ""))
        ts_str = r.get("started_at", "")[:19].replace("T", " ")
        
        err = r.get("error", "")
        if err:
            summary_val = str(err)
        else:
            stats = r.get("stats_json")
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

def equity_rich_view(days=7) -> str:
    from framework.risk.equity import latest_equity, day_start_equity, daily_pnl
    from earnings_edge.db import equity_snapshots_daily_avg
    
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
        out.append(f"<p>Latest: ${val:,.2f} | BP: ${bp:,.2f} | Day-start: ${start_val:,.2f} | Day-PnL: ${pnl:,.2f}</p>")
    else:
        out.append(f"<p>Latest: ${val:,.2f} | BP: ${bp:,.2f} | Day-PnL: ${pnl:,.2f}</p>")

    history = equity_snapshots_daily_avg(days=days)
    if not history:
        out.append("<p><i>No daily history.</i></p>")
        return "\n".join(out)

    out.append("<table bordered striped compact>")
    out.append("<tr><th>Date</th><th>Equity</th></tr>")
    for r in history:
        date = str(r.get("d", ""))
        eq = r.get("e")
        eq_str = f"${eq:,.2f}" if eq is not None else ""
        out.append(f"<tr><td>{html.escape(date)}</td><td>{html.escape(eq_str)}</td></tr>")
    out.append("</table>")

    return "\n".join(out)
