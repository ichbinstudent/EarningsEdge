"""Telegram command handlers extracted from TradingBot.

Functions take the ``bot`` (TradingBot) instance and the python-telegram-bot
``update``/``ctx`` pair. Logic is identical to the former TradingBot
methods — moved verbatim so behavior and the registration table in
``TradingBot.run`` stay unchanged. The bot keeps one-line ``_cmd_x``
delegates so tests that patch ``bot.TradingBot._cmd_x`` keep working.
"""

from __future__ import annotations

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode as _PM
from telegram.ext import ContextTypes

HTML = _PM.HTML

from earnings_edge import cards
from earnings_edge.bot_live import ProgressMessage
from earnings_edge.ops_auth import auth_message


def desk_refresh_kb(which: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [[InlineKeyboardButton("🔄 Refresh", callback_data=f"desk_{which}")]])


async def cmd_start(bot, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    from bot import _main_reply_kb
    kb = _main_reply_kb()
    from dashboard.tg_auth import webapp_url
    url = webapp_url()
    desk = ("• Open desk — Mini App book / inbox / halt\n" if url else "")
    await bot._send_panel(update,
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
    ikb = bot._desk_webapp_markup()
    if ikb:
        await update.message.reply_text(
            "Open the web desk from this button (not by pasting the URL):",
            reply_markup=ikb,
        )


async def cmd_help(bot, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    from bot import _main_reply_kb
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
    for name, sc in bot.scanners.items():
        lines.append(f"• {name}: {sc.schedule}")
    lines += [
        "\n📱 Use the keyboard at the bottom for quick access!",
    ]
    await bot._send_panel(update,
        "\n".join(lines), reply_markup=kb, parse_mode=HTML,
    )


async def cmd_scanners(bot, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    # Deprecated surface — folded into /signals (unified signal surface).
    await cmd_signals(bot, update, ctx)


async def cmd_subscriptions(bot, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    # Deprecated surface — folded into /signals.
    await cmd_signals(bot, update, ctx)


async def cmd_signals(bot, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    import asyncio

    uid = update.effective_user.id
    modes = await asyncio.to_thread(bot._effective_modes_sync)
    await bot._send_panel(update,
        bot._signals_intro(), reply_markup=bot._signals_kb(uid, modes), parse_mode=HTML)


async def cmd_setups(bot, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    from earnings_edge.bot_views import SETUP_STRATEGIES, setup_menu_text
    ikb = [[InlineKeyboardButton(name, callback_data=f"setup_{name}")]
           for name in SETUP_STRATEGIES]
    await bot._send_panel(update,
        setup_menu_text(), reply_markup=InlineKeyboardMarkup(ikb), parse_mode=HTML)


async def cmd_run(bot, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    msg, ikb = bot._run_panel()
    await bot._send_panel(update, msg, reply_markup=ikb, parse_mode=HTML)


async def cmd_halt(bot, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    import asyncio

    if not bot._risk_authorized(update.effective_chat.id):
        await update.message.reply_text(auth_message())
        return
    await asyncio.to_thread(bot._halt_sync, "operator")
    await bot._flush_alerts()
    await update.message.reply_text("🛑 KILL SWITCH TRIPPED — no orders will submit until /resume.")


async def cmd_resume(bot, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    import asyncio

    if not bot._risk_authorized(update.effective_chat.id):
        await update.message.reply_text(auth_message())
        return
    await asyncio.to_thread(bot._resume_sync, "operator")
    await update.message.reply_text("✅ Kill switch released — order submission re-enabled.")


async def cmd_risk(bot, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    import asyncio

    text = await asyncio.to_thread(bot._risk_status_sync)
    await update.message.reply_text(text, parse_mode=HTML)


async def _cmd_set_lifecycle(bot, update: Update, promote: bool) -> None:
    import asyncio

    if not bot._risk_authorized(update.effective_chat.id):
        await update.message.reply_text(auth_message())
        return
    args = (update.message.text or "").split()
    if len(args) != 2:
        await update.message.reply_text("Usage: /promote <strategy> | /demote <strategy>")
        return
    cur, new = await asyncio.to_thread(
        bot._lifecycle_step_sync, args[1], promote, "telegram")
    await update.message.reply_text(f"📊 <code>{cards.esc(args[1])}</code>: {cur} → {new}", parse_mode=HTML)


async def cmd_promote(bot, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    await _cmd_set_lifecycle(bot, update, promote=True)


async def cmd_demote(bot, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    await _cmd_set_lifecycle(bot, update, promote=False)


async def cmd_settings(bot, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    await bot._send_panel(update, "🛠 <b>Settings</b>", parse_mode=HTML,
                                    reply_markup=bot._settings_kb())


async def cmd_monitor(bot, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    import asyncio

    chat_id = update.effective_chat.id
    await bot._stop_monitor(chat_id)
    bot._monitors[chat_id] = asyncio.create_task(bot._monitor_loop(chat_id))


async def cmd_status(bot, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    import asyncio

    text = await asyncio.to_thread(bot._status_text_sync)
    await bot._flush_alerts()
    await bot._send_panel(update,
        text, reply_markup=desk_refresh_kb("st"), parse_mode=HTML)


async def cmd_positions(bot, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    import asyncio

    text, rows = await asyncio.to_thread(bot._positions_panel_sync)
    await bot._flush_alerts()
    markup = InlineKeyboardMarkup(rows) if rows else None
    # One message: book + actions. Reply keyboard stays MAIN_KB.
    await bot._send_panel(update, text, reply_markup=markup, parse_mode=HTML)


async def cmd_orders(bot, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    import asyncio

    text = await asyncio.to_thread(bot._orders_text_sync)
    await bot._send_panel(update, text, reply_markup=desk_refresh_kb("or"), parse_mode=HTML)


async def cmd_jobs(bot, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    import asyncio

    text = await asyncio.to_thread(bot._jobs_text_sync)
    await bot._send_panel(update, text, reply_markup=desk_refresh_kb("jb"), parse_mode=HTML)


async def cmd_equity(bot, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    import asyncio

    text = await asyncio.to_thread(bot._equity_text_sync)
    await bot._send_panel(update, text, reply_markup=desk_refresh_kb("eq"), parse_mode=HTML)


async def cmd_strategies(bot, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    import asyncio

    text, ikb = await asyncio.to_thread(bot._strategies_panel_sync)
    await bot._send_panel(update, text, reply_markup=InlineKeyboardMarkup(ikb))


async def cmd_exits(bot, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    import asyncio

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
    await bot._push_batches(update.effective_chat.id, rows, "exit")


async def cmd_pending(bot, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    import asyncio

    text, rows = await asyncio.to_thread(bot._pending_panel_sync)
    await bot._send_panel(update,
        text, parse_mode=HTML, reply_markup=InlineKeyboardMarkup(rows))


async def cmd_propose(bot, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    pm = ProgressMessage(bot.application.bot, update.effective_chat.id,
                         "Building trade proposals")
    await pm.start()
    try:
        await pm.set_stage("live signals → legs → risk gates → persist")
        await bot._propose_and_push()
        from earnings_edge import trade_approval
        from earnings_edge.trade_approval import funnel_line
        funnel = funnel_line(trade_approval.LAST_FUNNEL)
        tail = f"\n{funnel}" if funnel else ""
        await pm.finish(f"✅ Proposals built and pushed.{tail}")
    except Exception as exc:
        import logging
        logging.getLogger("trading_bot").exception("manual proposal build failed")
        await pm.finish(f"❌ Proposal build failed: {exc}")
