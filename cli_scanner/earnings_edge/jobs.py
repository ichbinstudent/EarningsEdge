"""Scheduled job implementations extracted from TradingBot.

Each ``*_job`` coroutine/function is the body the scheduler dispatches
via ``TradingBot._dispatch`` — identical logic to the former TradingBot
methods, now importable and unit-testable without a Telegram
Application. The bot keeps thin ``_x`` wrappers so scheduler
registration and monkeypatched-test seams stay unchanged.

Jobs in this module:
  equity_snapshot, reconcile, guard_eval, exit_eval, db_backup,
  db_health_check, chain_cache, picks_pipeline.

``bot`` is the TradingBot instance: jobs use only ``_flush_alerts``,
``_push_risk_alert``, ``_approval_chats``, ``_push_batches``, and
``application.bot`` — the same surface the inline methods used.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sqlite3
import sys
from datetime import datetime

from earnings_edge.alpaca_trading import create_client
from earnings_edge.db import managed_positions_list, persist_picks, snapshots_max_scan_date

logger = logging.getLogger("earnings_edge.jobs")


async def equity_snapshot_job(bot) -> dict | None:
    """Snapshot account equity; check the daily-loss kill switch."""
    from framework.core.calendar import get_calendar
    from framework.jobs import run_job
    from framework.risk.equity import snapshot_equity
    from framework.risk.manager import RiskManager

    def work():
        if not get_calendar().is_open_now():
            return {"skipped": "market closed"}
        row = snapshot_equity(create_client())
        halted = RiskManager().check_daily_loss(row["equity"])
        return {"equity": row["equity"], "halted": halted}

    try:
        stats = await asyncio.to_thread(run_job, "equity_snapshot", work)
    except Exception:
        return None  # run_job already logged
    if stats.get("halted"):
        from framework.alerts import DEDUPER
        DEDUPER.emit("daily_loss", "🛑 DAILY LOSS LIMIT BREACHED — kill switch tripped. "
                     "No orders will submit until /resume.")
    await bot._flush_alerts()
    return stats


async def reconcile_job(bot) -> None:
    """Diff broker positions vs local managed positions."""
    from framework.execution.reconcile import Reconciler
    from framework.jobs import run_job

    def work():
        report = Reconciler(create_client()).run()
        return {"summary": report.summary()}

    try:
        await asyncio.to_thread(run_job, "reconcile", work)
    except Exception as e:
        logger.error("Job failed: %s", e)
    await bot._flush_alerts()


async def guard_eval_job(bot) -> None:
    """Daily assignment-risk scan over open managed positions."""
    import json as _json

    from framework.jobs import run_job
    from framework.positions.guards import (
        LegView, check_assignment_risk, occ_underlying,
    )

    def work():
        client = create_client()
        spots: dict[str, float] = {}
        flags = []
        for pos in managed_positions_list():
            meta = _json.loads(pos.get("metadata") or "{}")
            if meta.get("leg_side") != "sell":
                continue  # only short legs can be assigned
            try:
                leg = LegView(
                    symbol=pos["symbol"], side="sell",
                    option_type=meta["option_type"], strike=float(meta["strike"]),
                    expiry=datetime.strptime(meta["expiry"], "%Y-%m-%d").date(),
                )
            except (KeyError, ValueError):
                continue
            und = occ_underlying(pos["symbol"])
            if not und:
                continue
            if und not in spots:
                px = client.get_stock_latest_trade(und)
                if px is None:
                    continue
                spots[und] = px
            # Dividend-aware checks arrive with the exit engine; the
            # near-expiry ITM check is the v1 guard.
            flags.extend(check_assignment_risk([leg], spots[und]))
        from earnings_edge.db import trade_events_insert
        for f in flags:
            trade_events_insert(
                "guard_flag", symbol=f.symbol, qty=1,
                detail=f"{f.reason} dte={f.dte}",
            )
        return {"flags": len(flags), "symbols": [f.symbol for f in flags]}

    try:
        stats = await asyncio.to_thread(run_job, "assignment_guard", work)
    except Exception:
        return
    if stats and stats.get("flags"):
        syms = ", ".join(stats["symbols"][:5])
        await bot._push_risk_alert(
            f"⚠️ Assignment risk on {stats['flags']} short leg(s): {syms}. "
            f"Review exits before expiry.")


async def exit_eval_job(bot) -> None:
    """Every 15 min in market hours: evaluate PT/SL (auto-close) and
    time exits (approval cards) over all open managed positions."""
    from framework.core.calendar import get_calendar
    from framework.jobs import run_job
    from framework.positions.manager import ExitManager

    def work():
        if not get_calendar().is_open_now():
            return {"skipped": "market closed"}
        mgr = ExitManager(create_client())
        stats = mgr.evaluate_all()
        stats["pending_cards"] = mgr.pending_exit_proposals()
        return stats

    try:
        stats = await asyncio.to_thread(run_job, "exit_eval", work)
    except Exception:
        return
    if not stats or stats.get("skipped"):
        return
    for msg in stats.get("auto_closed", []):
        await bot._push_risk_alert(msg)
    for err in stats.get("errors", []):
        logger.warning("exit eval: %s", err)
    await bot._flush_alerts()
    # Push any pending exit-approval cards not yet pushed this process,
    # grouped by strategy like entry proposals.
    pushed = getattr(bot, "_exit_cards_pushed", None)
    if pushed is None:
        pushed = bot._exit_cards_pushed = set()
    new_rows = [r for r in stats.get("pending_cards", []) if r["id"] not in pushed]
    for row in new_rows:
        pushed.add(row["id"])
    if new_rows:
        for uid in bot._approval_chats():
            await bot._push_batches(uid, new_rows, "exit")


def db_backup_job() -> None:
    from framework.backup import backup_db
    from framework.jobs import run_job

    def work():
        path = backup_db()
        return {"path": str(path)}

    try:
        run_job("db_backup", work)
    except Exception as e:
        logger.error("Job failed: %s", e)


async def db_health_check_job(bot) -> None:
    """Hourly PRAGMA integrity_check against the live DB (read-only conn).

    Corruption here in the past sat undetected for ~24h until the nightly
    backup job refused to run (2026-08-30/31 incident). This surfaces it
    within the hour via a direct Telegram alert instead.
    """
    from earnings_edge.db.engine import DEFAULT_DB_PATH
    from framework.jobs import run_job

    def work():
        conn = sqlite3.connect(f"file:{DEFAULT_DB_PATH}?mode=ro", uri=True, timeout=10)
        try:
            result = conn.execute("PRAGMA integrity_check(1)").fetchone()[0]
        finally:
            conn.close()
        if result.lower() != "ok":
            raise RuntimeError(f"DB integrity check FAILED: {result}")
        return {"status": "ok"}

    try:
        await asyncio.to_thread(lambda: run_job("db_health_check", work))
    except Exception as e:
        logger.error("DB health check failed: %s", e)
        text = f"🚨 DB CORRUPTION DETECTED\n\n{e}\n\nEarnings scanner DB may need repair — check data/earnings_ml.db immediately."
        for uid in bot._approval_chats():
            try:
                await bot.application.bot.send_message(chat_id=uid, text=text)
            except Exception as exc:
                logger.error("db health alert push to %d failed: %s", uid, exc)


async def chain_cache_job(bot) -> None:
    """Hourly Alpaca chain snapshot while the US session is open."""
    from framework.jobs import run_job

    def work():
        clock = create_client().get_clock()
        if not clock.get("is_open"):
            return {"skipped": "market closed"}
        from earnings_edge.chain_cache import HOURLY_MAX_TICKERS, run_hourly
        return run_hourly(max_tickers=HOURLY_MAX_TICKERS)

    try:
        await asyncio.to_thread(lambda: run_job("chain_cache", work))
    except Exception as e:
        logger.error("Job failed: %s", e)


async def picks_pipeline_job(bot) -> None:
    """Pre-market pipeline: refresh chains + signals, generate/persist picks,
    push a one-line summary. Picks are computed as of the latest snapshot
    scan_date (same convention as picks_report.py)."""
    from framework.jobs import run_job

    def work():
        import subprocess
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        py = sys.executable
        chain = subprocess.run(
            [py, "scripts/collect_options_snapshot.py", "--max-tickers", "250"],
            cwd=root, capture_output=True, text=True, timeout=1800)
        sig = subprocess.run(
            [py, "scripts/collect_daily_signals.py", "--max-tickers", "60"],
            cwd=root, capture_output=True, text=True, timeout=1800)
        if chain.returncode != 0:
            raise RuntimeError(f"chain collection failed: {chain.stderr[-300:]}")
        if sig.returncode != 0:
            raise RuntimeError(f"signals collection failed: {sig.stderr[-300:]}")

        from earnings_edge.picks import generate_picks
        latest = snapshots_max_scan_date()
        if not latest:
            return {"picks_written": 0, "counts": {}, "note": "no snapshots"}
        as_of = datetime.fromisoformat(str(latest)[:10]).date()
        picks = generate_picks(as_of)
        n = persist_picks(picks, as_of)
        return {"picks_written": n,
                "counts": {k: len(v) for k, v in picks.items()},
                "as_of": str(as_of)}

    try:
        result = await asyncio.to_thread(lambda: run_job("daily_picks", work))
    except Exception as e:
        logger.error("Job failed: %s", e)
        return

    counts = (result or {}).get("counts") or {}
    as_of = (result or {}).get("as_of", "?")
    breakdown = ", ".join(f"{k}: {v}" for k, v in counts.items()) or "none"
    text = (f"🎯 Daily picks refreshed ({as_of}) — {sum(counts.values())} picks "
            f"persisted ({breakdown}). Tap 🎯 Picks to browse.")
    for uid in bot._approval_chats():
        try:
            await bot.application.bot.send_message(chat_id=uid, text=text)
        except Exception as exc:
            logger.error("picks push to %s failed: %s", uid, exc)
