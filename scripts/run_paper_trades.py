#!/usr/bin/env python3
"""DEPRECATED. Daily operation is bot.py trade-approval cards, not this CLI.

The Hermes cron that called this is paused. Do not wire it back as the live path.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Ensure the project root is importable
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))

from earnings_edge.alpaca_bridge import run_auto_trade, BEST_STRATEGIES

logging.basicConfig(
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("paper_trader")


TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
LOCK_PATH = Path("/tmp/paper_trade.lock")
LOCK_STALE_S = 30 * 60


# ---------------------------------------------------------------------------
# Overlap guard
# ---------------------------------------------------------------------------

def acquire_lock() -> bool:
    """Create the run lock. Fresh lock -> skip run; stale lock -> take over."""
    try:
        if LOCK_PATH.exists():
            age = time.time() - LOCK_PATH.stat().st_mtime
            if age < LOCK_STALE_S:
                logger.warning("Fresh lock exists (age %.0fs) — another run is active, skipping", age)
                return False
            logger.warning("Stale lock (age %.0fs) — taking over", age)
            LOCK_PATH.unlink()
        LOCK_PATH.write_text(str(os.getpid()))
        return True
    except OSError as e:
        logger.warning("Lock handling failed (%s) — proceeding without guard", e)
        return True


def release_lock() -> None:
    try:
        LOCK_PATH.unlink(missing_ok=True)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Telegram
# ---------------------------------------------------------------------------

def telegram_token_valid() -> bool:
    """Preflight the bot token (it has been revoked repeatedly)."""
    if not TELEGRAM_BOT_TOKEN:
        return False
    try:
        import requests

        resp = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getMe", timeout=10)
        return resp.status_code == 200
    except Exception as e:
        logger.error("Telegram token check failed: %s", e)
        return False


def send_telegram_notification(chat_id: str, text: str) -> bool:
    """Send a Telegram message as PLAIN TEXT (Markdown + '$' = parse failures)."""
    if not TELEGRAM_BOT_TOKEN:
        logger.debug("TELEGRAM_BOT_TOKEN not set, skipping notification")
        return False

    try:
        import requests

        resp = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={"chat_id": int(chat_id), "text": text},
            timeout=15,
        )
        if resp.status_code != 200:
            logger.error("Telegram send failed: HTTP %s %s", resp.status_code, resp.text[:200])
            return False
        return True
    except Exception as e:
        logger.error("Telegram notification failed: %s", e)
        return False


def format_summary(summary: dict) -> str:
    """Format execution summary (plain text — safe for Telegram and logs)."""
    lines = [
        f"📊 Paper Trade Execution — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"Buying Power: ${summary.get('buying_power', 0):,.2f}",
        "Mode: LIVE",
        "",
    ]

    strat_results = summary.get("strategies", {})
    if not strat_results:
        lines.append("No strategies processed.")
        return "\n".join(lines)

    lines.append("Strategy Results:")
    total_submitted = 0
    for name, result in strat_results.items():
        status = result.get("status", "?")
        trades = result.get("trades", 0)
        submitted = result.get("submitted", 0)
        total_submitted += submitted
        if status == "ok":
            lines.append(f"  • {name}: {trades} signals → {submitted} submitted")
        else:
            lines.append(f"  • {name}: {status} ({trades} signals)")

    skip_reasons = summary.get("skip_reasons") or {}
    if skip_reasons:
        reasons = ", ".join(f"{k}={v}" for k, v in sorted(skip_reasons.items()))
        lines.append(f"Skips: {reasons}")

    orders = summary.get("orders", [])
    if orders:
        lines.append("")
        lines.append("Orders:")
        for o in orders[:10]:
            lines.append(f"  • {o['strategy']}: {o['symbol']} ({o['legs']} legs, {o['status']})")
        if len(orders) > 10:
            lines.append(f"  ... and {len(orders) - 10} more")

    lines.append("")
    lines.append(f"Total submitted: {total_submitted}")
    if total_submitted > 0:
        lines.append("⚠️ LIVE ORDERS SUBMITTED — monitor fills")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run paper-trading strategies")
    parser.add_argument("--strategies", nargs="*", default=None, help="Strategies to run (default: BEST_STRATEGIES)")
    parser.add_argument("--notify", default=None, help="Telegram chat ID to send notification to")
    parser.add_argument("--output", default=None, help="Write JSON output to file")
    parser.add_argument("--db", default=None, help="Path to earnings_ml.db")
    parser.add_argument("--api-key", default=None, help="Alpaca API key (or env APCA_API_KEY_ID)")
    parser.add_argument("--api-secret", default=None, help="Alpaca API secret (or env APCA_API_SECRET_KEY)")
    args = parser.parse_args()

    if not acquire_lock():
        return 0  # another run in progress — clean skip

    try:
        strategies = args.strategies or BEST_STRATEGIES
        logger.info("Paper trading run: strategies=%s", strategies)

        # Verify the bot token BEFORE the long scan, not after (recurring 401s)
        telegram_ok = True
        if args.notify:
            telegram_ok = telegram_token_valid()
            if not telegram_ok:
                logger.error("Telegram bot token invalid/revoked — will print report only")

        summary = run_auto_trade(
            strategies=strategies,
            db_path=args.db,
            api_key=args.api_key,
            api_secret=args.api_secret,
            max_per_ticker=5000,
            min_buying_power=10000,
            max_orders=20,
        )

        # Print formatted summary
        output_text = format_summary(summary)
        print(output_text)

        # Write JSON if requested
        if args.output:
            Path(args.output).write_text(json.dumps(summary, indent=2, default=str))
            print(f"\nJSON report written to {args.output}")

        # Telegram notification
        if args.notify:
            if telegram_ok:
                send_telegram_notification(args.notify, output_text)
            else:
                logger.error("Skipped Telegram delivery (token invalid). Message was:\n%s", output_text)

        if summary.get("total_submitted", 0) > 0:
            logger.warning("%d LIVE orders submitted", summary["total_submitted"])
            return 2  # signal: live orders placed

        return 0
    finally:
        release_lock()


if __name__ == "__main__":
    sys.exit(main())
