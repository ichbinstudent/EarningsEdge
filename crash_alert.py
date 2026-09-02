#!/usr/bin/env python3
"""German-venue crash alerts — own process, not the earnings Telegram bot.

Polls Gettex/Xetra/Frankfurt every 2 min 07:30–23:00 Europe/Berlin weekdays.
Telegram only (no orders). Does not long-poll Telegram, so it can share the
bot token with bot.py.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import urllib.error
import urllib.request

from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
import pytz

from earnings_edge.german_crash import build_monitor, format_alert
from earnings_edge.ops_auth import operator_chat_ids
from framework.jobs import run_job

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("crash_alert")

_MONITOR = None


def _chats() -> list[int]:
    ops = operator_chat_ids()
    if ops:
        return list(ops)
    raw = os.environ.get("TELEGRAM_APPROVAL_CHAT_ID", "").strip()
    if raw.isdigit():
        return [int(raw)]
    return []


def _send_html(chat_id: int, text: str) -> None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN unset")
    body = json.dumps(
        {"chat_id": chat_id, "text": text, "parse_mode": "HTML"}
    ).encode()
    req = urllib.request.Request(
        f"https://api.telegram.org/bot{token}/sendMessage",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        resp.read()


def _monitor():
    global _MONITOR
    if _MONITOR is None:
        data_dir = os.path.join(os.path.dirname(__file__), "data", "gettex_quotes")
        _MONITOR = build_monitor(data_dir)
    return _MONITOR


def poll_once() -> dict:
    def work():
        return _monitor().poll()

    return run_job("german_crash", work) or {}


def tick() -> None:
    try:
        result = poll_once()
    except Exception as exc:
        logger.error("German crash poll failed: %s", exc)
        return
    chats = _chats()
    for alert in (result or {}).get("alerts") or []:
        html = format_alert(alert)
        for uid in chats:
            try:
                _send_html(uid, html)
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                logger.error("crash alert push to %s failed: %s", uid, exc)


def main() -> None:
    tz = pytz.timezone("Europe/Berlin")
    sched = BlockingScheduler(timezone=tz)
    kw = dict(max_instances=1, coalesce=True, misfire_grace_time=90)
    sched.add_job(tick, CronTrigger.from_crontab("30-59/2 7 * * mon-fri", timezone=tz),
                  id="german_crash_pre", name="German crash alert (7:30-7:58)", **kw)
    sched.add_job(tick, CronTrigger.from_crontab("*/2 8-22 * * mon-fri", timezone=tz),
                  id="german_crash", name="German crash alert (8:00-22:58)", **kw)
    sched.add_job(tick, CronTrigger.from_crontab("0 23 * * mon-fri", timezone=tz),
                  id="german_crash_close", name="German crash alert (23:00)", **kw)
    logger.info("crash_alert scheduler 07:30–23:00 Europe/Berlin; pid=%s", os.getpid())
    sched.start()


if __name__ == "__main__":
    main()
