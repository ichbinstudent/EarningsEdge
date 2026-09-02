"""Live Telegram message primitives: progress animations + auto-updating monitor.

Telegram gives two "animation" channels: chat actions (typing…) which expire
after ~5s and must be re-sent, and message EDITS — a message whose text is
replaced every few seconds reads as a live animation. This module implements
both on top of pure text helpers (unit-tested; the telegram glue stays thin).

Usage:
    pm = ProgressMessage(bot, chat_id, "Building proposals")
    await pm.start()
    await pm.set_stage("running live signals")
    ...
    await pm.finish("✅ done")
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)

SPINNER = "⠋⠙⠹⠸⠼⠴⦷⦯⦿⦾"
SPARK_BLOCKS = "▁▂▃▄▅▆▇█"


# ---------------------------------------------------------------------------
# Pure text helpers (unit-tested)
# ---------------------------------------------------------------------------

def spinner_frame(tick: int) -> str:
    return SPINNER[tick % len(SPINNER)]


def fmt_duration(seconds: float) -> str:
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}m {s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m"


def sparkline(values: list[float], width: int = 16) -> str:
    """Unicode sparkline for a numeric series ('' when fewer than 2 points).

    Resamples to `width` buckets by taking every ceil(n/width)-th point.
    """
    vals = [float(v) for v in values if v is not None]
    if len(vals) < 2:
        return ""
    if len(vals) > width:
        step = max(1, len(vals) // width)
        vals = vals[::step][-width:]
    lo, hi = min(vals), max(vals)
    span = hi - lo
    if span <= 0:
        return SPARK_BLOCKS[3] * len(vals)
    return "".join(
        SPARK_BLOCKS[min(7, int((v - lo) / span * 7.999))] for v in vals
    )


def progress_text(title: str, stage: str, started: float, tick: int,
                  now: Optional[float] = None) -> str:
    """One frame of the animated progress message (plain text)."""
    elapsed = fmt_duration((now if now is not None else time.monotonic()) - started)
    bar = "▓" * (1 + tick % 3) + "░" * (3 - (1 + tick % 3))
    return (f"{spinner_frame(tick)} {title}\n"
            f"{bar} {stage}\n"
            f"⏱ {elapsed}")


# ---------------------------------------------------------------------------
# Telegram glue
# ---------------------------------------------------------------------------

class ProgressMessage:
    """A message that animates (spinner + stage + elapsed) until finished.

    Edits are throttled to `interval` seconds; Telegram edit failures
    (not-modified, transient rate limits) are swallowed — progress display
    must never break the underlying work.
    """

    def __init__(self, bot, chat_id: int, title: str, interval: float = 2.5):
        self._bot = bot
        self._chat_id = chat_id
        self._title = title
        self._interval = interval
        self._stage = "starting…"
        self._started = 0.0
        self._tick = 0
        self._msg = None
        self._task: Optional[asyncio.Task] = None
        self._done = False

    async def start(self) -> None:
        self._started = time.monotonic()
        self._msg = await self._bot.send_message(
            chat_id=self._chat_id, text=progress_text(self._title, self._stage, self._started, 0))
        self._task = asyncio.create_task(self._loop())

    async def attach(self, message) -> None:
        """Animate an existing Telegram message instead of sending a new one."""
        self._started = time.monotonic()
        self._msg = message
        try:
            await message.edit_text(
                progress_text(self._title, self._stage, self._started, 0))
        except Exception as exc:
            logger.debug("progress attach edit skipped: %s", exc)
        self._task = asyncio.create_task(self._loop())

    async def set_stage(self, stage: str) -> None:
        self._stage = stage

    async def finish(self, text: str, reply_markup=None) -> None:
        self._done = True
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
        if self._msg is not None:
            try:
                await self._msg.edit_text(text, reply_markup=reply_markup)
            except Exception as exc:
                logger.info("progress final edit failed (%s) — sending new message", exc)
                try:
                    await self._bot.send_message(
                        chat_id=self._chat_id, text=text, reply_markup=reply_markup)
                except Exception:
                    pass

    async def _loop(self) -> None:
        try:
            while not self._done:
                await asyncio.sleep(self._interval)
                self._tick += 1
                try:
                    await self._msg.edit_text(
                        progress_text(self._title, self._stage, self._started, self._tick))
                except Exception as exc:
                    # "Message is not modified" / transient — keep animating
                    logger.debug("progress edit skipped: %s", exc)
                try:
                    await self._bot.send_chat_action(chat_id=self._chat_id, action="typing")
                except Exception:
                    pass
        except asyncio.CancelledError:
            raise
