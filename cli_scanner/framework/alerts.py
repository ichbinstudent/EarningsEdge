"""De-duped exception alerts for the approval chat only."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

DEFAULT_WINDOW_S = 15 * 60

KEYS = (
    "kill_switch",
    "alpaca_401",
    "scan_fail",
    "clock_dns",
    "remaining_leg_exhaust",
    "orphan",
    "missing",
    "daily_loss",
)


class AlertDeduper:
    """Emit each key at most once per window. ``now`` injectable.

    ``emit`` also queues the message on ``_outbox`` so the bot can
    ``drain()`` and push to the approval chat after sync work.
    """

    def __init__(self, window_s: int = DEFAULT_WINDOW_S):
        self.window = timedelta(seconds=window_s)
        self._last: dict[str, datetime] = {}
        self._outbox: list[str] = []

    def should_emit(self, key: str, now: Optional[datetime] = None) -> bool:
        now = now or datetime.now(timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        prev = self._last.get(key)
        if prev is not None and now - prev < self.window:
            return False
        self._last[key] = now
        return True

    def emit(self, key: str, message: str, now: Optional[datetime] = None) -> Optional[str]:
        """Return ``message`` if it should be sent, else None."""
        if self.should_emit(key, now):
            self._outbox.append(message)
            return message
        return None

    def drain(self) -> list[str]:
        """Return and clear queued messages since the last drain."""
        out = list(self._outbox)
        self._outbox.clear()
        return out

    def reset(self) -> None:
        self._last.clear()
        self._outbox.clear()


# Process-wide default used by the bot.
DEDUPER = AlertDeduper()


def is_alpaca_401(exc: BaseException) -> bool:
    if getattr(exc, "status_code", None) == 401:
        return True
    if type(exc).__name__ == "AlpacaAuthError":
        return True
    text = str(exc).lower()
    return "401" in text or "invalid api" in text


def emit_clock_failure(exc: BaseException) -> Optional[str]:
    """Clock/DNS vs 401 at get_clock sites."""
    if is_alpaca_401(exc):
        return DEDUPER.emit(
            "alpaca_401",
            f"🛑 Alpaca 401 — invalid API keys ({exc})",
        )
    return DEDUPER.emit("clock_dns", f"⚠️ Clock check failed: {exc}")


def emit_broker_failure(exc: BaseException) -> Optional[str]:
    """401 on get_positions / other broker calls (non-401 is silent here)."""
    if is_alpaca_401(exc):
        return DEDUPER.emit(
            "alpaca_401",
            f"🛑 Alpaca 401 — invalid API keys ({exc})",
        )
    return None
