"""Process-level ops: single-instance lock and log redaction."""
from __future__ import annotations

import atexit
import logging
import os
import re
from pathlib import Path
from typing import Optional

try:
    import fcntl
except ImportError:  # pragma: no cover — Windows
    fcntl = None  # type: ignore


DEFAULT_LOCK = Path(__file__).resolve().parent.parent / "data" / "trading-bot.lock"

_TOKEN_RE = re.compile(r"bot\d+:[A-Za-z0-9_-]+")
_KEY_RE = re.compile(r"(APCA-API-(?:KEY-ID|SECRET-KEY)|token=)([^\s&]+)", re.I)


class SecretRedactFilter(logging.Filter):
    """Strip Telegram bot tokens and API key query/header crumbs from log lines."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
        except Exception:
            return True
        redacted = _TOKEN_RE.sub("bot<redacted>", msg)
        redacted = _KEY_RE.sub(r"\1<redacted>", redacted)
        if redacted != msg:
            record.msg = redacted
            record.args = ()
        return True


def install_secret_redaction(logger: Optional[logging.Logger] = None) -> None:
    filt = SecretRedactFilter()
    target = logger or logging.getLogger()
    target.addFilter(filt)
    for handler in target.handlers:
        handler.addFilter(filt)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


class InstanceLock:
    """Exclusive flock on ``data/trading-bot.lock``. Second process raises."""

    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else DEFAULT_LOCK
        self._fh = None

    def acquire(self) -> "InstanceLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a+")
        if fcntl is None:
            return self
        try:
            fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self._fh.close()
            self._fh = None
            raise RuntimeError(
                f"another trading-bot instance holds {self.path} — refusing to double-poll"
            ) from exc
        self._fh.seek(0)
        self._fh.truncate()
        self._fh.write(str(os.getpid()))
        self._fh.flush()
        atexit.register(self.release)
        return self

    def release(self) -> None:
        if self._fh is None:
            return
        try:
            if fcntl is not None:
                fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
        finally:
            try:
                self._fh.close()
            except Exception:
                pass
            self._fh = None
