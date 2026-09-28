"""Process-wide lock for local-book mutations.

Fill bookkeeping, reconcile-adopt, and exit closes can overlap on the
APScheduler thread pool (same clock minute). A reentrant lock keeps those
writers from racing even when unique indexes already reject duplicates.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Iterator

_LOCK = threading.RLock()


@contextmanager
def book_lock() -> Iterator[None]:
    with _LOCK:
        yield
