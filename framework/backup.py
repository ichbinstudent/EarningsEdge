"""Nightly SQLite backup via the online backup API (hot-DB safe)."""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime
from pathlib import Path

from earnings_edge.db.engine import DEFAULT_DB_PATH, wal_checkpoint

DEFAULT_SRC = DEFAULT_DB_PATH
DEFAULT_DEST = Path(__file__).resolve().parent.parent / "data" / "backups"
# One copy is enough for a thin disk; override with BACKUP_KEEP.
DEFAULT_KEEP = 1

logger = logging.getLogger("framework.backup")


def keep_count(override: int | None = None) -> int:
    if override is not None:
        return max(0, int(override))
    raw = os.environ.get("BACKUP_KEEP", str(DEFAULT_KEEP))
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return DEFAULT_KEEP


def _unlink_sidecars(tmp_dest: Path) -> None:
    """sqlite3 leaves -wal/-shm/-journal next to the .tmp file."""
    tmp_dest.unlink(missing_ok=True)
    for suffix in ("-wal", "-shm", "-journal"):
        tmp_dest.with_name(tmp_dest.name + suffix).unlink(missing_ok=True)


def prune_backups(dest_dir: Path, keep: int | None = None) -> list[Path]:
    """Keep the ``keep`` newest ``earnings_ml_*.db`` files; drop the rest and leftover tmps."""
    dest_dir = Path(dest_dir)
    if not dest_dir.exists():
        return []
    removed: list[Path] = []
    for p in dest_dir.glob(".earnings_ml_*.tmp*"):
        p.unlink(missing_ok=True)
        removed.append(p)
    keep_n = keep_count(keep)
    completed = sorted(
        (p for p in dest_dir.glob("earnings_ml_*.db") if p.is_file()),
        key=lambda p: p.name,
        reverse=True,
    )
    for p in completed[keep_n:]:
        p.unlink(missing_ok=True)
        removed.append(p)
    if removed:
        logger.info("pruned %d backup file(s); keeping %d", len(removed), keep_n)
    return removed


def backup_db(
    src: Path | None = None,
    dest_dir: Path | None = None,
    *,
    now: datetime | None = None,
    keep: int | None = None,
) -> Path:
    """Online-backup ``src`` (live, hot DB safe) to ``dest_dir/earnings_ml_YYYYMMDDTHHMMSSZ.db``.

    Uses sqlite3's online backup API (Connection.backup) instead of a raw file
    copy after WAL checkpointing. TRUNCATE-mode checkpointing against a hot,
    actively-written database requires an exclusive lock and can leave the
    WAL/db header inconsistent if interrupted (e.g. transient disk I/O error),
    which is exactly what corrupted this DB on 2026-08-30/31. The backup API
    reads consistent pages under a shared lock and never truncates the live
    WAL, so a live scanner process keeps running safely throughout.

    After a successful copy, older backups are pruned so only ``keep``
    completed files remain (default 1; ``BACKUP_KEEP`` env).
    """
    import sqlite3

    src = Path(src) if src else DEFAULT_SRC
    dest_dir = Path(dest_dir) if dest_dir else DEFAULT_DEST
    if not src.exists():
        raise FileNotFoundError(src)
    dest_dir.mkdir(parents=True, exist_ok=True)
    now = now or datetime.now(UTC)
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    dest = dest_dir / f"earnings_ml_{stamp}.db"
    tmp_dest = dest_dir / f".{dest.name}.tmp"

    src_conn = sqlite3.connect(f"file:{src}?mode=ro", uri=True, timeout=30)
    try:
        dst_conn = sqlite3.connect(str(tmp_dest), timeout=30)
        try:
            src_conn.backup(dst_conn)
        finally:
            dst_conn.close()
    finally:
        src_conn.close()

    check_conn = sqlite3.connect(f"file:{tmp_dest}?mode=ro", uri=True)
    try:
        check = check_conn.execute("PRAGMA integrity_check").fetchone()[0]
    finally:
        check_conn.close()
    if check.lower() != "ok":
        _unlink_sidecars(tmp_dest)
        raise RuntimeError(f"Database integrity check failed on backup copy: {check}")

    tmp_dest.rename(dest)
    _unlink_sidecars(tmp_dest)  # leftover -wal/-shm from the tmp name

    # Best-effort passive checkpoint on the live DB to keep the WAL from
    # growing unbounded. PASSIVE never blocks writers and never truncates,
    # so it cannot corrupt the live file the way TRUNCATE did.
    try:
        wal_checkpoint(src, mode="PASSIVE")
    except Exception:
        pass

    prune_backups(dest_dir, keep=keep_count(keep))
    return dest
