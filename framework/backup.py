"""Nightly SQLite backup via the online backup API (hot-DB safe)."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from earnings_edge.db.engine import DEFAULT_DB_PATH, wal_checkpoint

DEFAULT_SRC = DEFAULT_DB_PATH
DEFAULT_DEST = Path(__file__).resolve().parent.parent / "data" / "backups"


def backup_db(
    src: Optional[Path] = None,
    dest_dir: Optional[Path] = None,
    *,
    now: Optional[datetime] = None,
) -> Path:
    """Online-backup ``src`` (live, hot DB safe) to ``dest_dir/earnings_ml_YYYYMMDDTHHMMSSZ.db``.

    Uses sqlite3's online backup API (Connection.backup) instead of a raw file
    copy after WAL checkpointing. TRUNCATE-mode checkpointing against a hot,
    actively-written database requires an exclusive lock and can leave the
    WAL/db header inconsistent if interrupted (e.g. transient disk I/O error),
    which is exactly what corrupted this DB on 2026-08-30/31. The backup API
    reads consistent pages under a shared lock and never truncates the live
    WAL, so a live scanner process keeps running safely throughout.
    """
    import sqlite3

    src = Path(src) if src else DEFAULT_SRC
    dest_dir = Path(dest_dir) if dest_dir else DEFAULT_DEST
    if not src.exists():
        raise FileNotFoundError(src)
    dest_dir.mkdir(parents=True, exist_ok=True)
    now = now or datetime.now(timezone.utc)
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
        tmp_dest.unlink(missing_ok=True)
        raise RuntimeError(f"Database integrity check failed on backup copy: {check}")

    tmp_dest.rename(dest)

    # Best-effort passive checkpoint on the live DB to keep the WAL from
    # growing unbounded. PASSIVE never blocks writers and never truncates,
    # so it cannot corrupt the live file the way TRUNCATE did.
    try:
        wal_checkpoint(src, mode="PASSIVE")
    except Exception:
        pass

    return dest
