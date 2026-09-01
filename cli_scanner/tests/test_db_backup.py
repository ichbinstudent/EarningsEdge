"""Hot-DB-safe backup: online backup API, never TRUNCATE the live WAL."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from earnings_edge.db import engine as db_engine
from framework.backup import backup_db

NOW = datetime(2026, 9, 1, 6, 15, tzinfo=timezone.utc)


def test_backup_invoke(tmp_path):
    src = tmp_path / "src.db"
    dest = tmp_path / "backups"
    db_engine.configure(src)
    with db_engine.session_scope() as s:
        from sqlalchemy import text as sa_text
        s.execute(sa_text("CREATE TABLE IF NOT EXISTS t (x int)"))
        s.execute(sa_text("INSERT INTO t VALUES (1)"))
    out = backup_db(src, dest, now=NOW)
    assert out.exists() and out.stat().st_size > 0
    assert out.parent == dest
    assert out.name == "earnings_ml_20260901T061500Z.db"


def test_backup_does_not_truncate_live_wal(tmp_path):
    src = tmp_path / "src.db"
    dest = tmp_path / "backups"
    db_engine.configure(src)
    with db_engine.session_scope() as s:
        from sqlalchemy import text as sa_text
        s.execute(sa_text("CREATE TABLE IF NOT EXISTS t (x int)"))
        s.execute(sa_text("INSERT INTO t VALUES (1)"))

    with patch("earnings_edge.db.engine.wal_checkpoint") as ckpt:
        # import path used by backup.py
        with patch("framework.backup.wal_checkpoint") as ckpt2:
            backup_db(src, dest, now=NOW)
            ckpt2.assert_called_once()
            _, kwargs = ckpt2.call_args
            assert kwargs.get("mode") == "PASSIVE"
            assert ckpt.call_count == 0  # engine helper not used directly


def test_wal_checkpoint_default_is_passive(tmp_path):
    src = tmp_path / "src.db"
    db_engine.configure(src)
    executed = []

    class FakeConn:
        def execute(self, sql, *a, **k):
            executed.append(str(sql))
            return MagicMock()

        def close(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    with patch("sqlite3.connect", return_value=FakeConn()):
        db_engine.wal_checkpoint(src)
    assert executed == ["PRAGMA wal_checkpoint(PASSIVE)"]
    assert all("TRUNCATE" not in s for s in executed)


def test_backup_rejects_failed_integrity(tmp_path):
    src = tmp_path / "src.db"
    dest = tmp_path / "backups"
    db_engine.configure(src)
    with db_engine.session_scope() as s:
        from sqlalchemy import text as sa_text
        s.execute(sa_text("CREATE TABLE IF NOT EXISTS t (x int)"))

    import sqlite3
    real_connect = sqlite3.connect
    n = {"i": 0}

    def connect_wrapper(*args, **kwargs):
        n["i"] += 1
        # 1=src ro, 2=tmp dest, 3=integrity check on tmp
        if n["i"] >= 3:
            m = MagicMock()
            m.execute.return_value.fetchone.return_value = ["error in table t"]
            return m
        return real_connect(*args, **kwargs)

    with patch("sqlite3.connect", side_effect=connect_wrapper):
        with pytest.raises(RuntimeError, match="integrity check failed"):
            backup_db(src, dest, now=NOW)
    assert list(dest.glob("earnings_ml_*.db")) == []
