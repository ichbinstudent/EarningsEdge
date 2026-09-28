"""Hot-DB-safe backup: online backup API, never TRUNCATE the live WAL."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest

from earnings_edge.db import engine as db_engine
from framework.backup import backup_db, prune_backups

NOW = datetime(2026, 9, 1, 6, 15, tzinfo=UTC)


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


def test_prune_backups_keeps_newest_and_drops_tmps(tmp_path):
    dest = tmp_path / "backups"
    dest.mkdir()
    names = [
        "earnings_ml_20260901T061500Z.db",
        "earnings_ml_20260902T061500Z.db",
        "earnings_ml_20260903T061500Z.db",
    ]
    for name in names:
        (dest / name).write_bytes(b"x")
    (dest / ".earnings_ml_20260902T061500Z.db.tmp-wal").write_bytes(b"")
    (dest / ".earnings_ml_20260902T061500Z.db.tmp-shm").write_bytes(b"")
    (dest / ".earnings_ml_20260919T041500Z.db.tmp").write_bytes(b"partial")
    prune_backups(dest, keep=1)
    left = sorted(p.name for p in dest.iterdir())
    assert left == ["earnings_ml_20260903T061500Z.db"]


def test_backup_db_prunes_to_keep(tmp_path):
    src = tmp_path / "src.db"
    dest = tmp_path / "backups"
    db_engine.configure(src)
    with db_engine.session_scope() as s:
        from sqlalchemy import text as sa_text

        s.execute(sa_text("CREATE TABLE IF NOT EXISTS t (x int)"))
        s.execute(sa_text("INSERT INTO t VALUES (1)"))
    first = backup_db(src, dest, now=datetime(2026, 9, 1, 6, 15, tzinfo=UTC), keep=1)
    second = backup_db(src, dest, now=datetime(2026, 9, 2, 6, 15, tzinfo=UTC), keep=1)
    left = sorted(p.name for p in dest.glob("earnings_ml_*.db"))
    assert left == [second.name]
    assert not first.exists()
    assert not list(dest.glob(".*.tmp*"))
