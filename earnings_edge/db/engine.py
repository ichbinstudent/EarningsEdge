"""Centralized SQLAlchemy engine and session management.

Single SQLite database (WAL mode). Tests and CLI tools re-point the engine
with ``configure(path)``; production code uses the default path.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "earnings_ml.db"

_engine: Engine | None = None
_session_factory: sessionmaker | None = None
_lock = threading.Lock()


def _set_pragmas(dbapi_conn, _connection_record) -> None:
    # Take over transaction control from pysqlite: with the default
    # isolation_level, the driver never issues BEGIN before DDL, so
    # CREATE/ALTER autocommit and session.rollback() cannot undo them.
    dbapi_conn.isolation_level = None
    cur = dbapi_conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL")
    cur.execute("PRAGMA busy_timeout=30000")
    cur.execute("PRAGMA synchronous=NORMAL")
    cur.close()


def _begin(conn) -> None:
    # Emit BEGIN ourselves since pysqlite's implicit handling is disabled.
    conn.exec_driver_sql("BEGIN")


def configure(db_path: str | Path | None = None, read_only: bool = False) -> Engine:
    """(Re)create the engine bound to ``db_path`` (default: production path).

    Creates the directory, applies schema (create_all + column migrations),
    and resets the session factory. Safe to call repeatedly (tests).

    ``read_only=True`` opens the file in SQLite URI read-only mode (``mode=ro``):
    SQLite itself refuses every write, so off-bot processes (scripts, crons)
    can never mutate the production DB — the single-writer guarantee. The
    schema step is skipped for read-only engines (a ro connection cannot
    CREATE; the file must already exist and be migrated).
    """
    global _engine, _session_factory
    with _lock:
        path = Path(db_path) if db_path else DEFAULT_DB_PATH
        if _engine is not None:
            _engine.dispose()
        if read_only:
            if not path.exists():
                raise FileNotFoundError(f"read-only engine requested for missing DB: {path}")
            _engine = create_engine(
                f"sqlite+pysqlite:///file:{path}?mode=ro&uri=true",
                connect_args={"check_same_thread": False, "timeout": 30},
            )
            _session_factory = sessionmaker(bind=_engine, expire_on_commit=False)
            return _engine
        path.parent.mkdir(parents=True, exist_ok=True)
        _engine = create_engine(
            f"sqlite:///{path}",
            connect_args={"check_same_thread": False, "timeout": 30},
        )
        event.listen(_engine, "connect", _set_pragmas)
        event.listen(_engine, "begin", _begin)
        _session_factory = sessionmaker(bind=_engine, expire_on_commit=False)
        from .migrations import run_migrations
        from .models import Base

        Base.metadata.create_all(_engine)
        with _engine.begin() as conn:
            run_migrations(conn)
        return _engine


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        _engine = configure()
    return _engine


def get_session() -> Session:
    global _session_factory
    if _session_factory is None:
        configure()
    assert _session_factory is not None
    return _session_factory()


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional scope: commit on success, rollback on exception."""
    session = get_session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def wal_checkpoint(db_path: str | Path | None = None, mode: str = "PASSIVE") -> None:
    """PRAGMA wal_checkpoint(<mode>) against ``db_path`` (or the current engine file).

    Default is PASSIVE: never blocks writers, never truncates the WAL, safe to
    run against a hot/live database. TRUNCATE requires an exclusive lock and
    physically truncates the WAL file -- if interrupted mid-operation (e.g. a
    transient disk I/O error) it can leave the database corrupted. That is
    what happened on 2026-08-30/31; do not pass mode="TRUNCATE" against the
    live production DB. Only use TRUNCATE/RESTART on an already-stopped bot.
    """
    from sqlalchemy import text

    mode = mode.upper()
    assert mode in {"PASSIVE", "FULL", "RESTART", "TRUNCATE"}, mode

    if db_path is not None:
        import sqlite3

        path = Path(db_path)
        conn = sqlite3.connect(str(path), timeout=30)
        try:
            conn.execute(f"PRAGMA wal_checkpoint({mode})")
        finally:
            conn.close()
    else:
        # Use the existing SQLAlchemy engine to avoid dropping POSIX locks when closing an ad-hoc connection
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text(f"PRAGMA wal_checkpoint({mode})"))
