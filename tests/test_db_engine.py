"""Tests for the centralized engine/session handler."""

import pytest
import sqlalchemy

from earnings_edge.db import engine as db_engine


@pytest.fixture(autouse=True)
def fresh_engine(tmp_path):
    eng = db_engine.configure(tmp_path / "t.db")
    yield eng
    db_engine.configure(tmp_path / "reset.db")  # re-point so sessions don't leak


def test_configure_creates_wal_engine(tmp_path):
    eng = db_engine.configure(tmp_path / "w.db")
    with eng.connect() as conn:
        mode = conn.execute(sqlalchemy.text("PRAGMA journal_mode")).scalar()
        busy = conn.execute(sqlalchemy.text("PRAGMA busy_timeout")).scalar()
    assert mode.lower() == "wal"
    assert int(busy) == 30000


def test_migration_dedupes_open_managed_positions(tmp_path):
    from earnings_edge.db.migrations import run_migrations

    eng = db_engine.configure(tmp_path / "dup.db")
    with eng.begin() as conn:
        conn.execute(sqlalchemy.text("DROP INDEX IF EXISTS idx_uq_managed_positions_open_symbol"))
        conn.execute(
            sqlalchemy.text(
                "INSERT INTO managed_positions "
                "(symbol, strategy, group_id, qty, entry_price, status, opened_at) VALUES "
                "('PLAY260918C00008000', 'ff_ladder', 'g1', 1, 0.75, 'open', '2026-09-11T18:30:00.500'), "
                "('PLAY260918C00008000', 'ff_ladder', 'g1', 1, 0.35, 'open', '2026-09-11T18:30:00.942'), "
                "('PLAY261016C00008000', 'ff_ladder', 'g1', 1, 0.75, 'open', '2026-09-11T18:30:00.500'), "
                "('PLAY261016C00008000', 'ff_ladder', 'g1', 1, 0.35, 'open', '2026-09-11T18:30:00.942')"
            )
        )
    with eng.begin() as conn:
        run_migrations(conn)
    with eng.connect() as conn:
        open_rows = conn.execute(
            sqlalchemy.text(
                "SELECT symbol, entry_price FROM managed_positions WHERE status = 'open' ORDER BY symbol"
            )
        ).fetchall()
        idx = conn.execute(
            sqlalchemy.text(
                "SELECT name FROM sqlite_master WHERE type='index' "
                "AND name='idx_uq_managed_positions_open_symbol'"
            )
        ).scalar()
    assert [(r[0], r[1]) for r in open_rows] == [
        ("PLAY260918C00008000", 0.35),
        ("PLAY261016C00008000", 0.35),
    ]
    assert idx == "idx_uq_managed_positions_open_symbol"


def test_session_scope_commits():
    with db_engine.session_scope() as s:
        s.execute(sqlalchemy.text("CREATE TABLE t1 (id INTEGER PRIMARY KEY, v TEXT)"))
        s.execute(sqlalchemy.text("INSERT INTO t1 (v) VALUES (:v)"), {"v": "a"})
    with db_engine.get_session() as s:
        assert s.execute(sqlalchemy.text("SELECT v FROM t1")).scalar() == "a"


def test_session_scope_rolls_back_on_error():
    with pytest.raises(RuntimeError), db_engine.session_scope() as s:
        s.execute(sqlalchemy.text("CREATE TABLE t2 (id INTEGER PRIMARY KEY, v TEXT)"))
        s.execute(sqlalchemy.text("INSERT INTO t2 (v) VALUES (:v)"), {"v": "b"})
        raise RuntimeError("boom")
    # table creation rolled back too -> querying it must fail
    with db_engine.get_session() as s, pytest.raises(sqlalchemy.exc.SQLAlchemyError):
        s.execute(sqlalchemy.text("SELECT v FROM t2")).scalar()
