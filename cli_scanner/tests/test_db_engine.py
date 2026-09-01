"""Tests for the centralized engine/session handler."""
import sqlalchemy
import pytest

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


def test_session_scope_commits():
    with db_engine.session_scope() as s:
        s.execute(sqlalchemy.text("CREATE TABLE t1 (id INTEGER PRIMARY KEY, v TEXT)"))
        s.execute(sqlalchemy.text("INSERT INTO t1 (v) VALUES (:v)"), {"v": "a"})
    with db_engine.get_session() as s:
        assert s.execute(sqlalchemy.text("SELECT v FROM t1")).scalar() == "a"


def test_session_scope_rolls_back_on_error():
    with pytest.raises(RuntimeError):
        with db_engine.session_scope() as s:
            s.execute(sqlalchemy.text("CREATE TABLE t2 (id INTEGER PRIMARY KEY, v TEXT)"))
            s.execute(sqlalchemy.text("INSERT INTO t2 (v) VALUES (:v)"), {"v": "b"})
            raise RuntimeError("boom")
    # table creation rolled back too -> querying it must fail
    with db_engine.get_session() as s:
        with pytest.raises(sqlalchemy.exc.SQLAlchemyError):
            s.execute(sqlalchemy.text("SELECT v FROM t2")).scalar()
