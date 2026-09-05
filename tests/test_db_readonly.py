"""Single-writer discipline: read_only engine refuses every write.

The 2026-09-01 corruption came from concurrent writers (bot + recovery
scripts). configure(read_only=True) is the structural guard: SQLite itself
rejects writes in mode=ro, so an off-bot process physically cannot mutate
the production DB.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import text

from earnings_edge.db import engine as db_engine


@pytest.fixture
def seed_db(tmp_path: Path) -> Path:
    """Create a migrated DB with one row via the normal writable path."""
    path = tmp_path / "seeded.db"
    db_engine.configure(path)
    with db_engine.session_scope() as s:
        s.execute(
            text(
                "INSERT INTO job_runs (job_name, started_at, success, stats_json, error) "
                "VALUES ('probe', '2026-09-05T00:00:00', 1, '', '')"
            )
        )
    return path


def test_read_only_engine_refuses_writes(seed_db: Path) -> None:
    db_engine.configure(seed_db, read_only=True)
    with db_engine.session_scope() as s:
        rows = s.execute(text("SELECT COUNT(*) FROM job_runs")).scalar()
        assert rows == 1  # reads fine
    with pytest.raises(Exception) as excinfo:
        with db_engine.session_scope() as s:
            s.execute(
                text(
                    "INSERT INTO job_runs (job_name, started_at, success, stats_json, error) "
                    "VALUES ('evil', '2026-09-05T00:00:01', 0, '', '')"
                )
            )
    assert "readonly" in str(excinfo.value).lower() or "read-only" in str(excinfo.value).lower()


def test_read_only_engine_missing_db_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        db_engine.configure(tmp_path / "missing.db", read_only=True)


def test_writable_default_unchanged(seed_db: Path) -> None:
    """Default configure() still writes (the bot's own path)."""
    db_engine.configure(seed_db)
    with db_engine.session_scope() as s:
        s.execute(
            text(
                "INSERT INTO job_runs (job_name, started_at, success, stats_json, error) "
                "VALUES ('normal', '2026-09-05T00:00:02', 1, '', '')"
            )
        )
    db_engine.configure(seed_db, read_only=True)
    with db_engine.session_scope() as s:
        rows = s.execute(text("SELECT COUNT(*) FROM job_runs WHERE job_name IN ('probe', 'normal')")).scalar()
        assert rows == 2
