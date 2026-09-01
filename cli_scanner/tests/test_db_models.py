"""Model definitions must match the production schema exactly."""
import shutil
import sqlite3

import pytest

from earnings_edge.db import engine as db_engine
from earnings_edge.db.models import Base

PROD_DB = db_engine.DEFAULT_DB_PATH


def _live_schema(db_file) -> dict:
    con = sqlite3.connect(db_file)
    try:
        out = {}
        tables = [r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name != 'sqlite_sequence'")]
        for t in tables:
            # Full tuple: (name, type, notnull, dflt_value, pk). dflt_value is
            # compared verbatim — the generator emits server_default=text(...) so
            # create_all renders the identical SQL literal ('pending', 0, etc.).
            out[t] = [(r[1], r[2].upper(), r[3], r[4], r[5]) for r in con.execute(f'PRAGMA table_info("{t}")')]
        return out
    finally:
        con.close()


@pytest.mark.skipif(not PROD_DB.exists(), reason="production db is not in CI")
def test_models_match_production_schema(tmp_path):
    prod_copy = tmp_path / "prod.db"
    shutil.copy(PROD_DB, prod_copy)
    fresh = tmp_path / "fresh.db"
    db_engine.configure(fresh)

    live = _live_schema(prod_copy)
    created = _live_schema(fresh)

    assert sorted(live) == sorted(created), (
        f"table mismatch: only-live={sorted(set(live) - set(created))} "
        f"only-created={sorted(set(created) - set(live))}")
    for table, cols in live.items():
        assert cols == created[table], f"column mismatch in {table}"
