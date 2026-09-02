"""Tests for the database repository layer."""

import tempfile
import unittest
from pathlib import Path

from sqlalchemy import text

from earnings_edge.db import engine as db_engine
from earnings_edge.models import EarningsCandidate
from datetime import date


class TestDatabase(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = Path(self.tmpdir) / "test.db"
        db_engine.configure(self.db_path)

    def test_wal_mode_enabled(self):
        with db_engine.get_engine().connect() as conn:
            mode = conn.execute(text("PRAGMA journal_mode")).scalar()
        self.assertEqual(mode.lower(), "wal")

    def test_connection_sets_busy_timeout(self):
        with db_engine.get_engine().connect() as conn:
            timeout = conn.execute(text("PRAGMA busy_timeout")).scalar()
        self.assertGreater(timeout, 0)

    def test_scan_runs_table_exists(self):
        with db_engine.get_session() as s:
            tables = {r[0] for r in s.execute(
                text("SELECT name FROM sqlite_master WHERE type='table'")
            )}
        self.assertIn("scan_runs", tables)

    def test_insert_scan_run(self):
        from earnings_edge.db import insert_scan_run
        row_id = insert_scan_run({
            "scan_timestamp": "2026-06-17T12:00:00Z",
            "scanner_name": "Earnings Calendar",
            "trigger_type": "test",
            "candidate_count": 50,
            "tier1_count": 3,
            "tier2_count": 2,
            "take_count": 1,
            "duration_secs": 42.5,
            "success": 1,
        })
        with db_engine.get_session() as s:
            row = s.execute(
                text("SELECT * FROM scan_runs WHERE id = :id"), {"id": row_id}
            ).mappings().first()
        self.assertEqual(row["scanner_name"], "Earnings Calendar")
        self.assertEqual(row["candidate_count"], 50)

    def test_data_source_column_exists_on_snapshots(self):
        with db_engine.get_session() as s:
            cols = {r["name"] for r in s.execute(text("PRAGMA table_info(snapshots)")).mappings()}
        self.assertIn("data_source", cols)


class TestEarningsCandidateSource(unittest.TestCase):
    def test_candidate_carries_source(self):
        c = EarningsCandidate(ticker="AAPL", timing="Post Market", source="finnhub")
        self.assertEqual(c.source, "finnhub")

    def test_candidate_default_source(self):
        c = EarningsCandidate(ticker="AAPL", timing="Post Market")
        self.assertEqual(c.source, "unknown")


if __name__ == "__main__":
    unittest.main()
