import tempfile
import unittest
from pathlib import Path

from sqlalchemy import text

from calendar_call_backtest import ensure_schema, score_existing_trades
from earnings_edge.db import (
    calendar_call_trades_list,
    calendar_call_trades_upsert,
    configure,
    get_engine,
    insert_snapshot,
)


class FixedPipeline:
    def predict_proba(self, frame):
        return [[0.2, 0.8] for _ in range(len(frame))]


class CalendarBacktestModelScoringTests(unittest.TestCase):
    def test_ensure_schema_adds_model_score_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            configure(Path(tmp) / "t.db")
            ensure_schema()
            with get_engine().connect() as conn:
                columns = {row[1] for row in conn.execute(text("pragma table_info(calendar_call_trades)"))}

        self.assertIn("model_score", columns)
        self.assertIn("model_recommendation", columns)
        self.assertIn("model_reason", columns)
        self.assertIn("model_name", columns)
        self.assertIn("model_scored_at", columns)

    def test_score_existing_trades_updates_clean_rows_and_rejects_bad_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            configure(Path(tmp) / "t.db")
            ensure_schema()
            sid1 = insert_snapshot({
                "ticker": "AAA",
                "earnings_date": "2025-01-15",
                "scan_date": "2025-01-14",
                "price": 100,
                "avg_volume_30d": 1000000,
                "has_options": 1,
                "days_to_expiry": 2,
                "atm_iv_near": 0.8,
                "data_source": "t1",
            })
            sid2 = insert_snapshot({
                "ticker": "AAA",
                "earnings_date": "2025-01-15",
                "scan_date": "2025-01-14",
                "price": 100,
                "avg_volume_30d": 1000000,
                "has_options": 1,
                "days_to_expiry": 2,
                "atm_iv_near": 0.8,
                "data_source": "t2",
            })
            self.assertTrue(sid1)
            self.assertTrue(sid2)
            self.assertNotEqual(sid1, sid2)
            base = {
                "ticker": "AAA",
                "earnings_date": "2025-01-15",
                "scan_date": "2025-01-14",
                "near_expiry": "2025-01-17",
                "far_expiry": "2025-01-24",
                "near_call_ticker": "O:AAA250117C00100000",
                "far_call_ticker": "O:AAA250124C00100000",
                "near_entry": 2.0,
                "far_entry": 3.0,
                "near_exit": 0.5,
                "far_exit": 2.0,
                "net_debit": 1.0,
                "exit_value": 1.5,
                "pnl_dollars": 50.0,
                "return_on_debit": 0.5,
            }
            calendar_call_trades_upsert({**base, "snapshot_id": sid1, "strike": 100.0})
            calendar_call_trades_upsert({**base, "snapshot_id": sid2, "strike": 400.0})
            artifact = {
                "pipeline": FixedPipeline(),
                "features": ["price", "strike", "net_debit", "moneyness"],
                "target": "min_pnl",
            }

            summary = score_existing_trades(artifact, model_name="unit", threshold=0.55)

            self.assertEqual(summary, {"scored": 1, "rejected": 1})
            rows = sorted(calendar_call_trades_list(), key=lambda r: r["snapshot_id"])
            self.assertAlmostEqual(rows[0]["model_score"], 0.8)
            self.assertEqual(rows[0]["model_recommendation"], 1)
            self.assertEqual(rows[0]["model_reason"], "model_score>=0.55")
            self.assertIsNone(rows[1]["model_score"])
            self.assertEqual(rows[1]["model_recommendation"], 0)
            self.assertEqual(rows[1]["model_reason"], "bad_moneyness")


if __name__ == "__main__":
    unittest.main()
