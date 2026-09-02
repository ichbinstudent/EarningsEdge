import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from train_calendar_filter import (
    build_model_pipeline,
    cv_evaluate,
    is_regression_target,
    make_target,
)


def _synthetic_clean(n: int, pnl_dollars: list[float]) -> pd.DataFrame:
    """Build a synthetic gated dataframe (post apply_data_quality_gates shape)."""
    rng = np.random.default_rng(7)
    return pd.DataFrame(
        {
            "earnings_date": pd.date_range("2023-01-04", periods=n, freq="7D"),
            "pnl_dollars": pnl_dollars,
            "return_on_debit": [p / 100.0 for p in pnl_dollars],
            "net_debit": rng.uniform(0.5, 3.0, n),
            "debit_pct_price": rng.uniform(0.01, 0.05, n),
        }
    )


class CalendarFilterTrainingTests(unittest.TestCase):
    def test_expected_return_target_uses_continuous_return_on_debit(self):
        df = pd.DataFrame({"return_on_debit": [0.25, -0.10, 1.50]})

        target = make_target(df, "expected_return", min_pnl=10.0, min_return=0.10)

        self.assertEqual(target.tolist(), [0.25, -0.10, 1.50])

    def test_expected_return_uses_regression_model_with_predict_not_classifier(self):
        pipe = build_model_pipeline(["net_debit", "debit_pct_price"], target="expected_return", model_name="ridge", random_state=42)

        self.assertTrue(is_regression_target("expected_return"))
        self.assertTrue(hasattr(pipe.named_steps["model"], "predict"))
        self.assertFalse(hasattr(pipe.named_steps["model"], "predict_proba"))


class CvEvaluateTests(unittest.TestCase):
    def test_regression_cv_returns_finite_mae_and_r2_per_fold(self):
        n = 120
        rng = np.random.default_rng(11)
        clean = _synthetic_clean(n, rng.uniform(-150.0, 200.0, n).tolist())
        # Shuffle rows: cv_evaluate must re-sort by earnings_date defensively.
        clean = clean.sample(frac=1.0, random_state=3).reset_index(drop=True)

        result = cv_evaluate(
            clean,
            ["net_debit", "debit_pct_price"],
            target="expected_return",
            model_name="ridge",
            n_splits=4,
            random_state=42,
        )

        self.assertEqual(result["target"], "expected_return")
        self.assertEqual(result["model"], "ridge")
        self.assertEqual(result["n_splits"], 4)
        self.assertEqual(len(result["folds"]), 4)
        for i, fold in enumerate(result["folds"], 1):
            self.assertEqual(fold["fold"], i)
            self.assertGreater(fold["train_size"], 0)
            self.assertGreater(fold["test_size"], 0)
            self.assertIn("mae", fold["metrics"])
            self.assertIn("r2", fold["metrics"])
            self.assertTrue(math.isfinite(fold["metrics"]["mae"]))
            self.assertTrue(math.isfinite(fold["metrics"]["r2"]))
        self.assertTrue(math.isfinite(result["mean"]["mae"]))
        self.assertTrue(math.isfinite(result["mean"]["r2"]))
        self.assertTrue(math.isfinite(result["std"]["mae"]))
        self.assertTrue(math.isfinite(result["std"]["r2"]))

    def test_classification_cv_single_class_fold_yields_nan_metrics(self):
        # Sorted by earnings_date, TimeSeriesSplit(4) on 100 rows gives test
        # segments [20:40], [40:60], [60:80], [80:100]. Rows 20-39 are all
        # losses, so fold 1's test segment is single-class while every train
        # segment keeps both classes.
        wins = np.ones(100, dtype=bool)
        wins[1:20:2] = False
        wins[20:40] = False
        wins[41::2] = False
        pnl = np.where(wins, 120.0, -80.0).tolist()
        clean = _synthetic_clean(100, pnl)

        result = cv_evaluate(
            clean,
            ["net_debit", "debit_pct_price"],
            target="win",
            model_name="logistic",
            n_splits=4,
            random_state=42,
        )

        self.assertEqual(len(result["folds"]), 4)
        first = result["folds"][0]
        for key in ("accuracy", "f1", "auc"):
            self.assertTrue(math.isnan(first["metrics"][key]), f"fold 1 {key} should be NaN")
        for fold in result["folds"][1:]:
            for key in ("accuracy", "f1", "auc"):
                self.assertTrue(math.isfinite(fold["metrics"][key]))
        for key in ("accuracy", "f1", "auc"):
            expected_mean = float(
                np.nanmean([f["metrics"][key] for f in result["folds"]])
            )
            self.assertTrue(math.isfinite(result["mean"][key]))
            self.assertAlmostEqual(result["mean"][key], expected_mean)
            self.assertTrue(math.isfinite(result["std"][key]))


class CvCliTests(unittest.TestCase):
    def _write_test_db(self, db_path: Path, n: int = 60) -> None:
        from earnings_edge.db import calendar_call_trades_upsert, configure, insert_snapshot

        rng = np.random.default_rng(5)
        configure(db_path)
        base = pd.Timestamp("2023-01-04")
        for i in range(n):
            earnings_date = (base + pd.Timedelta(days=7 * i)).strftime("%Y-%m-%d")
            debit = float(rng.uniform(0.5, 3.0))
            ret = float(rng.uniform(-0.8, 1.2))
            exit_value = debit * (1.0 + ret)
            sid = insert_snapshot({
                "ticker": "SYN",
                "earnings_date": earnings_date,
                "scan_date": earnings_date,
                "price": 100.0,
                "avg_volume_30d": 1e6,
                "market_cap": 1e9,
                "has_options": 1,
                "days_to_expiry": 30,
                "total_open_interest": 5000.0,
                "atm_iv_near": 0.6,
                "rv30": 0.4,
                "iv30_rv30": 1.5,
                "hist_vol_3m": 0.45,
                "term_slope": 0.02,
                "term_structure_valid": 1,
                "expected_move_pct": 0.08,
                "expected_move_dollars": 8.0,
                "straddle_price": 5.0,
                "atm_call_delta": 0.5,
                "atm_put_delta": -0.5,
                "atm_call_iv": 0.6,
                "atm_put_iv": 0.6,
                "sigma_baseline_1y": 0.4,
                "sigma_short_leg": 0.6,
                "sigma_short_leg_fair": 0.5,
                "actual_to_fair_ratio": 1.2,
                "mc_win_rate": 0.5,
                "mc_quarters": 8,
                "data_source": f"t{i}",
            })
            calendar_call_trades_upsert({
                "snapshot_id": sid,
                "ticker": "SYN",
                "earnings_date": earnings_date,
                "scan_date": earnings_date,
                "near_expiry": "2023-01-20",
                "far_expiry": "2023-02-17",
                "strike": 100.0,
                "near_call_ticker": "O:SYNNEAR",
                "far_call_ticker": "O:SYNFAR",
                "near_entry": debit * 0.7,
                "far_entry": debit * 0.5,
                "near_exit": 0.1,
                "far_exit": exit_value,
                "net_debit": debit,
                "pnl_dollars": 100.0 * debit * ret,
                "return_on_debit": ret,
                "exit_value": exit_value,
            })

    def test_cv_cli_writes_no_artifacts(self):
        script = Path(__file__).resolve().parents[1] / "train_calendar_filter.py"
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            db_path = tmp_path / "test.db"
            self._write_test_db(db_path)
            output_path = tmp_path / "calendar_call_filter.joblib"

            proc = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--db",
                    str(db_path),
                    "--output",
                    str(output_path),
                    "--target",
                    "expected_return",
                    "--model",
                    "ridge",
                    "--min-rows",
                    "10",
                    "--cv-splits",
                    "3",
                    "--cv",
                ],
                cwd=tmp,
                capture_output=True,
                text=True,
            )

            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            result = json.loads(proc.stdout)
            self.assertEqual(result["target"], "expected_return")
            self.assertEqual(len(result["folds"]), 3)
            leftovers = [
                p for p in tmp_path.rglob("*") if p.suffix in {".joblib", ".json"}
            ]
            self.assertEqual(leftovers, [])


if __name__ == "__main__":
    unittest.main()
