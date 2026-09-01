#!/usr/bin/env python3
"""Cross-validation screening harness for the option models.

For each (target, model_type) combo, runs the existing cv_evaluate() from
train_option_models.py AND an identical time-series CV over a Dummy baseline
(same folds, same metric functions, same NaN handling for single-class test
folds), then reports model vs baseline mean+-std and the delta per metric.

A combo is flagged SIGNAL when the model beats the baseline by margin:
  - magnitude: r2 delta > 0.02
  - direction/vol_edge: accuracy AND f1 delta > 0.02

Read-only: opens the snapshots DB in SQLite read-only mode, never writes to
the DB, never persists model artifacts.

Usage:
    cd ~/EarningsEdgeDetection/cli_scanner
    .venv/bin/python cv_sweep.py
    .venv/bin/python cv_sweep.py --targets magnitude --models linear ridge
    .venv/bin/python cv_sweep.py --n-splits 5 --output data/cv_sweep_results.json
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from train_option_models import (
    TARGETS,
    build_baseline_pipeline,
    build_pipeline,
    prepare_dataset,
    run_cv_folds,
)

logger = logging.getLogger("cv_sweep")
logging.basicConfig(level=logging.INFO)

# Sensible per-target defaults: linear/ridge for magnitude, logistic for
# classification, plus gradient_boosting for all.
DEFAULT_MODELS: Dict[str, List[str]] = {
    "magnitude": ["linear", "ridge", "gradient_boosting"],
    "direction": ["logistic", "gradient_boosting"],
    "vol_edge": ["logistic", "gradient_boosting"],
}

ALLOWED_MODELS: Dict[str, List[str]] = {
    "magnitude": ["linear", "ridge", "gradient_boosting"],
    "direction": ["logistic", "gradient_boosting"],
    "vol_edge": ["logistic", "gradient_boosting"],
}

SIGNAL_MARGIN = 0.02

BASELINE_LABELS = {
    "magnitude": "dummy_mean",
    "direction": "dummy_most_frequent",
    "vol_edge": "dummy_most_frequent",
}


def compare_results(
    model_result: Dict[str, Any],
    baseline_result: Dict[str, Any],
) -> Dict[str, Dict[str, float]]:
    """Per-metric comparison of a model CV result against a baseline CV result."""
    comparison: Dict[str, Dict[str, float]] = {}
    for metric, model_mean in model_result["mean"].items():
        baseline_mean = baseline_result["mean"][metric]
        comparison[metric] = {
            "model_mean": model_mean,
            "model_std": model_result["std"][metric],
            "baseline_mean": baseline_mean,
            "baseline_std": baseline_result["std"][metric],
            "delta": model_mean - baseline_mean,
        }
    return comparison


def is_signal(
    comparison: Dict[str, Dict[str, float]],
    target: str,
    margin: float = SIGNAL_MARGIN,
) -> bool:
    """True when the model beats the baseline by margin on the decisive metric(s)."""
    if target == "magnitude":
        deltas = [comparison["r2"]["delta"]]
    else:
        deltas = [comparison["accuracy"]["delta"], comparison["f1"]["delta"]]
    if any(math.isnan(d) for d in deltas):
        return False
    return all(d > margin for d in deltas)


def run_sweep(
    prepared: pd.DataFrame,
    targets: Optional[List[str]] = None,
    models: Optional[Dict[str, List[str]]] = None,
    n_splits: int = 5,
) -> List[Dict[str, Any]]:
    """Run model + baseline CV for every (target, model_type) combo."""
    targets = targets or list(TARGETS)
    models = models or DEFAULT_MODELS

    rows: List[Dict[str, Any]] = []
    for target in targets:
        for model_type in models[target]:
            model_result = run_cv_folds(
                prepared,
                lambda mt=model_type, tgt=target: build_pipeline(mt, tgt),
                target,
                n_splits=n_splits,
                model_type=model_type,
            )
            baseline_result = run_cv_folds(
                prepared,
                lambda tgt=target: build_baseline_pipeline(tgt),
                target,
                n_splits=n_splits,
                model_type=BASELINE_LABELS[target],
            )
            comparison = compare_results(model_result, baseline_result)
            rows.append({
                "target": target,
                "model_type": model_type,
                "n_splits": n_splits,
                "model": model_result,
                "baseline": baseline_result,
                "comparison": comparison,
                "signal": is_signal(comparison, target),
            })
    return rows


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def format_markdown(rows: List[Dict[str, Any]]) -> str:
    """Render sweep rows as a markdown table (one line per combo and metric)."""
    lines = [
        "| Target | Model | Metric | Model mean±std | Baseline mean±std | Δ | Signal |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        for metric, comp in row["comparison"].items():
            model_cell = f"{_fmt(comp['model_mean'])}±{_fmt(comp['model_std'])}"
            baseline_cell = f"{_fmt(comp['baseline_mean'])}±{_fmt(comp['baseline_std'])}"
            lines.append(
                f"| {row['target']} | {row['model_type']} | {metric} "
                f"| {model_cell} | {baseline_cell} | {comp['delta']:+.4f} "
                f"| {'SIGNAL' if row['signal'] else ''} |"
            )
    return "\n".join(lines)


def load_prepared(db_path: Optional[Path] = None) -> pd.DataFrame:
    """Load snapshots from the DB and prepare the dataset."""
    from sqlalchemy import text

    from earnings_edge.db import DEFAULT_DB_PATH, configure, get_engine

    configure(db_path or DEFAULT_DB_PATH)
    snapshots = pd.read_sql(text("SELECT * FROM snapshots"), get_engine())
    return prepare_dataset(snapshots)


def _json_safe(obj: Any) -> Any:
    """Recursively convert NaN floats to None so output is valid JSON."""
    if isinstance(obj, float) and math.isnan(obj):
        return None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    return obj


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", nargs="+", default=list(TARGETS),
                        choices=list(TARGETS),
                        help="Targets to sweep (default: all three)")
    parser.add_argument("--models", nargs="+", default=None,
                        choices=["linear", "ridge", "logistic", "gradient_boosting"],
                        help="Model types applied to every selected target "
                             "(default: sensible per-target list)")
    parser.add_argument("--n-splits", type=int, default=5,
                        help="Number of TimeSeriesSplit folds (default: 5)")
    parser.add_argument("--output", default=None,
                        help="Path to write JSON results (default: no file)")
    args = parser.parse_args(argv)

    if args.models is None:
        models = DEFAULT_MODELS
    else:
        for target in args.targets:
            invalid = [m for m in args.models if m not in ALLOWED_MODELS[target]]
            if invalid:
                parser.error(f"models {invalid} not valid for target '{target}'; "
                             f"allowed: {ALLOWED_MODELS[target]}")
        models = {target: list(args.models) for target in args.targets}

    prepared = load_prepared()
    if prepared.empty:
        logger.error("No rows with outcomes found.")
        return 1

    rows = run_sweep(prepared, targets=args.targets, models=models, n_splits=args.n_splits)

    print(format_markdown(rows))
    n_signal = sum(1 for r in rows if r["signal"])
    print(f"\n{n_signal}/{len(rows)} combos flagged SIGNAL (margin={SIGNAL_MARGIN})")

    if args.output:
        payload = {
            "targets": args.targets,
            "n_splits": args.n_splits,
            "signal_margin": SIGNAL_MARGIN,
            "results": rows,
        }
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as fh:
            json.dump(_json_safe(payload), fh, indent=2)
        logger.info(f"Wrote {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
