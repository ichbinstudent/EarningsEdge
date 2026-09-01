#!/usr/bin/env python3
"""Train option models for earnings-related trading strategies.

Trains three models:
  1. move_magnitude — regression head: predicts |actual_move_pct| (how big the stock move will be)
  2. move_direction — classification head: UP vs DOWN vs FLAT
  3. vol_edge — binary: True if |actual_move| < implied_move (premium is overpriced → sell)

Also trains the combined move-direction model and persists artifacts to data/models/.

Usage:
    cd ~/EarningsEdgeDetection/cli_scanner
    .venv/bin/python train_option_models.py --model gradient_boosting --target magnitude
    .venv/bin/python train_option_models.py --model logistic --target direction
    .venv/bin/python train_option_models.py --model gradient_boosting --target vol_edge
    .venv/bin/python train_option_models.py --train-all
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import logging
logger = logging.getLogger("option_models")
logging.basicConfig(level=logging.INFO)


# ---------------------------------------------------------------------------
# Feature set
# ---------------------------------------------------------------------------

FEATURE_COLUMNS: List[str] = [
    "price",           # underlying price at scan
    "avg_volume_30d",  # 30-day avg volume (filter for liquidity)
    "atm_iv_near",     # ATM implied vol (nearest expiry)
    "rv30",            # 30-day realized vol
    "iv30_rv30",       # IV/RV ratio (vol richness)
    "hist_vol_3m",     # 3-month historical vol
    "sigma_baseline_1y",   # 1-year baseline vol
    "sigma_short_leg",     # short-leg implied vol
    "sigma_short_leg_fair", # fair-value short-leg vol
    "actual_to_fair_ratio", # actual-to-fair vol ratio
    "term_slope",      # term-structure slope
    "term_structure_valid", # is term structure monotonic
]

# Targets
TARGETS = {
    "magnitude": "abs_actual_move",
    "direction": "direction_label",
    "vol_edge": "vol_edge_flag",
}


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract only feature columns and coerce dtypes."""
    cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    return df[cols].copy()


def prepare_dataset(snapshots: pd.DataFrame) -> pd.DataFrame:
    """From raw snapshot dataframe, build modeling-ready dataframe with features and targets.
    Only rows with outcomes are used.

    Adds:
      abs_actual_move   — |actual_move_pct| (magnitude target)
      direction_label   — encoded direction (UP=1, DOWN=-1, FLAT=0)
      vol_edge_flag     — 1 if |actual_move_pct| < expected_move_pct (premium overpriced), else 0
    """
    has_outcome = snapshots[snapshots["actual_move_pct"].notna()].copy()
    if has_outcome.empty:
        return pd.DataFrame()

    has_outcome["abs_actual_move"] = has_outcome["actual_move_pct"].abs()
    has_outcome["direction_label"] = has_outcome["actual_move_direction"].map(
        {"UP": 1, "DOWN": -1, "FLAT": 0}
    )

    # Vol edge: where expected_move_pct is available, compute edge
    has_outcome["vol_edge_flag"] = np.nan
    mask = has_outcome["expected_move_pct"].notna() & (has_outcome["expected_move_pct"] > 0)
    actual = has_outcome.loc[mask, "actual_move_pct"].abs()
    implied = has_outcome.loc[mask, "expected_move_pct"].abs()
    # Edge = 1 means actual < implied → premium was overpriced → selling was profitable
    has_outcome.loc[mask, "vol_edge_flag"] = (actual < implied).astype(int)

    return has_outcome


def _build_preprocessor() -> ColumnTransformer:
    """Shared preprocessing for all option-model pipelines (models and baselines)."""
    numeric_features = [c for c in FEATURE_COLUMNS if c != "term_structure_valid"]
    categorical_features = ["term_structure_valid"] if "term_structure_valid" in FEATURE_COLUMNS else []

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
        ] + ([("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features)]
           if categorical_features else []),
    )


def build_pipeline(model_type: str, target: str) -> Pipeline:
    """Build a sklearn Pipeline with preprocessing and estimator."""
    preprocessor = _build_preprocessor()

    if target == "magnitude":
        if model_type == "gradient_boosting":
            estimator = GradientBoostingRegressor(
                n_estimators=80, max_depth=3, min_samples_leaf=15, random_state=42
            )
        elif model_type == "ridge":
            estimator = Ridge(random_state=42)
        else:
            estimator = LinearRegression()
    elif target == "direction":
        if model_type == "gradient_boosting":
            estimator = GradientBoostingClassifier(
                n_estimators=80, max_depth=3, min_samples_leaf=15, random_state=42
            )
        else:
            estimator = LogisticRegression(max_iter=1000, random_state=42)
    elif target == "vol_edge":
        if model_type == "gradient_boosting":
            estimator = GradientBoostingClassifier(
                n_estimators=80, max_depth=3, min_samples_leaf=15, random_state=42
            )
        else:
            estimator = LogisticRegression(max_iter=1000, random_state=42)
    else:
        raise ValueError(f"Unknown target: {target}")

    return Pipeline(steps=[("preprocessor", preprocessor), ("estimator", estimator)])


def build_baseline_pipeline(target: str) -> Pipeline:
    """Build a Dummy baseline pipeline reusing the same preprocessor as build_pipeline.

    magnitude -> DummyRegressor(strategy='mean'); classification targets ->
    DummyClassifier(strategy='most_frequent'). Used by cv_sweep.py to screen
    models against a no-skill reference over identical CV folds.
    """
    if target == "magnitude":
        estimator = DummyRegressor(strategy="mean")
    elif target in ("direction", "vol_edge"):
        estimator = DummyClassifier(strategy="most_frequent")
    else:
        raise ValueError(f"Unknown target: {target}")

    return Pipeline(steps=[("preprocessor", _build_preprocessor()), ("estimator", estimator)])


def run_cv_folds(
    df_prepared: pd.DataFrame,
    pipeline_factory,
    target: str,
    n_splits: int = 5,
    model_type: str = "custom",
) -> Dict[str, Any]:
    """Time-series CV shared by cv_evaluate and the baseline sweep.

    pipeline_factory is a zero-arg callable returning a fresh, unfitted
    pipeline per fold. Sorting, target filtering, fold splitting, metric
    functions, and single-class NaN handling are identical to cv_evaluate.
    """
    if target not in TARGETS:
        raise ValueError(f"Unknown target: {target}")

    data = df_prepared.sort_values("earnings_date").reset_index(drop=True)
    y_col = TARGETS[target]
    if target == "vol_edge":
        data = data[data["vol_edge_flag"].notna()].copy()
        data["vol_edge_flag"] = data["vol_edge_flag"].astype(int)
    if target == "direction":
        data = data[data["direction_label"].notna()].copy()
        data["direction_label"] = data["direction_label"].astype(int)

    X = _build_features(data)
    y = data[y_col].values

    folds: List[Dict[str, Any]] = []
    tscv = TimeSeriesSplit(n_splits=n_splits)
    for fold, (tr_idx, te_idx) in enumerate(tscv.split(X), 1):
        X_train, X_test = X.iloc[tr_idx], X.iloc[te_idx]
        y_train, y_test = y[tr_idx], y[te_idx]

        pipeline = pipeline_factory()
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        metrics: Dict[str, float] = {}
        if target == "magnitude":
            metrics["mae"] = float(mean_absolute_error(y_test, y_pred))
            metrics["r2"] = float(r2_score(y_test, y_pred))
        elif len(np.unique(y_test)) < 2:
            # Single-class test segment: classification metrics undefined.
            metrics["accuracy"] = float("nan")
            metrics["f1"] = float("nan")
        else:
            metrics["accuracy"] = float(accuracy_score(y_test, y_pred))
            if target == "direction":
                metrics["f1"] = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
            else:
                metrics["f1"] = float(f1_score(y_test, y_pred, zero_division=0))

        folds.append({
            "fold": fold,
            "train_size": len(tr_idx),
            "test_size": len(te_idx),
            "metrics": metrics,
        })

    metric_names = list(folds[0]["metrics"].keys()) if folds else []
    mean = {k: float(np.nanmean([f["metrics"][k] for f in folds])) for k in metric_names}
    std = {k: float(np.nanstd([f["metrics"][k] for f in folds])) for k in metric_names}

    return {
        "target": target,
        "model_type": model_type,
        "n_splits": n_splits,
        "folds": folds,
        "mean": mean,
        "std": std,
    }


def cv_evaluate(
    df_prepared: pd.DataFrame,
    model_type: str,
    target: str,
    n_splits: int = 5,
) -> Dict[str, Any]:
    """Time-series cross-validation over a prepared dataset (no artifact persistence).

    Sorts by earnings_date (date-monotonic, matching the project's split
    convention) and evaluates build_pipeline(model_type, target) across
    sklearn TimeSeriesSplit folds. Classification folds whose test segment
    contains a single class record NaN metrics and are excluded from the mean.
    """
    return run_cv_folds(
        df_prepared,
        lambda: build_pipeline(model_type, target),
        target,
        n_splits=n_splits,
        model_type=model_type,
    )


def _load_snapshots(db_path: Optional[Path] = None) -> pd.DataFrame:
    from sqlalchemy import text

    from earnings_edge.db import configure, get_engine

    if db_path is not None:
        configure(db_path)
    return pd.read_sql(text("SELECT * FROM snapshots"), get_engine())


def train_target(
    target: str,
    model_type: str = "gradient_boosting",
    db_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Train a single model for the given target."""
    snapshots = _load_snapshots(db_path)

    data = prepare_dataset(snapshots)
    if data.empty:
        logger.error("No rows with outcomes found.")
        return {}

    y_col = TARGETS[target]
    # For vol_edge, drop rows where edge is undefined
    if target == "vol_edge":
        data = data[data["vol_edge_flag"].notna()].copy()
        data["vol_edge_flag"] = data["vol_edge_flag"].astype(int)
    if target == "direction":
        data = data[data["direction_label"].notna()].copy()
        data["direction_label"] = data["direction_label"].astype(int)

    if len(data) < 50:
        logger.error(f"Only {len(data)} usable rows for {target}; need >= 50.")
        return {}

    X = _build_features(data)
    y = data[y_col].values

    # Time-ordered split: 70% train, 30% test
    split_idx = int(len(data) * 0.7)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    pipeline = build_pipeline(model_type, target)
    pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = pipeline.predict(X_test)
    metrics: Dict[str, float] = {}

    if target == "magnitude":
        metrics["r2"] = float(r2_score(y_test, y_pred))
        metrics["mae"] = float(mean_absolute_error(y_test, y_pred))
    else:
        metrics["accuracy"] = float(accuracy_score(y_test, y_pred))
        metrics["f1_macro"] = float(f1_score(y_test, y_pred, average="macro", zero_division=0))

    model_name = f"option_model_{target}_{model_type}"
    model_dir = Path("data/models")
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{model_name}.joblib"
    meta_path = model_dir / f"{model_name}.json"

    joblib.dump(pipeline, model_path)

    metadata = {
        "model_name": model_name,
        "target": target,
        "model_type": model_type,
        "features": FEATURE_COLUMNS,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "metrics": metrics,
        "target_mean": float(np.mean(y)),
        "target_std": float(np.std(y)),
    }
    with open(meta_path, "w") as fh:
        json.dump(metadata, fh, indent=2)

    logger.info(f"Trained {model_name}: {metrics}")
    return {"model_path": str(model_path), "meta_path": str(meta_path), "metrics": metrics, "metadata": metadata}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gradient_boosting",
                        choices=["gradient_boosting", "linear", "ridge", "logistic"])
    parser.add_argument("--target", default="all",
                        choices=["magnitude", "direction", "vol_edge", "all"])
    parser.add_argument("--train-all", action="store_true",
                        help="Train all three models with default settings")
    parser.add_argument("--cv", action="store_true",
                        help="Run time-series cross-validation only; no artifacts are written")
    args = parser.parse_args(argv)

    if args.train_all:
        args.target = "all"

    if args.cv:
        snapshots = _load_snapshots(None)

        prepared = prepare_dataset(snapshots)
        if prepared.empty:
            logger.error("No rows with outcomes found.")
            return 1

        cv_targets = list(TARGETS) if args.target == "all" else [args.target]
        for tgt in cv_targets:
            result = cv_evaluate(prepared, args.model, tgt)
            print(f"\n=== CV {tgt} ({args.model}, {result['n_splits']} folds) ===")
            for f in result["folds"]:
                metric_str = "  ".join(f"{k}={v:.4f}" for k, v in f["metrics"].items())
                print(f"  fold {f['fold']}: {metric_str}  (train={f['train_size']}, test={f['test_size']})")
            summary = "  ".join(
                f"{k}={result['mean'][k]:.4f}+-{result['std'][k]:.4f}" for k in result["mean"]
            )
            print(f"  mean+-std: {summary}")
        return 0

    targets = ["magnitude", "direction"] if args.target == "all" else [args.target]

    for tgt in targets:
        logger.info(f"Training {tgt} model ({args.model})...")
        result = train_target(target=tgt, model_type=args.model)
        if not result:
            logger.error(f"Failed to train {tgt}")
            return 1
        print(f"\n=== {result['metadata']['model_name']} ===")
        print(f"  Train: {result['metadata']['train_size']}  Test: {result['metadata']['test_size']}")
        for k, v in result["metrics"].items():
            print(f"  {k}: {v:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
