"""Unit tests for cv_sweep (synthetic dataframes, no DB, no network)."""

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor

from train_option_models import FEATURE_COLUMNS, build_baseline_pipeline
from cv_sweep import (
    DEFAULT_MODELS,
    compare_results,
    format_markdown,
    is_signal,
    run_sweep,
)


def _make_prepared_df(n: int = 60, vol_edge_flags=None) -> pd.DataFrame:
    """Small synthetic prepared dataframe (output shape of prepare_dataset)."""
    rng = np.random.default_rng(0)
    data = {col: rng.normal(size=n) for col in FEATURE_COLUMNS}
    data["term_structure_valid"] = rng.integers(0, 2, size=n)
    df = pd.DataFrame(data)
    df["earnings_date"] = pd.date_range("2024-01-01", periods=n, freq="D")
    df["abs_actual_move"] = np.abs(rng.normal(size=n))
    df["direction_label"] = rng.integers(-1, 2, size=n)
    if vol_edge_flags is None:
        df["vol_edge_flag"] = rng.integers(0, 2, size=n)
    else:
        df["vol_edge_flag"] = vol_edge_flags
    return df


def _fake_cv_result(mean, std):
    return {"mean": mean, "std": std}


# ---------------------------------------------------------------------------
# compare_results
# ---------------------------------------------------------------------------

def test_compare_results_computes_delta_per_metric():
    model = _fake_cv_result(mean={"r2": 0.10, "mae": 1.0}, std={"r2": 0.01, "mae": 0.1})
    baseline = _fake_cv_result(mean={"r2": 0.05, "mae": 1.5}, std={"r2": 0.02, "mae": 0.2})

    comp = compare_results(model, baseline)

    assert set(comp) == {"r2", "mae"}
    assert comp["r2"]["model_mean"] == 0.10
    assert comp["r2"]["baseline_mean"] == 0.05
    assert comp["r2"]["delta"] == 0.05
    assert comp["mae"]["delta"] == -0.5
    assert comp["mae"]["model_std"] == 0.1
    assert comp["mae"]["baseline_std"] == 0.2


# ---------------------------------------------------------------------------
# is_signal
# ---------------------------------------------------------------------------

def test_is_signal_magnitude_uses_r2_margin_strictly():
    comp = {"r2": {"delta": 0.03}, "mae": {"delta": -1.0}}
    assert is_signal(comp, "magnitude") is True

    comp = {"r2": {"delta": 0.01}, "mae": {"delta": -1.0}}
    assert is_signal(comp, "magnitude") is False

    # Strictly greater than the margin: exactly 0.02 does not flag.
    comp = {"r2": {"delta": 0.02}, "mae": {"delta": -1.0}}
    assert is_signal(comp, "magnitude") is False


def test_is_signal_classification_requires_accuracy_and_f1():
    both = {"accuracy": {"delta": 0.03}, "f1": {"delta": 0.05}}
    assert is_signal(both, "direction") is True
    assert is_signal(both, "vol_edge") is True

    only_acc = {"accuracy": {"delta": 0.03}, "f1": {"delta": 0.01}}
    assert is_signal(only_acc, "direction") is False

    only_f1 = {"accuracy": {"delta": 0.0}, "f1": {"delta": 0.03}}
    assert is_signal(only_f1, "vol_edge") is False


def test_is_signal_nan_delta_is_false():
    comp = {"r2": {"delta": float("nan")}, "mae": {"delta": 0.0}}
    assert is_signal(comp, "magnitude") is False

    comp = {"accuracy": {"delta": float("nan")}, "f1": {"delta": 0.5}}
    assert is_signal(comp, "direction") is False


# ---------------------------------------------------------------------------
# build_baseline_pipeline (helper in train_option_models)
# ---------------------------------------------------------------------------

def test_build_baseline_pipeline_uses_same_preprocessor_and_dummy():
    mag = build_baseline_pipeline("magnitude")
    assert isinstance(mag.named_steps["estimator"], DummyRegressor)
    assert mag.named_steps["estimator"].strategy == "mean"
    assert "preprocessor" in mag.named_steps

    for target in ("direction", "vol_edge"):
        pipe = build_baseline_pipeline(target)
        assert isinstance(pipe.named_steps["estimator"], DummyClassifier)
        assert pipe.named_steps["estimator"].strategy == "most_frequent"


def test_default_models_cover_all_targets():
    assert set(DEFAULT_MODELS) == {"magnitude", "direction", "vol_edge"}
    assert "ridge" in DEFAULT_MODELS["magnitude"]
    assert "gradient_boosting" in DEFAULT_MODELS["direction"]
    assert "logistic" in DEFAULT_MODELS["vol_edge"]


# ---------------------------------------------------------------------------
# run_sweep on synthetic data
# ---------------------------------------------------------------------------

def test_run_sweep_uses_same_folds_for_model_and_baseline():
    df = _make_prepared_df(n=60)

    rows = run_sweep(df, targets=["magnitude"], models={"magnitude": ["linear"]}, n_splits=3)

    assert len(rows) == 1
    row = rows[0]
    assert row["target"] == "magnitude"
    assert row["model_type"] == "linear"
    model_folds = row["model"]["folds"]
    base_folds = row["baseline"]["folds"]
    assert len(model_folds) == len(base_folds) == 3
    for mf, bf in zip(model_folds, base_folds):
        assert mf["train_size"] == bf["train_size"]
        assert mf["test_size"] == bf["test_size"]
    # Comparison and signal attached.
    assert set(row["comparison"]) == {"mae", "r2"}
    assert isinstance(row["signal"], bool)


def test_run_sweep_baseline_single_class_fold_nan_excluded_from_mean():
    # Last test segment is single-class -> NaN metrics, excluded from the mean.
    flags = np.array([0, 1] * 10 + [1] * 10)
    df = _make_prepared_df(n=30, vol_edge_flags=flags)

    rows = run_sweep(df, targets=["vol_edge"], models={"vol_edge": ["logistic"]}, n_splits=2)

    row = rows[0]
    first, second = row["baseline"]["folds"]
    assert np.isfinite(first["metrics"]["accuracy"])
    assert np.isnan(second["metrics"]["accuracy"])
    assert np.isnan(second["metrics"]["f1"])
    assert row["baseline"]["mean"]["accuracy"] == first["metrics"]["accuracy"]
    # Same NaN handling applies to the model side.
    assert np.isnan(row["model"]["folds"][1]["metrics"]["accuracy"])
    assert row["model"]["mean"]["accuracy"] == row["model"]["folds"][0]["metrics"]["accuracy"]


# ---------------------------------------------------------------------------
# format_markdown
# ---------------------------------------------------------------------------

def test_format_markdown_renders_table_and_signal_flag():
    rows = [{
        "target": "magnitude",
        "model_type": "linear",
        "n_splits": 3,
        "model": _fake_cv_result(mean={"r2": 0.05, "mae": 1.0}, std={"r2": 0.01, "mae": 0.1}),
        "baseline": _fake_cv_result(mean={"r2": 0.0, "mae": 1.2}, std={"r2": 0.01, "mae": 0.1}),
        "comparison": {
            "r2": {"model_mean": 0.05, "model_std": 0.01,
                   "baseline_mean": 0.0, "baseline_std": 0.01, "delta": 0.05},
            "mae": {"model_mean": 1.0, "model_std": 0.1,
                    "baseline_mean": 1.2, "baseline_std": 0.1, "delta": -0.2},
        },
        "signal": True,
    }]

    table = format_markdown(rows)

    assert "| Target | Model | Metric |" in table
    assert "magnitude" in table
    assert "linear" in table
    assert "r2" in table
    assert "SIGNAL" in table
