"""Unit tests for cv_evaluate in train_option_models (synthetic data, no DB)."""

import numpy as np
import pandas as pd

from train_option_models import FEATURE_COLUMNS, cv_evaluate


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


def test_cv_evaluate_returns_n_splits_folds_with_finite_mae():
    df = _make_prepared_df(n=60)

    result = cv_evaluate(df, model_type="linear", target="magnitude", n_splits=5)

    assert len(result["folds"]) == 5
    for fold in result["folds"]:
        assert np.isfinite(fold["metrics"]["mae"])
        assert "r2" in fold["metrics"]
    assert np.isfinite(result["mean"]["mae"])
    assert np.isfinite(result["std"]["mae"])


def test_cv_evaluate_single_class_fold_is_nan_and_excluded_from_mean():
    # n_splits=2 on 30 rows -> test segments are rows 10-19 and 20-29;
    # the last segment is single-class (all 1).
    flags = np.array([0, 1] * 10 + [1] * 10)
    df = _make_prepared_df(n=30, vol_edge_flags=flags)

    result = cv_evaluate(df, model_type="logistic", target="vol_edge", n_splits=2)

    assert len(result["folds"]) == 2
    first, second = result["folds"]
    assert np.isfinite(first["metrics"]["accuracy"])
    assert np.isnan(second["metrics"]["accuracy"])
    assert np.isnan(second["metrics"]["f1"])
    # Mean excludes the NaN fold.
    assert result["mean"]["accuracy"] == first["metrics"]["accuracy"]
    assert result["mean"]["f1"] == first["metrics"]["f1"]
    assert result["std"]["accuracy"] == 0.0
