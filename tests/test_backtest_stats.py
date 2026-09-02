"""Tests for backtest statistics: trade stats, portfolio metrics, splits, cross-section."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from earnings_edge.backtest.stats import (
    cross_sectional_test,
    portfolio_metrics,
    trade_stats,
    train_test_report,
)


# ── trade_stats --------------------------------------------------------------

def test_trade_stats_known_list():
    stats = trade_stats([0.10, -0.05, 0.20, -0.02])
    assert stats.count == 4
    assert stats.mean == pytest.approx(0.0575)
    # sample std (ddof=1): sum of squared deviations = 0.039675, /3, sqrt
    assert stats.std == pytest.approx(math.sqrt(0.039675 / 3))
    assert stats.min == pytest.approx(-0.05)
    assert stats.max == pytest.approx(0.20)
    assert stats.median == pytest.approx(0.04)
    assert stats.p25 == pytest.approx(-0.0275)
    assert stats.p75 == pytest.approx(0.125)
    assert stats.win_rate == pytest.approx(0.5)
    # kelly = mean / sample variance
    assert stats.kelly_fraction == pytest.approx(0.0575 / (0.039675 / 3))


def test_trade_stats_all_winners():
    stats = trade_stats([0.05, 0.10, 0.02])
    assert stats.win_rate == pytest.approx(1.0)
    assert stats.kelly_fraction > 0


def test_trade_stats_empty():
    stats = trade_stats([])
    assert stats.count == 0
    assert math.isnan(stats.mean)
    assert math.isnan(stats.win_rate)


def test_trade_stats_single_observation():
    stats = trade_stats([0.07])
    assert stats.count == 1
    assert stats.mean == pytest.approx(0.07)
    assert stats.std == 0.0
    assert stats.median == pytest.approx(0.07)


# ── portfolio_metrics --------------------------------------------------------

def test_portfolio_metrics_hand_computed():
    # equity 100 -> 110 -> 99 -> 121; period returns 0.10, -0.10, 0.2222...
    m = portfolio_metrics([100.0, 110.0, 99.0, 121.0], periods_per_year=252)
    assert m.final_value == pytest.approx(121.0)
    assert m.total_return == pytest.approx(0.21)
    assert m.cagr == pytest.approx(1.21 ** (252 / 3) - 1)
    # drawdown: peak 110, trough 99 -> 10%
    assert m.max_drawdown == pytest.approx(0.10)
    assert m.total_trades == 3  # all three periods moved

    rets = np.array([0.10, -0.10, 121.0 / 99.0 - 1])
    expected_sharpe = rets.mean() / rets.std(ddof=1) * math.sqrt(252)
    assert m.sharpe == pytest.approx(expected_sharpe)


def test_portfolio_metrics_monotone_equity_no_drawdown():
    m = portfolio_metrics([100.0, 110.0, 120.0])
    assert m.max_drawdown == pytest.approx(0.0)
    assert m.total_return == pytest.approx(0.20)


def test_portfolio_metrics_with_benchmark():
    m = portfolio_metrics([100.0, 110.0, 99.0, 121.0],
                          benchmark_curve=[100.0, 105.0, 100.0, 110.0])
    assert m.benchmark_total_return == pytest.approx(0.10)
    assert m.excess_return == pytest.approx(0.21 - 0.10)


def test_portfolio_metrics_degenerate():
    m = portfolio_metrics([100.0])
    assert m.final_value == pytest.approx(100.0)
    assert m.total_return == pytest.approx(0.0)
    assert m.total_trades == 0


# ── train_test_report --------------------------------------------------------

def test_train_test_split_counts():
    returns = [0.01 * i for i in range(10)]
    report = train_test_report(returns, split=0.7)
    assert report["train"].count == 7
    assert report["test"].count == 3


def test_train_test_split_is_chronological():
    returns = [0.01] * 7 + [0.05] * 3
    report = train_test_report(returns, split=0.7)
    assert report["train"].mean == pytest.approx(0.01)
    assert report["test"].mean == pytest.approx(0.05)


# ── cross_sectional_test -----------------------------------------------------

def test_cross_sectional_monotone_signal_significant():
    rng = np.random.default_rng(42)
    n = 200
    signal = pd.Series(rng.normal(size=n))
    fwd = pd.Series(0.05 * signal + rng.normal(scale=0.01, size=n))
    res = cross_sectional_test(signal, fwd, n_buckets=10)
    assert res.n == n
    assert res.spearman_rho > 0.9
    assert res.p_value < 0.05
    assert len(res.bucket_means) == 10
    # decile means strictly increasing for a clean monotone signal
    diffs = np.diff(res.bucket_means.values)
    assert (diffs > 0).all()


def test_cross_sectional_noise_signal_not_significant():
    rng = np.random.default_rng(7)
    n = 100
    signal = pd.Series(rng.normal(size=n))
    fwd = pd.Series(rng.normal(size=n))
    res = cross_sectional_test(signal, fwd, n_buckets=5)
    assert len(res.bucket_means) == 5
    assert 0.0 <= res.p_value <= 1.0
    assert -1.0 <= res.spearman_rho <= 1.0
