"""Backtest statistics: per-trade stats, portfolio metrics, splits, cross-section.

Replicates the oquants.com research/backtest reporting layer:
  - strategy stats table (count, mean, std, min, median, percentiles, max,
    win rate, Kelly fraction) over per-trade returns;
  - portfolio metrics (final value, total return, CAGR, Sharpe, max drawdown,
    total trades) over an equity curve, optionally vs a buy-and-hold benchmark;
  - chronological train/test split discipline (research/set-train-split);
  - cross-sectional signal testing: decile mean returns plus a Spearman rank
    correlation with p-value between a signal and subsequent returns across
    the ticker universe.

All functions are pure — no I/O — so the whole module is unit-testable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


# ── Per-trade statistics ─────────────────────────────────────────────

@dataclass(frozen=True)
class TradeStats:
    """Summary statistics over a sequence of per-trade returns (fractions).

    ``kelly_fraction`` is the growth-optimal leverage ``mean / variance``
    (Gaussian approximation of the Kelly criterion for a return stream);
    positive edge -> positive fraction. ``std`` and the Kelly variance use the
    sample estimator (ddof=1); with a single observation std is 0.0 and Kelly
    is undefined (nan). All fields except ``count`` are nan when empty.
    """

    count: int
    mean: float
    std: float
    min: float
    median: float
    p25: float
    p75: float
    max: float
    win_rate: float
    kelly_fraction: float


def trade_stats(returns: Sequence[float]) -> TradeStats:
    """Compute TradeStats over per-trade returns (e.g. return_on_debit)."""
    arr = np.asarray(list(returns), dtype=float)
    arr = arr[np.isfinite(arr)]
    n = arr.size
    if n == 0:
        nan = math.nan
        return TradeStats(0, nan, nan, nan, nan, nan, nan, nan, nan, nan)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if n > 1 else 0.0
    var = float(arr.var(ddof=1)) if n > 1 else 0.0
    kelly = mean / var if var > 0 else math.nan
    return TradeStats(
        count=n,
        mean=mean,
        std=std,
        min=float(arr.min()),
        median=float(np.median(arr)),
        p25=float(np.percentile(arr, 25)),
        p75=float(np.percentile(arr, 75)),
        max=float(arr.max()),
        win_rate=float((arr > 0).mean()),
        kelly_fraction=kelly,
    )


# ── Portfolio metrics ────────────────────────────────────────────────

@dataclass(frozen=True)
class PortfolioMetrics:
    """Metrics over an equity curve (one value per period).

    ``total_trades`` is approximated as the number of periods in which the
    equity value changed (non-zero period return) — an equity curve alone does
    not carry trade counts. ``max_drawdown`` is the worst peak-to-trough
    decline as a positive fraction of the peak. ``sharpe`` annualizes the mean
    excess period return (rf=0) by ``sqrt(periods_per_year)``; it is 0.0 when
    the period returns have zero variance. ``cagr`` compounds
    ``(final/initial) ** (periods_per_year / n_periods) - 1``.
    """

    final_value: float
    total_return: float
    cagr: float
    sharpe: float
    max_drawdown: float
    total_trades: int
    benchmark_total_return: Optional[float] = None
    excess_return: Optional[float] = None


def portfolio_metrics(
    equity_curve: Sequence[float],
    *,
    periods_per_year: int = 252,
    benchmark_curve: Optional[Sequence[float]] = None,
) -> PortfolioMetrics:
    """Compute PortfolioMetrics over an equity curve (at least one point)."""
    eq = np.asarray(list(equity_curve), dtype=float)
    if eq.size == 0:
        raise ValueError("equity_curve must contain at least one value")
    initial = eq[0]
    final = float(eq[-1])
    total_return = final / initial - 1.0 if initial > 0 else math.nan

    rets = np.diff(eq) / eq[:-1] if eq.size > 1 else np.array([])
    n_periods = rets.size
    if n_periods > 0 and initial > 0 and final > 0:
        cagr = (final / initial) ** (periods_per_year / n_periods) - 1.0
    else:
        cagr = math.nan
    if n_periods > 1:
        std = float(rets.std(ddof=1))
        sharpe = float(rets.mean() / std * math.sqrt(periods_per_year)) if std > 0 else 0.0
    else:
        sharpe = 0.0

    peaks = np.maximum.accumulate(eq)
    drawdowns = np.where(peaks > 0, (peaks - eq) / peaks, 0.0)
    max_drawdown = float(drawdowns.max()) if drawdowns.size else 0.0
    total_trades = int((rets != 0).sum())

    benchmark_total_return: Optional[float] = None
    excess_return: Optional[float] = None
    if benchmark_curve is not None:
        bench = np.asarray(list(benchmark_curve), dtype=float)
        if bench.size > 0 and bench[0] > 0:
            benchmark_total_return = float(bench[-1] / bench[0] - 1.0)
            excess_return = total_return - benchmark_total_return

    return PortfolioMetrics(
        final_value=final,
        total_return=total_return,
        cagr=cagr,
        sharpe=sharpe,
        max_drawdown=max_drawdown,
        total_trades=total_trades,
        benchmark_total_return=benchmark_total_return,
        excess_return=excess_return,
    )


# ── Train/test split ─────────────────────────────────────────────────

def train_test_report(
    returns_or_trades: Sequence[float],
    split: float = 0.7,
) -> dict[str, TradeStats]:
    """Chronological train/test split, matching oquants' split discipline.

    The first ``split`` fraction of observations (in the order given — callers
    must pass them chronologically) is the training set, the remainder the
    out-of-sample test set. Returns ``{"train": TradeStats, "test": TradeStats}``.
    """
    if not 0.0 < split < 1.0:
        raise ValueError(f"split must be in (0, 1), got {split}")
    arr = list(returns_or_trades)
    cut = int(len(arr) * split)
    return {"train": trade_stats(arr[:cut]), "test": trade_stats(arr[cut:])}


# ── Cross-sectional signal test ──────────────────────────────────────

@dataclass(frozen=True)
class CrossSectionalResult:
    """Signal vs subsequent return across the ticker universe.

    ``bucket_means`` holds the mean forward return per signal bucket (deciles
    by default, ordered from lowest to highest signal). ``spearman_rho`` /
    ``p_value`` are the rank correlation between signal and forward return and
    its two-sided p-value. ``n`` is the number of paired observations used.
    """

    bucket_means: pd.Series
    spearman_rho: float
    p_value: float
    n: int
    n_buckets: int


def cross_sectional_test(
    signal: pd.Series,
    forward_returns: pd.Series,
    n_buckets: int = 10,
) -> CrossSectionalResult:
    """Cross-sectional test: bucketed mean returns + Spearman rank correlation.

    Pairs each ticker's signal value with its subsequent return (aligned on
    index, NaN pairs dropped), ranks the signal into ``n_buckets`` quantile
    buckets, and reports the mean forward return per bucket plus the Spearman
    rank correlation with its p-value. A monotone signal shows monotonically
    increasing bucket means and a significant positive rho.
    """
    df = pd.concat(
        [signal.rename("signal"), forward_returns.rename("fwd")], axis=1
    ).dropna()
    if df.shape[0] < 3:
        raise ValueError("need at least 3 paired observations")
    rho, p_value = spearmanr(df["signal"], df["fwd"])
    buckets = pd.qcut(df["signal"], n_buckets, labels=False, duplicates="drop")
    bucket_means = df.groupby(buckets)["fwd"].mean()
    return CrossSectionalResult(
        bucket_means=bucket_means,
        spearman_rho=float(rho),
        p_value=float(p_value),
        n=int(df.shape[0]),
        n_buckets=int(bucket_means.size),
    )
