#!/usr/bin/env python3
"""Backtest report for stored calendar-call trades, with oquants-style realism.

Loads `calendar_call_trades` from data/earnings_ml.db READ-ONLY and reports:
  - trade_stats over return_on_debit, overall and with a 70/30 chronological
    train/test split by scan_date;
  - the same stats net of worst-case IBKR commissions (4 fills per calendar:
    entry near/far, exit near/far — each leg priced at its recorded premium);
  - portfolio metrics over the cumulative-PnL equity curve.

NOTE: This report is deliberately left as a manual CLI tool rather than a scheduled
cron/systemd job. There is currently no existing weekly-review cadence or email/reporting
infrastructure in `deploy/` to attach it to, and building bespoke infra for a single
terminal report is anti-pattern. Run this manually when reviewing model performance.

Usage:
    ./.venv/bin/python backtest_report.py
    ./.venv/bin/python backtest_report.py --db path/to/earnings_ml.db --initial-capital 50000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from earnings_edge.backtest.realism import ibkr_commission
from earnings_edge.backtest.stats import (
    TradeStats,
    portfolio_metrics,
    trade_stats,
    train_test_report,
)

DEFAULT_DB = Path(__file__).parent / "data" / "earnings_ml.db"

COLUMNS = (
    "ticker, scan_date, net_debit, near_entry, far_entry, near_exit, far_exit, "
    "pnl_dollars, return_on_debit, model_score"
)


def load_trades(db_path: Path) -> pd.DataFrame:
    """Load calendar_call_trades; empty frame if missing/empty."""
    from sqlalchemy import text

    from earnings_edge.db import configure, get_engine

    configure(db_path)
    engine = get_engine()
    with engine.connect() as con:
        tables = {
            row[0]
            for row in con.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
        }
        if "calendar_call_trades" not in tables:
            return pd.DataFrame()
    return pd.read_sql(
        text(f"SELECT {COLUMNS} FROM calendar_call_trades ORDER BY scan_date, ticker"),
        engine,
    )


def net_of_cost_returns(df: pd.DataFrame) -> pd.Series:
    """Per-trade return on debit after worst-case IBKR commissions.

    Cost model: a calendar is 4 fills (entry near + far, exit near + far),
    each charged the tier for its own recorded premium. Commissions are
    subtracted from pnl_dollars and the result re-expressed on the original
    debit (net_debit * 100).
    """
    commissions = (
        df["near_entry"].apply(ibkr_commission)
        + df["far_entry"].apply(ibkr_commission)
        + df["near_exit"].apply(ibkr_commission)
        + df["far_exit"].apply(ibkr_commission)
    )
    net_pnl = df["pnl_dollars"] - commissions
    debit_dollars = df["net_debit"] * 100.0
    return (net_pnl / debit_dollars).where(debit_dollars > 0)


def print_stats(label: str, stats: TradeStats) -> None:
    """Pretty-print one TradeStats block."""
    if stats.count == 0:
        print(f"  {label:<12} (no trades)")
        return
    print(
        f"  {label:<12} n={stats.count:<5} mean={stats.mean:+.4f} std={stats.std:.4f} "
        f"win%={stats.win_rate * 100:5.1f} median={stats.median:+.4f} "
        f"p25={stats.p25:+.4f} p75={stats.p75:+.4f} "
        f"min={stats.min:+.4f} max={stats.max:+.4f} kelly={stats.kelly_fraction:+.3f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Realism-adjusted backtest report")
    parser.add_argument("--db", default=str(DEFAULT_DB), help="Path to earnings_ml.db")
    parser.add_argument("--split", type=float, default=0.7, help="Chronological train fraction")
    parser.add_argument("--initial-capital", type=float, default=100_000.0,
                        help="Starting equity for the portfolio metrics")
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"database not found: {db_path}")
        return 0

    df = load_trades(db_path)
    if df.empty:
        print(f"calendar_call_trades is empty or missing in {db_path} — nothing to report")
        return 0

    returns = df["return_on_debit"].dropna()
    print(f"\n=== Calendar Call Backtest Report ({len(df)} trades, "
          f"{df['scan_date'].min()} .. {df['scan_date'].max()}) ===")

    print("\nGross returns on debit:")
    print_stats("overall", trade_stats(returns))
    split_report = train_test_report(list(returns), split=args.split)
    print_stats("train", split_report["train"])
    print_stats("test", split_report["test"])

    net_returns = net_of_cost_returns(df).dropna()
    print("\nNet of IBKR commissions (4 fills per calendar, worst-case tiers):")
    print_stats("overall", trade_stats(net_returns))
    net_split = train_test_report(list(net_returns), split=args.split)
    print_stats("train", net_split["train"])
    print_stats("test", net_split["test"])

    by_date = df.groupby("scan_date")["pnl_dollars"].sum().sort_index()
    equity = [args.initial_capital] + list(args.initial_capital + by_date.cumsum())
    pm = portfolio_metrics(equity, periods_per_year=252)
    print(f"\nPortfolio (initial capital ${args.initial_capital:,.0f}, daily equity from summed PnL):")
    print(f"  final=${pm.final_value:,.0f} total_return={pm.total_return:+.2%} "
          f"cagr={pm.cagr:+.2%} sharpe={pm.sharpe:.2f} max_dd={pm.max_drawdown:.2%} "
          f"active_days={pm.total_trades}")

    if "model_score" in df and df["model_score"].notna().sum() >= 3:
        from earnings_edge.backtest.stats import cross_sectional_test
        cs = cross_sectional_test(df["model_score"], df["return_on_debit"], n_buckets=5)
        print(f"\nCross-sectional (model_score vs return_on_debit, n={cs.n}): "
              f"spearman={cs.spearman_rho:+.3f} p={cs.p_value:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
