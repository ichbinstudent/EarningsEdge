#!/usr/bin/env python3
"""Daily signals collector — derive per-ticker signals and persist them.

Run after the options-chain collector (scripts/collect_options_snapshot.py).
For each ticker in the universe it:
  1. loads that ticker's latest options_chain rows (<= --as-of) and computes
     option_volume / atm_iv / skew_25d (earnings_edge.signals);
  2. fetches ~1y of stock daily bars via Polygon for ts_momentum, and the
     benchmark's bars once for relative_momentum;
  3. computes iv_pctl_1y / skew_zscore against the ticker's own daily_signals
     history (None until >= 20 observations have accrued — never fabricated);
  4. upserts one row into daily_signals.

Usage:
    ./.venv/bin/python scripts/collect_daily_signals.py --max-tickers 50
    ./.venv/bin/python scripts/collect_daily_signals.py --tickers AAPL MSFT --dry-run
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date, timedelta
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Must run before any earnings_edge import: config.py constructs the Settings
# singleton at import time, which caches os.environ.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import pandas as pd

from earnings_edge.config import get_logger, setup_logging
from earnings_edge.db import (
    configure,
    daily_signals_history,
    options_chain_df_latest,
    snapshots_latest_price,
    snapshots_tickers_as_of,
    upsert_daily_signals,
)
from earnings_edge.signals import (
    compute_chain_signals,
    compute_iv_percentile,
    compute_ts_momentum,
    compute_zscore,
    enrich_chain_with_bs,
    relative_momentum,
)

setup_logging()
logger = get_logger("collect_daily_signals")


def _parse_args():
    p = argparse.ArgumentParser(description="Daily signals collector")
    p.add_argument("--tickers", nargs="*", default=[],
                   help="Universe; default = tickers from the latest snapshot scan_date.")
    p.add_argument("--max-tickers", type=int, default=100)
    p.add_argument("--as-of", default=date.today().isoformat())
    p.add_argument("--benchmark", default="SPY")
    p.add_argument("--rate", type=float, default=0.045, help="Risk-free rate for BSM IV solve")
    p.add_argument("--db", default=None, help="Path to earnings_ml.db")
    p.add_argument("--dry-run", action="store_true", help="Compute only; don't persist.")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    if args.db:
        configure(Path(args.db))

    tickers = args.tickers or snapshots_tickers_as_of(args.as_of, args.max_tickers)
    tickers = tickers[: args.max_tickers]
    if not tickers:
        logger.warning("empty universe — nothing to do")
        return 0
    logger.info("universe: %d tickers (as_of=%s)", len(tickers), args.as_of)

    # Polygon client for stock bars (momentum). Optional: signals degrade to
    # None when the key is missing, matching the module's no-fabrication rule.
    poly = None
    if os.environ.get("POLYGON_API_KEY"):
        from earnings_edge.collectors.polygon import PolygonClient
        poly = PolygonClient()
    else:
        logger.warning("POLYGON_API_KEY not set — momentum signals will be null")

    bench_mom = None
    from_date = (date.fromisoformat(args.as_of) - timedelta(days=400)).isoformat()
    if poly is not None:
        bench_bars = poly.get_daily_bars(args.benchmark, from_date, args.as_of, limit=300)
        bench_mom = compute_ts_momentum(bench_bars)
        logger.info("benchmark %s ts_momentum=%s", args.benchmark, bench_mom)

    rows = []
    for ticker in tickers:
        chain = options_chain_df_latest(ticker, args.as_of)

        mom = None
        spot = None
        if poly is not None:
            bars = poly.get_daily_bars(ticker, from_date, args.as_of, limit=300)
            mom = compute_ts_momentum(bars)
            if bars:
                spot = float(bars[-1]["c"])
        if spot is None:
            spot = snapshots_latest_price(ticker)

        if spot and not chain.empty:
            chain = enrich_chain_with_bs(chain, spot=spot, r=args.rate, as_of=args.as_of)
        sig = compute_chain_signals(chain, as_of=args.as_of)

        iv_pctl = compute_iv_percentile(
            daily_signals_history(ticker, "atm_iv", args.as_of), sig["atm_iv"])
        skew_z, skew_mean = compute_zscore(
            daily_signals_history(ticker, "skew_25d", args.as_of), sig["skew_25d"])

        rows.append({
            "ticker": ticker,
            "signal_date": args.as_of,
            "option_volume": sig["option_volume"],
            "atm_iv": sig["atm_iv"],
            "skew_25d": sig["skew_25d"],
            "skew_zscore": skew_z,
            "skew_mean": skew_mean,
            "iv_pctl_1y": iv_pctl,
            "ts_momentum": mom,
            "relative_momentum": relative_momentum(mom, bench_mom),
        })

    if args.dry_run:
        for r in rows:
            logger.info("dry-run %s", r)
        print(pd.DataFrame(rows).to_string(index=False))
        return 0

    n = upsert_daily_signals(rows)
    logger.info("upserted %d daily_signals rows", n)
    print(f"daily_signals: upserted {n} rows for {args.as_of}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
