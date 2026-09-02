#!/usr/bin/env python3
"""Live options-chain collector via Alpaca (hourly-capable).

Pulls /v1beta1/options/snapshots/{underlying} and persists into
options_chain keyed by (contract, captured_hour). The bot runs this
hourly during RTH; this CLI is the same path for manual/backfill runs.

Rate limit: Alpaca data tier allows ~5 req/sec sustained.
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from earnings_edge.config import get_logger, setup_logging
from earnings_edge.chain_cache import (
    DEFAULT_MAX_TICKERS,
    collect,
    default_underlyings,
)

setup_logging()
logger = get_logger("collect_options_snapshot")

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def _parse_args():
    p = argparse.ArgumentParser(description="Live options-chain collector")
    p.add_argument("--api-key", default=os.environ.get("APCA_API_KEY_ID"))
    p.add_argument("--api-secret", default=os.environ.get("APCA_API_SECRET_KEY"))
    p.add_argument("--underlyings", nargs="*", default=[],
                   help="Underlying tickers; default from DB.")
    p.add_argument("--max-tickers", type=int, default=DEFAULT_MAX_TICKERS,
                   help="Cap tickers to avoid long runs.")
    p.add_argument("--dry-run", action="store_true",
                   help="Pull only; don't persist.")
    return p.parse_args()


def main():
    args = _parse_args()
    if not args.api_key or not args.api_secret:
        raise RuntimeError("Must pass --api-key + --api-secret or set env APCA_API_KEY+SECRET")

    from earnings_edge.collectors.alpaca_options import AlpacaOptionsClient
    client = AlpacaOptionsClient(api_key=args.api_key, api_secret=args.api_secret)

    if args.underlyings:
        underlyings = list(args.underlyings)[: args.max_tickers]
    else:
        underlyings = default_underlyings(args.max_tickers)

    if not underlyings:
        logger.warning("No underlyings to collect")
        return

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    logger.info("Starting options-chain collection run=%s (%d underlyings)",
                run_id, len(underlyings))
    stats = collect(client, underlyings, run_id=run_id, dry_run=args.dry_run)
    logger.info("Run %s complete: %s", run_id, stats)


if __name__ == "__main__":
    main()
