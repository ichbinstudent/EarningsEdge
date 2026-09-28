#!/usr/bin/env python3
"""Re-measure legacy hist-move backfill rows with the announcement timing.

Rows the FF hist backfill wrote before 2026-09-29 carry ``timing='Backfill'``
and were measured with the before-open window for every name: for after-close
reporters that is the session BEFORE the announcement, so their realized
moves (and the FF ladder's hist RMS gate) were understated — a fake edge.
This drive recomputes them via fwd_factor_ladder.refresh_legacy_hist_moves
(Yahoo announcement time → AMC/BMO window; unknown time → two-session window)
and relabels them ``Backfill:AMC`` / ``Backfill:BMO`` / ``Backfill:2d``.

The bot also refreshes a ticker lazily the first time it builds an FF
candidate for it; this script does the whole table at once.

Single-writer discipline: run it while the bot is stopped (or outside the
13:45-16:00 ET FF window) — it writes to the snapshots table.

Usage:
  .venv/bin/python3.12 scripts/refresh_hist_moves.py            # dry run: counts only
  .venv/bin/python3.12 scripts/refresh_hist_moves.py --apply [--limit N] [--sleep 0.3]
"""

import argparse
import logging
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("refresh_hist")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="write changes (default: dry run)")
    ap.add_argument("--limit", type=int, default=None, help="max tickers to refresh")
    ap.add_argument("--sleep", type=float, default=0.3, help="pause between tickers (Yahoo/LSE politeness)")
    args = ap.parse_args()

    from earnings_edge.db.repositories import snapshots_legacy_backfill_rows

    rows = snapshots_legacy_backfill_rows()
    tickers = list(dict.fromkeys(r["ticker"] for r in rows))
    logger.info("legacy backfill rows: %d across %d tickers", len(rows), len(tickers))
    if not args.apply:
        logger.info("dry run — pass --apply to re-measure them")
        return 0

    from earnings_edge.fwd_factor_ladder import refresh_legacy_hist_moves

    if args.limit:
        tickers = tickers[: args.limit]
    refreshed = 0
    for i, t in enumerate(tickers, 1):
        refreshed += refresh_legacy_hist_moves(t)
        if i % 25 == 0:
            logger.info("[%d/%d] rows refreshed so far: %d", i, len(tickers), refreshed)
        if args.sleep:
            time.sleep(args.sleep)
    left = len(snapshots_legacy_backfill_rows())
    logger.info(
        "done: %d rows refreshed; %d legacy rows remain (Yahoo no longer lists them)", refreshed, left
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
