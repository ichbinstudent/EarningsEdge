#!/usr/bin/env python3
"""Warm hist-move coverage for the FF ladder hist gate.

Two modes:

* ``upcoming`` (default, original behavior): for each reporter in the next
  N days, ensure_hist_moves() backfills realized earnings moves into
  snapshots (timing='Backfill') so later 13:45 ET proposal runs are free.
* ``universe``: systematic repair drive over every under-covered ticker in
  the snapshots table (not just next week). Candidates come from
  earnings_edge.coverage.repair_candidates — liquid names first, cheapest
  top-ups first. ``--liquid-only`` (default on) skips the OTC/zero-option
  tail that never feeds training or FF candidates.

Usage:
  PYTHONUNBUFFERED=1 .venv/bin/python3.12 scripts/warm_hist_coverage.py [days]
  PYTHONUNBUFFERED=1 .venv/bin/python3.12 scripts/warm_hist_coverage.py \
      --mode universe [--all] [--limit N] [--min-events 3] [--sleep 0.3]
"""
import argparse
import logging
import sys
import time
from datetime import date, timedelta
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("warm_hist")


def repair_universe(tickers, sleep: float = 0.3, progress_every: int = 25,
                    ensure_fn=None) -> dict:
    """Run ensure_hist_moves over an explicit ticker list.

    ``ensure_fn`` is injectable for tests; defaults to
    fwd_factor_ladder.ensure_hist_moves. Returns before/after coverage over
    the repaired list plus per-ticker failure count.
    """
    if ensure_fn is None:
        from earnings_edge.fwd_factor_ladder import ensure_hist_moves
        ensure_fn = ensure_hist_moves
    from earnings_edge.coverage import hist_move_coverage

    before = hist_move_coverage(universe=list(tickers))
    done = failed = 0
    start = time.time()
    for i, t in enumerate(tickers, 1):
        try:
            ensure_fn(t)
            done += 1
        except Exception as exc:
            logger.info("%s: backfill failed (%s)", t, exc)
            failed += 1
        if i % progress_every == 0:
            rate = i / max(time.time() - start, 1)
            logger.info("[%d/%d] %.1f tickers/min", i, len(tickers), rate * 60)
        if sleep:
            time.sleep(sleep)  # be polite to Yahoo/LSE
    after = hist_move_coverage(universe=list(tickers))
    return {
        "processed": done,
        "failed": failed,
        "elapsed_s": time.time() - start,
        "coverage_before": before,
        "coverage_after": after,
    }


def _upcoming_tickers(days: int) -> list[str]:
    from earnings_edge.collectors.earnings_calendar import EarningsCalendarCollector

    collector = EarningsCalendarCollector()
    tickers: list[str] = []
    for i in range(days):
        d = date.today() + timedelta(days=i)
        if d.weekday() >= 5:
            continue
        try:
            cands = collector.fetch(d)
        except Exception as exc:
            logger.error("calendar fetch %s failed: %s", d, exc)
            continue
        logger.info("%s: %d reporters", d, len(cands))
        tickers.extend(c.ticker for c in cands)
    return list(dict.fromkeys(tickers))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("days", nargs="?", type=int, default=7,
                    help="upcoming mode: days ahead to warm")
    ap.add_argument("--mode", choices=["upcoming", "universe"], default="upcoming")
    ap.add_argument("--all", action="store_true",
                    help="universe mode: include non-liquid tickers (default: liquid only)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--min-events", type=int, default=3)
    ap.add_argument("--sleep", type=float, default=0.3)
    args = ap.parse_args()

    if args.mode == "universe":
        from earnings_edge.coverage import repair_candidates
        tickers = repair_candidates(
            min_events=args.min_events, liquid_only=not args.all
        )
        logger.info("universe repair: %d under-covered tickers (liquid_only=%s)",
                    len(tickers), not args.all)
    else:
        tickers = _upcoming_tickers(args.days)
        logger.info("unique upcoming tickers: %d", len(tickers))

    if args.limit:
        tickers = tickers[: args.limit]

    result = repair_universe(tickers, sleep=args.sleep)
    logger.info("done: %d processed, %d failed, %.0fs",
                result["processed"], result["failed"], result["elapsed_s"])
    if args.mode == "universe":
        b, a = result["coverage_before"], result["coverage_after"]
        logger.info("hist-gate coverage over repaired universe: %d/%d (%.1f%%) -> %d/%d (%.1f%%)",
                    b["covered"], b["universe"], b["pct"],
                    a["covered"], a["universe"], a["pct"])


if __name__ == "__main__":
    main()
