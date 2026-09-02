#!/usr/bin/env python3
"""Forward-factor backfill: implied vs realized earnings-event vol at 30-60 DTE.

For each historical (ticker, scan_date) snapshot with a realized outcome:

  T1 = expiry in [30, 60] DTE (target 45), contains the earnings event
  T2 = expiry 21-42 days after T1 (target +28), event-free
  sigma_fwd  = sqrt((T2*s2^2 - T1*s1^2) / (T2 - T1))   (event-free baseline)
  implied_event_move = sqrt(s1^2*T1 - sigma_fwd^2*(T1 - tau_eff))
  premium_ratio = implied_event_move / median(|actual_move_pct| of other events)

Results land in the `ff_snapshots` table (resumable: processed pairs are
skipped on re-run). Rate: one call every --rate-sleep seconds (default 13),
matching the original Polygon backfill discipline.

Usage:
  PYTHONUNBUFFERED=1 .venv/bin/python3.12 scripts/ff_backfill.py [--limit N] [--rate-sleep 13]
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from polygon_backfill import PolygonClient, implied_vol  # noqa: E402
from earnings_edge.db import (  # noqa: E402
    ff_snapshots_pending_pairs,
    ff_snapshots_upsert_many,
    snapshots_hist_move_abs,
)

# Bump when the T1/T2 selection rule changes — stale rows are reprocessed.
# v1: T1 closest to 45, T2 +28. v2: T1 closest to 30, T2 +30 (2026-07-25).
SELECTOR_VERSION = 2

def get_pairs() -> list[dict]:
    """(ticker, scan_date) pairs with outcomes and >=3 realized events per ticker."""
    return ff_snapshots_pending_pairs(SELECTOR_VERSION)


def hist_move_stats(ticker: str, exclude_scan: str) -> tuple[float | None, float | None, int]:
    """Median and RMS |actual_move_pct| over the ticker's other events (leave-one-out).

    RMS is the right benchmark for implied vol (both are second moments);
    the median is kept for reference only.
    """
    vals = sorted(snapshots_hist_move_abs(ticker, exclude_scan))
    if not vals:
        return None, None, 0
    n = len(vals)
    med = vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2
    rms = math.sqrt(sum(v * v for v in vals) / n)
    return med, rms, n


def pick_expiries(contracts: list[dict], scan: date) -> tuple[dict | None, dict | None]:
    """T1: DTE in [30,60] closest to 30. T2: 21-42 days after T1, closest to +30."""
    by_expiry: dict[str, list[dict]] = {}
    for c in contracts:
        exp = c.get("expiration_date")
        if exp:
            by_expiry.setdefault(exp, []).append(c)

    t1_cands = []
    for exp, cs in by_expiry.items():
        dte = (datetime.strptime(exp, "%Y-%m-%d").date() - scan).days
        if 30 <= dte <= 60:
            t1_cands.append((abs(dte - 30), exp, dte, cs))
    if not t1_cands:
        return None, None
    t1_cands.sort()
    _, t1_exp, t1_dte, t1_cs = t1_cands[0]

    t2_cands = []
    for exp, cs in by_expiry.items():
        dte = (datetime.strptime(exp, "%Y-%m-%d").date() - scan).days
        gap = dte - t1_dte
        if 21 <= gap <= 42:
            t2_cands.append((abs(gap - 30), exp, dte, cs))
    if not t2_cands:
        return None, None
    t2_cands.sort()
    _, t2_exp, t2_dte, t2_cs = t2_cands[0]

    return (
        {"expiry": t1_exp, "dte": t1_dte, "contracts": t1_cs},
        {"expiry": t2_exp, "dte": t2_dte, "contracts": t2_cs},
    )


def atm_contract(contracts: list[dict], spot: float) -> dict | None:
    calls = [c for c in contracts if c.get("contract_type") == "call" and c.get("strike_price")]
    if not calls:
        return None
    return min(calls, key=lambda c: abs(float(c["strike_price"]) - spot))


def process_pair(pg: PolygonClient, pair: dict) -> dict:
    ticker = pair["ticker"]
    scan = datetime.strptime(pair["scan_date"], "%Y-%m-%d").date()
    earnings = datetime.strptime(pair["earnings_date"], "%Y-%m-%d").date()
    spot = float(pair["price"])
    row: dict = {
        "ticker": ticker, "scan_date": pair["scan_date"],
        "earnings_date": pair["earnings_date"], "spot": spot,
        "skip_reason": None,
    }

    med, rms, n_hist = hist_move_stats(ticker, pair["scan_date"])
    row["hist_median_move_pct"] = med
    row["hist_rms_move_pct"] = rms
    row["n_hist_events"] = n_hist
    if rms is None or rms <= 0:
        row["skip_reason"] = "no_hist_events"
        return row

    contracts = pg.option_contracts(
        ticker, as_of=scan,
        expiry_gte=scan + timedelta(days=25),
        expiry_lte=scan + timedelta(days=110),
        contract_type="call",
    )
    if not contracts:
        row["skip_reason"] = "no_contracts"
        return row

    t1, t2 = pick_expiries(contracts, scan)
    if not t1:
        row["skip_reason"] = "no_t1_in_30_60"
        return row
    if not t2:
        row["skip_reason"] = "no_t2_pair"
        return row

    c1 = atm_contract(t1["contracts"], spot)
    c2 = atm_contract(t2["contracts"], spot)
    if not c1 or not c2:
        row["skip_reason"] = "no_atm_contract"
        return row

    close1 = pg.option_close(c1["ticker"], scan)
    close2 = pg.option_close(c2["ticker"], scan)

    T1 = t1["dte"] / 365.0
    T2 = t2["dte"] / 365.0
    iv1 = implied_vol(close1, spot, float(c1["strike_price"]), T1, "call") if close1 else None
    iv2 = implied_vol(close2, spot, float(c2["strike_price"]), T2, "call") if close2 else None

    row.update({
        "t1_expiry": t1["expiry"], "t1_dte": t1["dte"],
        "t1_strike": float(c1["strike_price"]), "t1_contract": c1["ticker"],
        "t1_close": close1, "t1_iv": iv1,
        "t2_expiry": t2["expiry"], "t2_dte": t2["dte"],
        "t2_strike": float(c2["strike_price"]), "t2_contract": c2["ticker"],
        "t2_close": close2, "t2_iv": iv2,
    })

    if not iv1 or not iv2:
        row["skip_reason"] = "iv_unsolvable"
        return row

    var_diff = iv2 ** 2 * T2 - iv1 ** 2 * T1
    if var_diff <= 0:
        row["skip_reason"] = "negative_fwd_variance"
        return row
    sigma_fwd = math.sqrt(var_diff / (T2 - T1))
    row["sigma_fwd"] = sigma_fwd

    # Event window: days from scan to earnings, +1 session for the move to realize
    tau_days = max((earnings - scan).days, 0) + 1
    tau = tau_days / 365.0
    row["tau_days"] = tau_days

    event_var = iv1 ** 2 * T1 - sigma_fwd ** 2 * (T1 - tau)
    if event_var <= 0:
        row["skip_reason"] = "negative_event_variance"
        return row

    implied_move = math.sqrt(event_var) * 100.0  # pct of spot
    row["implied_event_move_pct"] = implied_move
    row["premium_ratio"] = implied_move / rms
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="max pairs to process (0=all)")
    ap.add_argument("--rate-sleep", type=float, default=13.0)
    args = ap.parse_args()

    api_key = os.environ.get("POLYGON_API_KEY", "")
    if not api_key:
        sys.exit("POLYGON_API_KEY not set")

    pg = PolygonClient(api_key, sleep=args.rate_sleep)

    pairs = get_pairs()
    if args.limit:
        pairs = pairs[: args.limit]
    total = len(pairs)
    print(f"ff_backfill: {total} (ticker, scan_date) pairs to process", flush=True)
    if not total:
        return

    t0 = time.time()
    done = skipped = failed = 0
    batch: list[dict] = []
    for i, pair in enumerate(pairs):
        try:
            row = process_pair(pg, pair)
        except Exception as exc:
            row = {"ticker": pair["ticker"], "scan_date": pair["scan_date"],
                   "earnings_date": pair["earnings_date"], "spot": pair["price"],
                   "skip_reason": f"error:{exc}"[:80]}
        row["selector_version"] = SELECTOR_VERSION
        batch.append(row)
        if row.get("skip_reason"):
            skipped += 1
        else:
            done += 1
        if (i + 1) % 10 == 0:
            ff_snapshots_upsert_many(batch)
            batch = []
            rate = (i + 1) / (time.time() - t0)
            eta = (total - i - 1) / rate / 60 if rate else 0
            print(f"  {i+1}/{total} ok={done} skip={skipped} eta={eta:.0f}min", flush=True)
    ff_snapshots_upsert_many(batch)
    print(f"DONE ok={done} skip={skipped} fail={failed} elapsed={(time.time()-t0)/60:.0f}min", flush=True)


if __name__ == "__main__":
    main()
