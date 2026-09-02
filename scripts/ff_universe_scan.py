#!/usr/bin/env python3
"""Forward-factor general universe scan: earnings-agnostic forward vol.

This script scans the entire optionable universe and computes the pure
forward-factor (term structure richness) for T1/T2 near/far pairs, completely
ignoring whether an earnings event is present.

If an earnings date happens to fall in the T1 window, it adds the
event-adjusted metrics as a bonus. The results land in the
`ff_universe_snapshots` table.
"""

import argparse
import math
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from earnings_edge.db.repositories import (
    ff_universe_snapshots_upsert_many,
    snapshots_optionable_universe,
    options_chain_df_latest,
    snapshots_hist_move_abs,
    _fetchall,
)
from earnings_edge.fwd_factor import forward_iv, required_near_iv
from earnings_edge.option_math import implied_volatility
from scripts.ff_backfill import pick_expiries, atm_contract

SELECTOR_VERSION = 2

def get_latest_chain(ticker: str, as_of: str):
    return _fetchall(
        None,
        "SELECT expiry AS expiration_date, strike AS strike_price, contract_type, close, midpoint, contract_ticker as ticker "
        "FROM options_chain WHERE ticker = :ticker AND scan_date = ("
        "  SELECT MAX(scan_date) FROM options_chain WHERE ticker = :ticker AND scan_date <= :as_of"
        ")",
        {"ticker": ticker, "as_of": as_of},
    )

def get_earnings_in_window(ticker: str, scan_date: str, t1_expiry: str):
    """Finds earnings date strictly after scan_date and on or before t1_expiry."""
    rows = _fetchall(
        None,
        "SELECT earnings_date FROM snapshots "
        "WHERE ticker = :ticker AND earnings_date > :scan_date AND earnings_date <= :t1_expiry "
        "ORDER BY earnings_date ASC LIMIT 1",
        {"ticker": ticker, "scan_date": scan_date, "t1_expiry": t1_expiry}
    )
    return rows[0]["earnings_date"] if rows else None

def hist_move_stats(ticker: str, exclude_scan: str) -> tuple[float | None, float | None, int]:
    vals = sorted(snapshots_hist_move_abs(ticker, exclude_scan))
    if not vals:
        return None, None, 0
    n = len(vals)
    med = vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2
    rms = math.sqrt(sum(v * v for v in vals) / n)
    return med, rms, n

def get_spot(ticker: str, scan_date: str) -> float | None:
    rows = _fetchall(
        None,
        "SELECT price FROM snapshots WHERE ticker = :ticker AND scan_date = :scan_date AND price IS NOT NULL",
        {"ticker": ticker, "scan_date": scan_date}
    )
    if rows:
        return rows[0]["price"]
    
    # Fallback to most recent
    rows = _fetchall(
        None,
        "SELECT price FROM snapshots WHERE ticker = :ticker AND price IS NOT NULL ORDER BY scan_date DESC LIMIT 1",
        {"ticker": ticker}
    )
    return rows[0]["price"] if rows else None


def process_ticker(ticker: str, scan: date) -> dict:
    scan_date_str = scan.strftime("%Y-%m-%d")
    
    row = {
        "ticker": ticker,
        "scan_date": scan_date_str,
        "has_earnings_in_window": 0,
        "skip_reason": None,
        "selector_version": SELECTOR_VERSION,
    }
    
    spot = get_spot(ticker, scan_date_str); spot = float(spot) if spot else None
    if not spot:
        row["skip_reason"] = "no_spot_price"
        return row
    row["spot"] = float(spot)

    contracts = get_latest_chain(ticker, scan_date_str)
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

    close1 = c1.get("close")
    if close1 is None or close1 == 0:
        close1 = c1.get("midpoint")
    close2 = c2.get("close")
    if close2 is None or close2 == 0:
        close2 = c2.get("midpoint")

    T1 = t1["dte"] / 365.0
    T2 = t2["dte"] / 365.0
    
    iv1 = implied_volatility(close1, spot, float(c1["strike_price"]), T1, 0.045, "call") if close1 else None
    iv2 = implied_volatility(close2, spot, float(c2["strike_price"]), T2, 0.045, "call") if close2 else None

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

    sigma_fwd = forward_iv(iv1, T1, iv2, T2)
    if sigma_fwd is None:
        row["skip_reason"] = "negative_fwd_variance"
        return row
    
    row["sigma_fwd"] = sigma_fwd
    row["forward_factor"] = (iv1 - sigma_fwd) / sigma_fwd

    earnings_date_str = get_earnings_in_window(ticker, scan_date_str, t1["expiry"])
    if earnings_date_str:
        row["has_earnings_in_window"] = 1
        row["earnings_date"] = earnings_date_str
        
        earnings_dt = datetime.strptime(earnings_date_str, "%Y-%m-%d").date()
        tau_days = max((earnings_dt - scan).days, 0) + 1
        tau = tau_days / 365.0
        row["tau_days"] = tau_days

        med, rms, n_hist = hist_move_stats(ticker, scan_date_str)
        row["hist_median_move_pct"] = med
        row["hist_rms_move_pct"] = rms
        row["n_hist_events"] = n_hist

        event_var = iv1 ** 2 * T1 - sigma_fwd ** 2 * (T1 - tau)
        if event_var > 0:
            implied_move = math.sqrt(event_var) * 100.0
            row["implied_event_move_pct"] = implied_move
            if rms and rms > 0:
                row["premium_ratio"] = implied_move / rms

    return row

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="max tickers to process (0=all)")
    ap.add_argument("--dry-run", action="store_true", help="Do not write to DB")
    args = ap.parse_args()

    # Determine latest scan date in options chain
    scan_date_str = date.today().strftime("%Y-%m-%d")
    rows = _fetchall(None, "SELECT MAX(scan_date) as sd FROM options_chain", {})
    if rows and rows[0]["sd"]:
        scan_date_str = rows[0]["sd"]
    
    scan = datetime.strptime(scan_date_str, "%Y-%m-%d").date()

    tickers = snapshots_optionable_universe(10000)
    if args.limit:
        tickers = tickers[:args.limit]
        
    total = len(tickers)
    print(f"ff_universe_scan: {total} tickers to process for scan_date {scan_date_str}", flush=True)
    if not total:
        return

    t0 = time.time()
    done = skipped = failed = 0
    batch: list[dict] = []
    
    for i, ticker in enumerate(tickers):
        try:
            row = process_ticker(ticker, scan)
        except Exception as exc:
            row = {
                "ticker": ticker, "scan_date": scan_date_str,
                "has_earnings_in_window": 0,
                "selector_version": SELECTOR_VERSION,
                "skip_reason": f"error:{exc}"[:80]
            }
        batch.append(row)
        
        if row.get("skip_reason"):
            skipped += 1
        else:
            done += 1
            
        if (i + 1) % 50 == 0:
            if not args.dry_run:
                ff_universe_snapshots_upsert_many(batch)
            if not args.dry_run:
                batch = []
            rate = (i + 1) / (time.time() - t0)
            eta = (total - i - 1) / rate / 60 if rate else 0
            print(f"  {i+1}/{total} ok={done} skip={skipped} eta={eta:.0f}min", flush=True)

    if batch and not args.dry_run:
        ff_universe_snapshots_upsert_many(batch)

    if args.dry_run:
        print("\n--- DRY RUN OUTPUT ---")
        valid = [r for r in batch if not r.get("skip_reason")]
        with_earning = [r for r in valid if r.get("has_earnings_in_window")]
        no_earning = [r for r in valid if not r.get("has_earnings_in_window")]
        
        print(f"Total processed: {len(batch)}")
        print(f"Valid forward_factor: {len(valid)}")
        print(f"Earnings in window: {len(with_earning)}")
        print(f"No earnings in window: {len(no_earning)}")
        
        print("\nExample With Earnings:")
        if with_earning:
            print(with_earning[0])
            
        print("\nExample Without Earnings:")
        if no_earning:
            print(no_earning[0])

    print(f"DONE ok={done} skip={skipped} fail={failed} elapsed={(time.time()-t0)/60:.0f}min", flush=True)

if __name__ == "__main__":
    main()
