#!/usr/bin/env python3
"""CLI entrypoint for Position Designer (designer.py)."""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from pprint import pprint

import numpy as np

from earnings_edge.designer import Leg, analyze, pnl_at_expiry, pnl_at_front_expiry, rv_scenario

def parse_leg(s: str) -> Leg:
    # format: action kind strike expiry qty price iv
    # e.g.: "buy call 150.0 2026-09-18 1 5.20 0.45"
    # price/iv may be "auto" (resolved from the persisted options chain;
    # requires --ticker)
    parts = s.split()
    if len(parts) != 7:
        raise ValueError(f"Leg string must have 7 parts: action kind strike expiry qty price iv. Got: {s}")

    action = parts[0].lower()
    kind = parts[1].lower()
    strike = float(parts[2])
    expiry = datetime.strptime(parts[3], "%Y-%m-%d").date()
    qty = int(parts[4])
    price = parts[5] if parts[5].lower() == "auto" else float(parts[5])
    iv = parts[6] if parts[6].lower() == "auto" else float(parts[6])

    return Leg(action, kind, strike, expiry, qty, price, iv)

def print_risk_curve(title: str, grid: np.ndarray, pnl: np.ndarray, spot: float,
                     breakevens: list[float] | None = None) -> None:
    print("\n" + "=" * 50)
    print(title)
    print("=" * 50)
    span = max(float(np.max(np.abs(pnl))), 1e-9)
    width = 20
    breakevens = breakevens or []
    step = float(grid[1] - grid[0]) if len(grid) > 1 else 1.0
    for s, p in zip(grid, pnl):
        n = int(round(width * float(p) / span))
        bar = (" " * (width + n) + "-" * (-n) + "|") if n < 0 else (" " * width + "|" + "+" * n)
        mark = ""
        if any(abs(s - be) <= step / 2 for be in breakevens if np.isfinite(be)):
            mark = " BE"
        print(f"  {s:8.2f} {bar} {float(p):+10.2f}{mark}")
    print(f"  {'':8} (zero line at '|', grid ±20% around spot {spot:.2f})")


def main() -> int:
    parser = argparse.ArgumentParser(description="Options Position Designer")
    parser.add_argument("--spot", type=float, required=True, help="Underlying spot price")
    parser.add_argument("--leg", action="append", required=True,
                        help="Leg format: 'action kind strike expiry qty price iv' (can be repeated)")
    parser.add_argument("--forecast-rv", type=float, help="Forecast RV for simulation (e.g. 0.45)")
    parser.add_argument("--r", type=float, default=0.045, help="Risk free rate (default 0.045)")
    parser.add_argument("--ticker", help="Underlying ticker — required when a leg uses 'auto'")
    parser.add_argument("--db", default=None,
                        help="Path to earnings_ml.db for 'auto' price/iv (default: data/earnings_ml.db)")
    args = parser.parse_args()

    try:
        legs = [parse_leg(l) for l in args.leg]
    except Exception as e:
        print(f"Error parsing legs: {e}")
        return 1

    needs_market = any(l.price == "auto" or l.iv == "auto" for l in legs)
    if needs_market:
        if not args.ticker:
            print("Error: 'auto' price/iv requires --ticker")
            return 1
        from datetime import date as date_cls

        from earnings_edge.db import configure, options_chain_latest_contract

        db = args.db or str(Path(__file__).parent / "data" / "earnings_ml.db")
        configure(db)
        as_of = date_cls.today().isoformat()
        resolved = []
        for l in legs:
            if l.price == "auto" or l.iv == "auto":
                expiry_iso = l.expiry.isoformat() if hasattr(l.expiry, "isoformat") else str(l.expiry)
                row = options_chain_latest_contract(
                    args.ticker, l.kind, float(l.strike), expiry_iso, as_of,
                )
                if row is None:
                    print(f"Error: no chain data for {args.ticker} {l.kind} "
                          f"{l.strike} {l.expiry}")
                    return 1
                mid, close, stored_iv = row["midpoint"], row["close"], row["implied_volatility"]
                m_price = mid if mid is not None else close
                m_iv = stored_iv
                if m_iv is None and m_price and m_price > 0:
                    T = max((l.expiry - date_cls.today()).days, 0) / 365.0
                    if T > 0:
                        from earnings_edge.option_math import implied_volatility
                        solved = implied_volatility(
                            float(m_price), args.spot, float(l.strike), T, args.r, l.kind,
                        )
                        if solved is not None and not (
                            isinstance(solved, float) and np.isnan(solved)
                        ):
                            m_iv = float(solved)
                price = m_price if l.price == "auto" else l.price
                iv = m_iv if l.iv == "auto" else l.iv
                if price is None or iv is None:
                    print(f"Error: chain row for {args.ticker} {l.kind} {l.strike} "
                          f"{l.expiry} lacks {'price' if price is None else 'iv'}")
                    return 1
                l = Leg(l.action, l.kind, l.strike, l.expiry, l.quantity, price, iv)
            resolved.append(l)
        legs = resolved
    
    print("\n" + "=" * 50)
    print("POSITION SUMMARY")
    print("=" * 50)
    summary = analyze(legs, args.spot, args.r)
    
    for k, v in summary.items():
        if isinstance(v, dict):
            print(f"{k}:")
            for sub_k, sub_v in v.items():
                print(f"  {sub_k}: {sub_v}")
        elif isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")

    from datetime import date as _date
    grid = np.linspace(args.spot * 0.8, args.spot * 1.2, 41)
    breakevens = summary.get("breakevens") or []
    expiries = {l.expiry for l in legs if l.is_option}
    if len(expiries) > 1:
        # Multi-expiry structure (calendar/diagonal): the front-expiry curve is
        # the meaningful risk profile; final-expiry curve shown for reference.
        front = min(expiries)
        pnl_front = pnl_at_front_expiry(legs, grid, args.r, _date.today())
        print_risk_curve(f"RISK CURVE (P&L AT FRONT EXPIRY {front})", grid, pnl_front,
                         args.spot, breakevens)
    print_risk_curve("RISK CURVE (P&L AT EXPIRATION)", grid, pnl_at_expiry(legs, grid),
                     args.spot, breakevens)

    if args.forecast_rv:
        print("\n" + "=" * 50)
        print(f"RV SCENARIO SIMULATION (forecast_rv={args.forecast_rv})")
        print("=" * 50)
        scenario = rv_scenario(legs, args.spot, args.r, args.forecast_rv)
        for k, v in scenario.items():
            if isinstance(v, float):
                print(f"{k}: {v:.4f}")
            else:
                print(f"{k}: {v}")

    return 0

if __name__ == "__main__":
    sys.exit(main())
