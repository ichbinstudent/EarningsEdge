#!/usr/bin/env python3
"""Backtest the intended earnings calendar-call strategy.

Strategy modeled per event:
- Enter on the snapshot scan_date.
- Sell the nearest post-earnings ATM call.
- Buy the same/closest strike call roughly one month beyond the near expiry.
- Exit both legs on the first available option close on/after earnings_date.

PnL is per one calendar spread contract, multiplied by 100. This uses Polygon EOD
option aggregate closes and skips rows where either leg lacks entry/exit prices.
"""
from __future__ import annotations

import argparse
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Mapping, Optional

import joblib
import numpy as np
import pandas as pd
from sqlalchemy import text

from polygon_backfill import PolygonClient, choose_atm_pair
from earnings_edge.calendar_spread import select_calendar_expiries
from earnings_edge.db import (
    DEFAULT_DB_PATH,
    calendar_call_trades_list,
    calendar_call_trades_update_model,
    calendar_call_trades_upsert,
    calendar_call_trades_with_snapshots,
    configure,
    get_engine,
)
from earnings_edge.calendar_filter import (
    data_quality_rejection_reasons,
    score_calendar_trade,
    utc_now_iso,
)


@dataclass(frozen=True)
class CalendarCallTrade:
    snapshot_id: int
    ticker: str
    earnings_date: str
    scan_date: str
    near_expiry: str
    far_expiry: str
    strike: float
    near_call_ticker: str
    far_call_ticker: str
    near_entry: float
    far_entry: float
    near_exit: float
    far_exit: float
    net_debit: float
    exit_value: float
    pnl_dollars: float
    return_on_debit: Optional[float]
    model_score: Optional[float] = None
    model_recommendation: Optional[int] = None
    model_reason: Optional[str] = None
    model_name: Optional[str] = None
    model_scored_at: Optional[str] = None


def ensure_schema() -> None:
    """Ensure calendar_call_trades exists (create_all + column migrations)."""
    get_engine()


def parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def first_option_close_on_or_after(pg: PolygonClient, option_ticker: str, start: date, max_days: int = 5) -> Optional[float]:
    bars = pg.daily_bars(option_ticker, start, start + timedelta(days=max_days), limit=10)
    bars = [b for b in bars if b.get("c") is not None]
    if not bars:
        return None
    return float(bars[0]["c"])


def select_calendar_calls(pg: PolygonClient, ticker: str, spot: float, as_of: date, earnings_date: date) -> Optional[tuple[dict, dict, date, date]]:
    contracts = pg.option_contracts(
        ticker,
        as_of=as_of,
        expiry_gte=earnings_date,
        expiry_lte=earnings_date + timedelta(days=70),
        contract_type="call",
    )
    if not contracts:
        return None
    expiries = sorted({parse_date(c["expiration_date"]) for c in contracts if c.get("expiration_date")})
    if len(expiries) < 2:
        return None
    near_exp, far_exp = select_calendar_expiries(expiries)
    near_calls = [c for c in contracts if c.get("expiration_date") == near_exp.isoformat()]
    far_calls = [c for c in contracts if c.get("expiration_date") == far_exp.isoformat()]
    if not near_calls or not far_calls:
        return None
    near = min(near_calls, key=lambda c: abs(float(c.get("strike_price") or 0) - spot))
    strike = float(near.get("strike_price") or 0)
    far = min(far_calls, key=lambda c: abs(float(c.get("strike_price") or 0) - strike))
    return near, far, near_exp, far_exp


def model_feature_row(row: Mapping, trade: CalendarCallTrade) -> dict:
    """Combine snapshot features and calendar-entry fields for model scoring."""

    data = dict(row)
    data.update(
        {
            "strike": trade.strike,
            "near_entry": trade.near_entry,
            "far_entry": trade.far_entry,
            "net_debit": trade.net_debit,
            "near_expiry": trade.near_expiry,
            "far_expiry": trade.far_expiry,
        }
    )
    return data


def _trade_with_model_fields(trade: CalendarCallTrade, **fields) -> CalendarCallTrade:
    data = asdict(trade)
    data.update(fields)
    return CalendarCallTrade(**data)


def score_trade(
    row: Mapping,
    trade: CalendarCallTrade,
    artifact: dict,
    *,
    model_name: str,
    threshold: float,
) -> CalendarCallTrade:
    """Return a copy of a trade populated with model score fields."""

    feature_row = model_feature_row(row, trade)
    reasons = data_quality_rejection_reasons(feature_row)
    if reasons:
        return _trade_with_model_fields(
            trade,
            model_recommendation=0,
            model_reason=",".join(reasons),
            model_name=model_name,
            model_scored_at=utc_now_iso(),
        )
    score = score_calendar_trade(artifact, feature_row, threshold=threshold)
    return _trade_with_model_fields(
        trade,
        model_score=score.probability,
        model_recommendation=int(score.recommended),
        model_reason=score.reason,
        model_name=model_name,
        model_scored_at=utc_now_iso(),
    )


def build_trade(
    pg: PolygonClient,
    row: Mapping,
    artifact: Optional[dict] = None,
    model_name: str = "",
    threshold: float = 0.55,
) -> Optional[CalendarCallTrade]:
    ticker = row["ticker"]
    ed = parse_date(row["earnings_date"])
    as_of = parse_date(row["scan_date"])
    spot = float(row["price"])
    selected = select_calendar_calls(pg, ticker, spot, as_of, ed)
    if not selected:
        return None
    near, far, near_exp, far_exp = selected
    near_ticker = near["ticker"]
    far_ticker = far["ticker"]
    near_entry = pg.option_close(near_ticker, as_of)
    far_entry = pg.option_close(far_ticker, as_of)
    if near_entry is None or far_entry is None:
        return None
    near_exit = first_option_close_on_or_after(pg, near_ticker, ed)
    far_exit = first_option_close_on_or_after(pg, far_ticker, ed)
    if near_exit is None or far_exit is None:
        return None
    net_debit = far_entry - near_entry
    exit_value = far_exit - near_exit
    pnl = (exit_value - net_debit) * 100.0
    trade = CalendarCallTrade(
        snapshot_id=int(row["id"]),
        ticker=ticker,
        earnings_date=row["earnings_date"],
        scan_date=row["scan_date"],
        near_expiry=near_exp.isoformat(),
        far_expiry=far_exp.isoformat(),
        strike=float(near["strike_price"]),
        near_call_ticker=near_ticker,
        far_call_ticker=far_ticker,
        near_entry=float(near_entry),
        far_entry=float(far_entry),
        near_exit=float(near_exit),
        far_exit=float(far_exit),
        net_debit=float(net_debit),
        exit_value=float(exit_value),
        pnl_dollars=float(pnl),
        return_on_debit=float(pnl / (net_debit * 100.0)) if net_debit > 0 else None,
    )
    if artifact:
        trade = score_trade(row, trade, artifact, model_name=model_name, threshold=threshold)
    return trade


def insert_trade(trade: CalendarCallTrade) -> None:
    calendar_call_trades_upsert(asdict(trade))


def load_model_artifact(path: Optional[Path]) -> Optional[dict]:
    if path is None:
        return None
    artifact = joblib.load(path)
    if "pipeline" not in artifact or "features" not in artifact:
        raise RuntimeError(f"Invalid calendar filter artifact: {path}")
    return artifact


def score_existing_trades(artifact: dict, *, model_name: str, threshold: float = 0.55) -> dict[str, int]:
    """Score stored calendar trades in-place using snapshot + entry features."""

    rows = calendar_call_trades_with_snapshots()
    scored = 0
    rejected = 0
    now = utc_now_iso()
    for row in rows:
        feature_row = dict(row)
        feature_row["id"] = row["trade_snapshot_id"]
        reasons = data_quality_rejection_reasons(feature_row)
        if reasons:
            calendar_call_trades_update_model(
                row["trade_snapshot_id"],
                model_score=None,
                model_recommendation=0,
                model_reason=",".join(reasons),
                model_name=model_name,
                model_scored_at=now,
            )
            rejected += 1
            continue
        score = score_calendar_trade(artifact, feature_row, threshold=threshold)
        calendar_call_trades_update_model(
            row["trade_snapshot_id"],
            model_score=score.probability,
            model_recommendation=int(score.recommended),
            model_reason=score.reason,
            model_name=model_name,
            model_scored_at=now,
        )
        scored += 1
    return {"scored": scored, "rejected": rejected}


def summarize() -> None:
    rows = calendar_call_trades_list()
    if not rows:
        print("No calendar-call trades stored yet")
        return
    pnl = np.array([float(r["pnl_dollars"]) for r in rows])
    rets = np.array([float(r["return_on_debit"]) for r in rows if r["return_on_debit"] is not None])
    print(f"calendar_call_trades: {len(rows)}")
    print(f"total_pnl: ${pnl.sum():.2f}")
    print(f"avg_pnl: ${pnl.mean():.2f}")
    print(f"median_pnl: ${np.median(pnl):.2f}")
    print(f"win_rate: {(pnl > 0).mean():.3f}")
    if len(rets):
        print(f"avg_return_on_debit: {rets.mean():.3f}")
        print(f"median_return_on_debit: {np.median(rets):.3f}")
    print("recent trades:")
    for r in rows[-10:]:
        score = ""
        if "model_score" in r and r["model_score"] is not None:
            rec = "TAKE" if r["model_recommendation"] else "SKIP"
            score = f" model={r['model_score']:.3f}/{rec}"
        print(f"  {r['ticker']} {r['earnings_date']} strike={r['strike']} debit={r['net_debit']:.2f} exit={r['exit_value']:.2f} pnl=${r['pnl_dollars']:.2f}{score}")

    model_rows = [r for r in rows if "model_recommendation" in r and r["model_recommendation"] is not None]
    if model_rows:
        taken = [r for r in model_rows if int(r["model_recommendation"]) == 1]
        rejected = [r for r in model_rows if int(r["model_recommendation"]) == 0]
        print("model_filter:")
        print(f"  scored_or_rejected: {len(model_rows)}")
        print(f"  take: {len(taken)} skip: {len(rejected)}")
        if taken:
            taken_pnl = np.array([float(r["pnl_dollars"]) for r in taken])
            print(f"  take_total_pnl: ${taken_pnl.sum():.2f}")
            print(f"  take_avg_pnl: ${taken_pnl.mean():.2f}")
            print(f"  take_median_pnl: ${np.median(taken_pnl):.2f}")
            print(f"  take_win_rate: {(taken_pnl > 0).mean():.3f}")


def main() -> None:
    p = argparse.ArgumentParser(description="Backtest earnings calendar-call trades from earnings_ml.db snapshots")
    p.add_argument("--limit", type=int, help="Max new snapshots to process")
    p.add_argument("--rate-sleep", type=float, default=13.0)
    p.add_argument("--api-key", help="Polygon API key; defaults to POLYGON_API_KEY env via PolygonClient")
    p.add_argument("--summary-only", action="store_true")
    p.add_argument("--model", type=Path, help="Trained calendar-call filter .joblib artifact to score trades")
    p.add_argument("--model-threshold", type=float, default=0.55, help="Probability threshold for TAKE/SKIP")
    p.add_argument("--score-existing", action="store_true", help="Score already-stored trades with --model before summarizing")
    args = p.parse_args()

    import os
    key = args.api_key or os.environ.get("POLYGON_API_KEY")
    if not key and not args.summary_only:
        raise RuntimeError("POLYGON_API_KEY not set")

    configure(DEFAULT_DB_PATH)
    ensure_schema()

    artifact = load_model_artifact(args.model)
    model_name = args.model.name if args.model else ""
    if args.score_existing and artifact is None:
        raise RuntimeError("--score-existing requires --model")
    if args.score_existing and artifact is not None:
        score_summary = score_existing_trades(artifact, model_name=model_name, threshold=args.model_threshold)
        print(f"Scored existing trades: {score_summary['scored']}; rejected by data gates: {score_summary['rejected']}")

    if not args.summary_only:
        assert key is not None
        pg = PolygonClient(key, sleep=args.rate_sleep)
        sql = """
            SELECT s.* FROM snapshots s
            LEFT JOIN calendar_call_trades t ON t.snapshot_id = s.id
            WHERE t.snapshot_id IS NULL
              AND s.collection_error IS NULL
              AND s.actual_move_pct IS NOT NULL
              AND s.price IS NOT NULL
            ORDER BY s.earnings_date, s.ticker
        """
        rows = pd.read_sql(text(sql), get_engine()).to_dict(orient="records")
        if args.limit:
            rows = rows[:args.limit]
        print(f"Processing {len(rows)} snapshots")
        ok = 0
        skipped = 0
        for i, row in enumerate(rows, start=1):
            print(f"[{i}/{len(rows)}] {row['ticker']} {row['earnings_date']}")
            trade = build_trade(pg, row, artifact=artifact, model_name=model_name, threshold=args.model_threshold)
            if not trade:
                skipped += 1
                print("  skip: missing calendar-call leg/price")
                continue
            insert_trade(trade)
            ok += 1
            print(f"  pnl=${trade.pnl_dollars:.2f} debit={trade.net_debit:.2f} exit={trade.exit_value:.2f}")
            time.sleep(0.05)
        print(f"Inserted/updated {ok}; skipped {skipped}")

    summarize()


if __name__ == "__main__":
    main()
