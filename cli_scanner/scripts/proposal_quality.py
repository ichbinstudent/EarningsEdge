#!/usr/bin/env python3
"""Measure approved trade proposals against realized earnings outcomes.

Answers: "did the bot's operator-approved proposals beat the expected move?"

Proposal sources (both live in the shared SQLite DB):
- ``pending_trades`` (calendar-ML approval flow): approved == status 'executed'.
- ``ff_ladders`` (forward-factor ladder): approved == armed_by IS NOT NULL;
  executed == status 'filled' (fill price from managed_positions entry_price).

Realized outcomes join on snapshots (ticker, earnings_date):
expected_move_pct (scanner straddle-implied move at scan time) and
actual_move_pct / actual_move_direction (post-earnings realized move).

Win convention (long calendars): the thesis profits when the realized move is
SMALLER than the expected/implied move (IV crush exceeds realized). A proposal
"beats the expected move" when |actual_move_pct| < expected_move_pct. For ff
candidates we additionally derive the implied event move from the stored
candidate quotes (BS-solved near-leg IV minus forward variance), which covers
trades whose snapshots row never got an expected_move_pct.

P&L per executed trade under the strategy's fill/exit assumptions:
entry = recorded fill debit (managed_positions), exit = both legs priced at the
first available option close ON or AFTER the earnings date (same convention as
calendar_call_backtest). Exit prices need Polygon and are only fetched with
--fetch-exit-prices; without it, P&L uses recorded exit prices for closed
positions and is reported as unavailable otherwise.

Usage:
    .venv/bin/python3.12 scripts/proposal_quality.py [--db PATH] [--json OUT]
        [--fetch-exit-prices] [--rate-sleep 13]

Stdlib-only by default so the script is hermetic and unit-testable; the
Polygon client is imported lazily inside the fetch path.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import text
from sqlalchemy.orm import Session

from earnings_edge.db import configure, get_session

DEFAULT_DB = Path(__file__).resolve().parents[1] / "data" / "earnings_ml.db"
RISK_FREE_RATE = 0.045  # match fwd_factor / polygon_backfill convention
MIN_SIGNIFICANT_N = 20  # below this, report descriptive stats only


# ---------------------------------------------------------------------------
# Minimal Black-Scholes (bisection IV solver)
# ---------------------------------------------------------------------------

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return max(0.0, S - K) if T == 0 else float("nan")
    if S <= 0 or K <= 0:
        return float("nan")
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)


def solve_call_iv(price: float, S: float, K: float, T: float,
                  r: float = RISK_FREE_RATE) -> Optional[float]:
    """Bisection IV solve; None when the price is outside no-arb bounds."""
    if T <= 0 or S <= 0 or K <= 0 or price <= 0:
        return None
    lo, hi = 1e-4, 5.0
    p_lo, p_hi = bs_call_price(S, K, T, r, lo), bs_call_price(S, K, T, r, hi)
    if not (math.isfinite(p_lo) and math.isfinite(p_hi)):
        return None
    if price < p_lo or price > p_hi:
        return None
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        p_mid = bs_call_price(S, K, T, r, mid)
        if abs(p_mid - price) < 1e-6:
            return mid
        if p_mid < price:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# Data access
# ---------------------------------------------------------------------------

def connect_ro(db_path: Path) -> Session:
    configure(db_path)
    return get_session()


def _table_exists(conn: Session, name: str) -> bool:
    row = conn.execute(
        text("SELECT 1 FROM sqlite_master WHERE type='table' AND name=:name"),
        {"name": name},
    ).fetchone()
    return row is not None


def load_ff_proposals(conn: Session) -> list[dict[str, Any]]:
    """Approved ff-ladder proposals (operator clicked Arm)."""
    if not _table_exists(conn, "ff_ladders"):
        return []
    out = []
    for row in conn.execute(text(
        "SELECT id, ticker, candidate_json, order_id, status, armed_by, created_at "
        "FROM ff_ladders WHERE armed_by IS NOT NULL ORDER BY id"
    )).mappings():
        cand = json.loads(row["candidate_json"])
        out.append({
            "source": "ff_ladder",
            "proposal_id": row["id"],
            "ticker": row["ticker"],
            "earnings_date": cand.get("earnings_date"),
            "strategy": "ff_ladder",
            "status": row["status"],
            "executed": row["status"] == "filled",
            "order_id": row["order_id"],
            "created_at": row["created_at"],
            "candidate": cand,
        })
    return out


def load_pending_proposals(conn: Session) -> list[dict[str, Any]]:
    """Approved calendar-ML proposals (operator clicked Execute)."""
    if not _table_exists(conn, "pending_trades"):
        return []
    out = []
    for row in conn.execute(text(
        "SELECT id, strategy, ticker, side, trade_json, status, order_json, created_at "
        "FROM pending_trades WHERE status IN ('executed','error') ORDER BY id"
    )).mappings():
        trade = json.loads(row["trade_json"])
        out.append({
            "source": "pending_trades",
            "proposal_id": row["id"],
            "ticker": row["ticker"],
            "earnings_date": trade.get("earnings_date"),
            "strategy": row["strategy"],
            "status": row["status"],
            "executed": row["status"] == "executed",
            "order_id": (json.loads(row["order_json"]) or {}).get("order_id")
            if row["order_json"] else None,
            "created_at": row["created_at"],
            "candidate": trade,
        })
    return out


def load_fills(conn: Session) -> dict[str, dict[str, Any]]:
    """group_id/order_id -> {entry_debit, qty, exit_credit?} from managed_positions.

    Both legs of a combo carry the same net entry price (verified on prod rows),
    so the group value is taken once. exit_price is only set when the exit
    engine actually closed the position.
    """
    if not _table_exists(conn, "managed_positions"):
        return {}
    fills: dict[str, dict[str, Any]] = {}
    for row in conn.execute(text(
        "SELECT group_id, entry_price, exit_price, qty, status FROM managed_positions"
    )).mappings():
        gid = row["group_id"]
        if not gid:
            continue
        f = fills.setdefault(gid, {"entry_debit": None, "qty": 0.0, "closed": True})
        if row["entry_price"] is not None:
            f["entry_debit"] = float(row["entry_price"])
        f["qty"] = max(f["qty"], float(row["qty"] or 0.0))
        if row["status"] != "closed":
            f["closed"] = False
        if row["exit_price"] is not None:
            f["recorded_exit"] = float(row["exit_price"])
    return fills


def load_outcome(conn: Session, ticker: str,
                 earnings_date: str) -> dict[str, Any]:
    """Best available outcome for (ticker, earnings_date) from snapshots.

    Multiple rows can exist per event (different scan_date / data_source);
    take the latest non-null value per field. Includes ``timing`` (BMO/AMC)
    when the scanner recorded it — needed for event-move alignment.
    """
    if not _table_exists(conn, "snapshots"):
        return {}
    rows = conn.execute(
        text(
            "SELECT scan_date, timing, expected_move_pct, actual_move_pct, "
            "actual_move_direction, pre_earnings_close, post_earnings_close "
            "FROM snapshots WHERE ticker=:ticker AND earnings_date=:earnings_date "
            "ORDER BY scan_date ASC"
        ),
        {"ticker": ticker, "earnings_date": earnings_date},
    ).mappings().fetchall()
    out: dict[str, Any] = {}
    for r in rows:
        for col in ("timing", "expected_move_pct", "actual_move_pct",
                    "actual_move_direction", "pre_earnings_close",
                    "post_earnings_close"):
            if r[col] is not None:
                out[col] = r[col]
    return out


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def implied_event_move_pct(candidate: dict[str, Any],
                           proposal_date: date) -> Optional[float]:
    """Implied earnings-event move (%) from stored candidate quotes.

    event_var = iv_near^2 * T1 - sigma_fwd^2 * (T1 - tau), solved from the
    near-leg quote mid. None when quotes/inputs are degenerate.
    """
    try:
        spot = float(candidate["spot"])
        strike = float(candidate["strike"])
        near_bid = float(candidate["near_bid"])
        near_ask = float(candidate["near_ask"])
        sigma_fwd = float(candidate["sigma_fwd"])
        tau = float(candidate["tau_days"]) / 365.0
        near_expiry = date.fromisoformat(candidate["near_expiry"])
    except (KeyError, TypeError, ValueError):
        return None
    if near_bid <= 0 or near_ask <= 0 or near_ask < near_bid or sigma_fwd <= 0:
        return None
    T1 = (near_expiry - proposal_date).days / 365.0
    if T1 <= 0 or tau < 0 or tau > T1:
        return None
    iv_near = solve_call_iv((near_bid + near_ask) / 2.0, spot, strike, T1)
    if iv_near is None:
        return None
    event_var = iv_near ** 2 * T1 - sigma_fwd ** 2 * (T1 - tau)
    if event_var <= 0:
        return None
    return math.sqrt(event_var) * 100.0


def _binomial_p_two_sided(k: int, n: int, p: float = 0.5) -> float:
    """Two-sided binomial test P(|X - np| >= |k - np|) under H0: hit rate = p."""
    if n <= 0:
        return 1.0
    probs = [math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(n + 1)]
    observed = probs[k]
    return min(1.0, sum(pr for pr in probs if pr <= observed + 1e-12))


# ---------------------------------------------------------------------------
# Timing-aware realized event move (pure — unit-testable)
# ---------------------------------------------------------------------------
# The stored snapshots.actual_move_pct uses the outcome-service convention
# (post_bar = first close ON/AFTER the earnings date). For AMC ("Post Market")
# reporters that close is the PRE-announcement session: the realized earnings
# move only shows up the next trading day. Measured 2026-08-07 on the four
# filled ff_ladder trades (all AMC 2026-07-29): stored moves were -1.31/-0.71/
# -4.42/-1.14% while the actual post-announcement moves were -7.95/+15.51/
# -2.62/-7.10%. Hit rates on the stored convention are therefore flattering
# for AMC names; compute the event-window move explicitly.

def is_post_market(timing: Optional[str]) -> bool:
    """True when the announcement is after the close on earnings_date."""
    return bool(timing) and "post" in timing.lower()


def compute_event_move(bars: list[dict[str, Any]], ed: date,
                       timing: Optional[str]) -> Optional[float]:
    """Close-to-close realized move (%) across the announcement.

    bars: Polygon-style agg dicts (``t`` ms epoch, ``c`` close), ascending.
    AMC ("Post Market"): last close ON/BEFORE ed -> first close AFTER ed.
    BMO / unknown:       last close BEFORE ed  -> first close ON/AFTER ed
                         (matches the stored outcome-service convention).
    """
    dated = []
    for b in bars:
        if b.get("c") is None or b.get("t") is None:
            continue
        dated.append((datetime.fromtimestamp(b["t"] / 1000).date(), float(b["c"])))
    if len(dated) < 2:
        return None
    if is_post_market(timing):
        pre = [d for d in dated if d[0] <= ed]
        post = [d for d in dated if d[0] > ed]
    else:
        pre = [d for d in dated if d[0] < ed]
        post = [d for d in dated if d[0] >= ed]
    if not pre or not post:
        return None
    pre_close, post_close = pre[-1][1], post[0][1]
    if pre_close <= 0:
        return None
    return (post_close - pre_close) / pre_close * 100.0


def exit_window_start(ed: date, timing: Optional[str]) -> date:
    """First date whose close reflects the announcement (exit pricing basis)."""
    return ed + timedelta(days=1) if is_post_market(timing) else ed


def aggregate(trades: list[dict[str, Any]]) -> dict[str, Any]:
    approved = len(trades)
    executed = [t for t in trades if t["executed"]]
    with_outcome = [t for t in trades if t.get("actual_move_pct") is not None]
    scorable = [t for t in with_outcome if t.get("hit") is not None]
    hits = sum(1 for t in scorable if t["hit"])
    n = len(scorable)
    # timing-aware event-move stats (only present after --fetch-exit-prices)
    ev_scorable = [t for t in trades if t.get("hit_event") is not None]
    ev_hits = sum(1 for t in ev_scorable if t["hit_event"])
    ev_n = len(ev_scorable)
    pnl_trades = [t for t in executed if t.get("pnl") is not None]
    ratios = [abs(t["actual_move_pct"]) / t["expected_move_pct"]
              for t in scorable if t.get("expected_move_pct")]
    return {
        "approved": approved,
        "executed": len(executed),
        "with_outcome": len(with_outcome),
        "scorable": n,
        "hits": hits,
        "hit_rate": (hits / n) if n else None,
        "binomial_p": _binomial_p_two_sided(hits, n) if n else None,
        "mean_actual_over_expected": (sum(ratios) / len(ratios)) if ratios else None,
        "event_scorable": ev_n,
        "event_hits": ev_hits,
        "event_hit_rate": (ev_hits / ev_n) if ev_n else None,
        "event_binomial_p": _binomial_p_two_sided(ev_hits, ev_n) if ev_n else None,
        "pnl_trades": len(pnl_trades),
        "total_pnl": sum(t["pnl"] for t in pnl_trades) if pnl_trades else None,
        "significant": n >= MIN_SIGNIFICANT_N,
    }


# ---------------------------------------------------------------------------
# Optional Polygon exit pricing (network — opt-in only)
# ---------------------------------------------------------------------------

def _fetch_exit_credit(pg: Any, near_symbol: str, far_symbol: str,
                       ed: date, timing: Optional[str]) -> tuple[Optional[float], Optional[str]]:
    """Combo credit at the first close reflecting the announcement.

    AMC names are priced from the first session AFTER earnings_date; BMO from
    the first session ON/AFTER (the calendar-backtest convention). Returns
    (exit_credit, exit_date_iso).
    """
    start = exit_window_start(ed, timing)
    end = ed + timedelta(days=6)

    def first_close(symbol: str) -> tuple[Optional[float], Optional[date]]:
        bars = pg.daily_bars(f"O:{symbol}", start, end, limit=10)
        bars = [b for b in bars if b.get("c") is not None]
        if not bars:
            return None, None
        first = bars[0]  # daily_bars is sorted ascending
        ts = first.get("t")
        d = datetime.fromtimestamp(ts / 1000).date() if ts else None
        return float(first["c"]), d

    near, near_d = first_close(near_symbol)
    far, far_d = first_close(far_symbol)
    if near is None or far is None:
        return None, None
    exit_date = max(d for d in (near_d, far_d) if d is not None)
    return far - near, exit_date.isoformat() if exit_date else None


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def build_report(db_path: Path, *, fetch_exit_prices: bool = False,
                 rate_sleep: float = 13.0) -> dict[str, Any]:
    conn = connect_ro(db_path)
    try:
        proposals = load_ff_proposals(conn) + load_pending_proposals(conn)
        fills = load_fills(conn)
        trades: list[dict[str, Any]] = []
        for p in proposals:
            t = {k: p[k] for k in ("source", "proposal_id", "ticker", "earnings_date",
                                   "strategy", "status", "executed")}
            cand = p["candidate"]
            outcome = load_outcome(conn, p["ticker"], p["earnings_date"]) \
                if p["earnings_date"] else {}
            t["expected_move_pct"] = outcome.get("expected_move_pct")
            t["actual_move_pct"] = outcome.get("actual_move_pct")
            t["actual_move_direction"] = outcome.get("actual_move_direction")
            t["timing"] = outcome.get("timing")
            t["event_move_pct"] = None
            t["hit_event"] = None

            # implied event move from candidate quotes (ff_ladder only)
            proposal_date = None
            if p["created_at"]:
                try:
                    proposal_date = datetime.fromisoformat(
                        str(p["created_at"]).replace("Z", "+00:00")).date()
                except ValueError:
                    proposal_date = None
            t["implied_event_move_pct"] = (
                implied_event_move_pct(cand, proposal_date)
                if (p["source"] == "ff_ladder" and proposal_date) else None
            )
            t["hist_rms_move_pct"] = (
                round(float(cand["hist_rms_move"]) * 100.0, 4)
                if cand.get("hist_rms_move") is not None else None
            )

            # win convention: realized move smaller than expected/implied
            actual = t["actual_move_pct"]
            if actual is not None and t["expected_move_pct"]:
                t["hit"] = abs(actual) < t["expected_move_pct"]
            elif actual is not None and t["implied_event_move_pct"]:
                t["hit"] = abs(actual) < t["implied_event_move_pct"]
            else:
                t["hit"] = None

            # fills / P&L
            fill = fills.get(p["order_id"] or "", {})
            t["entry_debit"] = fill.get("entry_debit")
            if t["entry_debit"] is None and p["executed"]:
                t["entry_debit"] = cand.get("entry_price")  # pending_trades est.
            t["qty"] = fill.get("qty") or 1.0
            t["position_status"] = "closed" if fill.get("closed") else (
                "open" if fill else None)
            t["pnl"] = None
            t["exit_date"] = None
            t["_near_symbol"] = cand.get("near_symbol")
            t["_far_symbol"] = cand.get("far_symbol")
            trades.append(t)

        notes: list[str] = []
        if fetch_exit_prices:
            _apply_market_data(trades, rate_sleep, notes)

        by_source: dict[str, Any] = {}
        sources = sorted({t["source"] for t in trades})
        for src in sources:
            by_source[src] = aggregate([t for t in trades if t["source"] == src])
        by_strategy: dict[str, Any] = {}
        for strat in sorted({t["strategy"] for t in trades}):
            by_strategy[strat] = aggregate([t for t in trades if t["strategy"] == strat])

        summary = aggregate(trades)
        if not summary["significant"]:
            notes.append(
                f"Sample too small for significance (scorable n="
                f"{summary['scorable']} < {MIN_SIGNIFICANT_N}); descriptive stats only."
            )
        for t in trades:  # strip internal fields
            t.pop("_near_symbol", None)
            t.pop("_far_symbol", None)

        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "db_path": str(db_path),
            "summary": summary,
            "by_source": by_source,
            "by_strategy": by_strategy,
            "trades": trades,
            "notes": notes,
        }
    finally:
        conn.close()


def _apply_market_data(trades: list[dict[str, Any]], rate_sleep: float,
                       notes: list[str]) -> None:
    """Fetch (a) timing-aware realized event moves for ALL approved trades and
    (b) exit leg prices for executed trades, via Polygon (network, paced)."""
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    import os

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from polygon_backfill import PolygonClient  # lazy: network-capable import

    api_key = os.environ.get("POLYGON_API_KEY", "")
    if not api_key:
        notes.append("POLYGON_API_KEY not set — market data not fetched.")
        return
    pg = PolygonClient(api_key, sleep=rate_sleep)
    for t in trades:
        if not t["earnings_date"]:
            continue
        ed = date.fromisoformat(t["earnings_date"])
        timing = t.get("timing")
        # (a) timing-aware realized event move (stock aggs — fast endpoint)
        bars = pg.daily_bars(t["ticker"], ed - timedelta(days=7),
                             ed + timedelta(days=6), limit=20)
        move = compute_event_move(bars, ed, timing)
        if move is not None:
            t["event_move_pct"] = round(move, 4)
            ref = t["expected_move_pct"] or t["implied_event_move_pct"]
            t["hit_event"] = abs(move) < ref if ref else None
        else:
            notes.append(f"{t['ticker']}: no stock bars around {ed} (Polygon).")
        # (b) exit pricing for executed trades
        if not t["executed"] or t["entry_debit"] is None:
            continue
        near, far = t.get("_near_symbol"), t.get("_far_symbol")
        if not near or not far:
            continue
        credit, exit_date = _fetch_exit_credit(pg, near, far, ed, timing)
        if credit is None:
            notes.append(f"{t['ticker']}: no exit prices after {ed} (Polygon).")
            continue
        t["exit_date"] = exit_date
        t["exit_credit"] = round(credit, 4)
        t["pnl"] = round((credit - t["entry_debit"]) * 100.0 * t["qty"], 2)


def format_text(report: dict[str, Any]) -> str:
    s = report["summary"]
    lines = [
        "Proposal quality report",
        f"  db: {report['db_path']}",
        f"  approved proposals: {s['approved']}  executed: {s['executed']}  "
        f"with realized outcome: {s['with_outcome']}",
    ]
    if s["scorable"]:
        lines.append(
            f"  beat-expected-move hit rate (stored outcome convention): "
            f"{s['hits']}/{s['scorable']} "
            f"({s['hit_rate']:.0%})  binomial p={s['binomial_p']:.3f} vs 50%"
        )
    if s["event_scorable"]:
        lines.append(
            f"  beat-expected-move hit rate (timing-aware event move): "
            f"{s['event_hits']}/{s['event_scorable']} "
            f"({s['event_hit_rate']:.0%})  binomial p={s['event_binomial_p']:.3f} vs 50%"
        )
    if s["mean_actual_over_expected"] is not None:
        lines.append(
            f"  mean |actual|/expected move: {s['mean_actual_over_expected']:.2f}"
        )
    if s["total_pnl"] is not None:
        lines.append(
            f"  P&L over {s['pnl_trades']} executed trades: ${s['total_pnl']:+,.2f}"
        )
    else:
        lines.append("  P&L: unavailable (run with --fetch-exit-prices)")
    for src, agg in report["by_source"].items():
        hr = f"{agg['hits']}/{agg['scorable']}" if agg["scorable"] else "n/a"
        pnl = (f"${agg['total_pnl']:+,.2f}" if agg["total_pnl"] is not None else "n/a")
        lines.append(
            f"  [{src}] approved={agg['approved']} executed={agg['executed']} "
            f"hits={hr} pnl={pnl}"
        )
    for note in report["notes"]:
        lines.append(f"  NOTE: {note}")
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> dict[str, Any]:
    ap = argparse.ArgumentParser(
        description="Measure approved trade proposals against realized earnings outcomes."
    )
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--json", type=Path, default=None, help="write full report JSON")
    ap.add_argument("--fetch-exit-prices", action="store_true",
                    help="fetch leg exit prices from Polygon (network, rate-limited)")
    ap.add_argument("--rate-sleep", type=float, default=13.0)
    args = ap.parse_args(argv)

    configure(args.db)
    report = build_report(args.db, fetch_exit_prices=args.fetch_exit_prices,
                          rate_sleep=args.rate_sleep)
    print(format_text(report))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2))
        print(f"\nJSON written to {args.json}")
    return report


if __name__ == "__main__":
    main()
