"""Live signal layer: latest scan session -> executable Trade objects.

The backtest strategies (earnings_edge.backtest)
replay HISTORICAL rows and can never produce a live trade: their legs fall
back to expiry = earnings_date (DTE 0) and some require the outcome to
already be known. This module maps the most recent 14:00 ET earnings scan
(scanner_scan_outputs, enriched from live_calendar_candidates) into Trade
objects with real legs — real strikes, real near/far expiries, real quoted
combo-ask debits — for the strategies that have a pre-event live mapping.

LIVE_STRATEGIES (mapped):
  - calendar_call_ml     model_decision == TAKE          -> CALENDAR
  - vol_risk_premium     iv_rv >= 1.4 and EM >= 6%       -> SHORT_STRADDLE
  - short_straddle       iv_rv >= 1.2 and EM >= 6%       -> SHORT_STRADDLE
    (filter-only: the magnitude model's training features are scan-time
    snapshot fields not all present on live rows, mirroring the backtest's
    own filter-only fallback when the model artifact is absent)

NOT mapped:
  - earnings_quality is a POST-event strategy (trades the surprise after the
    move is known, LONG/SHORT stock). No pre-event live mapping exists; it
    stays backtest-only until someone designs the post-event entry path.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from typing import Optional

import pandas as pd

from earnings_edge.trading_types import Trade

logger = logging.getLogger(__name__)

# Entry thresholds mirrored from the backtest strategy classes
# (positional_strategies.VolRiskPremium / .ShortStraddle,
#  strategies.DebitSizeExploit).
VRP_IV_RV_MIN = 1.4
VRP_MIN_EXPECTED_MOVE = 6.0
SS_IV_RV_MIN = 1.2
SS_MIN_EXPECTED_MOVE = 6.0
DSE_MAX_DEBIT_PCT = 0.03

LIVE_STRATEGIES = [
    "calendar_call_ml",
    "vol_risk_premium",
    "short_straddle",
]


def _parse_ts(raw) -> Optional[datetime]:
    try:
        ts = pd.to_datetime(raw)
    except Exception:
        return None
    if ts.tzinfo is None:
        ts = ts.tz_localize(timezone.utc)
    return ts.to_pydatetime()


def _parse_date(raw) -> Optional[date]:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    try:
        return pd.to_datetime(raw).date()
    except Exception:
        return None


def latest_scan_frame(db_path=None, max_age_hours: float = 30.0) -> pd.DataFrame:
    """Rows of the single most recent scan session, upcoming earnings only.

    Reads scanner_scan_outputs (every candidate the scanner saw) and LEFT
    JOINs live_calendar_candidates for straddle_price (only quoted candidates
    have it). Deduped per ticker (best row: passed first, then lowest tier).
    Returns an empty DataFrame when there is no fresh session.
    """
    from sqlalchemy import text

    from earnings_edge.db import configure, get_engine

    if db_path is not None:
        configure(db_path)
    engine = get_engine()
    with engine.connect() as conn:
        tables = {
            r[0]
            for r in conn.execute(
                text("SELECT name FROM sqlite_master WHERE type='table'")
            ).fetchall()
        }
        if "scanner_scan_outputs" not in tables:
            return pd.DataFrame()
        latest = conn.execute(
            text("SELECT MAX(scan_timestamp) FROM scanner_scan_outputs")
        ).scalar()
    if not latest:
        return pd.DataFrame()
    ts = _parse_ts(latest)
    if ts is None:
        return pd.DataFrame()
    if datetime.now(timezone.utc) - ts > timedelta(hours=max_age_hours):
        logger.info("latest scan session %s is stale (>%.0fh)", latest, max_age_hours)
        return pd.DataFrame()
    df = pd.read_sql(
        text("SELECT * FROM scanner_scan_outputs WHERE scan_timestamp = :latest"),
        engine,
        params={"latest": latest},
    )
    if "live_calendar_candidates" in tables:
        live = pd.read_sql(
            text(
                "SELECT ticker, earnings_date, straddle_price "
                "FROM live_calendar_candidates WHERE scan_timestamp = :latest"
            ),
            engine,
            params={"latest": latest},
        )
        if not live.empty:
            live = live.drop_duplicates(subset=["ticker", "earnings_date"])
            df = df.merge(live, on=["ticker", "earnings_date"], how="left")

    if df.empty:
        return df

    scan_day = ts.date()
    df["_earnings"] = df["earnings_date"].map(_parse_date)
    df = df[df["_earnings"].notna() & (df["_earnings"] >= scan_day)]
    if df.empty:
        return df
    # best row per ticker: passed candidates first, then lowest tier
    df["_passed"] = df["passed"].fillna(0).astype(int)
    df["_tier"] = df["tier"].fillna(99).astype(int)
    df = df.sort_values(["_passed", "_tier"], ascending=[False, True])
    df = df.drop_duplicates(subset=["ticker"], keep="first")
    return df.drop(columns=["_passed", "_tier"])


def _first_positive(*vals) -> float:
    for v in vals:
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if f > 0:
            return f
    return 0.0


def _calendar_trade(row, strategy: str, decision: str = "TAKE") -> Optional[Trade]:
    strike = _first_positive(row.get("strike"))
    price = _first_positive(row.get("price"))
    near = _parse_date(row.get("near_expiry"))
    far = _parse_date(row.get("far_expiry"))
    earnings = row.get("_earnings")
    if not (strike and price and near and far and earnings):
        return None
    debit = _first_positive(
        row.get("net_debit_ask"), row.get("net_debit"), row.get("net_debit_mid")
    )
    if debit <= 0:
        return None
    score = row.get("model_expected_return")
    try:
        score = float(score) if score is not None and pd.notna(score) else None
    except (TypeError, ValueError):
        score = None
    return Trade(
        ticker=row["ticker"],
        earnings_date=earnings,
        scan_date=_parse_date(row.get("scan_timestamp")) or earnings,
        strategy=strategy,
        side="CALENDAR",
        entry_price=debit,
        features={
            "near_strike": strike,
            "far_strike": strike,
            "atm_strike": strike,
            "near_expiry": near.isoformat(),
            "far_expiry": far.isoformat(),
            "net_debit": debit,
            "price": price,
        },
        model_score=score,
        ml_decision=decision,
        notes=f"live scan; exp_ret={score}; iv_rv={row.get('iv_rv_ratio')}",
    )


def _straddle_trade(row, strategy: str) -> Optional[Trade]:
    strike = _first_positive(row.get("strike"))
    price = _first_positive(row.get("price"))
    near = _parse_date(row.get("near_expiry"))
    earnings = row.get("_earnings")
    if not (strike and price and near and earnings):
        return None
    em_dollars = _first_positive(row.get("expected_move_dollars"))
    if em_dollars <= 0:
        em_pct = _first_positive(row.get("expected_move_pct"))
        em_dollars = price * em_pct / 100.0 if em_pct else 0.0
    credit = _first_positive(row.get("straddle_price"), em_dollars)
    if credit <= 0 or em_dollars <= 0:
        return None
    return Trade(
        ticker=row["ticker"],
        earnings_date=earnings,
        scan_date=_parse_date(row.get("scan_timestamp")) or earnings,
        strategy=strategy,
        side="SHORT_STRADDLE",
        entry_price=credit,
        features={
            "atm_strike": strike,
            "expiry": near.isoformat(),
            "expected_move_dollars": em_dollars,
            "iv_rv": row.get("iv_rv_ratio"),
        },
        model_score=None,
        ml_decision="TAKE",
        notes=(
            f"live scan; iv_rv={row.get('iv_rv_ratio')}; "
            f"em={row.get('expected_move_pct')}%"
        ),
    )


def calendar_row_reason(row) -> str:
    """Why a calendar_call_ml row died or passed: take | model_skip | no_quote | no_decision."""
    get = row.get if hasattr(row, "get") else lambda k, default=None: (
        row[k] if k in row else default
    )
    strike = _first_positive(get("strike"))
    near = _parse_date(get("near_expiry"))
    far = _parse_date(get("far_expiry"))
    debit = _first_positive(get("net_debit_ask"), get("net_debit"), get("net_debit_mid"))
    decision = get("model_decision")
    if not (strike and near and far and debit > 0):
        return "no_quote"
    if decision == "SKIP":
        return "model_skip"
    if decision == "TAKE":
        return "take"
    return "no_decision"


def calendar_funnel_reasons(df: pd.DataFrame) -> dict[str, int]:
    """Count SKIP vs no-quote vs take for calendar_call_ml screening."""
    counts = {"take": 0, "model_skip": 0, "no_quote": 0, "no_decision": 0}
    if df is None or df.empty:
        return counts
    for _, row in df.iterrows():
        counts[calendar_row_reason(row)] += 1
    return counts


def build_live_trades(df: pd.DataFrame, strategy_name: str) -> list[Trade]:
    """Map a latest_scan_frame onto one strategy's live entry rules."""
    if df is None or df.empty:
        return []
    trades: list[Trade] = []

    if strategy_name == "calendar_call_ml":
        rows = df[df["model_decision"] == "TAKE"]
        for _, row in rows.iterrows():
            t = _calendar_trade(row, strategy_name, decision="TAKE")
            if t:
                trades.append(t)

    elif strategy_name == "vol_risk_premium":
        rows = df[
            (df["iv_rv_ratio"].fillna(0) >= VRP_IV_RV_MIN)
            & (df["expected_move_pct"].fillna(0) >= VRP_MIN_EXPECTED_MOVE)
        ]
        for _, row in rows.iterrows():
            t = _straddle_trade(row, strategy_name)
            if t:
                trades.append(t)

    elif strategy_name == "short_straddle":
        rows = df[
            (df["iv_rv_ratio"].fillna(0) >= SS_IV_RV_MIN)
            & (df["expected_move_pct"].fillna(0) >= SS_MIN_EXPECTED_MOVE)
        ]
        for _, row in rows.iterrows():
            t = _straddle_trade(row, strategy_name)
            if t:
                trades.append(t)

    else:
        logger.warning("no live mapping for strategy %s", strategy_name)
    return trades
