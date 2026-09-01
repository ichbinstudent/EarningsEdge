"""Repository functions — the only SQL-bearing call sites after the migration.

Each helper is a port of the matching function in ``db_legacy.py`` (or
``picks.py`` / ``signals.py``) with the ``conn`` parameter dropped. During
the incremental migration a leading sqlite3 connection is still accepted
and used (same-connection writes) so existing ``fn(conn, ...)`` call sites
keep working until later tasks convert them.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime, timedelta, timezone
from typing import Any, Optional

import pandas as pd
from sqlalchemy import func, select, text, update
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import class_mapper

from earnings_edge.config import get_logger

from .engine import get_engine, session_scope
from .models import (
    AdoptedPosition,
    AlpacaPosition,
    CalendarCallTrade,
    DataCatalog,
    EquitySnapshot,
    ExitProposal,
    FfLadder,
    JobRun,
    ManagedPosition,
    ModelRegistry,
    OptionsChain,
    PendingTrade,
    ProposalFunnel,
    RiskEvent,
    RiskState,
    ScanRun,
    ScannerScanOutput,
    Snapshot,
    StrategyState,
    TradeEvent,
)

logger = get_logger("db")

_DAILY_SIGNAL_COLS = [
    "ticker", "signal_date", "option_volume", "atm_iv", "skew_25d",
    "skew_zscore", "skew_mean", "iv_pctl_1y", "ts_momentum", "relative_momentum",
]

_SNAPSHOT_COLS = [
    "ticker", "earnings_date", "scan_date", "timing",
    "price", "avg_volume_30d", "market_cap",
    "has_options", "nearest_expiry", "days_to_expiry", "total_open_interest",
    "atm_iv_near", "rv30", "iv30_rv30", "hist_vol_3m",
    "term_slope", "term_structure_valid",
    "expected_move_pct", "expected_move_dollars", "straddle_price",
    "atm_call_delta", "atm_put_delta", "atm_call_iv", "atm_put_iv",
    "sigma_baseline_1y", "sigma_short_leg", "sigma_short_leg_fair", "actual_to_fair_ratio",
    "recommendation",
    "mc_win_rate", "mc_quarters",
    "collection_error",
    "data_source",
]

_LIVE_CANDIDATE_COLS = [
    "scan_timestamp", "ticker", "earnings_date",
    "tier", "passed", "near_miss", "scanner_reason", "display_status",
    "price", "volume", "market_cap",
    "strike", "near_expiry", "far_expiry",
    "days_to_expiry", "total_open_interest",
    "near_bid", "near_ask", "far_bid", "far_ask",
    "near_entry", "far_entry",
    "net_debit", "net_debit_bid", "net_debit_mid", "net_debit_ask",
    "atm_iv_near", "sigma_baseline_1y", "sigma_short_leg", "sigma_short_leg_fair",
    "actual_to_fair_ratio", "iv_rv_ratio", "hist_vol_3m",
    "term_slope", "term_structure_valid",
    "expected_move_pct", "expected_move_dollars", "straddle_price",
    "atm_call_delta", "atm_put_delta", "atm_call_iv", "atm_put_iv",
    "win_rate", "win_quarters",
    "model_expected_return", "model_decision", "model_rejection_reasons",
    "selected_by_bot", "features_json",
    "exit_value", "pnl_dollars", "return_on_debit", "outcome_fetched_at",
]

_SCANNER_OUTPUT_COLS = [
    "scan_timestamp", "ticker", "earnings_date",
    "tier", "passed", "near_miss", "scanner_reason", "display_status",
    "price", "volume", "market_cap",
    "strike", "near_expiry", "far_expiry",
    "days_to_expiry", "total_open_interest",
    "net_debit", "net_debit_bid", "net_debit_mid", "net_debit_ask",
    "atm_iv_near", "sigma_baseline_1y", "sigma_short_leg", "sigma_short_leg_fair",
    "actual_to_fair_ratio", "iv_rv_ratio",
    "term_slope", "term_structure_valid",
    "expected_move_pct", "expected_move_dollars",
    "win_rate", "win_quarters",
    "model_expected_return", "model_decision", "model_rejection_reasons",
    "selected_by_bot", "features_json",
    "exit_value", "pnl_dollars", "return_on_debit", "outcome_fetched_at",
]

_OPTIONS_CHAIN_COLS = [
    "collector_run_id", "ticker", "scan_date", "contract_ticker",
    "underlying", "expiry", "strike", "contract_type", "style",
    "bid", "ask", "bid_size", "ask_size", "midpoint",
    "close", "open_price", "high", "low",
    "trade_count", "volume", "vwap",
    "implied_volatility", "delta", "gamma", "theta", "vega",
    "captured_at", "captured_hour",
]

_SCAN_RUN_COLS = [
    "scan_timestamp", "scanner_name", "trigger_type",
    "candidate_count", "tier1_count", "tier2_count", "take_count",
    "duration_secs", "success", "error_message",
]


def _is_connection(obj: Any) -> bool:
    if obj is None or isinstance(obj, (dict, str, int, float, bool, list, tuple, pd.DataFrame)):
        return False
    if isinstance(obj, sqlite3.Connection):
        return True
    return hasattr(obj, "cursor") and hasattr(obj, "commit") and hasattr(obj, "rollback")


def _split_conn(args: tuple) -> tuple[Any, tuple]:
    if args and _is_connection(args[0]):
        return args[0], args[1:]
    return None, args


def _execute(conn, sql: str, params, *, many: bool = False, commit: bool = True):
    if conn is not None:
        cur = conn.executemany(sql, params) if many else conn.execute(sql, params)
        if commit:
            conn.commit()
        return cur
    with session_scope() as s:
        result = s.execute(text(sql), params)
        return result


def _fetchall(conn, sql: str, params) -> list[dict]:
    if conn is not None:
        return [dict(r) for r in conn.execute(sql, params).fetchall()]
    with session_scope() as s:
        return [dict(r) for r in s.execute(text(sql), params).mappings().all()]


def _insert_row(conn, table: str, cols: list[str], row: dict, *, or_ignore: bool = False) -> int:
    verb = "INSERT OR IGNORE" if or_ignore else "INSERT"
    placeholders = ", ".join(f":{c}" for c in cols)
    sql = f"{verb} INTO {table} ({', '.join(cols)}) VALUES ({placeholders})"
    result = _execute(conn, sql, {c: row.get(c) for c in cols})
    return (result.lastrowid or 0) if result is not None else 0


def insert_snapshot(*args, row: dict | None = None) -> int:
    """Insert a single feature snapshot and return the row id."""
    conn, rest = _split_conn(args)
    if row is None:
        row = rest[0]
    rid = _insert_row(conn, "snapshots", _SNAPSHOT_COLS, row, or_ignore=True)
    if rid == 0:
        logger.debug(
            "Skipping duplicate snapshot for %s @ %s",
            row.get("ticker"), row.get("scan_date"),
        )
    return rid


def fetch_pending_outcomes(*args, min_age_days: int = 2) -> list[dict]:
    """Return snapshots where earnings_date is past and outcome not yet fetched."""
    conn, rest = _split_conn(args)
    if rest:
        min_age_days = rest[0]
    cutoff = date.today().isoformat()
    age_cutoff = (date.today() - timedelta(days=min_age_days)).isoformat()
    return _fetchall(
        conn,
        "SELECT * FROM snapshots "
        "WHERE outcome_fetched_at IS NULL "
        "  AND earnings_date <= :cutoff "
        "  AND earnings_date <= :age_cutoff "
        "ORDER BY CASE WHEN has_options = 1 THEN 0 ELSE 1 END, earnings_date",
        {"cutoff": cutoff, "age_cutoff": age_cutoff},
    )


def fetch_pending_live_candidates(*args, min_age_days: int = 2) -> list[dict]:
    """Return live_calendar_candidates needing a stock-move outcome."""
    conn, rest = _split_conn(args)
    if rest:
        min_age_days = rest[0]
    age_cutoff = (date.today() - timedelta(days=min_age_days)).isoformat()
    return _fetchall(
        conn,
        "SELECT * FROM live_calendar_candidates "
        "WHERE outcome_fetched_at IS NULL "
        "  AND earnings_date IS NOT NULL "
        "  AND earnings_date <= :age_cutoff "
        "ORDER BY earnings_date",
        {"age_cutoff": age_cutoff},
    )


def update_live_candidate_move(*args, cid: int | None = None, outcome: dict | None = None) -> None:
    """Write a stock-move outcome row to a live_calendar_candidates row."""
    conn, rest = _split_conn(args)
    if cid is None:
        cid = rest[0]
        outcome = rest[1] if len(rest) > 1 else outcome
    _execute(
        conn,
        """UPDATE live_calendar_candidates SET
            pre_earnings_close = :pre_earnings_close,
            post_earnings_close = :post_earnings_close,
            actual_move_pct = :actual_move_pct,
            actual_move_direction = :actual_move_direction,
            max_intraday_range_pct = :max_intraday_range_pct,
            outcome_fetched_at = :outcome_fetched_at
        WHERE id = :id""",
        {**outcome, "id": cid},
    )


def record_live_candidate_failure(*args, cid: int | None = None, max_retries: int | None = None) -> None:
    """Bump attempt_count; mark unavailable once retries are exhausted."""
    conn, rest = _split_conn(args)
    if cid is None:
        cid = rest[0]
        max_retries = rest[1] if len(rest) > 1 else max_retries
    rows = _fetchall(
        conn,
        "SELECT outcome_attempt_count FROM live_calendar_candidates WHERE id = :id",
        {"id": cid},
    )
    attempt_count = (rows[0]["outcome_attempt_count"] if rows else None) or 0
    attempt_count += 1
    if attempt_count >= max_retries:
        _execute(
            conn,
            "UPDATE live_calendar_candidates "
            "SET outcome_fetched_at = 'unavailable', "
            "outcome_attempt_count = :n WHERE id = :id",
            {"n": attempt_count, "id": cid},
        )
    else:
        _execute(
            conn,
            "UPDATE live_calendar_candidates "
            "SET outcome_attempt_count = :n WHERE id = :id",
            {"n": attempt_count, "id": cid},
        )


def insert_live_calendar_candidate(*args, row: dict | None = None) -> int:
    """Insert a live call-calendar candidate quote/model snapshot."""
    conn, rest = _split_conn(args)
    if row is None:
        row = rest[0]
    return _insert_row(conn, "live_calendar_candidates", _LIVE_CANDIDATE_COLS, row)


def insert_scanner_output(*args, row: dict | None = None) -> int:
    """Insert a scanner output row for backtest/audit purposes."""
    conn, rest = _split_conn(args)
    if row is None:
        row = rest[0]
    return _insert_row(conn, "scanner_scan_outputs", _SCANNER_OUTPUT_COLS, row)


def insert_options_chain_rows(*args, rows: list[dict] | None = None) -> int:
    """Bulk-insert Alpaca options-chain snapshot rows.

    Uses INSERT OR IGNORE so re-running on the same contracts skips duplicates.
    Returns number of rows actually inserted.
    """
    conn, rest = _split_conn(args)
    if rows is None:
        rows = rest[0]
    if not rows:
        return 0
    placeholders = ", ".join(f":{c}" for c in _OPTIONS_CHAIN_COLS)
    sql = (
        f"INSERT OR IGNORE INTO options_chain ({', '.join(_OPTIONS_CHAIN_COLS)}) "
        f"VALUES ({placeholders})"
    )
    payload = [{c: r.get(c) for c in _OPTIONS_CHAIN_COLS} for r in rows]
    result = _execute(conn, sql, payload, many=True)
    return result.rowcount or 0


def fetch_chain_for_ticker(*args, ticker: str | None = None, scan_date: str | None = None) -> list[dict]:
    """Return all options_chain rows for one ticker/scan_date."""
    conn, rest = _split_conn(args)
    if ticker is None:
        ticker = rest[0]
        scan_date = rest[1] if len(rest) > 1 else scan_date
    return _fetchall(
        conn,
        "SELECT * FROM options_chain WHERE ticker = :ticker AND scan_date = :scan_date",
        {"ticker": ticker, "scan_date": scan_date},
    )


def fetch_chain_for_ticker_date(date: str) -> list:  # noqa: ARG001
    """Return all options_chain rows with expiry on or after ``date``."""
    return []  # placeholder — kept for type-checker parity with other fetch funcs


def update_outcome(*args, snapshot_id: int | None = None, outcome: dict | None = None) -> None:
    """Write outcome data back to a snapshot row."""
    conn, rest = _split_conn(args)
    if snapshot_id is None:
        snapshot_id = rest[0]
        outcome = rest[1] if len(rest) > 1 else outcome
    _execute(
        conn,
        """UPDATE snapshots SET
            pre_earnings_close = :pre_earnings_close,
            post_earnings_close = :post_earnings_close,
            actual_move_pct = :actual_move_pct,
            actual_move_direction = :actual_move_direction,
            max_intraday_range_pct = :max_intraday_range_pct,
            outcome_fetched_at = :outcome_fetched_at
        WHERE id = :id""",
        {**outcome, "id": snapshot_id},
    )


def insert_scan_run(*args, row: dict | None = None) -> int:
    """Insert a scan-run audit row and return its id."""
    conn, rest = _split_conn(args)
    if row is None:
        row = rest[0]
    return _insert_row(conn, "scan_runs", _SCAN_RUN_COLS, row)


def persist_picks(*args, picks: dict | None = None, as_of=None) -> int:
    """Persist one day's pick lists (insert-or-replace per date/strategy/ticker).

    Stores each pick as (pick_date, strategy, rank, ticker, signals_json) so
    historical pick performance can be evaluated later. Returns rows written.
    """
    conn, rest = _split_conn(args)
    if picks is None:
        picks = rest[0]
        as_of = rest[1] if len(rest) > 1 else as_of
    pick_date = as_of.isoformat() if isinstance(as_of, (date, datetime)) else str(as_of)
    payload = []
    for strategy, df in picks.items():
        for rank, (_, row) in enumerate(df.iterrows(), start=1):
            signals = {
                k: (None if pd.isna(v) else v.item() if hasattr(v, "item") else v)
                for k, v in row.items()
            }
            payload.append({
                "pick_date": pick_date,
                "strategy": strategy,
                "rank": rank,
                "ticker": str(row["ticker"]),
                "signals_json": json.dumps(signals, default=str),
            })
    if not payload:
        return 0
    sql = (
        "INSERT OR REPLACE INTO picks "
        "(pick_date, strategy, rank, ticker, signals_json) "
        "VALUES (:pick_date, :strategy, :rank, :ticker, :signals_json)"
    )
    result = _execute(conn, sql, payload, many=True)
    return result.rowcount or 0


def load_picks(*args, pick_date: str | None = None, strategy: Optional[str] = None) -> pd.DataFrame:
    """Read persisted picks for one date (optionally one strategy)."""
    conn, rest = _split_conn(args)
    if pick_date is None:
        pick_date = rest[0]
        if len(rest) > 1:
            strategy = rest[1]
    sql = "SELECT * FROM picks WHERE pick_date = :pick_date"
    params: dict = {"pick_date": pick_date}
    if strategy:
        sql += " AND strategy = :strategy"
        params["strategy"] = strategy
    sql += " ORDER BY strategy, rank"
    if conn is not None:
        return pd.read_sql_query(sql, conn, params=params)
    with session_scope() as s:
        return pd.read_sql_query(text(sql), s.connection(), params=params)


def upsert_daily_signals(*args, rows: list[dict] | None = None) -> int:
    """Insert-or-replace daily signal rows. Returns affected row count."""
    conn, rest = _split_conn(args)
    if rows is None:
        rows = rest[0]
    if not rows:
        return 0
    cols = ", ".join(_DAILY_SIGNAL_COLS)
    placeholders = ", ".join(f":{c}" for c in _DAILY_SIGNAL_COLS)
    sql = f"INSERT OR REPLACE INTO daily_signals ({cols}) VALUES ({placeholders})"
    payload = [{c: r.get(c) for c in _DAILY_SIGNAL_COLS} for r in rows]
    result = _execute(conn, sql, payload, many=True)
    return result.rowcount or 0


def record_snapshot_outcome_failure(
    *args, snapshot_id: int | None = None, max_retries: int | None = None
) -> None:
    """Bump snapshots.outcome_attempt_count; mark unavailable once retries are exhausted."""
    conn, rest = _split_conn(args)
    if snapshot_id is None:
        snapshot_id = rest[0]
        max_retries = rest[1] if len(rest) > 1 else max_retries
    rows = _fetchall(
        conn,
        "SELECT outcome_attempt_count FROM snapshots WHERE id = :id",
        {"id": snapshot_id},
    )
    attempt_count = (rows[0]["outcome_attempt_count"] if rows else None) or 0
    attempt_count += 1
    if attempt_count >= max_retries:
        _execute(
            conn,
            "UPDATE snapshots "
            "SET outcome_fetched_at = 'unavailable', "
            "outcome_attempt_count = :n WHERE id = :id",
            {"n": attempt_count, "id": snapshot_id},
        )
    else:
        _execute(
            conn,
            "UPDATE snapshots "
            "SET outcome_attempt_count = :n WHERE id = :id",
            {"n": attempt_count, "id": snapshot_id},
        )


def snapshots_earnings_on_date(*args, earnings_date: str | None = None) -> list[tuple[str, str]]:
    """(ticker, timing) pairs with earnings on exactly ``earnings_date``,
    optionable only. Used by the FF ladder/arb proposal builder to source
    tomorrow's earnings names without re-running the full scan pipeline."""
    conn, rest = _split_conn(args)
    if earnings_date is None:
        earnings_date = rest[0]
    rows = _fetchall(
        conn,
        "SELECT DISTINCT ticker, timing FROM snapshots "
        "WHERE has_options = 1 AND earnings_date = :earnings_date",
        {"earnings_date": earnings_date},
    )
    return [(r["ticker"], r["timing"]) for r in rows]


def snapshots_optionable_universe(*args, max_tickers: int | None = None) -> list[str]:
    """Upcoming optionable earnings first, then recently-optionable names."""
    conn, rest = _split_conn(args)
    if max_tickers is None:
        max_tickers = rest[0]
    upcoming_rows = _fetchall(
        conn,
        "SELECT DISTINCT ticker FROM snapshots "
        "WHERE has_options = 1 "
        "AND earnings_date >= date('now') "
        "AND earnings_date <= date('now', '+21 days')",
        {},
    )
    recent_rows = _fetchall(
        conn,
        "SELECT DISTINCT ticker FROM snapshots "
        "WHERE has_options = 1 "
        "ORDER BY rowid DESC LIMIT :max_tickers",
        {"max_tickers": max_tickers},
    )
    upcoming = [r["ticker"] for r in upcoming_rows]
    recent = [r["ticker"] for r in recent_rows]
    out: list[str] = []
    seen: set[str] = set()
    for t in upcoming + recent:
        if not t or t in seen:
            continue
        seen.add(t)
        out.append(t)
        if len(out) >= max_tickers:
            break
    return out


_CALENDAR_CALL_TRADE_COLS = [
    "snapshot_id", "ticker", "earnings_date", "scan_date",
    "near_expiry", "far_expiry", "strike",
    "near_call_ticker", "far_call_ticker",
    "near_entry", "far_entry", "near_exit", "far_exit",
    "net_debit", "exit_value", "pnl_dollars", "return_on_debit",
    "model_score", "model_recommendation", "model_reason",
    "model_name", "model_scored_at",
]


def calendar_call_trades_upsert(row: dict) -> None:
    """Insert-or-replace one calendar-call backtest trade."""
    placeholders = ", ".join(f":{c}" for c in _CALENDAR_CALL_TRADE_COLS)
    sql = (
        f"INSERT OR REPLACE INTO calendar_call_trades "
        f"({', '.join(_CALENDAR_CALL_TRADE_COLS)}) VALUES ({placeholders})"
    )
    _execute(None, sql, {c: row.get(c) for c in _CALENDAR_CALL_TRADE_COLS})


def calendar_call_trades_list() -> list[dict]:
    """All calendar-call trades ordered by earnings_date, ticker."""
    return _fetchall(
        None,
        "SELECT * FROM calendar_call_trades ORDER BY earnings_date, ticker",
        {},
    )


def calendar_call_trades_with_snapshots() -> list[dict]:
    """Calendar-call trades LEFT JOINed to snapshot features."""
    return _fetchall(
        None,
        "SELECT c.*, s.*, c.snapshot_id AS trade_snapshot_id "
        "FROM calendar_call_trades c "
        "LEFT JOIN snapshots s ON s.id = c.snapshot_id "
        "ORDER BY c.earnings_date, c.ticker",
        {},
    )


def calendar_call_trades_update_model(
    snapshot_id: int,
    *,
    model_score,
    model_recommendation,
    model_reason,
    model_name,
    model_scored_at,
) -> None:
    """Write model score fields onto one stored calendar-call trade."""
    _execute(
        None,
        "UPDATE calendar_call_trades "
        "SET model_score = :model_score, "
        "    model_recommendation = :model_recommendation, "
        "    model_reason = :model_reason, "
        "    model_name = :model_name, "
        "    model_scored_at = :model_scored_at "
        "WHERE snapshot_id = :snapshot_id",
        {
            "snapshot_id": snapshot_id,
            "model_score": model_score,
            "model_recommendation": model_recommendation,
            "model_reason": model_reason,
            "model_name": model_name,
            "model_scored_at": model_scored_at,
        },
    )


def options_chain_latest_contract(
    ticker: str,
    kind: str,
    strike: float,
    expiry: str,
    as_of: str,
) -> dict | None:
    """Latest options_chain quote for one contract on or before ``as_of``."""
    rows = _fetchall(
        None,
        "SELECT midpoint, close, implied_volatility, delta, scan_date "
        "FROM options_chain "
        "WHERE ticker = :ticker AND contract_type = :kind AND strike = :strike "
        "AND expiry = :expiry AND scan_date <= :as_of "
        "ORDER BY scan_date DESC LIMIT 1",
        {
            "ticker": ticker,
            "kind": kind,
            "strike": strike,
            "expiry": expiry,
            "as_of": as_of,
        },
    )
    return rows[0] if rows else None


def _row_dict(obj) -> dict:
    """Model instance -> dict keyed by SQLite column names (not ORM attrs).

    Uses the mapper so ``ManagedPosition.metadata_`` lands under ``"metadata"``
    rather than colliding with ``Base.metadata``.
    """
    out: dict = {}
    for prop in class_mapper(type(obj)).column_attrs:
        value = getattr(obj, prop.key)
        for col in prop.columns:
            out[col.name] = value
    return out


# ---------------------------------------------------------------------------
# pending_trades
# ---------------------------------------------------------------------------

def pending_trades_insert(
    *,
    created_at: str,
    strategy: str,
    ticker: str,
    side: str,
    trade_json: str,
    card_text: str,
    model_score: Optional[float] = None,
) -> Optional[int]:
    """Insert a pending proposal; None if strategy+ticker+side is already pending."""
    with session_scope() as s:
        dup = s.execute(
            select(PendingTrade.id).where(
                PendingTrade.strategy == strategy,
                PendingTrade.ticker == ticker,
                PendingTrade.side == side,
                PendingTrade.status == "pending",
            )
        ).scalar_one_or_none()
        if dup is not None:
            return None
        obj = PendingTrade(
            created_at=created_at,
            strategy=strategy,
            ticker=ticker,
            side=side,
            trade_json=trade_json,
            card_text=card_text,
            model_score=model_score,
            status="pending",
        )
        s.add(obj)
        s.flush()
        return int(obj.id) if obj.id is not None else None


def pending_trades_update_card(proposal_id: int, card_text: str) -> None:
    """UPDATE pending_trades SET card_text=? WHERE id=?"""
    with session_scope() as s:
        s.execute(
            update(PendingTrade)
            .where(PendingTrade.id == proposal_id)
            .values(card_text=card_text)
        )


def pending_trades_get(proposal_id: int) -> Optional[dict]:
    """SELECT * FROM pending_trades WHERE id=?"""
    with session_scope() as s:
        obj = s.get(PendingTrade, proposal_id)
        return _row_dict(obj) if obj is not None else None


def pending_trades_list_pending() -> list[dict]:
    """Pending rows ordered by model_score DESC, id ASC."""
    with session_scope() as s:
        rows = s.execute(
            select(PendingTrade)
            .where(PendingTrade.status == "pending")
            .order_by(PendingTrade.model_score.desc(), PendingTrade.id.asc())
        ).scalars().all()
        return [_row_dict(r) for r in rows]


def pending_trades_mark_decided(
    proposal_id: int,
    status: str,
    *,
    order_json: Optional[str] = None,
    note: Optional[str] = None,
    decided_by: Optional[int] = None,
    decided_at: Optional[str] = None,
) -> None:
    """UPDATE pending_trades SET status, order_json, note, decided_by, decided_at WHERE id=?"""
    if decided_at is None:
        decided_at = datetime.now(timezone.utc).isoformat()
    with session_scope() as s:
        s.execute(
            update(PendingTrade)
            .where(PendingTrade.id == proposal_id)
            .values(
                status=status,
                order_json=order_json,
                note=note,
                decided_by=decided_by,
                decided_at=decided_at,
            )
        )


def proposal_funnel_insert(
    *,
    created_at: str,
    strategies: str,
    counts: str,
    proposals_total: int,
) -> int:
    """INSERT INTO proposal_funnel (created_at, strategies, counts, proposals_total)."""
    with session_scope() as s:
        obj = ProposalFunnel(
            created_at=created_at,
            strategies=strategies,
            counts=counts,
            proposals_total=proposals_total,
        )
        s.add(obj)
        s.flush()
        return int(obj.id) if obj.id is not None else 0


# ---------------------------------------------------------------------------
# exit_proposals
# ---------------------------------------------------------------------------

def exit_proposals_insert(
    *,
    group_id: str,
    strategy: str,
    ticker: str,
    rule: str,
    reason: Optional[str] = None,
    card_text: Optional[str] = None,
    created_at: Optional[str] = None,
) -> Optional[int]:
    """Insert a pending exit card; None if group_id already has a pending row."""
    if created_at is None:
        created_at = datetime.now(timezone.utc).isoformat()
    with session_scope() as s:
        dup = s.execute(
            select(ExitProposal.id).where(
                ExitProposal.group_id == group_id,
                ExitProposal.status == "pending",
            )
        ).scalar_one_or_none()
        if dup is not None:
            return None
        obj = ExitProposal(
            created_at=created_at,
            group_id=group_id,
            strategy=strategy,
            ticker=ticker,
            rule=rule,
            reason=reason,
            card_text=card_text,
            status="pending",
        )
        s.add(obj)
        s.flush()
        return int(obj.id) if obj.id is not None else None


def exit_proposals_get(proposal_id: int) -> Optional[dict]:
    """SELECT * FROM exit_proposals WHERE id=?"""
    with session_scope() as s:
        obj = s.get(ExitProposal, proposal_id)
        return _row_dict(obj) if obj is not None else None


def exit_proposals_list_pending() -> list[dict]:
    """SELECT * FROM exit_proposals WHERE status='pending' ORDER BY id"""
    with session_scope() as s:
        rows = s.execute(
            select(ExitProposal)
            .where(ExitProposal.status == "pending")
            .order_by(ExitProposal.id.asc())
        ).scalars().all()
        return [_row_dict(r) for r in rows]


def exit_proposals_mark(
    proposal_id: int,
    status: str,
    *,
    snoozed_until: Optional[str] = None,
    decided_by: Optional[int] = None,
    decided_at: Optional[str] = None,
) -> None:
    """Update an exit proposal's decision columns."""
    if decided_at is None:
        decided_at = datetime.now(timezone.utc).isoformat()
    with session_scope() as s:
        s.execute(
            update(ExitProposal)
            .where(ExitProposal.id == proposal_id)
            .values(
                status=status,
                snoozed_until=snoozed_until,
                decided_by=decided_by,
                decided_at=decided_at,
            )
        )


# ---------------------------------------------------------------------------
# managed_positions
# ---------------------------------------------------------------------------

def managed_positions_open(
    legs: list[dict],
    strategy: str,
    group_id: str,
    *,
    order_id: Optional[str] = None,
    entry_price: Optional[float] = None,
    metadata: Optional[dict] = None,
    exit_by: Optional[date] = None,
) -> int:
    """Insert one open row per leg. Returns the number of legs written."""
    ts = datetime.now(timezone.utc).isoformat()
    base_meta = dict(metadata or {})
    exit_by_s = exit_by.isoformat() if exit_by else None
    with session_scope() as s:
        for leg in legs:
            meta = json.dumps({
                **base_meta,
                "leg_side": leg.get("side"),
                "option_type": leg.get("option_type"),
                "strike": leg.get("strike"),
                "expiry": str(leg.get("expiry")),
            }, default=str)
            obj = ManagedPosition(
                symbol=leg["symbol"],
                strategy=strategy,
                group_id=group_id,
                qty=float(leg.get("ratio_qty", 1)),
                entry_price=entry_price,
                status="open",
                order_id=order_id,
                opened_at=ts,
                metadata_=meta,
                exit_by=exit_by_s,
            )
            s.add(obj)
            s.flush()
    return len(legs)


def managed_positions_list(strategy: Optional[str] = None) -> list[dict]:
    """SELECT * FROM managed_positions WHERE status='open' [AND strategy=?]"""
    with session_scope() as s:
        stmt = select(ManagedPosition).where(ManagedPosition.status == "open")
        if strategy:
            stmt = stmt.where(ManagedPosition.strategy == strategy)
        rows = s.execute(stmt).scalars().all()
        return [_row_dict(r) for r in rows]


def managed_positions_close(
    group_id: str,
    *,
    exit_price: Optional[float] = None,
    closed_at: Optional[str] = None,
) -> int:
    """Mark all open rows in a group closed. Returns rows updated."""
    with session_scope() as s:
        result = s.execute(
            update(ManagedPosition)
            .where(
                ManagedPosition.group_id == group_id,
                ManagedPosition.status == "open",
            )
            .values(
                status="closed",
                closed_at=closed_at or datetime.now(timezone.utc).isoformat(),
                exit_price=exit_price,
            )
        )
        return result.rowcount or 0


# ---------------------------------------------------------------------------
# snapshots / scan_runs (bot helpers)
# ---------------------------------------------------------------------------

def snapshots_max_scan_date() -> Optional[str]:
    """SELECT MAX(scan_date) FROM snapshots"""
    with session_scope() as s:
        return s.execute(select(func.max(Snapshot.scan_date))).scalar()


def scan_runs_latest_success() -> Optional[str]:
    """scan_timestamp of the latest successful scan_run, else None."""
    with session_scope() as s:
        return s.execute(
            select(ScanRun.scan_timestamp)
            .where(ScanRun.success == 1)
            .order_by(ScanRun.id.desc())
            .limit(1)
        ).scalar_one_or_none()


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_df(sql: str, params: Optional[dict] = None) -> pd.DataFrame:
    return pd.read_sql(text(sql), get_engine(), params=params or {})


def table_exists(name: str) -> bool:
    """SELECT 1 FROM sqlite_master WHERE type='table' AND name=?"""
    with session_scope() as s:
        row = s.execute(
            text("SELECT 1 FROM sqlite_master WHERE type='table' AND name=:name"),
            {"name": name},
        ).first()
        return row is not None


# ---------------------------------------------------------------------------
# risk_state
# ---------------------------------------------------------------------------

def risk_state_get(*, ensure: bool = True) -> dict:
    """Singleton risk_state row (id=1). Inserts halted=0 if missing (unless ensure=False)."""
    with session_scope() as s:
        obj = s.get(RiskState, 1)
        if obj is None:
            if not ensure:
                return {"halted": 0, "reason": None, "tripped_at": None, "tripped_by": None}
            obj = RiskState(id=1, halted=0)
            s.add(obj)
            s.flush()
        return _row_dict(obj)


def risk_state_set_halted(
    halted: bool,
    *,
    reason: Optional[str] = None,
    tripped_at: Optional[str] = None,
    tripped_by: Optional[str] = None,
) -> None:
    """Trip (halted=1 + reason/at/by) or resume (halted=0, fields NULL)."""
    with session_scope() as s:
        obj = s.get(RiskState, 1)
        if obj is None:
            obj = RiskState(id=1, halted=0)
            s.add(obj)
        obj.halted = 1 if halted else 0
        if halted:
            obj.reason = reason
            obj.tripped_at = tripped_at
            obj.tripped_by = tripped_by
        else:
            obj.reason = None
            obj.tripped_at = None
            obj.tripped_by = None


# ---------------------------------------------------------------------------
# risk_events
# ---------------------------------------------------------------------------

def risk_events_insert(
    event_type: str,
    detail: str,
    *,
    strategy: Optional[str] = None,
    ts: Optional[str] = None,
) -> int:
    """INSERT INTO risk_events (ts, event_type, strategy, detail)."""
    with session_scope() as s:
        obj = RiskEvent(
            ts=ts or _utcnow(),
            event_type=event_type,
            strategy=strategy,
            detail=detail,
        )
        s.add(obj)
        s.flush()
        return int(obj.id) if obj.id is not None else 0


def risk_events_list(
    *,
    limit: Optional[int] = None,
    event_type: Optional[str] = None,
    strategy: Optional[str] = None,
    since: Optional[str] = None,
    newest_first: bool = True,
) -> list[dict]:
    """Filtered risk_events rows."""
    with session_scope() as s:
        stmt = select(RiskEvent)
        if event_type is not None:
            stmt = stmt.where(RiskEvent.event_type == event_type)
        if strategy is not None:
            stmt = stmt.where(RiskEvent.strategy == strategy)
        if since is not None:
            stmt = stmt.where(RiskEvent.ts >= since)
        stmt = stmt.order_by(RiskEvent.id.desc() if newest_first else RiskEvent.id.asc())
        if limit is not None:
            stmt = stmt.limit(limit)
        return [_row_dict(r) for r in s.execute(stmt).scalars().all()]


# ---------------------------------------------------------------------------
# equity_snapshots
# ---------------------------------------------------------------------------

def equity_snapshots_insert(
    *,
    ts: str,
    equity: float,
    buying_power: float,
    portfolio_value: float,
    source: str = "alpaca",
) -> int:
    """INSERT INTO equity_snapshots."""
    with session_scope() as s:
        obj = EquitySnapshot(
            ts=ts,
            equity=equity,
            buying_power=buying_power,
            portfolio_value=portfolio_value,
            source=source,
        )
        s.add(obj)
        s.flush()
        return int(obj.id) if obj.id is not None else 0


def equity_snapshots_latest() -> Optional[dict]:
    """Latest snapshot (id DESC). Columns used by latest_equity()."""
    with session_scope() as s:
        obj = s.execute(
            select(EquitySnapshot).order_by(EquitySnapshot.id.desc()).limit(1)
        ).scalar_one_or_none()
        if obj is None:
            return None
        return {
            "ts": obj.ts,
            "equity": obj.equity,
            "buying_power": obj.buying_power,
            "portfolio_value": obj.portfolio_value,
        }


def equity_snapshots_day_start(on: Optional[date] = None) -> Optional[float]:
    """First snapshot equity of the given UTC day (ts >= ISO date)."""
    on = on or datetime.now(timezone.utc).date()
    with session_scope() as s:
        val = s.execute(
            select(EquitySnapshot.equity)
            .where(EquitySnapshot.ts >= on.isoformat())
            .order_by(EquitySnapshot.id.asc())
            .limit(1)
        ).scalar_one_or_none()
        return float(val) if val is not None else None


def equity_snapshots_equities(limit: int = 16) -> list[float]:
    """Most recent `limit` equity values, oldest first (sparkline)."""
    with session_scope() as s:
        rows = s.execute(
            select(EquitySnapshot.equity)
            .order_by(EquitySnapshot.id.desc())
            .limit(limit)
        ).scalars().all()
    return [float(v) for v in reversed(rows) if v is not None]


def equity_snapshots_daily_avg(days: int = 7) -> list[dict]:
    """Avg equity per UTC date, newest first, limited to ``days`` days."""
    with session_scope() as s:
        rows = s.execute(
            text(
                "SELECT substr(ts, 1, 10) AS d, AVG(equity) AS e "
                "FROM equity_snapshots GROUP BY d ORDER BY d DESC LIMIT :n"
            ),
            {"n": days},
        ).mappings().all()
        return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# strategy_state
# ---------------------------------------------------------------------------

def strategy_state_get(name: str) -> Optional[dict]:
    """SELECT * FROM strategy_state WHERE name=?"""
    with session_scope() as s:
        obj = s.get(StrategyState, name)
        return _row_dict(obj) if obj is not None else None


def strategy_state_list() -> list[dict]:
    """SELECT * FROM strategy_state ORDER BY name"""
    with session_scope() as s:
        rows = s.execute(
            select(StrategyState).order_by(StrategyState.name.asc())
        ).scalars().all()
        return [_row_dict(r) for r in rows]


def strategy_state_upsert(
    name: str,
    *,
    lifecycle: str,
    updated_at: Optional[str] = None,
    updated_by: Optional[str] = None,
) -> None:
    """INSERT lifecycle row; ON CONFLICT update lifecycle/updated_* only."""
    ts = updated_at or _utcnow()
    with session_scope() as s:
        stmt = sqlite_insert(StrategyState).values(
            name=name,
            lifecycle=lifecycle,
            updated_at=ts,
            updated_by=updated_by,
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=["name"],
            set_={
                "lifecycle": stmt.excluded.lifecycle,
                "updated_at": stmt.excluded.updated_at,
                "updated_by": stmt.excluded.updated_by,
            },
        )
        s.execute(stmt)


def strategy_state_insert_ignore(
    name: str,
    lifecycle: str,
    *,
    updated_at: Optional[str] = None,
    updated_by: str = "config",
) -> int:
    """INSERT OR IGNORE into strategy_state. Returns 1 if inserted else 0."""
    with session_scope() as s:
        if s.get(StrategyState, name) is not None:
            return 0
        s.add(StrategyState(
            name=name,
            lifecycle=lifecycle,
            updated_at=updated_at or _utcnow(),
            updated_by=updated_by,
        ))
        return 1


def strategy_state_set_enabled(
    name: str,
    enabled: bool,
    *,
    updated_at: Optional[str] = None,
    updated_by: str = "operator",
) -> None:
    """INSERT (lifecycle=paper, enabled) ON CONFLICT UPDATE enabled/updated_*."""
    ts = updated_at or _utcnow()
    with session_scope() as s:
        stmt = sqlite_insert(StrategyState).values(
            name=name,
            lifecycle="paper",
            enabled=1 if enabled else 0,
            updated_at=ts,
            updated_by=updated_by,
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=["name"],
            set_={
                "enabled": stmt.excluded.enabled,
                "updated_at": stmt.excluded.updated_at,
                "updated_by": stmt.excluded.updated_by,
            },
        )
        s.execute(stmt)


def strategy_state_clear_enabled(
    name: str,
    *,
    updated_at: Optional[str] = None,
    updated_by: str = "operator",
) -> None:
    """UPDATE strategy_state SET enabled=NULL ... WHERE name=?"""
    with session_scope() as s:
        s.execute(
            update(StrategyState)
            .where(StrategyState.name == name)
            .values(
                enabled=None,
                updated_at=updated_at or _utcnow(),
                updated_by=updated_by,
            )
        )


def strategy_state_enabled_overrides() -> dict[str, bool]:
    """name -> bool for rows where enabled IS NOT NULL."""
    with session_scope() as s:
        rows = s.execute(
            select(StrategyState.name, StrategyState.enabled)
            .where(StrategyState.enabled.is_not(None))
        ).all()
        return {r[0]: bool(r[1]) for r in rows}


def strategy_state_set_execution_mode(
    name: str,
    mode: str,
    *,
    updated_at: Optional[str] = None,
    updated_by: str = "operator",
) -> None:
    """INSERT (lifecycle=paper, execution_mode) ON CONFLICT UPDATE mode/updated_*."""
    ts = updated_at or _utcnow()
    with session_scope() as s:
        stmt = sqlite_insert(StrategyState).values(
            name=name,
            lifecycle="paper",
            execution_mode=mode,
            updated_at=ts,
            updated_by=updated_by,
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=["name"],
            set_={
                "execution_mode": stmt.excluded.execution_mode,
                "updated_at": stmt.excluded.updated_at,
                "updated_by": stmt.excluded.updated_by,
            },
        )
        s.execute(stmt)


def strategy_state_clear_execution_mode(
    name: str,
    *,
    updated_at: Optional[str] = None,
    updated_by: str = "operator",
) -> None:
    """UPDATE strategy_state SET execution_mode=NULL ... WHERE name=?"""
    with session_scope() as s:
        s.execute(
            update(StrategyState)
            .where(StrategyState.name == name)
            .values(
                execution_mode=None,
                updated_at=updated_at or _utcnow(),
                updated_by=updated_by,
            )
        )


def strategy_state_execution_mode_overrides() -> dict[str, str]:
    """name -> mode for rows where execution_mode IS NOT NULL."""
    with session_scope() as s:
        rows = s.execute(
            select(StrategyState.name, StrategyState.execution_mode)
            .where(StrategyState.execution_mode.is_not(None))
        ).all()
        return {r[0]: r[1] for r in rows}


# ---------------------------------------------------------------------------
# job_runs
# ---------------------------------------------------------------------------

def job_runs_start(job_name: str, *, started_at: Optional[str] = None) -> int:
    """INSERT INTO job_runs (job_name, started_at). Returns id."""
    with session_scope() as s:
        obj = JobRun(job_name=job_name, started_at=started_at or _utcnow())
        s.add(obj)
        s.flush()
        return int(obj.id) if obj.id is not None else 0


def job_runs_finish(
    run_id: int,
    *,
    success: int,
    stats_json: Optional[str] = None,
    error: Optional[str] = None,
    finished_at: Optional[str] = None,
) -> None:
    """UPDATE job_runs SET finished_at, success, stats_json, error WHERE id=?"""
    with session_scope() as s:
        s.execute(
            update(JobRun)
            .where(JobRun.id == run_id)
            .values(
                finished_at=finished_at or _utcnow(),
                success=success,
                stats_json=stats_json,
                error=error,
            )
        )


def job_runs_list(*, name: Optional[str] = None, limit: int = 20,
                  success: Optional[int] = None) -> list[dict]:
    """Recent job_runs, optionally filtered by job_name / success."""
    with session_scope() as s:
        stmt = select(JobRun)
        if name:
            stmt = stmt.where(JobRun.job_name == name)
        if success is not None:
            stmt = stmt.where(JobRun.success == success)
        stmt = stmt.order_by(JobRun.id.desc()).limit(limit)
        return [_row_dict(r) for r in s.execute(stmt).scalars().all()]


def job_runs_latest(job_name: str, *, success: Optional[int] = None) -> Optional[dict]:
    """Most recent row for a job_name."""
    rows = job_runs_list(name=job_name, limit=1, success=success)
    return rows[0] if rows else None


def job_runs_failed(limit: int = 10) -> list[dict]:
    """SELECT ... FROM job_runs WHERE success=0 ORDER BY id DESC LIMIT ?"""
    with session_scope() as s:
        rows = s.execute(
            select(JobRun)
            .where(JobRun.success == 0)
            .order_by(JobRun.id.desc())
            .limit(limit)
        ).scalars().all()
        return [_row_dict(r) for r in rows]


# ---------------------------------------------------------------------------
# data_catalog
# ---------------------------------------------------------------------------

def data_catalog_upsert(
    dataset: str,
    *,
    symbol: Optional[str] = None,
    as_of_date: Optional[str] = None,
    source: str = "unknown",
    available_at: Optional[str] = None,
    pit_safe: bool = True,
    ingested_at: Optional[str] = None,
) -> int:
    """INSERT INTO data_catalog (always-insert; name matches the plan)."""
    ts = _utcnow()
    with session_scope() as s:
        obj = DataCatalog(
            dataset=dataset,
            symbol=symbol,
            as_of_date=as_of_date,
            available_at=available_at or ts,
            source=source,
            pit_safe=1 if pit_safe else 0,
            ingested_at=ingested_at or ts,
        )
        s.add(obj)
        s.flush()
        return int(obj.id) if obj.id is not None else 0


def data_catalog_query(
    dataset: str,
    decision_time: str,
    *,
    symbol: Optional[str] = None,
    as_of_start: Optional[str] = None,
    as_of_end: Optional[str] = None,
    pit_only: bool = True,
) -> list[str]:
    """Distinct as_of_dates knowable at decision_time."""
    sql = (
        "SELECT DISTINCT as_of_date FROM data_catalog "
        "WHERE dataset = :dataset AND available_at <= :decision_time "
        "AND as_of_date IS NOT NULL"
    )
    params: dict = {"dataset": dataset, "decision_time": decision_time}
    if pit_only:
        sql += " AND pit_safe = 1"
    if symbol is not None:
        sql += " AND symbol = :symbol"
        params["symbol"] = symbol
    if as_of_start is not None:
        sql += " AND as_of_date >= :as_of_start"
        params["as_of_start"] = as_of_start
    if as_of_end is not None:
        sql += " AND as_of_date <= :as_of_end"
        params["as_of_end"] = as_of_end
    sql += " ORDER BY as_of_date"
    with session_scope() as s:
        return [r[0] for r in s.execute(text(sql), params).all()]


def data_catalog_latest(dataset: str, symbol: Optional[str] = None) -> Optional[dict]:
    """Most recent catalog row for a dataset (optional symbol)."""
    with session_scope() as s:
        stmt = select(DataCatalog).where(DataCatalog.dataset == dataset)
        if symbol is not None:
            stmt = stmt.where(DataCatalog.symbol == symbol)
        stmt = stmt.order_by(DataCatalog.id.desc()).limit(1)
        obj = s.execute(stmt).scalar_one_or_none()
        return _row_dict(obj) if obj is not None else None


# ---------------------------------------------------------------------------
# model_registry
# ---------------------------------------------------------------------------

def model_registry_register(
    name: str,
    path: str,
    sha256: str,
    *,
    trained_at: Optional[str] = None,
) -> int:
    """INSERT OR IGNORE then return the row id for (name, sha256)."""
    with session_scope() as s:
        existing = s.execute(
            select(ModelRegistry).where(
                ModelRegistry.name == name,
                ModelRegistry.sha256 == sha256,
            )
        ).scalar_one_or_none()
        if existing is not None:
            return int(existing.id)
        obj = ModelRegistry(
            name=name,
            path=path,
            sha256=sha256,
            trained_at=trained_at or _utcnow(),
        )
        s.add(obj)
        s.flush()
        return int(obj.id) if obj.id is not None else 0


def model_registry_promote(name: str, sha256: str, *, promoted_at: Optional[str] = None) -> None:
    """UPDATE model_registry SET promoted_at=? WHERE name=? AND sha256=?"""
    with session_scope() as s:
        s.execute(
            update(ModelRegistry)
            .where(ModelRegistry.name == name, ModelRegistry.sha256 == sha256)
            .values(promoted_at=promoted_at or _utcnow())
        )


def model_registry_get_active(name: str) -> Optional[dict]:
    """Most recently promoted (fallback: latest trained) artifact for name."""
    with session_scope() as s:
        obj = s.execute(
            select(ModelRegistry)
            .where(ModelRegistry.name == name)
            .order_by(func.coalesce(ModelRegistry.promoted_at, ModelRegistry.trained_at).desc())
            .limit(1)
        ).scalar_one_or_none()
        return _row_dict(obj) if obj is not None else None


def model_registry_list() -> list[dict]:
    """SELECT * FROM model_registry."""
    with session_scope() as s:
        rows = s.execute(select(ModelRegistry).order_by(ModelRegistry.id.asc())).scalars().all()
        return [_row_dict(r) for r in rows]


# ---------------------------------------------------------------------------
# trade_events
# ---------------------------------------------------------------------------

def trade_events_insert(
    event_type: str,
    *,
    symbol: Optional[str] = None,
    strategy: Optional[str] = None,
    qty: Optional[float] = None,
    price: Optional[float] = None,
    detail: Optional[str] = None,
    ts: Optional[str] = None,
) -> int:
    """INSERT INTO trade_events."""
    with session_scope() as s:
        obj = TradeEvent(
            ts=ts or _utcnow(),
            event_type=event_type,
            symbol=symbol,
            strategy=strategy,
            qty=qty,
            price=price,
            detail=detail,
        )
        s.add(obj)
        s.flush()
        return int(obj.id) if obj.id is not None else 0


def trade_events_list(
    *,
    event_type: Optional[str] = None,
    limit: int = 20,
) -> list[dict]:
    """SELECT * FROM trade_events [WHERE event_type=?] ORDER BY id DESC LIMIT ?"""
    with session_scope() as s:
        stmt = select(TradeEvent)
        if event_type is not None:
            stmt = stmt.where(TradeEvent.event_type == event_type)
        stmt = stmt.order_by(TradeEvent.id.desc()).limit(limit)
        return [_row_dict(r) for r in s.execute(stmt).scalars().all()]


# ---------------------------------------------------------------------------
# adopted_positions / alpaca_positions
# ---------------------------------------------------------------------------

def adopted_positions_insert(symbol: str, adopted_at: Optional[str] = None) -> None:
    """INSERT OR IGNORE INTO adopted_positions."""
    with session_scope() as s:
        stmt = sqlite_insert(AdoptedPosition).values(
            symbol=symbol,
            adopted_at=adopted_at or _utcnow(),
        ).on_conflict_do_nothing(index_elements=["symbol"])
        s.execute(stmt)


def adopted_positions_symbols() -> set[str]:
    """SELECT symbol FROM adopted_positions."""
    with session_scope() as s:
        rows = s.execute(select(AdoptedPosition.symbol)).scalars().all()
        return {r for r in rows if r}


def alpaca_positions_insert(
    *,
    ts: str,
    symbol: Optional[str] = None,
    qty: Optional[float] = None,
    side: Optional[str] = None,
    avg_entry_price: Optional[float] = None,
    current_price: Optional[float] = None,
    market_value: Optional[float] = None,
    unrealized_pl: Optional[float] = None,
    strategy: Optional[str] = None,
    managed: int = 0,
    run_id: Optional[int] = None,
) -> int:
    """INSERT INTO alpaca_positions."""
    with session_scope() as s:
        obj = AlpacaPosition(
            ts=ts,
            symbol=symbol,
            qty=qty,
            side=side,
            avg_entry_price=avg_entry_price,
            current_price=current_price,
            market_value=market_value,
            unrealized_pl=unrealized_pl,
            strategy=strategy,
            managed=managed,
            run_id=run_id,
        )
        s.add(obj)
        s.flush()
        return int(obj.id) if obj.id is not None else 0


def alpaca_positions_list(limit: int = 20) -> list[dict]:
    """SELECT * FROM alpaca_positions ORDER BY id DESC LIMIT ?"""
    with session_scope() as s:
        rows = s.execute(
            select(AlpacaPosition).order_by(AlpacaPosition.id.desc()).limit(limit)
        ).scalars().all()
        return [_row_dict(r) for r in rows]


# ---------------------------------------------------------------------------
# managed_positions extras
# ---------------------------------------------------------------------------

def managed_positions_close_by_id(row_id: int, closed_at: Optional[str] = None) -> int:
    """Mark one managed_positions row closed by id."""
    with session_scope() as s:
        result = s.execute(
            update(ManagedPosition)
            .where(ManagedPosition.id == row_id)
            .values(status="closed", closed_at=closed_at or _utcnow())
        )
        return result.rowcount or 0


def managed_positions_set_exit_by(group_id: str, exit_by: str) -> int:
    """UPDATE managed_positions SET exit_by=? WHERE group_id=? AND status='open'."""
    with session_scope() as s:
        result = s.execute(
            update(ManagedPosition)
            .where(
                ManagedPosition.group_id == group_id,
                ManagedPosition.status == "open",
            )
            .values(exit_by=exit_by)
        )
        return result.rowcount or 0


def managed_positions_set_opened_at(group_id: str, opened_at: str) -> int:
    """UPDATE managed_positions SET opened_at=? WHERE group_id=? (tests/backfill)."""
    with session_scope() as s:
        result = s.execute(
            update(ManagedPosition)
            .where(ManagedPosition.group_id == group_id)
            .values(opened_at=opened_at)
        )
        return result.rowcount or 0


# ---------------------------------------------------------------------------
# ff_ladders
# ---------------------------------------------------------------------------

def ff_ladders_recent(limit: int = 80) -> list[dict]:
    """SELECT id, ticker, candidate_json, status, order_id FROM ff_ladders ORDER BY id DESC."""
    with session_scope() as s:
        rows = s.execute(
            select(
                FfLadder.id,
                FfLadder.ticker,
                FfLadder.candidate_json,
                FfLadder.status,
                FfLadder.order_id,
                FfLadder.rung,
                FfLadder.updated_at,
            )
            .order_by(FfLadder.id.desc())
            .limit(limit)
        ).all()
        return [
            {
                "id": r.id,
                "ticker": r.ticker,
                "candidate_json": r.candidate_json,
                "status": r.status,
                "order_id": r.order_id,
                "rung": r.rung,
                "updated_at": r.updated_at,
            }
            for r in rows
        ]


def ff_ladders_count_armed() -> int:
    """SELECT COUNT(*) FROM ff_ladders WHERE status='armed'."""
    with session_scope() as s:
        return int(
            s.execute(
                select(func.count()).select_from(FfLadder).where(FfLadder.status == "armed")
            ).scalar() or 0
        )


def ff_ladders_armed_id_for_ticker(ticker: str) -> Optional[int]:
    """id of an armed ladder for ``ticker``, or None."""
    with session_scope() as s:
        row = s.execute(
            select(FfLadder.id).where(FfLadder.status == "armed", FfLadder.ticker == ticker)
        ).first()
        return int(row[0]) if row else None


def ff_ladders_insert(ticker: str, candidate_json: str, armed_by: Optional[int] = None) -> int:
    """INSERT a new armed ladder; returns the new id."""
    with session_scope() as s:
        obj = FfLadder(ticker=ticker, candidate_json=candidate_json, armed_by=armed_by)
        s.add(obj)
        s.flush()
        return int(obj.id or 0)


def ff_ladders_load_armed() -> list[dict]:
    """Armed ladders for LadderRunner."""
    with session_scope() as s:
        rows = s.execute(
            select(
                FfLadder.id,
                FfLadder.candidate_json,
                FfLadder.order_id,
                FfLadder.rung,
                FfLadder.status,
                FfLadder.created_at,
            ).where(FfLadder.status == "armed")
        ).all()
        return [
            {
                "id": r.id,
                "candidate_json": r.candidate_json,
                "order_id": r.order_id,
                "rung": r.rung,
                "status": r.status,
                "created_at": r.created_at,
            }
            for r in rows
        ]


def ff_ladders_update_state(
    ladder_id: int,
    order_id: Optional[str],
    rung: int,
    status: str,
) -> None:
    """UPDATE order_id/rung/status and bump updated_at."""
    with session_scope() as s:
        s.execute(
            text(
                "UPDATE ff_ladders SET order_id=:order_id, rung=:rung, status=:status, "
                "updated_at=datetime('now') WHERE id=:id"
            ),
            {"order_id": order_id, "rung": rung, "status": status, "id": ladder_id},
        )


# ---------------------------------------------------------------------------
# dashboard / streamlit analytics
# ---------------------------------------------------------------------------

def scan_runs_recent(limit: int = 10) -> list[dict]:
    """Latest scan_runs for the dashboard table."""
    with session_scope() as s:
        rows = s.execute(
            text(
                "SELECT scan_timestamp, scanner_name, trigger_type, candidate_count, "
                "take_count, printf('%.0f', duration_secs) AS secs, success "
                "FROM scan_runs ORDER BY id DESC LIMIT :n"
            ),
            {"n": limit},
        ).mappings().all()
        return [dict(r) for r in rows]


def scanner_scan_outputs_latest() -> tuple[Optional[str], list[dict]]:
    """Rows for the latest scan_timestamp in scanner_scan_outputs."""
    with session_scope() as s:
        latest = s.execute(
            select(func.max(ScannerScanOutput.scan_timestamp))
        ).scalar()
        if not latest:
            return None, []
        rows = s.execute(
            text(
                "SELECT ticker, earnings_date, tier, display_status, "
                "printf('%.2f', price) AS price "
                "FROM scanner_scan_outputs "
                "WHERE scan_timestamp = :ts "
                "ORDER BY CASE tier WHEN 'TAKE' THEN 0 WHEN 'tier1' THEN 1 "
                "WHEN 'tier2' THEN 2 ELSE 3 END, ticker "
                "LIMIT 30"
            ),
            {"ts": latest},
        ).mappings().all()
        return latest, [dict(r) for r in rows]


def pending_trades_recent(limit: int = 20) -> list[dict]:
    """Dashboard: recent pending_trades with formatted score."""
    with session_scope() as s:
        rows = s.execute(
            text(
                "SELECT id, created_at, strategy, ticker, status, "
                "printf('%.3f', model_score) AS score, decided_at "
                "FROM pending_trades ORDER BY id DESC LIMIT :n"
            ),
            {"n": limit},
        ).mappings().all()
        return [dict(r) for r in rows]


def pending_trades_brief(limit: int = 30) -> list[dict]:
    """Desk inbox: pending rows, newest first."""
    with session_scope() as s:
        rows = s.execute(
            select(
                PendingTrade.id,
                PendingTrade.created_at,
                PendingTrade.strategy,
                PendingTrade.ticker,
                PendingTrade.side,
                PendingTrade.status,
            )
            .where(PendingTrade.status == "pending")
            .order_by(PendingTrade.id.desc())
            .limit(limit)
        ).all()
        return [
            {
                "id": r.id,
                "created_at": r.created_at,
                "strategy": r.strategy,
                "ticker": r.ticker,
                "side": r.side,
                "status": r.status,
            }
            for r in rows
        ]


def calendar_call_trades_stats() -> dict:
    """Aggregate stats for the dashboard calendar panel."""
    with session_scope() as s:
        r = s.execute(
            text(
                "SELECT COUNT(*) AS n, "
                "printf('%.2f', AVG(net_debit)) AS avg_debit, "
                "SUM(CASE WHEN near_exit IS NOT NULL THEN 1 ELSE 0 END) AS closed, "
                "printf('%.2f', AVG(CASE WHEN near_exit IS NOT NULL "
                "THEN (far_exit - far_entry) - (near_exit - near_entry) END)) AS avg_pnl "
                "FROM calendar_call_trades"
            )
        ).mappings().one()
        return dict(r)


def calendar_call_trades_ml_frame() -> pd.DataFrame:
    """Calendar trades for backtest summary / Streamlit ML tab."""
    return _read_df(
        "SELECT ticker, scan_date, net_debit, near_entry, far_entry, near_exit, far_exit, "
        "pnl_dollars, return_on_debit, model_score "
        "FROM calendar_call_trades ORDER BY scan_date, ticker"
    )


def calendar_call_trades_with_features_df() -> pd.DataFrame:
    """JOIN used by the Streamlit dashboard load_calendar_trades()."""
    return _read_df(
        """
        SELECT c.ticker, c.earnings_date, c.scan_date, c.near_expiry, c.far_expiry,
               c.strike, c.near_entry, c.far_entry, c.net_debit, c.exit_value,
               c.pnl_dollars, c.return_on_debit, c.model_score, c.model_recommendation,
               s.price, s.avg_volume_30d, s.market_cap, s.has_options, s.days_to_expiry,
               s.total_open_interest, s.atm_iv_near, s.rv30, s.iv30_rv30, s.hist_vol_3m,
               s.term_slope, s.term_structure_valid, s.expected_move_pct,
               s.expected_move_dollars, s.straddle_price, s.atm_call_delta, s.atm_put_delta,
               s.atm_call_iv, s.atm_put_iv, s.sigma_baseline_1y, s.sigma_short_leg,
               s.sigma_short_leg_fair, s.actual_to_fair_ratio, s.mc_win_rate, s.mc_quarters
        FROM calendar_call_trades c
        JOIN snapshots s ON c.snapshot_id = s.id
        WHERE c.return_on_debit IS NOT NULL
        ORDER BY c.scan_date
        """
    )


def calendar_call_trades_count() -> int:
    with session_scope() as s:
        return int(s.execute(select(func.count()).select_from(CalendarCallTrade)).scalar() or 0)


def calendar_call_trades_backfill_stats() -> dict:
    """Counts used by the Streamlit queue tab."""
    with session_scope() as s:
        total = s.execute(select(func.count()).select_from(CalendarCallTrade)).scalar() or 0
        with_exit = s.execute(
            select(func.count()).select_from(CalendarCallTrade).where(
                CalendarCallTrade.exit_value.is_not(None)
            )
        ).scalar() or 0
        no_exit = s.execute(
            select(func.count()).select_from(CalendarCallTrade).where(
                CalendarCallTrade.exit_value.is_(None)
            )
        ).scalar() or 0
        date_range = s.execute(
            select(
                func.min(CalendarCallTrade.scan_date),
                func.max(CalendarCallTrade.scan_date),
            )
        ).one()
        return {
            "total": int(total),
            "with_exit": int(with_exit),
            "no_exit": int(no_exit),
            "min_scan_date": date_range[0],
            "max_scan_date": date_range[1],
        }


def snapshots_status_counts() -> dict:
    """total / labeled / unavailable / pending snapshot counts."""
    with session_scope() as s:
        total = s.execute(select(func.count()).select_from(Snapshot)).scalar() or 0
        labeled = s.execute(
            text(
                "SELECT COUNT(*) FROM snapshots "
                "WHERE outcome_fetched_at IS NOT NULL AND outcome_fetched_at != 'unavailable'"
            )
        ).scalar() or 0
        unavail = s.execute(
            text("SELECT COUNT(*) FROM snapshots WHERE outcome_fetched_at = 'unavailable'")
        ).scalar() or 0
        pending = s.execute(
            text("SELECT COUNT(*) FROM snapshots WHERE outcome_fetched_at IS NULL")
        ).scalar() or 0
        return {
            "total": int(total),
            "labeled": int(labeled),
            "unavailable": int(unavail),
            "pending": int(pending),
        }


def snapshots_outcome_moves_df() -> pd.DataFrame:
    return _read_df(
        """
        SELECT actual_move_pct, expected_move_pct, ticker, earnings_date
        FROM snapshots
        WHERE actual_move_pct IS NOT NULL
          AND outcome_fetched_at IS NOT NULL AND outcome_fetched_at != 'unavailable'
        """
    )


def live_calendar_candidates_recent_df(limit: int = 100) -> pd.DataFrame:
    return _read_df(
        """
        SELECT scan_timestamp, ticker, strike, net_debit, model_expected_return,
               model_decision, tier, sigma_short_leg_fair as fair_iv,
               atm_iv_near as actual_iv, iv_rv_ratio,
               term_slope as term_structure, expected_move_pct, win_rate
        FROM live_calendar_candidates
        ORDER BY scan_timestamp DESC LIMIT :n
        """,
        {"n": limit},
    )


def scanner_scan_outputs_recent_df(limit: int = 100) -> pd.DataFrame:
    return _read_df(
        """
        SELECT scan_timestamp, ticker, tier, model_decision, model_expected_return,
               atm_iv_near as actual_iv, expected_move_pct, win_rate
        FROM scanner_scan_outputs
        ORDER BY scan_timestamp DESC LIMIT :n
        """,
        {"n": limit},
    )


_QUEUE_FILTERS = {
    "Pending": "outcome_fetched_at IS NULL",
    "Unavailable": "outcome_fetched_at = 'unavailable'",
    "Labeled": "outcome_fetched_at IS NOT NULL AND outcome_fetched_at != 'unavailable'",
    "All": "1=1",
}


def snapshots_queue_df(
    view: str,
    *,
    ticker_filter: str = "",
    limit: int = 200,
) -> pd.DataFrame:
    where = _QUEUE_FILTERS.get(view, "1=1")
    sql = (
        "SELECT id, ticker, earnings_date, scan_date, timing, price, "
        "atm_iv_near, expected_move_pct, outcome_fetched_at, "
        "outcome_attempt_count, actual_move_pct, actual_move_direction, "
        "pre_earnings_close, post_earnings_close "
        f"FROM snapshots WHERE {where}"
    )
    params: dict = {"limit": int(limit)}
    if ticker_filter:
        sql += " AND ticker LIKE :ticker"
        params["ticker"] = f"%{ticker_filter}%"
    sql += " ORDER BY earnings_date DESC LIMIT :limit"
    return _read_df(sql, params)


def snapshots_reset_outcomes(ids: list[int]) -> int:
    """Reset listed snapshot ids back to pending."""
    if not ids:
        return 0
    with session_scope() as s:
        result = s.execute(
            update(Snapshot)
            .where(Snapshot.id.in_(ids))
            .values(outcome_fetched_at=None, outcome_attempt_count=0)
        )
        return result.rowcount or 0


def snapshots_reset_outcomes_view(view: str, *, ticker_filter: str = "") -> int:
    """Reset all snapshots matching a queue view back to pending."""
    where = _QUEUE_FILTERS.get(view, "1=1")
    sql = (
        "UPDATE snapshots SET outcome_fetched_at = NULL, outcome_attempt_count = 0 "
        f"WHERE {where}"
    )
    params: dict = {}
    if ticker_filter:
        sql += " AND ticker LIKE :ticker"
        params["ticker"] = f"%{ticker_filter}%"
    with session_scope() as s:
        result = s.execute(text(sql), params)
        return result.rowcount or 0


def ff_backfill_progress() -> dict:
    """FF backfill stats for the live dashboard panel."""
    with session_scope() as s:
        total_pairs = s.execute(
            text(
                """
                SELECT COUNT(*) FROM (
                    SELECT ticker, scan_date
                    FROM snapshots
                    WHERE has_options=1 AND price>=3 AND actual_move_pct IS NOT NULL
                      AND outcome_fetched_at IS NOT NULL AND outcome_fetched_at != 'unavailable'
                      AND ticker IN (
                          SELECT ticker FROM snapshots
                          WHERE actual_move_pct IS NOT NULL
                            AND outcome_fetched_at IS NOT NULL AND outcome_fetched_at != 'unavailable'
                          GROUP BY ticker HAVING COUNT(*) >= 3
                      )
                    GROUP BY ticker, scan_date
                )
                """
            )
        ).scalar() or 0
        row = s.execute(
            text(
                """
                SELECT COUNT(*) AS done,
                       SUM(CASE WHEN skip_reason IS NULL THEN 1 ELSE 0 END) AS ok,
                       SUM(CASE WHEN skip_reason IS NOT NULL THEN 1 ELSE 0 END) AS skipped,
                       MAX(created_at) AS last_at
                FROM ff_snapshots WHERE selector_version = 2
                """
            )
        ).mappings().one()
        out = {
            "total_pairs": int(total_pairs),
            "done": int(row["done"] or 0),
            "ok": int(row["ok"] or 0),
            "skipped": int(row["skipped"] or 0),
            "last_at": row["last_at"],
        }
        if out["done"] >= out["total_pairs"] and out["total_pairs"]:
            pr = s.execute(
                text(
                    """
                    SELECT COUNT(*) AS n,
                           AVG(premium_ratio) AS avg_pr,
                           SUM(CASE WHEN premium_ratio >= 1.2 THEN 1 ELSE 0 END) AS rich
                    FROM ff_snapshots
                    WHERE selector_version = 2 AND premium_ratio IS NOT NULL
                    """
                )
            ).mappings().one()
            out["premium_n"] = int(pr["n"] or 0)
            out["avg_pr"] = pr["avg_pr"]
            out["rich"] = int(pr["rich"] or 0)
        return out


# ---------------------------------------------------------------------------
# Backfill / collector helpers (Task 9)
# ---------------------------------------------------------------------------

_USABLE_OUTCOME_SQL = (
    "actual_move_pct IS NOT NULL AND outcome_fetched_at IS NOT NULL "
    "AND outcome_fetched_at != 'unavailable'"
)

_IV_FEATURE_FIELDS = [
    "atm_iv_near", "atm_call_iv", "atm_put_iv",
    "rv30", "hist_vol_3m", "iv30_rv30",
    "term_slope", "term_structure_valid",
    "expected_move_pct", "expected_move_dollars",
    "straddle_price", "atm_call_delta", "atm_put_delta",
    "sigma_baseline_1y", "sigma_short_leg", "sigma_short_leg_fair",
    "actual_to_fair_ratio",
]

_SNAPSHOT_COALESCE_FIELDS = [
    "price", "avg_volume_30d", "rv30", "hist_vol_3m", "has_options",
    "nearest_expiry", "days_to_expiry",
    "atm_call_iv", "atm_put_iv", "atm_iv_near", "atm_call_delta", "atm_put_delta",
    "straddle_price", "expected_move_dollars", "expected_move_pct",
    "iv30_rv30", "term_slope", "term_structure_valid",
    "sigma_baseline_1y", "sigma_short_leg", "sigma_short_leg_fair",
    "actual_to_fair_ratio",
]

_FF_SNAPSHOT_COLS = [
    "ticker", "scan_date", "earnings_date", "spot",
    "t1_expiry", "t1_dte", "t1_strike", "t1_contract", "t1_close", "t1_iv",
    "t2_expiry", "t2_dte", "t2_strike", "t2_contract", "t2_close", "t2_iv",
    "sigma_fwd", "tau_days", "implied_event_move_pct",
    "hist_median_move_pct", "hist_rms_move_pct", "n_hist_events",
    "premium_ratio", "skip_reason", "selector_version",
]

_FF_UNIVERSE_SNAPSHOT_COLS = [
    "ticker", "scan_date", "has_earnings_in_window", "earnings_date", "spot",
    "t1_expiry", "t1_dte", "t1_strike", "t1_contract", "t1_close", "t1_iv",
    "t2_expiry", "t2_dte", "t2_strike", "t2_contract", "t2_close", "t2_iv",
    "forward_factor", "sigma_fwd", "tau_days", "implied_event_move_pct",
    "hist_median_move_pct", "hist_rms_move_pct", "n_hist_events",
    "premium_ratio", "skip_reason", "selector_version",
]

_SNAPSHOT_MODEL_COLS = {c.key for c in class_mapper(Snapshot).columns}


def snapshots_exists(ticker: str, earnings_date: str) -> bool:
    """True if a snapshot already exists for ticker + earnings_date."""
    with session_scope() as s:
        row = s.execute(
            select(Snapshot.id)
            .where(Snapshot.ticker == ticker.upper(), Snapshot.earnings_date == earnings_date)
            .limit(1)
        ).first()
        return row is not None


def snapshots_iv_pending_groups(limit: int = 0) -> list[dict]:
    """Unique (ticker, earnings_date, scan_date) groups missing atm_iv_near."""
    sql = (
        "SELECT ticker, earnings_date, scan_date, GROUP_CONCAT(id) AS ids "
        "FROM snapshots "
        "WHERE scan_date >= '2026-05-01' AND has_options = 1 AND atm_iv_near IS NULL "
        "GROUP BY ticker, earnings_date, scan_date "
        "ORDER BY earnings_date, ticker"
    )
    params: dict = {}
    if limit:
        sql += " LIMIT :limit"
        params["limit"] = int(limit)
    return _fetchall(None, sql, params)


def snapshots_update_iv(snapshot_ids: list[int], features: dict) -> None:
    """Write IV feature columns onto a batch of snapshot ids (one transaction)."""
    if not snapshot_ids:
        return
    values = {f: features.get(f) for f in _IV_FEATURE_FIELDS}
    with session_scope() as s:
        s.execute(update(Snapshot).where(Snapshot.id.in_(snapshot_ids)).values(**values))


def snapshots_mark_iv_skip(
    snapshot_ids: list[int],
    error: str,
    partial: Optional[dict] = None,
) -> None:
    """Record collection_error (and optional rv/hv) on a batch of ids."""
    if not snapshot_ids:
        return
    values: dict[str, Any] = {"collection_error": error}
    if partial:
        for key, val in partial.items():
            if key in _SNAPSHOT_MODEL_COLS:
                values[key] = val
    with session_scope() as s:
        s.execute(update(Snapshot).where(Snapshot.id.in_(snapshot_ids)).values(**values))


def snapshots_rv_pending_pairs() -> list[dict]:
    """Distinct (ticker, scan_date) pairs needing rv30 backfill."""
    return _fetchall(
        None,
        "SELECT ticker, scan_date, MIN(id) AS sample_id "
        "FROM snapshots "
        "WHERE scan_date >= '2026-05-01' AND has_options = 1 AND rv30 IS NULL "
        "GROUP BY ticker, scan_date "
        "ORDER BY ticker, scan_date",
        {},
    )


def snapshots_apply_rv(ticker: str, scan_date: str, rv30, hist_vol_3m) -> None:
    """Set rv30/hist_vol_3m and recompute iv30_rv30 for one ticker+scan_date."""
    with session_scope() as s:
        s.execute(
            update(Snapshot)
            .where(
                Snapshot.ticker == ticker,
                Snapshot.scan_date == scan_date,
                Snapshot.scan_date >= "2026-05-01",
            )
            .values(rv30=rv30, hist_vol_3m=hist_vol_3m)
        )
        s.execute(
            text(
                "UPDATE snapshots SET iv30_rv30 = atm_iv_near / rv30 "
                "WHERE ticker = :ticker AND scan_date = :scan_date "
                "AND scan_date >= '2026-05-01' "
                "AND atm_iv_near IS NOT NULL AND rv30 IS NOT NULL AND rv30 > 0 "
                "AND (iv30_rv30 IS NULL OR iv30_rv30 != atm_iv_near / rv30)"
            ),
            {"ticker": ticker, "scan_date": scan_date},
        )


def snapshots_count_null_rv30() -> int:
    with session_scope() as s:
        return int(
            s.execute(
                text(
                    "SELECT COUNT(*) FROM snapshots "
                    "WHERE scan_date >= '2026-05-01' AND has_options = 1 AND rv30 IS NULL"
                )
            ).scalar()
            or 0
        )


def snapshots_missing_atm_iv() -> list[dict]:
    """Snapshots with options but no atm_iv_near."""
    return _fetchall(
        None,
        "SELECT id, ticker, earnings_date, price FROM snapshots "
        "WHERE has_options = 1 AND atm_iv_near IS NULL "
        "ORDER BY ticker, earnings_date",
        {},
    )


def snapshots_update_fields(snapshot_ids: list[int], fields: dict) -> None:
    """SET given columns on a batch of snapshot ids (one transaction)."""
    if not snapshot_ids or not fields:
        return
    values = {k: v for k, v in fields.items() if k in _SNAPSHOT_MODEL_COLS}
    if not values:
        return
    with session_scope() as s:
        s.execute(update(Snapshot).where(Snapshot.id.in_(snapshot_ids)).values(**values))


def snapshots_coalesce_features(snapshot_id: int, feats: dict) -> list[str]:
    """Fill NULL snapshot feature columns; never overwrite existing values."""
    values = {}
    filled: list[str] = []
    for field in _SNAPSHOT_COALESCE_FIELDS:
        val = feats.get(field)
        if val is None:
            continue
        col = getattr(Snapshot, field)
        values[field] = func.coalesce(col, val)
        filled.append(field)
    if not values:
        return []
    with session_scope() as s:
        s.execute(update(Snapshot).where(Snapshot.id == snapshot_id).values(**values))
    return filled


def snapshots_iv_gap_rows(
    *,
    with_outcomes_only: bool = False,
    scan_date_since: Optional[str] = None,
    limit: Optional[int] = None,
) -> list[dict]:
    """has_options=1 rows with NULL atm_iv_near; labeled / recent first."""
    where = "atm_iv_near IS NULL AND has_options = 1"
    params: dict = {}
    if with_outcomes_only:
        where += f" AND {_USABLE_OUTCOME_SQL}"
    if scan_date_since:
        where += " AND scan_date >= :since"
        params["since"] = scan_date_since
    sql = (
        f"SELECT id, ticker, earnings_date, scan_date FROM snapshots WHERE {where} "
        "ORDER BY (actual_move_pct IS NOT NULL) DESC, scan_date DESC, ticker"
    )
    rows = _fetchall(None, sql, params)
    if limit:
        rows = rows[: int(limit)]
    return rows


def ff_snapshots_pending_pairs(selector_version: int = 2) -> list[dict]:
    """(ticker, scan_date) pairs still needing an ff_snapshots row."""
    pairs = _fetchall(
        None,
        "SELECT ticker, scan_date, earnings_date, price, "
        "AVG(ABS(actual_move_pct)) AS _x "
        "FROM snapshots "
        "WHERE has_options=1 AND price>=3 AND actual_move_pct IS NOT NULL "
        f"AND {_USABLE_OUTCOME_SQL} "
        "AND ticker IN ("
        "  SELECT ticker FROM snapshots "
        f"  WHERE {_USABLE_OUTCOME_SQL} "
        "  GROUP BY ticker HAVING COUNT(*) >= 3"
        ") "
        "GROUP BY ticker, scan_date "
        "ORDER BY scan_date",
        {},
    )
    done = {
        (r["ticker"], r["scan_date"])
        for r in _fetchall(
            None,
            "SELECT ticker, scan_date FROM ff_snapshots "
            "WHERE (hist_rms_move_pct IS NOT NULL OR skip_reason IS NOT NULL) "
            "AND selector_version = :v",
            {"v": selector_version},
        )
    }
    return [p for p in pairs if (p["ticker"], p["scan_date"]) not in done]


def snapshots_hist_move_abs(ticker: str, exclude_scan: str) -> list[float]:
    """|actual_move_pct| for a ticker excluding one scan_date."""
    rows = _fetchall(
        None,
        "SELECT ABS(actual_move_pct) AS mag FROM snapshots "
        "WHERE ticker = :ticker AND actual_move_pct IS NOT NULL "
        "AND outcome_fetched_at IS NOT NULL AND outcome_fetched_at != 'unavailable' "
        "AND scan_date != :exclude_scan",
        {"ticker": ticker, "exclude_scan": exclude_scan},
    )
    return [float(r["mag"]) for r in rows if r["mag"] is not None]


def ff_snapshots_upsert_many(rows: list[dict]) -> None:
    """INSERT OR REPLACE a batch of ff_snapshots rows in one transaction."""
    if not rows:
        return
    placeholders = ", ".join(f":{c}" for c in _FF_SNAPSHOT_COLS)
    sql = (
        f"INSERT OR REPLACE INTO ff_snapshots ({', '.join(_FF_SNAPSHOT_COLS)}) "
        f"VALUES ({placeholders})"
    )
    payload = [{c: r.get(c) for c in _FF_SNAPSHOT_COLS} for r in rows]
    _execute(None, sql, payload, many=True)

def ff_universe_snapshots_upsert_many(rows: list[dict]) -> None:
    """INSERT OR REPLACE a batch of ff_universe_snapshots rows in one transaction."""
    if not rows:
        return
    placeholders = ", ".join(f":{c}" for c in _FF_UNIVERSE_SNAPSHOT_COLS)
    sql = (
        f"INSERT OR REPLACE INTO ff_universe_snapshots ({', '.join(_FF_UNIVERSE_SNAPSHOT_COLS)}) "
        f"VALUES ({placeholders})"
    )
    payload = [{c: r.get(c) for c in _FF_UNIVERSE_SNAPSHOT_COLS} for r in rows]
    _execute(None, sql, payload, many=True)


def snapshots_clear_iv_since(scan_date: str) -> int:
    """NULL IV/RV fields on rows scanned on/after ``scan_date``."""
    with session_scope() as s:
        result = s.execute(
            update(Snapshot)
            .where(Snapshot.scan_date >= scan_date)
            .where((Snapshot.atm_iv_near.is_not(None)) | (Snapshot.rv30.is_not(None)))
            .values(
                atm_iv_near=None,
                atm_call_iv=None,
                atm_put_iv=None,
                rv30=None,
                hist_vol_3m=None,
            )
        )
        return result.rowcount or 0


def snapshots_iv_presence_counts() -> dict:
    """Counts used by the yfinance-IV rollback script."""
    with session_scope() as s:
        return {
            "iv_before": int(
                s.execute(
                    text(
                        "SELECT COUNT(*) FROM snapshots "
                        "WHERE atm_iv_near IS NOT NULL AND scan_date < '2026-05-01'"
                    )
                ).scalar()
                or 0
            ),
            "iv_after": int(
                s.execute(
                    text(
                        "SELECT COUNT(*) FROM snapshots "
                        "WHERE atm_iv_near IS NOT NULL AND scan_date >= '2026-05-01'"
                    )
                ).scalar()
                or 0
            ),
            "total_may": int(
                s.execute(
                    text("SELECT COUNT(*) FROM snapshots WHERE scan_date >= '2026-05-01'")
                ).scalar()
                or 0
            ),
            "need_backfill": int(
                s.execute(
                    text(
                        "SELECT COUNT(*) FROM snapshots "
                        "WHERE scan_date >= '2026-05-01' AND has_options = 1 "
                        "AND collection_error IS NULL"
                    )
                ).scalar()
                or 0
            ),
        }


def snapshots_dedup() -> tuple[int, int]:
    """Delete duplicate snapshots, keeping the lowest id. Returns (deleted, remaining_groups)."""
    with session_scope() as s:
        groups = s.execute(
            text(
                "SELECT MIN(id) AS keep_id, ticker, earnings_date, scan_date, timing, data_source "
                "FROM snapshots "
                "GROUP BY ticker, earnings_date, scan_date, timing, data_source "
                "HAVING COUNT(*) > 1"
            )
        ).mappings().all()
        deleted = 0
        for g in groups:
            result = s.execute(
                text(
                    "DELETE FROM snapshots WHERE id > :keep_id AND ticker = :ticker "
                    "AND earnings_date = :ed AND scan_date = :sd "
                    "AND timing IS :timing AND data_source IS :ds"
                ),
                {
                    "keep_id": g["keep_id"],
                    "ticker": g["ticker"],
                    "ed": g["earnings_date"],
                    "sd": g["scan_date"],
                    "timing": g["timing"],
                    "ds": g["data_source"],
                },
            )
            deleted += result.rowcount or 0
        remaining = s.execute(
            text(
                "SELECT COUNT(*) FROM ("
                "  SELECT ticker, earnings_date, scan_date, timing, data_source "
                "  FROM snapshots "
                "  GROUP BY ticker, earnings_date, scan_date, timing, data_source "
                "  HAVING COUNT(*) > 1"
                ")"
            )
        ).scalar()
        return deleted, int(remaining or 0)


def snapshots_usable_outcome_count(*args, ticker: str | None = None) -> int:
    """Usable realized-outcome count for one ticker (hist-gate numerator)."""
    conn, rest = _split_conn(args)
    if ticker is None:
        ticker = rest[0]
    if conn is not None:
        row = conn.execute(
            "SELECT COUNT(*) FROM snapshots "
            "WHERE ticker=? AND actual_move_pct IS NOT NULL "
            "AND outcome_fetched_at IS NOT NULL AND outcome_fetched_at != 'unavailable'",
            (ticker,),
        ).fetchone()
        return int((row[0] if row else 0) or 0)
    rows = _fetchall(
        conn,
        "SELECT COUNT(*) AS n FROM snapshots "
        "WHERE ticker = :ticker AND actual_move_pct IS NOT NULL "
        "AND outcome_fetched_at IS NOT NULL AND outcome_fetched_at != 'unavailable'",
        {"ticker": ticker},
    )
    return int(rows[0]["n"] if rows else 0)


def snapshots_outcome_row(
    *args, ticker: str | None = None, earnings_date: str | None = None
) -> Optional[dict]:
    """id / actual_move_pct / outcome_fetched_at for ticker+earnings_date."""
    conn, rest = _split_conn(args)
    if ticker is None:
        ticker = rest[0]
        earnings_date = rest[1] if len(rest) > 1 else earnings_date
    if conn is not None:
        row = conn.execute(
            "SELECT id, actual_move_pct, outcome_fetched_at FROM snapshots "
            "WHERE ticker=? AND earnings_date=? ORDER BY id LIMIT 1",
            (ticker, earnings_date),
        ).fetchone()
        if row is None:
            return None
        try:
            return dict(row)
        except TypeError:
            return {
                "id": row[0],
                "actual_move_pct": row[1],
                "outcome_fetched_at": row[2],
            }
    rows = _fetchall(
        conn,
        "SELECT id, actual_move_pct, outcome_fetched_at FROM snapshots "
        "WHERE ticker = :ticker AND earnings_date = :earnings_date "
        "ORDER BY id LIMIT 1",
        {"ticker": ticker, "earnings_date": earnings_date},
    )
    return rows[0] if rows else None


def snapshots_apply_hist_backfill_batch(*args, writes: list[dict] | None = None) -> None:
    """Apply collected hist-backfill INSERT/UPDATE writes in one transaction.

    Each write is ``{existing_id?, ticker, earnings_date, outcome, fetched_at}``.
    A leading sqlite3 connection is still accepted so in-memory tests keep working.
    """
    conn, rest = _split_conn(args)
    if writes is None:
        writes = rest[0]
    if not writes:
        return
    def _write_one(execute, w: dict) -> None:
        outcome = w["outcome"]
        if w.get("existing_id"):
            execute(
                """UPDATE snapshots SET
                    pre_earnings_close=:pre, post_earnings_close=:post,
                    actual_move_pct=:move, actual_move_direction=:direction,
                    max_intraday_range_pct=:rng, outcome_fetched_at=:fetched
                   WHERE id=:id""",
                {
                    "pre": outcome["pre_earnings_close"],
                    "post": outcome["post_earnings_close"],
                    "move": outcome["actual_move_pct"],
                    "direction": outcome["actual_move_direction"],
                    "rng": outcome["max_intraday_range_pct"],
                    "fetched": w["fetched_at"],
                    "id": w["existing_id"],
                },
            )
        else:
            execute(
                """INSERT INTO snapshots
                    (ticker, earnings_date, scan_date, timing, has_options,
                     pre_earnings_close, post_earnings_close,
                     actual_move_pct, actual_move_direction,
                     max_intraday_range_pct, outcome_fetched_at)
                   VALUES (:ticker, :ed, :ed, 'Backfill', 0,
                           :pre, :post, :move, :direction, :rng, :fetched)""",
                {
                    "ticker": w["ticker"],
                    "ed": w["earnings_date"],
                    "pre": outcome["pre_earnings_close"],
                    "post": outcome["post_earnings_close"],
                    "move": outcome["actual_move_pct"],
                    "direction": outcome["actual_move_direction"],
                    "rng": outcome["max_intraday_range_pct"],
                    "fetched": w["fetched_at"],
                },
            )

    if conn is not None:
        for w in writes:
            try:
                _write_one(conn.execute, w)
            except Exception as exc:
                logger.info("hist backfill write failed: %s", exc)
        conn.commit()
        return
    with session_scope() as s:
        for w in writes:
            try:
                _write_one(lambda sql, params: s.execute(text(sql), params), w)
            except Exception as exc:
                logger.info("hist backfill write failed: %s", exc)


def snapshots_distinct_tickers(*args) -> list[str]:
    conn, _rest = _split_conn(args)
    return [r["ticker"] for r in _fetchall(conn, "SELECT DISTINCT ticker FROM snapshots", {})]


def snapshots_usable_counts_by_ticker(
    *args, universe: Optional[list[str]] = None
) -> dict[str, int]:
    conn, rest = _split_conn(args)
    if universe is None and rest:
        universe = rest[0]
    if not universe:
        return {}
    params = {f"t{i}": t for i, t in enumerate(universe)}
    placeholders = ", ".join(f":t{i}" for i in range(len(universe)))
    rows = _fetchall(
        conn,
        f"SELECT ticker, COUNT(*) AS n FROM snapshots "
        f"WHERE ticker IN ({placeholders}) AND {_USABLE_OUTCOME_SQL} "
        f"GROUP BY ticker",
        params,
    )
    return {r["ticker"]: int(r["n"]) for r in rows}


def snapshots_hist_repair_stats(*args) -> list[dict]:
    """Per-ticker usable-outcome count and has_options flag for repair ranking."""
    conn, _rest = _split_conn(args)
    return _fetchall(
        conn,
        "SELECT s.ticker AS ticker, "
        f"SUM(CASE WHEN {_USABLE_OUTCOME_SQL} THEN 1 ELSE 0 END) AS usable, "
        "MAX(s.has_options) AS liquid "
        "FROM snapshots s GROUP BY s.ticker",
        {},
    )


def snapshots_hist_abs_moves(ticker: str) -> list[float]:
    """|actual_move_pct| for usable realized outcomes of one ticker."""
    rows = _fetchall(
        None,
        "SELECT ABS(actual_move_pct) AS mag FROM snapshots "
        "WHERE ticker = :ticker AND actual_move_pct IS NOT NULL "
        "AND outcome_fetched_at IS NOT NULL AND outcome_fetched_at != 'unavailable'",
        {"ticker": ticker},
    )
    return [float(r["mag"]) for r in rows if r["mag"] is not None]


def snapshots_as_of_df(cutoff: str) -> pd.DataFrame:
    """SELECT * FROM snapshots WHERE scan_date <= cutoff. Missing table -> empty."""
    try:
        return _read_df(
            "SELECT * FROM snapshots WHERE scan_date <= :cutoff", {"cutoff": cutoff}
        )
    except Exception:
        return pd.DataFrame()

def ff_universe_snapshots_as_of_df(cutoff: str) -> pd.DataFrame:
    """SELECT * FROM ff_universe_snapshots WHERE scan_date <= cutoff. Missing table -> empty."""
    try:
        return _read_df(
            "SELECT * FROM ff_universe_snapshots WHERE scan_date <= :cutoff",
            {"cutoff": cutoff},
        )
    except Exception:
        return pd.DataFrame()

def daily_signals_as_of_df(cutoff: str) -> pd.DataFrame:
    """SELECT * FROM daily_signals WHERE signal_date <= cutoff. Missing table -> empty."""
    try:
        return _read_df(
            "SELECT * FROM daily_signals WHERE signal_date <= :cutoff",
            {"cutoff": cutoff},
        )
    except Exception:
        return pd.DataFrame()


def ff_snapshots_as_of_df(cutoff: str) -> pd.DataFrame:
    """SELECT * FROM ff_snapshots WHERE scan_date <= cutoff. Missing table -> empty."""
    try:
        return _read_df(
            "SELECT * FROM ff_snapshots WHERE scan_date <= :cutoff",
            {"cutoff": cutoff},
        )
    except Exception:
        return pd.DataFrame()


def snapshots_tickers_as_of(as_of: str, limit: int) -> list[str]:
    """Tickers from the most recent snapshot scan_date <= as_of."""
    with session_scope() as s:
        latest = s.execute(
            select(func.max(Snapshot.scan_date)).where(Snapshot.scan_date <= as_of)
        ).scalar()
        if not latest:
            return []
        rows = s.execute(
            select(Snapshot.ticker)
            .where(Snapshot.scan_date == latest)
            .distinct()
            .order_by(Snapshot.ticker)
        ).all()
        return [r[0] for r in rows][:limit]


def options_chain_df_latest(ticker: str, as_of: str) -> pd.DataFrame:
    """Latest options_chain rows for ticker with scan_date <= as_of."""
    with session_scope() as s:
        latest = s.execute(
            select(func.max(OptionsChain.scan_date)).where(
                OptionsChain.ticker == ticker, OptionsChain.scan_date <= as_of
            )
        ).scalar()
        if not latest:
            return pd.DataFrame()
    return _read_df(
        "SELECT expiry, strike, contract_type, volume, implied_volatility, delta, "
        "midpoint, close FROM options_chain WHERE ticker = :ticker AND scan_date = :scan_date",
        {"ticker": ticker, "scan_date": latest},
    )


def daily_signals_history(
    ticker: str, column: str, as_of: str, limit: int = 260
) -> list[float]:
    """Prior daily_signals values for ``column`` (allowlisted)."""
    if column not in _DAILY_SIGNAL_COLS:
        raise ValueError(f"unknown daily_signals column: {column}")
    rows = _fetchall(
        None,
        f"SELECT {column} AS v FROM daily_signals "
        "WHERE ticker = :ticker AND signal_date < :as_of "
        f"AND {column} IS NOT NULL ORDER BY signal_date DESC LIMIT :limit",
        {"ticker": ticker, "as_of": as_of, "limit": limit},
    )
    return [r["v"] for r in rows]


def snapshots_latest_price(ticker: str) -> Optional[float]:
    """Most recent non-null snapshots.price for ticker."""
    with session_scope() as s:
        row = s.execute(
            select(Snapshot.price)
            .where(Snapshot.ticker == ticker, Snapshot.price.is_not(None))
            .order_by(Snapshot.scan_date.desc())
            .limit(1)
        ).first()
        return float(row[0]) if row and row[0] is not None else None

