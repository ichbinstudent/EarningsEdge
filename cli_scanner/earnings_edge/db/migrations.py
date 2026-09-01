"""Idempotent schema migrations for existing databases.

Ported from ``earnings_edge.db_legacy`` and ``framework.storage``; each
private migration checks ``pragma table_info`` before issuing ALTERs, so
``run_migrations`` is safe to run on every ``engine.configure()`` call.
"""

from __future__ import annotations

import sqlalchemy
from sqlalchemy.engine import Connection


def run_migrations(conn: Connection) -> None:
    """Apply all column/table migrations (idempotent)."""
    tables = {r[0] for r in conn.execute(sqlalchemy.text(
        "SELECT name FROM sqlite_master WHERE type='table'"))}
    if "snapshots" in tables:
        _migrate_snapshots(conn)
    if "calendar_call_trades" in tables:
        _migrate_calendar_call_trades(conn)
    if "options_chain" in tables:
        _migrate_options_chain_hourly(conn)
    if "live_calendar_candidates" in tables:
        _migrate_live_calendar_candidates(conn)
    _migrate_ff_universe_snapshots(conn, tables)
    _migrate_framework(conn, tables)
    _create_indexes(conn, tables)


def _migrate_snapshots(conn: Connection) -> None:
    """Add missing columns to snapshots for existing databases."""
    existing = {r[1] for r in conn.execute(sqlalchemy.text('pragma table_info(snapshots)'))}
    migrations = {
        'outcome_attempt_count': 'INTEGER DEFAULT 0',
        'data_source': 'TEXT DEFAULT "unknown"',
    }
    for col, col_type in migrations.items():
        if col not in existing:
            conn.execute(sqlalchemy.text(f"ALTER TABLE snapshots ADD COLUMN {col} {col_type}"))


def _migrate_calendar_call_trades(conn: Connection) -> None:
    """Add model-score columns to calendar_call_trades for existing databases."""
    existing = {r[1] for r in conn.execute(sqlalchemy.text('pragma table_info(calendar_call_trades)'))}
    migrations = {
        "model_score": "REAL",
        "model_recommendation": "INTEGER",
        "model_reason": "TEXT",
        "model_name": "TEXT",
        "model_scored_at": "TEXT",
    }
    for col, col_type in migrations.items():
        if col not in existing:
            conn.execute(sqlalchemy.text(
                f"ALTER TABLE calendar_call_trades ADD COLUMN {col} {col_type}"
            ))


def _migrate_options_chain_hourly(conn: Connection) -> None:
    """Allow one snapshot per contract per hour (was one per day via close)."""
    tables = {r[0] for r in conn.execute(sqlalchemy.text(
        "SELECT name FROM sqlite_master WHERE type='table'"))}
    if "options_chain" not in tables:
        return
    cols = {r[1] for r in conn.execute(sqlalchemy.text("pragma table_info(options_chain)"))}
    ddl_row = conn.execute(sqlalchemy.text(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='options_chain'"
    )).fetchone()
    ddl = (ddl_row[0] if ddl_row else "") or ""
    compact = ddl.replace(" ", "")
    already_hourly = (
        "captured_hour" in cols
        and "UNIQUE(contract_ticker,captured_hour)" in compact
    )
    if already_hourly:
        return
    if "captured_at" not in cols:
        conn.execute(sqlalchemy.text("ALTER TABLE options_chain ADD COLUMN captured_at TEXT"))
    if "captured_hour" not in cols:
        conn.execute(sqlalchemy.text("ALTER TABLE options_chain ADD COLUMN captured_hour TEXT"))
    conn.execute(sqlalchemy.text(
        "UPDATE options_chain SET captured_hour = scan_date || 'T16', "
        "captured_at = COALESCE(created_at, scan_date || 'T16:00:00') "
        "WHERE captured_hour IS NULL OR captured_hour = ''"
    ))
    for statement in (
        """
        CREATE TABLE IF NOT EXISTS options_chain_v2 (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            collector_run_id TEXT,
            ticker TEXT NOT NULL,
            scan_date TEXT NOT NULL,
            contract_ticker TEXT NOT NULL,
            underlying TEXT,
            expiry TEXT,
            strike REAL,
            contract_type TEXT,
            style TEXT,
            bid REAL, ask REAL, bid_size INTEGER, ask_size INTEGER,
            midpoint REAL, close REAL, open_price REAL, high REAL, low REAL,
            trade_count INTEGER, volume INTEGER, vwap REAL,
            implied_volatility REAL, delta REAL, gamma REAL, theta REAL, vega REAL,
            created_at TEXT DEFAULT (datetime('now')),
            captured_at TEXT,
            captured_hour TEXT,
            UNIQUE(contract_ticker, captured_hour)
        )
        """,
        """
        INSERT OR IGNORE INTO options_chain_v2 (
            collector_run_id, ticker, scan_date, contract_ticker, underlying,
            expiry, strike, contract_type, style, bid, ask, bid_size, ask_size,
            midpoint, close, open_price, high, low, trade_count, volume, vwap,
            implied_volatility, delta, gamma, theta, vega, created_at,
            captured_at, captured_hour
        )
        SELECT
            collector_run_id, ticker, scan_date, contract_ticker, underlying,
            expiry, strike, contract_type, style, bid, ask, bid_size, ask_size,
            midpoint, close, open_price, high, low, trade_count, volume, vwap,
            implied_volatility, delta, gamma, theta, vega, created_at,
            captured_at, captured_hour
        FROM options_chain
        """,
        "DROP TABLE options_chain",
        "ALTER TABLE options_chain_v2 RENAME TO options_chain",
        "CREATE INDEX IF NOT EXISTS idx_chain_underlying ON options_chain(underlying)",
        "CREATE INDEX IF NOT EXISTS idx_chain_scan_date ON options_chain(scan_date)",
        "CREATE INDEX IF NOT EXISTS idx_chain_ticker_date ON options_chain(ticker, scan_date)",
        "CREATE INDEX IF NOT EXISTS idx_chain_captured_hour ON options_chain(captured_hour)",
    ):
        conn.execute(sqlalchemy.text(statement))


def _migrate_live_calendar_candidates(conn: Connection) -> None:
    """Add missing columns to live_calendar_candidates for existing databases."""
    existing = {r[1] for r in conn.execute(sqlalchemy.text('pragma table_info(live_calendar_candidates)'))}
    needed = {
        'tier': 'INTEGER',
        'passed': 'INTEGER',
        'near_miss': 'INTEGER DEFAULT 0',
        'scanner_reason': 'TEXT',
        'display_status': 'TEXT',
        'volume': 'REAL',
        'market_cap': 'REAL',
        'days_to_expiry': 'INTEGER',
        'total_open_interest': 'INTEGER',
        'atm_iv_near': 'REAL',
        'sigma_baseline_1y': 'REAL',
        'sigma_short_leg': 'REAL',
        'sigma_short_leg_fair': 'REAL',
        'actual_to_fair_ratio': 'REAL',
        'iv_rv_ratio': 'REAL',
        'hist_vol_3m': 'REAL',
        'term_slope': 'REAL',
        'term_structure_valid': 'INTEGER',
        'expected_move_pct': 'REAL',
        'expected_move_dollars': 'REAL',
        'straddle_price': 'REAL',
        'atm_call_delta': 'REAL',
        'atm_put_delta': 'REAL',
        'atm_call_iv': 'REAL',
        'atm_put_iv': 'REAL',
        'win_rate': 'REAL',
        'win_quarters': 'INTEGER',
        # Stock-move outcome columns (filled by outcomes.py alongside exit_value/pnl)
        'pre_earnings_close': 'REAL',
        'post_earnings_close': 'REAL',
        'actual_move_pct': 'REAL',
        'actual_move_direction': 'TEXT',
        'max_intraday_range_pct': 'REAL',
        'outcome_attempt_count': 'INTEGER DEFAULT 0',
    }
    for col, col_type in needed.items():
        if col not in existing:
            conn.execute(sqlalchemy.text(f"ALTER TABLE live_calendar_candidates ADD COLUMN {col} {col_type}"))

def _migrate_ff_universe_snapshots(conn: Connection, tables: set) -> None:
    if "ff_universe_snapshots" not in tables:
        conn.execute(sqlalchemy.text('''
            CREATE TABLE ff_universe_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                scan_date TEXT NOT NULL,
                has_earnings_in_window INTEGER NOT NULL DEFAULT 0,
                earnings_date TEXT,
                spot REAL,
                t1_expiry TEXT,
                t1_dte INTEGER,
                t1_strike REAL,
                t1_contract TEXT,
                t1_close REAL,
                t1_iv REAL,
                t2_expiry TEXT,
                t2_dte INTEGER,
                t2_strike REAL,
                t2_contract TEXT,
                t2_close REAL,
                t2_iv REAL,
                forward_factor REAL,
                sigma_fwd REAL,
                tau_days INTEGER,
                implied_event_move_pct REAL,
                hist_median_move_pct REAL,
                n_hist_events INTEGER,
                premium_ratio REAL,
                skip_reason TEXT,
                created_at TEXT DEFAULT (datetime('now')),
                hist_rms_move_pct REAL,
                selector_version INTEGER,
                UNIQUE(ticker, scan_date)
            )
        '''))
        conn.execute(sqlalchemy.text('CREATE UNIQUE INDEX IF NOT EXISTS idx_uq_ff_universe_snapshots_ticker_date ON ff_universe_snapshots(ticker, scan_date)'))


def _migrate_framework(conn: Connection, tables: set) -> None:
    """Idempotent column additions for databases created by older versions."""
    if "strategy_state" in tables:
        cols = {r[1] for r in conn.execute(sqlalchemy.text("PRAGMA table_info(strategy_state)"))}
        # Runtime enable/disable override: NULL = follow the TOML [strategy] enabled
        # flag, 0/1 = operator override via the bot (framework.core.control).
        if "enabled" not in cols:
            conn.execute(sqlalchemy.text("ALTER TABLE strategy_state ADD COLUMN enabled INTEGER"))
        # Runtime execution-mode override: NULL = follow TOML execution_mode,
        # 'approval' | 'auto' = operator override via the bot.
        if "execution_mode" not in cols:
            conn.execute(sqlalchemy.text("ALTER TABLE strategy_state ADD COLUMN execution_mode TEXT"))
    if "managed_positions" in tables:
        # Structural exit deadline (e.g. a calendar spread's near-leg expiry),
        # computed once at entry — see ScheduledExit in framework.positions.exits.
        pos_cols = {r[1] for r in conn.execute(sqlalchemy.text("PRAGMA table_info(managed_positions)"))}
        if "exit_by" not in pos_cols:
            conn.execute(sqlalchemy.text("ALTER TABLE managed_positions ADD COLUMN exit_by TEXT"))


def _create_indexes(conn: Connection, tables: set) -> None:
    """Unique indexes the model generator does not emit (PRAGMA table_info only).

    ``idx_snap_dedup`` was created by ``db_legacy._create_indexes``. The other
    unique indexes reconstruct table-level UNIQUE(...) constraints so
    INSERT OR IGNORE / INSERT OR REPLACE keep their original semantics on
    databases created via ``create_all``.
    """
    statements = []
    if "snapshots" in tables:
        statements.append(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_snap_dedup "
            "ON snapshots(ticker, earnings_date, scan_date, timing, data_source)"
        )
    if "daily_signals" in tables:
        statements.append(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_uq_daily_signals_ticker_date "
            "ON daily_signals(ticker, signal_date)"
        )
    if "picks" in tables:
        statements.append(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_uq_picks_date_strategy_ticker "
            "ON picks(pick_date, strategy, ticker)"
        )
    if "options_chain" in tables:
        statements.append(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_uq_options_chain_contract_hour "
            "ON options_chain(contract_ticker, captured_hour)"
        )
    if "model_registry" in tables:
        statements.append(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_uq_model_registry_name_sha "
            "ON model_registry(name, sha256)"
        )
    if "ff_snapshots" in tables:
        statements.append(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_uq_ff_snapshots_ticker_date "
            "ON ff_snapshots(ticker, scan_date)"
        )
    if "ff_universe_snapshots" in tables:
        statements.append(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_uq_ff_universe_snapshots_ticker_date ON ff_universe_snapshots(ticker, scan_date)"
        )
    for sql in statements:
        conn.execute(sqlalchemy.text(sql))
