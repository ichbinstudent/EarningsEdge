"""E2E: full backtest CLI run on a small fixture DB produces the report artifact.

Spawns ``backtest.py`` as a real subprocess against a temp fixture DB and
asserts the JSON report lands on disk with deterministic, hand-computed
summaries for two strategies (calendar_call_no_ml, earnings_quality).
"""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

import pytest

pytestmark = pytest.mark.e2e

CLI_ROOT = Path(__file__).resolve().parents[2]  # cli_scanner/
EARNINGS_DATE = date(2026, 7, 15)
SCAN_DATE = date(2026, 7, 14)
N_TICKERS = 35  # CalendarCallStrategy.min_rows default is 30 — clear it

_CALENDAR_TRADES_DDL = """
CREATE TABLE IF NOT EXISTS calendar_call_trades (
    snapshot_id INTEGER PRIMARY KEY,
    ticker TEXT NOT NULL,
    earnings_date TEXT NOT NULL,
    scan_date TEXT NOT NULL,
    near_expiry TEXT NOT NULL,
    far_expiry TEXT NOT NULL,
    strike REAL NOT NULL,
    near_call_ticker TEXT NOT NULL,
    far_call_ticker TEXT NOT NULL,
    near_entry REAL NOT NULL,
    far_entry REAL NOT NULL,
    near_exit REAL NOT NULL,
    far_exit REAL NOT NULL,
    net_debit REAL NOT NULL,
    exit_value REAL NOT NULL,
    pnl_dollars REAL NOT NULL,
    return_on_debit REAL,
    model_score REAL,
    model_recommendation INTEGER,
    model_reason TEXT,
    model_name TEXT,
    model_scored_at TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
)
"""


def _seed_fixture_db(db_path) -> None:
    """35 deterministic rows in both snapshots and calendar_call_trades.

    pnl_dollars: +12 on even i (18 rows), -6 on odd i (17 rows) -> total 114.
    actual_move_pct: +8 on even i, -8 on odd i (all clear the 5% gate).
    """
    import sqlite3
    from earnings_edge.db import insert_snapshot

    near_expiry = (EARNINGS_DATE + timedelta(days=7)).isoformat()
    far_expiry = (EARNINGS_DATE + timedelta(days=35)).isoformat()

    conn = sqlite3.connect(str(db_path), timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute(_CALENDAR_TRADES_DDL)
        for i in range(N_TICKERS):
            ticker = f"E2EBT{i:02d}"
            price = 100.0 + i
            net_debit = 0.60
            pnl = 12.0 if i % 2 == 0 else -6.0
            move = 8.0 if i % 2 == 0 else -8.0
            insert_snapshot(conn, {
                "ticker": ticker,
                "earnings_date": EARNINGS_DATE.isoformat(),
                "scan_date": SCAN_DATE.isoformat(),
                "timing": "Post Market",
                "price": price,
                "avg_volume_30d": 5_000_000,
                "has_options": 1,
                "data_source": "e2e_fixture",
            })
            conn.execute(
                "UPDATE snapshots SET pre_earnings_close = ?, "
                "post_earnings_close = ?, actual_move_pct = ? WHERE ticker = ?",
                (price, price * (1 + move / 100), move, ticker),
            )
            conn.execute(
                """
                INSERT INTO calendar_call_trades (
                    ticker, earnings_date, scan_date,
                    near_expiry, far_expiry, strike,
                    near_call_ticker, far_call_ticker,
                    near_entry, far_entry, near_exit, far_exit,
                    net_debit, exit_value, pnl_dollars, return_on_debit
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    ticker, EARNINGS_DATE.isoformat(), SCAN_DATE.isoformat(),
                    near_expiry, far_expiry, price,
                    f"O:{ticker}NEAR", f"O:{ticker}FAR",
                    0.40, 1.00,
                    0.05, 0.90,
                    net_debit, 100.0, pnl, pnl / (net_debit * 100),
                ),
            )
        conn.commit()
    finally:
        conn.close()


def test_backtest_cli_produces_report_artifact(tmp_db_path, tmp_path):
    _seed_fixture_db(tmp_db_path)
    report = tmp_path / "backtest_report.json"

    proc = subprocess.run(
        [
            sys.executable, "backtest.py",
            "--db", str(tmp_db_path),
            "--strategies", "calendar_call_no_ml", "earnings_quality",
            "--output", str(report),
        ],
        cwd=CLI_ROOT, capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"

    # Report artifact exists and parses.
    assert report.exists()
    data = json.loads(report.read_text())
    assert set(data) == {"calendar_call_no_ml", "earnings_quality"}

    # calendar_call_no_ml takes every gated row: 35/35, pnl = 18*12 - 17*6.
    cal = data["calendar_call_no_ml"]["summary"]
    assert cal["total"] == N_TICKERS
    assert cal["taken"] == N_TICKERS
    assert cal["pnl"] == pytest.approx(114.0)
    assert cal["win_rate"] == pytest.approx(18 / 35)

    # earnings_quality: 35 surprises, 18 up / 17 down.
    eq = data["earnings_quality"]["summary"]
    assert eq["total"] == N_TICKERS
    assert eq["long_trades"] == 18
    assert eq["short_trades"] == 17
    assert eq["total_pnl_pct"] == pytest.approx(8.0)  # 18*8 - 17*8
    assert eq["win_rate"] == pytest.approx(18 / 35)
