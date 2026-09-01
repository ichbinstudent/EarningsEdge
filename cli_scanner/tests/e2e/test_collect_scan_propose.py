"""E2E: collect -> persist -> strategy scan -> proposal store.

Runs the full pipeline on fixture data into a temp DB:
1. EarningsCalendarCollector with mocked investing.com/Finnhub sources
   produces candidates (collect phase).
2. Candidates are persisted as snapshots; matching calendar_call_trades
   fixture rows are seeded (this is what the backtester would have written).
3. The calendar_call_no_ml strategy runs over DataBundle.from_db — the same
   code path the bot's proposal builder uses (scan phase).
4. TAKE trades are persisted via PendingTradeStore and the stored proposal
   row is asserted for shape (propose phase).

No network, no Alpaca, no Telegram — the pipeline stops at the proposal
store, which is exactly where the human approval gate sits in production.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import date, timedelta

import pytest

pytestmark = pytest.mark.e2e

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


def _tickers() -> list[str]:
    return [f"E2E{i:02d}" for i in range(N_TICKERS)]


def _collect_phase(collector, monkeypatch) -> list:
    """Collect candidates with mocked sources (investing primary, finnhub fallback)."""
    from earnings_edge.models import EarningsCandidate

    monkeypatch.setattr(
        collector, "_investing_fetch",
        lambda d: [
            EarningsCandidate(ticker=t, timing="Post Market", source="investing")
            for t in _tickers()
        ],
    )

    def _finnhub_down(d):
        raise ValueError("FINNHUB_API_KEY not set")

    monkeypatch.setattr(collector, "_finnhub_fetch", _finnhub_down)
    return collector.fetch(EARNINGS_DATE)


def _persist_phase(tmp_db_path, candidates) -> None:
    """Write snapshots + matching calendar_call_trades fixture rows."""
    import sqlite3
    from earnings_edge.db import insert_snapshot

    near_expiry = (EARNINGS_DATE + timedelta(days=7)).isoformat()
    far_expiry = (EARNINGS_DATE + timedelta(days=35)).isoformat()

    conn = sqlite3.connect(str(tmp_db_path), timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute(_CALENDAR_TRADES_DDL)
        for i, c in enumerate(candidates):
            price = 100.0 + i  # vary prices so rows aren't identical
            insert_snapshot(conn, {
                "ticker": c.ticker,
                "earnings_date": EARNINGS_DATE.isoformat(),
                "scan_date": SCAN_DATE.isoformat(),
                "timing": c.timing,
                "price": price,
                "avg_volume_30d": 5_000_000,
                "has_options": 1,
                "data_source": "e2e_fixture",
            })
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
                    c.ticker, EARNINGS_DATE.isoformat(), SCAN_DATE.isoformat(),
                    near_expiry, far_expiry, price,  # ATM: strike == price
                    f"O:{c.ticker}{near_expiry.replace('-', '')}C{int(price):08d}",
                    f"O:{c.ticker}{far_expiry.replace('-', '')}C{int(price):08d}",
                    0.40, 1.00,   # near_entry, far_entry
                    0.05, 0.90,   # near_exit, far_exit (IV crush on near leg)
                    0.60,         # net_debit (combo ask convention)
                    0.85,         # exit_value
                    25.0,         # pnl_dollars
                    0.42,         # return_on_debit
                ),
            )
        conn.commit()
    finally:
        conn.close()


def test_collect_scan_propose_pipeline(tmp_db_path, tmp_path, monkeypatch, test_settings):
    from earnings_edge.collectors.earnings_calendar import EarningsCalendarCollector
    from earnings_edge.backtest.calendar import CalendarCallNoML
    from earnings_edge.trading_types import DataBundle
    from earnings_edge.trade_approval import PendingTradeStore

    # -- 1. collect ------------------------------------------------------
    collector = EarningsCalendarCollector()
    candidates = _collect_phase(collector, monkeypatch)
    assert len(candidates) == N_TICKERS
    assert all(c.source == "investing" for c in candidates)

    # -- 2. persist fixture data ------------------------------------------
    _persist_phase(tmp_db_path, candidates)

    # -- 3. scan: real strategy over DataBundle.from_db --------------------
    bundle = DataBundle.from_db(str(tmp_db_path))
    assert len(bundle.snapshots) == N_TICKERS
    assert len(bundle.calendar_trades) == N_TICKERS

    strategy = CalendarCallNoML(
        model_path=str(tmp_path / "no_such_model.joblib"),  # hermetic: no real artifact
        min_rows=30,
    )
    result = strategy.run(bundle)

    assert len(result.trades) == N_TICKERS
    assert all(t.ml_decision == "TAKE" for t in result.trades)
    assert all(t.side == "CALENDAR" for t in result.trades)
    assert all(t.strategy == "calendar_call_no_ml" for t in result.trades)
    assert result.summary["taken"] == N_TICKERS

    # -- 4. propose: persist and assert proposal row shape ------------------
    store = PendingTradeStore(db_path=str(tmp_db_path))
    takes = [t for t in result.trades if t.ml_decision == "TAKE"]
    ids = [store.add(t, f"card for {t.ticker}") for t in takes]
    assert all(pid is not None for pid in ids)

    first_id = ids[0]
    assert first_id is not None
    row = store.get(first_id)
    assert row is not None
    # DB row shape
    assert row["status"] == "pending"
    assert row["strategy"] == "calendar_call_no_ml"
    assert row["ticker"] == takes[0].ticker
    assert row["side"] == "CALENDAR"
    assert row["created_at"]  # ISO timestamp present
    assert row["decided_at"] is None
    assert row["card_text"].startswith("card for ")
    # trade_json payload shape — what execute_proposal() will rehydrate
    payload = json.loads(row["trade_json"])
    assert {
        "ticker", "earnings_date", "scan_date", "strategy", "side",
        "entry_price", "exit_price", "pnl", "pnl_pct",
        "features", "model_score", "ml_decision", "notes",
    } <= set(payload)
    assert payload["earnings_date"] == EARNINGS_DATE.isoformat()
    assert payload["scan_date"] == SCAN_DATE.isoformat()
    assert payload["ml_decision"] == "TAKE"
    assert payload["entry_price"] == pytest.approx(0.60)

    # Dedupe contract: an identical pending proposal is never double-asked
    assert store.add(takes[0], "duplicate card") is None
    assert len(store.list_pending()) == N_TICKERS
