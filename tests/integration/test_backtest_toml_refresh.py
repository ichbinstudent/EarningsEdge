"""Integration: backtest engine over TOML-registered strategies + exit engine.

Pins two contracts on fixture data (temp DB, no network, no model artifacts):

1. The backtest engine path (``DataBundle.from_db`` -> strategy registry ->
   ``run`` -> summary) produces deterministic, hand-computable metrics for
   TOML-registered strategies (``debit_size_exploit``, ``earnings_quality``).
2. The exit engine drives fixture positions using the REAL
   ``strategies/calendar_call_ml.toml`` exit rules loaded through
   ``StrategyRegistry`` — profit target auto-closes, time exits land in
   ``exit_proposals``.
"""

from __future__ import annotations

import sqlite3
from datetime import date, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.integration

EARNINGS_DATE = date(2026, 7, 15)
SCAN_DATE = date(2026, 7, 14)
N_TICKERS = 35  # CalendarCallStrategy.min_rows default is 30 — clear it
TODAY = date(2026, 7, 27)  # Monday

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


def _seed_calendar_fixture(db_path) -> None:
    """35 deterministic snapshot + calendar_call_trades row pairs.

    i < 30:  net_debit 0.60 (debit <= 3% of price -> passes debit_size_exploit)
    i >= 30: net_debit 5.00 (debit > 3% of price  -> filtered out)
    pnl_dollars: +12 on even i, -6 on odd i.
    """
    from sqlalchemy import text
    from earnings_edge.db import engine as db_engine
    from earnings_edge.db import insert_snapshot

    near_expiry = (EARNINGS_DATE + timedelta(days=7)).isoformat()
    far_expiry = (EARNINGS_DATE + timedelta(days=35)).isoformat()

    db_engine.configure(db_path)
    with db_engine.session_scope() as s:
        s.execute(text(_CALENDAR_TRADES_DDL))
    for i in range(N_TICKERS):
        ticker = f"FIX{i:02d}"
        price = 100.0 + i
        insert_snapshot({
            "ticker": ticker,
            "earnings_date": EARNINGS_DATE.isoformat(),
            "scan_date": SCAN_DATE.isoformat(),
            "timing": "Post Market",
            "price": price,
            "avg_volume_30d": 5_000_000,
            "has_options": 1,
            "data_source": "integration_fixture",
        })
    with db_engine.session_scope() as s:
        for i in range(N_TICKERS):
            ticker = f"FIX{i:02d}"
            price = 100.0 + i
            net_debit = 0.60 if i < 30 else 5.00
            pnl = 12.0 if i % 2 == 0 else -6.0
            s.execute(text(
                """
                INSERT INTO calendar_call_trades (
                    ticker, earnings_date, scan_date,
                    near_expiry, far_expiry, strike,
                    near_call_ticker, far_call_ticker,
                    near_entry, far_entry, near_exit, far_exit,
                    net_debit, exit_value, pnl_dollars, return_on_debit
                ) VALUES (:tk,:ed,:sc,:ne,:fe,:st,:nc,:fc,:n1,:f1,:n2,:f2,:nd,:ev,:pnl,:rod)
                """),
                {
                    "tk": ticker, "ed": EARNINGS_DATE.isoformat(), "sc": SCAN_DATE.isoformat(),
                    "ne": near_expiry, "fe": far_expiry, "st": price,  # ATM: strike == price
                    "nc": f"O:{ticker}NEAR", "fc": f"O:{ticker}FAR",
                    "n1": 0.40, "f1": 1.00,
                    "n2": 0.05, "f2": 0.90,
                    "nd": net_debit, "ev": 100.0, "pnl": pnl,
                    "rod": pnl / (net_debit * 100),
                },
            )


def test_toml_strategy_backtest_deterministic_metrics(tmp_db_path, monkeypatch):
    """debit_size_exploit (TOML-registered) on fixture data: exact metrics."""
    _seed_calendar_fixture(tmp_db_path)

    import earnings_edge.backtest.calendar as strat_mod
    from earnings_edge.trading_types import DataBundle
    from earnings_edge.backtest.calendar import get_strategy

    # Neutralise the ML artifact: the engine path (join -> quality gates ->
    # debit filter -> summarize) is what is under test, not the model.
    stub_score = SimpleNamespace(probability=1.0, recommended=True, reason="stub")
    monkeypatch.setattr(
        strat_mod, "score_calendar_trade", lambda **kwargs: stub_score
    )
    monkeypatch.setattr(
        strat_mod.CalendarCallStrategy, "_load_model",
        lambda self: {"features": [], "pipeline": None},
    )

    data = DataBundle.from_db(str(tmp_db_path))
    strat = get_strategy("debit_size_exploit")  # name exists in strategies/*.toml

    res1 = strat.run(data)
    res2 = strat.run(data)
    assert res1.summary == res2.summary  # deterministic across runs

    s = res1.summary
    # 5 high-debit rows (net_debit 5.00 on ~$130 prices > 3%) filtered out.
    assert s["total"] == 30
    assert s["taken"] == 30
    # i in 0..29: 15 even rows (+12) and 15 odd rows (-6) -> 15*12 - 15*6 = 90
    assert s["pnl"] == pytest.approx(90.0)
    assert s["win_rate"] == pytest.approx(0.5)


def test_earnings_quality_toml_strategy_exact_metrics(tmp_db_path):
    """earnings_quality (TOML-registered) on fixture snapshots: exact metrics."""
    import sqlite3
    from earnings_edge.trading_types import DataBundle
    from earnings_edge.backtest.calendar import get_strategy

    conn = sqlite3.connect(str(tmp_db_path), timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute(_CALENDAR_TRADES_DDL)  # DataBundle.from_db reads all 5 tables
        rows = [
            ("FIXU", 100.0, 108.0, 8.0),    # |move| > 5 -> LONG_STOCK trade
            ("FIXD", 100.0, 92.0, -8.0),    # |move| > 5 -> SHORT_STOCK trade
            ("FIXF", 100.0, 103.0, 3.0),    # |move| < 5 -> no trade
        ]
        for ticker, pre, post, move in rows:
            conn.execute(
                """
                INSERT INTO snapshots (
                    ticker, earnings_date, scan_date, timing, price,
                    pre_earnings_close, post_earnings_close, actual_move_pct,
                    data_source
                ) VALUES (?,?,?,?,?,?,?,?,?)
                """,
                (
                    ticker, EARNINGS_DATE.isoformat(), SCAN_DATE.isoformat(),
                    "Post Market", pre, pre, post, move, "integration_fixture",
                ),
            )
        conn.commit()
    finally:
        conn.close()

    data = DataBundle.from_db(str(tmp_db_path))
    res = get_strategy("earnings_quality").run(data)

    assert res.summary["total"] == 2
    assert res.summary["long_trades"] == 1
    assert res.summary["short_trades"] == 1
    assert res.summary["total_pnl_pct"] == pytest.approx(0.0)   # +8 + (-8)
    assert res.summary["avg_return_pct"] == pytest.approx(0.0)
    assert res.summary["win_rate"] == pytest.approx(0.5)


# ── exit engine on fixture positions, real TOML rules ────────────────────────


def _snaps(prices: dict[str, float], width: float = 0.1) -> dict[str, dict]:
    return {
        sym: {"latestQuote": {"bp": px - width / 2, "ap": px + width / 2}}
        for sym, px in prices.items()
    }


def _stub_client(snaps: dict, fill_price: float | None = None):
    client = MagicMock()
    client.get_option_snapshots_bulk.return_value = snaps
    seq = {"n": 0}

    def submit_order(symbol, qty, side, order_type, limit_price, time_in_force, client_order_id):
        seq["n"] += 1
        return {"id": f"e{seq['n']}", "status": "filled",
                "filled_qty": qty, "filled_avg_price": fill_price or limit_price or 1.0}

    def submit_multi_leg_order(legs, qty, order_type, limit_price, time_in_force, client_order_id):
        return submit_order(legs[0]["symbol"], qty, legs[0]["side"], order_type,
                            limit_price, time_in_force, client_order_id)

    client.submit_order.side_effect = submit_order
    client.submit_multi_leg_order.side_effect = submit_multi_leg_order
    client.get_order.side_effect = lambda oid: {
        "id": oid, "status": "filled", "filled_qty": 1, "filled_avg_price": fill_price or 1.0}
    client.cancel_order.return_value = {}
    return client


def _seed_calendar_position(group_id="g1", entry=1.85,
                            opened="2026-07-24T14:00:00+00:00", exit_by=None):
    from sqlalchemy import text
    from framework.execution.managed import record_open_positions
    from earnings_edge.db import engine as db_engine

    legs = [
        {"symbol": "AAPL260731C00190000", "side": "sell", "ratio_qty": 1,
         "option_type": "call", "strike": 190.0, "expiry": date(2026, 7, 31)},
        {"symbol": "AAPL260828C00190000", "side": "buy", "ratio_qty": 1,
         "option_type": "call", "strike": 190.0, "expiry": date(2026, 8, 28)},
    ]
    record_open_positions(legs, "calendar_call_ml", group_id=group_id,
                          entry_price=entry, exit_by=exit_by,
                          metadata={"side": "CALENDAR", "credit": False,
                                    "earnings_date": "2026-07-29"})
    with db_engine.session_scope() as s:
        s.execute(
            text("UPDATE managed_positions SET opened_at = :opened WHERE group_id = :gid"),
            {"opened": opened, "gid": group_id},
        )


def _exit_manager(client, registry):
    from framework.execution.order_manager import OrderManager
    from framework.positions.manager import ExitManager

    return ExitManager(client, registry=registry,
                       order_manager=OrderManager(client, poll_secs=0,
                                                  sleep=lambda s: None),
                       today=TODAY)


def test_exit_engine_auto_closes_on_real_toml_profit_target(tmp_path):
    """Fixture position + real calendar_call_ml.toml exits -> auto close."""
    from sqlalchemy import text
    from framework.core.registry import StrategyRegistry
    from framework.execution.managed import open_groups
    from earnings_edge.db import configure, engine as db_engine

    configure(tmp_path / "fw.db")
    registry = StrategyRegistry()  # loads real strategies/*.toml
    cfg = registry.get("calendar_call_ml")
    assert cfg is not None
    assert [r["rule"] for r in cfg.exits] == ["scheduled", "profit_target", "stop_loss"]

    _seed_calendar_position(entry=1.85)
    # value now 3.0 vs entry 1.85 -> +62% >= 50% profit target
    snaps = _snaps({"AAPL260731C00190000": 1.0, "AAPL260828C00190000": 4.0})
    client = _stub_client(snaps, fill_price=3.0)

    out = _exit_manager(client, registry).evaluate_all()
    assert len(out["auto_closed"]) == 1
    assert "profit_target" in out["auto_closed"][0]
    assert open_groups() == []
    with db_engine.get_session() as s:
        ev = s.execute(
            text("SELECT * FROM trade_events WHERE event_type = 'exit_filled'")
        ).mappings().first()
    assert ev is not None and "realized_pnl" in ev["detail"]


def test_exit_engine_scheduled_exit_auto_closes_on_real_toml_rules(tmp_path):
    """Structural deadline (exit_by, computed at entry — the near leg's
    expiry) reached with the session close approaching -> real
    calendar_call_ml.toml's "scheduled" rule auto-closes, no approval card.
    This replaced a fixed days_after_entry TOML rule specifically because it
    didn't track the near leg's real expiry (see ScheduledExit docstring)."""
    from framework.core.registry import StrategyRegistry
    from framework.execution.managed import open_groups
    from earnings_edge.db import configure

    configure(tmp_path / "fw.db")
    registry = StrategyRegistry()
    _seed_calendar_position(opened="2026-07-21T14:00:00+00:00",
                            exit_by=date(2026, 7, 31))
    # value ~entry (1.80 vs 1.85): neither profit target nor stop loss —
    # only the scheduled deadline should trigger this
    snaps = _snaps({"AAPL260731C00190000": 1.0, "AAPL260828C00190000": 2.8})
    client = _stub_client(snaps, fill_price=1.8)
    client.get_clock.return_value = {
        "is_open": True,
        "timestamp": "2026-07-31T14:30:00-04:00",
        "next_close": "2026-07-31T16:00:00-04:00",  # 90min to close
    }

    from framework.execution.order_manager import OrderManager
    from framework.positions.manager import ExitManager
    mgr = ExitManager(client, registry=registry,
                      order_manager=OrderManager(client, poll_secs=0, sleep=lambda s: None),
                      today=date(2026, 7, 31))
    out = mgr.evaluate_all()
    assert len(out["auto_closed"]) == 1
    assert "scheduled" in out["auto_closed"][0]
    assert open_groups() == []  # closed, not left for approval
    assert not mgr.pending_exit_proposals()


def test_exit_engine_scheduled_exit_no_op_without_exit_by(tmp_path):
    """A position opened before exit_by tracking existed (or any structure
    without a differential-expiry deadline) must not get a phantom exit
    signal now that calendar_call_ml has no day-count time exit fallback."""
    from framework.core.registry import StrategyRegistry
    from framework.execution.managed import open_groups
    from earnings_edge.db import configure

    configure(tmp_path / "fw.db")
    registry = StrategyRegistry()
    _seed_calendar_position(opened="2026-07-21T14:00:00+00:00", exit_by=None)
    snaps = _snaps({"AAPL260731C00190000": 1.0, "AAPL260828C00190000": 2.8})
    client = _stub_client(snaps)
    client.get_clock.return_value = {
        "is_open": True,
        "timestamp": "2026-07-27T15:00:00-04:00",
        "next_close": "2026-07-27T16:00:00-04:00",
    }

    mgr = _exit_manager(client, registry)
    out = mgr.evaluate_all()
    assert out["held"] == 1
    assert not out["auto_closed"] and not out["proposed"]
    assert open_groups() != []
