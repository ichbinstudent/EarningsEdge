"""Tests for the final wiring layer: TOML sizers in the execution path,
runtime strategy enable/disable overrides, and the bot's operational views."""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from earnings_edge.alpaca_bridge import BridgeConfig, StrategyBridge
from earnings_edge.trading_types import Trade
from framework.core.control import (
    clear_override, effective_enabled, enabled_overrides, filter_enabled,
    set_enabled,
)
from framework.core.config import StrategyConfig
from framework.core.registry import StrategyRegistry
from framework.risk.manager import RiskLimits, RiskManager
from sqlalchemy import text
from earnings_edge.db import configure, engine as db_engine


def _calendar_trade(entry_price=1.85):
    return Trade(
        ticker="AAPL", earnings_date=date(2026, 7, 29), scan_date=date(2026, 7, 28),
        strategy="calendar_call_ml", side="CALENDAR", entry_price=entry_price,
        features={
            "near_strike": 190.0, "far_strike": 190.0,
            "near_expiry": date(2026, 7, 31), "far_expiry": date(2026, 8, 28),
        },
        model_score=0.7, ml_decision="TAKE",
    )


def _client(equity="100000", buying_power="50000"):
    client = MagicMock()
    client.position_symbols.return_value = set()
    client.get_account.return_value = {"equity": equity, "buying_power": buying_power}
    client.get_positions.return_value = []
    client.get_option_snapshot.return_value = {}
    client.submit_multi_leg_order.return_value = {"id": "o1", "status": "accepted", "legs": []}
    return client


def _bridge(tmp_path, client, sizer=None, limits=None):
    configure(tmp_path / "fw.db")
    return StrategyBridge(
        client=client, config=BridgeConfig(),
        risk_manager=RiskManager(limits=limits),
        sizer_resolver=(lambda name: sizer) if sizer else None,
    )


# ── Sizer wiring -------------------------------------------------------------

def test_sizer_pct_portfolio_scales_qty(tmp_path):
    # 5% of 100k equity = $5000 budget; unit cost $185 → 27, capped at 25.
    # Per-trade cap: 10% of 50k BP = $5000 → 25 * 185 = $4625 passes.
    bridge = _bridge(tmp_path, _client(), sizer={"name": "pct_portfolio", "pct": 0.05})
    result = bridge.execute_trade(_calendar_trade())
    assert result is not None
    call = bridge.client.submit_multi_leg_order.call_args.kwargs
    # legs keep their base ratio (1:1) — Alpaca requires ratio_qty values to
    # be relatively prime; the sizer's contract count is the order-level qty.
    assert all(l["ratio_qty"] == 1 for l in call["legs"])
    assert call["qty"] == 25


def test_sizer_fixed_dollar_qty(tmp_path):
    # $2000 budget / $185 unit = 10 contracts.
    bridge = _bridge(tmp_path, _client(), sizer={"name": "fixed_dollar", "budget": 2000.0})
    result = bridge.execute_trade(_calendar_trade())
    assert result is not None
    call = bridge.client.submit_multi_leg_order.call_args.kwargs
    assert all(l["ratio_qty"] == 1 for l in call["legs"])
    assert call["qty"] == 10


def test_sizer_qty_flows_into_result_legs_for_position_tracking(tmp_path):
    """record_open_positions derives each leg's tracked quantity from
    result.legs[i]['ratio_qty'] — this must carry the ACTUAL total contracts
    (base ratio × sizer qty), not the reduced ratio sent to Alpaca, or exits
    would only ever try to close 1 contract regardless of size. Deliberately
    doesn't depend on Alpaca's own response shape (the mock returns "legs":
    [] here, same as it would if a real response didn't need trusting)."""
    bridge = _bridge(tmp_path, _client(), sizer={"name": "fixed_dollar", "budget": 2000.0})
    result = bridge.execute_trade(_calendar_trade())
    assert result is not None
    assert len(result.legs) == 2
    assert all(l["ratio_qty"] == 10 for l in result.legs)


def test_sizer_veto_when_budget_below_unit_cost(tmp_path):
    # $100 budget < $185 unit cost → qty 0 → veto, no order.
    bridge = _bridge(tmp_path, _client(), sizer={"name": "fixed_dollar", "budget": 100.0})
    assert bridge.execute_trade(_calendar_trade()) is None
    assert bridge.skip_reasons["size_veto"] == 1
    bridge.client.submit_multi_leg_order.assert_not_called()


def test_sized_cost_flows_into_risk_gate(tmp_path):
    # 2% of 100k = $2000 → 10 contracts at $185 = $1850; tight per-trade cap
    # of 0.5% of 50k BP = $250 vetoes the SIZED cost ($185 would pass at qty 1).
    bridge = _bridge(tmp_path, _client(),
                     sizer={"name": "fixed_dollar", "budget": 2000.0},
                     limits=RiskLimits(max_pct_per_trade=0.005))
    assert bridge.execute_trade(_calendar_trade()) is None
    assert bridge.skip_reasons["risk_veto"] == 1


def test_no_sizer_keeps_qty_one(tmp_path):
    bridge = _bridge(tmp_path, _client())  # no sizer resolver
    result = bridge.execute_trade(_calendar_trade())
    assert result is not None
    legs = bridge.client.submit_multi_leg_order.call_args.kwargs["legs"]
    assert all(l["ratio_qty"] == 1 for l in legs)


def test_probation_multiplier_scales_sized_qty_down(tmp_path):
    configure(tmp_path / "fw.db")
    from framework.execution.lifecycle import LifecycleManager
    lm = LifecycleManager()
    lm.set_state("calendar_call_ml", "probation", by="test")
    bridge = StrategyBridge(
        client=_client(), config=BridgeConfig(),
        risk_manager=RiskManager(),
        lifecycle_manager=lm,
        sizer_resolver=lambda name: {"name": "fixed_dollar", "budget": 2000.0},
    )
    result = bridge.execute_trade(_calendar_trade())
    assert result is not None
    call = bridge.client.submit_multi_leg_order.call_args.kwargs
    # 10 contracts × 0.5 probation = 5; legs still keep their base ratio (1:1)
    assert all(l["ratio_qty"] == 1 for l in call["legs"])
    assert call["qty"] == 5


def test_record_entry_uses_scaled_cost(tmp_path):
    configure(tmp_path / "fw.db")
    bridge = StrategyBridge(
        client=_client(), config=BridgeConfig(),
        risk_manager=RiskManager(),
        sizer_resolver=lambda name: {"name": "fixed_dollar", "budget": 2000.0},
    )
    assert bridge.execute_trade(_calendar_trade()) is not None
    with db_engine.get_session() as s:
        row = s.execute(
            text("SELECT detail FROM risk_events WHERE event_type = 'entry'")
        ).mappings().first()
    assert "cost=1850.00" in row["detail"]  # 10 × $185, not $185


def test_account_fetched_once_for_sizing_and_risk(tmp_path):
    client = _client()
    bridge = _bridge(tmp_path, client,
                     sizer={"name": "fixed_dollar", "budget": 2000.0})
    bridge.execute_trade(_calendar_trade())
    assert client.get_account.call_count == 1


# ── Runtime enable/disable overrides -----------------------------------------

def test_control_set_and_effective(tmp_path):
    configure(tmp_path / "fw.db")
    assert effective_enabled("s1", True) is True     # no row → TOML default
    set_enabled("s1", False, by="test")
    assert effective_enabled("s1", True) is False    # override wins
    assert enabled_overrides() == {"s1": False}
    clear_override("s1", by="test")
    assert effective_enabled("s1", True) is True     # back to TOML
    assert enabled_overrides() == {}


def test_control_preserves_lifecycle_on_toggle(tmp_path):
    configure(tmp_path / "fw.db")
    from framework.execution.lifecycle import LifecycleManager
    LifecycleManager().set_state("s1", "live", by="test")
    set_enabled("s1", False, by="test")
    assert LifecycleManager().state("s1") == "live"    # toggle doesn't demote
    set_enabled("s1", True, by="test")
    assert LifecycleManager().state("s1") == "live"


def test_filter_enabled_layers_db_on_toml(tmp_path):
    configure(tmp_path / "fw.db")
    set_enabled("b", False, by="test")
    names = filter_enabled(["a", "b", "c"], toml_enabled=lambda n: n != "c")
    assert names == ["a"]  # c off via TOML, b off via override


def test_migration_adds_enabled_column_to_old_db(tmp_path):
    # Simulate a pre-migration DB: strategy_state without the enabled column.
    import sqlite3
    path = tmp_path / "fw.db"
    conn = sqlite3.connect(str(path))
    conn.execute(
        "CREATE TABLE IF NOT EXISTS strategy_state ("
        " name TEXT PRIMARY KEY, lifecycle TEXT NOT NULL DEFAULT 'paper',"
        " updated_at TEXT, updated_by TEXT)")
    conn.execute("INSERT INTO strategy_state (name, lifecycle) VALUES ('old', 'live')")
    conn.commit()
    conn.close()

    configure(path)  # migration runs here
    with db_engine.get_session() as s:
        cols = {r["name"] for r in s.execute(text("PRAGMA table_info(strategy_state)")).mappings()}
    assert "enabled" in cols
    # existing row survives, lifecycle intact, override unset (→ TOML default)
    assert effective_enabled("old", True) is True
    from framework.execution.lifecycle import LifecycleManager
    assert LifecycleManager().state("old") == "live"


def test_build_proposals_respects_db_override(tmp_path):
    from earnings_edge.trading_types import StrategyResult
    from earnings_edge.trade_approval import PendingTradeStore, build_proposals

    store = PendingTradeStore(str(tmp_path / "test.db"))
    configure(store._db_path)
    set_enabled("calendar_call_ml", False, by="test")

    source = MagicMock(return_value=[_calendar_trade()])
    reg = StrategyRegistry(configs={})
    with patch("framework.core.registry.get_registry", lambda: reg):
        rows = build_proposals(store, strategies=["calendar_call_ml"],
                               trade_source=source)
    assert rows == []
    source.assert_not_called()  # filtered before the signal source is consulted


# ── Bot views ------------------------------------------------------------------

def _seed_fw(tmp_path):
    from datetime import datetime, timezone
    today = datetime.now(timezone.utc).date().isoformat()
    configure(tmp_path / "fw.db")
    with db_engine.session_scope() as s:
        s.execute(text(
            "INSERT INTO equity_snapshots (ts, equity, buying_power, portfolio_value) "
            f"VALUES ('{today}T14:00:00+00:00', 100000, 50000, 100000)"))
        s.execute(text(
            "INSERT INTO equity_snapshots (ts, equity, buying_power, portfolio_value) "
            f"VALUES ('{today}T15:00:00+00:00', 99500, 50000, 99500)"))
        s.execute(text(
            "INSERT INTO job_runs (job_name, started_at, finished_at, success, stats_json) "
            f"VALUES ('equity_snapshot', '{today}T15:00:00', '{today}T15:00:01', 1, '{{\"equity\": 99500}}')"))
        s.execute(text(
            "INSERT INTO job_runs (job_name, started_at, finished_at, success, error) "
            f"VALUES ('reconcile', '{today}T15:30:00', '{today}T15:30:01', 0, 'boom')"))
        s.execute(text(
            "INSERT INTO trade_events (ts, event_type, symbol, strategy, qty, price, detail) "
            f"VALUES ('{today}T15:00:00', 'exit_filled', 'AAPL', 'calendar_call_ml', 1, 2.1, 'PT hit')"))


def test_status_view_renders(tmp_path):
    from earnings_edge.bot_views import status_view
    _seed_fw(tmp_path)
    text = status_view( market_open=True, pending_proposals=2, pending_exits=1)
    assert "SYSTEM STATUS" in text
    assert "🟢 open" in text
    assert "🟢 armed" in text
    assert "<b>Equity:</b> $99,500" in text
    assert "<b>Pending proposals: 2</b>" in text
    assert "reconcile" in text and "boom" in text  # failure surfaced


def test_positions_view_empty_and_seeded(tmp_path):
    from earnings_edge.bot_views import positions_view
    from framework.execution.managed import record_open_positions
    _seed_fw(tmp_path)
    assert "No open managed positions" in positions_view()
    record_open_positions([{"symbol": "AAPL260731C00190000", "side": "sell", "ratio_qty": 2,
          "option_type": "call", "strike": 190.0, "expiry": date(2026, 7, 31)},
         {"symbol": "AAPL260828C00190000", "side": "buy", "ratio_qty": 2,
          "option_type": "call", "strike": 190.0, "expiry": date(2026, 8, 28)}],
        "calendar_call_ml", group_id="ord-1",
        entry_price=1.85, metadata={"side": "CALENDAR", "earnings_date": "2026-07-29"},
    )
    text = positions_view()
    assert "calendar_call_ml" in text and "AAPL" in text
    assert "x1" in text or "x2" in text
    assert "AAPL260731C00190000" in text


def test_orders_and_jobs_views(tmp_path):
    from earnings_edge.bot_views import jobs_view, orders_view
    _seed_fw(tmp_path)
    orders = orders_view()
    assert "exit_filled" in orders and "AAPL" in orders
    jobs = jobs_view()
    assert "equity_snapshot" in jobs and "reconcile" in jobs
    assert "boom" in jobs


def test_equity_view(tmp_path):
    from earnings_edge.bot_views import equity_view
    _seed_fw(tmp_path)
    text = equity_view()
    assert "$99,500" in text
    assert "<b>Day PnL:</b> $-500" in text  # 100000 day start → 99500


def test_strategies_view_and_buttons(tmp_path):
    from earnings_edge.bot_views import strategies_view
    _seed_fw(tmp_path)
    cfg = StrategyConfig(name="s1", enabled=True, execution_mode="approval",
                         lifecycle="paper", sizer={"name": "fixed_dollar", "budget": 1000.0},
                         limits={}, exits=[])
    reg = StrategyRegistry(configs={"s1": cfg})
    set_enabled("s1", False, by="test")
    text, buttons = strategies_view( registry=reg)
    assert "s1" in text and "⏸" in text
    assert "fixed_dollar" in text
    assert buttons == [{"name": "s1", "enabled": False}]
