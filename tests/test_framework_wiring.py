"""Tests for the framework wiring: registry consumption, bridge risk gate
(exposure, cost fallback, per-strategy limits), FF ladder risk integration,
and proposal config gating."""

from __future__ import annotations

import sqlite3
from datetime import date, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from earnings_edge.alpaca_bridge import BridgeConfig, StrategyBridge
from earnings_edge.fwd_factor import ET, LadderSpec, occ_symbol
from earnings_edge.fwd_factor_ladder import LadderRunner, build_candidate
from earnings_edge.trading_types import StrategyResult, Trade
from earnings_edge.trade_approval import (
    PendingTradeStore, build_proposals, execute_proposal,
)
from framework.core.config import StrategyConfig
from framework.core.registry import StrategyRegistry
from framework.execution.managed import open_positions
from framework.risk.killswitch import KillSwitch
from framework.risk.manager import RiskLimits, RiskManager
from sqlalchemy import text
from earnings_edge.db import configure, engine as db_engine

from test_fwd_factor_ladder import FakeAlpaca, _bs  # noqa: E402


# ── Registry -----------------------------------------------------------------

def _cfg(name, **over):
    base = dict(name=name, enabled=True, execution_mode="approval",
                lifecycle="paper", sizer={}, limits={}, exits=[])
    base.update(over)
    return StrategyConfig(**base)


def test_registry_limits_override_and_fallback():
    reg = StrategyRegistry(configs={
        "tight": _cfg("tight", limits={"max_pct_per_trade": 0.02}),
    })
    assert reg.limits_for("tight").max_pct_per_trade == 0.02
    # unconfigured strategy → base defaults
    assert reg.limits_for("unknown").max_pct_per_trade == RiskLimits().max_pct_per_trade
    # override keeps other fields at defaults
    assert reg.limits_for("tight").max_pct_per_underlying == RiskLimits().max_pct_per_underlying


def test_registry_enabled_filter():
    reg = StrategyRegistry(configs={
        "on": _cfg("on"), "off": _cfg("off", enabled=False),
    })
    assert reg.enabled_strategies(["on", "off", "unconfigured"]) == ["on", "unconfigured"]


def test_registry_sync_lifecycle_preserves_operator_changes(tmp_path):
    configure(tmp_path / "fw.db")
    reg = StrategyRegistry(configs={
        "s1": _cfg("s1", lifecycle="paper"),
        "s2": _cfg("s2", lifecycle="probation"),
    })
    assert reg.sync_lifecycle() == 2
    with db_engine.session_scope() as s:
        s.execute(text("UPDATE strategy_state SET lifecycle = 'live' WHERE name = 's1'"))
    assert reg.sync_lifecycle() == 0  # nothing re-seeded
    with db_engine.get_session() as s:
        rows = {r["name"]: r["lifecycle"] for r in s.execute(text("SELECT * FROM strategy_state")).mappings()}
    assert rows == {"s1": "live", "s2": "probation"}


# ── Bridge risk-gate wiring ----------------------------------------------------

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


def _risk_client():
    client = MagicMock()
    client.position_symbols.return_value = set()
    client.get_account.return_value = {"equity": "100000", "buying_power": "50000"}
    client.get_positions.return_value = []
    client.get_option_snapshot.return_value = {}
    client.submit_multi_leg_order.return_value = {"id": "o1", "status": "accepted", "legs": []}
    return client


def _risk_bridge(tmp_path, client, limits=None, resolver_limits=None):
    configure(tmp_path / "fw.db")
    resolver = (lambda name: resolver_limits) if resolver_limits else None
    return StrategyBridge(
        client=client, config=BridgeConfig(),
        risk_manager=RiskManager(limits=limits), limits_resolver=resolver,
    )


def test_bridge_exposure_counts_stock_and_occ_options(tmp_path):
    client = _risk_client()
    client.get_positions.return_value = [
        {"symbol": "AAPL", "market_value": "10000"},
        {"symbol": "AAPL260731C00190000", "market_value": "-5000"},
        {"symbol": "MSFT", "market_value": "99999"},
    ]
    bridge = _risk_bridge(tmp_path, client)
    assert bridge._underlying_exposure("AAPL") == 15000.0
    assert bridge._underlying_exposure("MSFT") == 99999.0
    assert bridge._underlying_exposure("NVDA") == 0.0


def test_bridge_unpriced_trade_vetoed(tmp_path):
    client = _risk_client()  # snapshots empty → no midpoint
    bridge = _risk_bridge(tmp_path, client)
    result = bridge.execute_trade(_calendar_trade(entry_price=0.0))
    assert result is None
    assert bridge.skip_reasons["risk_unpriced"] == 1


def test_bridge_est_cost_falls_back_to_midpoint(tmp_path):
    client = _risk_client()
    # midpoint quote: (1.2 + 1.3) / 2 = 1.25 per leg; calendar nets 0 here,
    # so use a DIRECTIONAL_CALL for a positive net debit
    client.get_option_snapshot.return_value = {
        "latestQuote": {"bp": 1.2, "ap": 1.3}}
    trade = Trade(
        ticker="AAPL", earnings_date=date(2026, 7, 29), scan_date=date(2026, 7, 28),
        strategy="calendar_call_ml", side="DIRECTIONAL_CALL", entry_price=0.0,
        features={"strike": 190.0, "expiry": date(2026, 8, 21)},
        model_score=0.7, ml_decision="TAKE",
    )
    # tight per-trade cap: est_cost = 1.25 * 100 = 125 > 0.1% of 50k BP = 50
    tight = RiskLimits(max_pct_per_trade=0.001)
    bridge = _risk_bridge(tmp_path, client, limits=tight)
    result = bridge.execute_trade(trade)
    assert result is None
    assert bridge.skip_reasons["risk_veto"] == 1


def test_bridge_limits_resolver_overrides(tmp_path):
    client = _risk_client()
    trade = _calendar_trade(entry_price=1.85)  # est_cost = 185
    # global limits would pass (185 < 10% of 50k); resolver tightens to 0.1%
    bridge = _risk_bridge(tmp_path, client,
                          resolver_limits=RiskLimits(max_pct_per_trade=0.001))
    assert bridge.execute_trade(trade) is None
    assert bridge.skip_reasons["risk_veto"] == 1


def test_bridge_halted_killswitch_vetoes(tmp_path):
    configure(tmp_path / "fw.db")
    KillSwitch().trip("test halt", by="test")
    client = _risk_client()
    bridge = StrategyBridge(client=client, config=BridgeConfig(),
                            risk_manager=RiskManager())
    assert bridge.execute_trade(_calendar_trade()) is None
    assert bridge.skip_reasons["risk_veto"] == 1


# ── FF ladder wiring -------------------------------------------------------------

TODAY = date(2026, 7, 27)
EARNINGS = TODAY + timedelta(days=1)
# Frozen clock for LadderRunner.arm() — see test_fwd_factor_ladder.py
FROZEN_NOW = datetime(2026, 7, 27, 14, 0, tzinfo=ET)


@pytest.fixture
def fw_conn(tmp_path):
    """Framework-capable conn (earnings + framework schema) with hist rows."""
    configure(tmp_path / "fw.db")
    conn = sqlite3.connect(str(tmp_path / "fw.db"), timeout=30)
    conn.row_factory = sqlite3.Row
    for i, mv in enumerate((3.5, 4.0, 4.5, 3.8, 4.2)):
        conn.execute(
            "INSERT INTO snapshots (ticker, earnings_date, scan_date, timing, "
            "actual_move_pct, outcome_fetched_at) "
            "VALUES ('TEST', ?, '2026-01-14', 'Post Market', ?, '2026-01-16')",
            (f"2026-0{i + 1}-15", mv))
    conn.commit()
    yield conn
    conn.close()


def _ladder_runner(conn):
    al = FakeAlpaca()
    runner = LadderRunner(al, spec=LadderSpec(), now_fn=lambda: FROZEN_NOW)
    cand = build_candidate(al, "TEST", EARNINGS, today=TODAY)
    return runner, al, cand


def test_ladder_arm_vetoed_when_halted(fw_conn):
    KillSwitch().trip("test halt", by="test")
    runner, al, cand = _ladder_runner(fw_conn)
    assert runner.arm(cand) is None
    assert any("kill switch" in e for e in runner.drain_events())


def test_ladder_arm_risk_veto_with_tiny_limits(fw_conn):
    tiny = StrategyRegistry(configs={}, base_limits=RiskLimits(max_pct_per_trade=1e-9))
    with patch("framework.core.registry.get_registry", lambda: tiny):
        runner, al, cand = _ladder_runner(fw_conn)
        assert runner.arm(cand) is None
        assert any("risk veto" in e for e in runner.drain_events())


def test_ladder_arm_ok_under_default_limits(fw_conn):
    runner, al, cand = _ladder_runner(fw_conn)
    lid = runner.arm(cand)
    assert lid is not None


def test_ladder_fill_records_positions_and_risk_entry(fw_conn):
    runner, al, cand = _ladder_runner(fw_conn)
    runner.arm(cand)
    al._fill_next = True
    runner.step(datetime(2026, 7, 27, 14, 0, tzinfo=ET))
    positions = open_positions(strategy="ff_ladder")
    assert len(positions) == 2
    sides = sorted(p["symbol"] for p in positions)
    assert sides == sorted([cand.near_symbol, cand.far_symbol])
    with db_engine.get_session() as s:
        entry = s.execute(
            text("SELECT * FROM risk_events WHERE event_type = 'entry' AND strategy = 'ff_ladder'")
        ).mappings().first()
    assert entry is not None and "cost=" in entry["detail"]


# ── Proposal config gating -------------------------------------------------------

@pytest.fixture
def store(tmp_path):
    return PendingTradeStore(str(tmp_path / "test.db"))


def _trade(ticker="AAPL", score=0.61, decision="TAKE"):
    return Trade(
        ticker=ticker, earnings_date=date(2026, 7, 29), scan_date=date(2026, 7, 28),
        strategy="calendar_call_ml", side="CALENDAR", entry_price=1.85,
        features={
            "near_strike": 190.0, "far_strike": 190.0,
            "near_expiry": date(2026, 7, 31), "far_expiry": date(2026, 8, 28),
        },
        model_score=score, ml_decision=decision,
    )


def _fake_strategy(trades):
    strat = MagicMock()
    strat.run.return_value = StrategyResult(name="calendar_call_ml", trades=trades)
    return strat


def test_build_proposals_respects_disabled_config(store):
    reg = StrategyRegistry(configs={"calendar_call_ml": _cfg("calendar_call_ml", enabled=False)})
    source = MagicMock(return_value=[_trade()])
    with patch("framework.core.registry.get_registry", lambda: reg):
        rows = build_proposals(store, strategies=["calendar_call_ml"],
                               trade_source=source)
    assert rows == []
    source.assert_not_called()  # disabled before the signal source is consulted


def test_proposal_card_tagged_when_halted(store, tmp_path):
    configure(store._db_path)
    KillSwitch().trip("unit test", by="test")
    reg = StrategyRegistry(configs={})
    client = MagicMock()
    client.position_symbols.return_value = set()
    # preflight_combo requires a live Alpaca book per leg (near sell, far buy).
    def _bulk(*symbols):
        out = {}
        for i, s in enumerate(symbols):
            mid = 5.04 if i == 0 else 5.54
            out[s] = {"latestQuote": {"bp": round(mid - 0.04, 2), "ap": round(mid + 0.04, 2)}}
        return out
    client.get_option_snapshots_bulk.side_effect = _bulk
    bridge = StrategyBridge(client=client, config=BridgeConfig())
    with patch("framework.core.registry.get_registry", lambda: reg):
        rows = build_proposals(store, strategies=["calendar_call_ml"],
                               bridge=bridge,
                               trade_source=lambda name: [_trade()])
    assert len(rows) == 1
    assert "KILL SWITCH HALTED" in rows[0]["card_text"]


def test_execute_proposal_records_managed_positions(store):
    pid = store.add(_trade(), "card")
    client = MagicMock()
    client.position_symbols.return_value = set()
    client.get_option_snapshot.return_value = {}
    client.submit_multi_leg_order.return_value = {
        "id": "ord-1", "status": "filled", "filled_qty": 1, "filled_avg_price": 1.85,
        "legs": [
            {"symbol": "AAPL260731C00190000", "side": "sell", "ratio_qty": 1,
             "option_type": "call", "strike": 190.0, "expiry": date(2026, 7, 31)},
            {"symbol": "AAPL260828C00190000", "side": "buy", "ratio_qty": 1,
             "option_type": "call", "strike": 190.0, "expiry": date(2026, 8, 28)},
        ],
    }
    bridge = StrategyBridge(client=client, config=BridgeConfig())
    result = execute_proposal(store, pid, bridge=bridge, decided_by=7)
    assert result["ok"] is True
    positions = open_positions(strategy="calendar_call_ml")
    assert len(positions) == 2
    assert positions[0]["group_id"] == "ord-1"
    # near-leg expiry (2026-07-31), not the far leg (2026-08-28) — the
    # structural exit deadline computed at entry (ScheduledExit).
    assert all(p["exit_by"] == "2026-07-31" for p in positions)


# ── Credit risk basis (Part B) -------------------------------------------------

def test_structure_cost_debit_unchanged(tmp_path):
    client = _risk_client()
    bridge = _risk_bridge(tmp_path, client)
    legs = bridge._build_legs(_calendar_trade(entry_price=1.85))
    cost = bridge._structure_cost(_calendar_trade(entry_price=1.85), legs, 1)
    assert cost == pytest.approx(185.0)


def _condor_trade(credit=1.50):
    return Trade(
        ticker="AAPL", earnings_date=date(2026, 7, 29), scan_date=date(2026, 7, 28),
        strategy="short_straddle", side="IRON_CONDOR", entry_price=credit,
        features={
            "short_put": 180.0, "long_put": 170.0,
            "short_call": 200.0, "long_call": 210.0,
            "expiry": date(2026, 8, 21),
        },
        model_score=0.6, ml_decision="TAKE",
    )


def _straddle_trade(credit=4.0):
    return Trade(
        ticker="AAPL", earnings_date=date(2026, 7, 29), scan_date=date(2026, 7, 28),
        strategy="short_straddle", side="SHORT_STRADDLE", entry_price=credit,
        features={"atm_strike": 190.0, "expiry": date(2026, 8, 21)},
        model_score=0.6, ml_decision="TAKE",
    )


def test_structure_cost_defined_risk_condor(tmp_path):
    client = _risk_client()
    bridge = _risk_bridge(tmp_path, client)
    trade = _condor_trade(credit=1.50)
    legs = bridge._build_legs(trade)
    # wings 10 wide, credit 1.50 → max loss (10 − 1.5) × 100 = 850
    cost = bridge._structure_cost(trade, legs, 1)
    assert cost == pytest.approx(850.0)


def test_structure_cost_undefined_risk_straddle(tmp_path):
    client = _risk_client()
    bridge = _risk_bridge(tmp_path, client)
    trade = _straddle_trade(credit=4.0)
    legs = bridge._build_legs(trade)
    # no wings → notional proxy: 190 × 100 × 0.20 = 3800 (> premium 400)
    cost = bridge._structure_cost(trade, legs, 1)
    assert cost == pytest.approx(3800.0)


def test_straddle_vetoed_by_per_trade_cap_where_premium_would_pass(tmp_path):
    client = _risk_client()
    # cap 5% of 50k BP = 2500: premium basis (400) would pass, notional (3800) vetoes
    bridge = _risk_bridge(tmp_path, client, limits=RiskLimits(max_pct_per_trade=0.05))
    result = bridge.execute_trade(_straddle_trade())
    assert result is None
    assert bridge.skip_reasons["risk_veto"] == 1
