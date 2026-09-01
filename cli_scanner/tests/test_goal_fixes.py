"""Gating tests for the Aug-15 16-issue fix set.

Each test drives a shipped function (bridge submit, quote sanity, equity
snapshot, reconcile, remaining-leg plan, AMC bars). No mocked unit-under-test.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from unittest.mock import MagicMock

import pytest

from earnings_edge.alpaca_bridge import (
    MAX_DEBIT_VS_MID,
    StrikeChangedError,
    StrategyBridge,
    debit_within_mid_cap,
    resolved_keeps_strike,
)
from earnings_edge.bot_scanner import quote_is_sane
from earnings_edge.fwd_factor import occ_symbol
from earnings_edge.services.outcome_service import OutcomeService
from framework.execution.reconcile import Reconciler
from framework.jobs import run_job
from framework.risk.equity import snapshot_equity
from sqlalchemy import text
from earnings_edge.db import engine as db_engine


def test_qty_11_keeps_base_ratio_and_order_qty():
    client = MagicMock()
    client.submit_multi_leg_order.return_value = {
        "id": "oid", "status": "accepted", "legs": [],
    }
    bridge = StrategyBridge(client=client)
    legs = [
        {"symbol": "AAPL260731C00190000", "side": "sell", "ratio_qty": 1,
         "strike": 190.0, "expiry": date(2026, 7, 31), "option_type": "call"},
        {"symbol": "AAPL260828C00190000", "side": "buy", "ratio_qty": 1,
         "strike": 190.0, "expiry": date(2026, 8, 28), "option_type": "call"},
    ]
    bridge._submit_legs(legs, None, "cid", qty=11)
    call = client.submit_multi_leg_order.call_args
    kwargs = call.kwargs
    assert kwargs["qty"] == 11
    assert [l["ratio_qty"] for l in kwargs["legs"]] == [1, 1]


def test_resolve_131_to_130_raises_and_does_not_submit():
    client = MagicMock()
    from earnings_edge.alpaca_trading import AlpacaError
    client.submit_multi_leg_order.side_effect = [
        AlpacaError(422, "unknown contract"),
    ]
    # Catalog returns a 130 strike for a 131 request.
    client.get_option_contracts.return_value = {
        "option_contracts": [{
            "symbol": occ_symbol("TPR", date(2026, 9, 11), 130.0, "call"),
            "type": "call",
            "strike_price": 130.0,
        }],
    }
    bridge = StrategyBridge(client=client)
    from earnings_edge.trading_types import Trade
    trade = Trade(
        ticker="TPR", earnings_date=date(2026, 8, 13),
        scan_date=date(2026, 8, 13), strategy="debit_size_exploit",
        side="CALENDAR", entry_price=3.17,
        features={"near_strike": 131, "far_strike": 131, "atm_strike": 131,
                  "near_expiry": "2026-08-14", "far_expiry": "2026-09-11"},
        ml_decision="DEBIT_GATE",
    )
    legs = [
        {"symbol": occ_symbol("TPR", date(2026, 8, 14), 131.0, "call"),
         "side": "sell", "ratio_qty": 1, "strike": 131.0,
         "expiry": date(2026, 8, 14), "option_type": "call"},
        {"symbol": occ_symbol("TPR", date(2026, 9, 11), 131.0, "call"),
         "side": "buy", "ratio_qty": 1, "strike": 131.0,
         "expiry": date(2026, 9, 11), "option_type": "call"},
    ]
    with pytest.raises(StrikeChangedError):
        bridge._submit_with_resolution(trade, legs, None, "cid", qty=1)
    # First OCC submit rejected; the retry must not go out with strike 130.
    assert client.submit_multi_leg_order.call_count == 1


def test_resolved_keeps_strike_and_mid_cap_helpers():
    sym_131 = occ_symbol("TPR", date(2026, 9, 11), 131.0, "call")
    sym_130 = occ_symbol("TPR", date(2026, 9, 11), 130.0, "call")
    assert resolved_keeps_strike(131.0, sym_131)
    assert not resolved_keeps_strike(131.0, sym_130)
    assert debit_within_mid_cap(6.39 * MAX_DEBIT_VS_MID, 6.39)
    assert not debit_within_mid_cap(6.39 * 1.6, 6.39)


def test_quote_sanity_rejects_fat_debit_and_inverted():
    assert quote_is_sane(1.0, 1.2, 3.0, 3.4, 2.2, 100.0)
    assert not quote_is_sane(1.2, 1.0, 3.0, 3.4, 2.2, 100.0)  # inverted near
    assert not quote_is_sane(1.0, 1.2, 3.0, 3.4, 20.0, 100.0)  # 20% of spot


def test_equity_snapshot_and_reconcile_insert(tmp_path):
    db_engine.configure(tmp_path / "jobs.db")
    client = MagicMock()
    client.get_account.return_value = {
        "equity": 100_000, "buying_power": 80_000, "portfolio_value": 100_000,
    }
    client.get_positions.return_value = [{
        "symbol": "AAPL260828C00200000", "qty": "1", "side": "long",
        "avg_entry_price": "2.5", "current_price": "2.0",
        "market_value": "200", "unrealized_pl": "-50",
    }]

    def work():
        snapshot_equity(client)
        return Reconciler(client).run().summary()

    stats = run_job("equity_and_reconcile", work)
    with db_engine.get_session() as s:
        assert s.execute(text("SELECT COUNT(*) FROM equity_snapshots")).scalar() >= 1
        assert s.execute(text("SELECT COUNT(*) FROM alpaca_positions")).scalar() >= 1
        assert s.execute(text("SELECT COUNT(*) FROM job_runs")).scalar() >= 1
        row = s.execute(text("SELECT success FROM job_runs ORDER BY id DESC LIMIT 1")).mappings().first()
    assert row["success"] == 1
    assert "broker=" in stats


def test_amc_outcome_from_bars_uses_next_session():
    def _bar(day, c):
        ts = int(datetime(day.year, day.month, day.day).timestamp() * 1000)
        return {"t": ts, "c": c, "h": c + 1, "l": c - 1}

    ed = date(2026, 7, 29)
    bars = [
        _bar(ed - timedelta(days=1), 100.0),
        _bar(ed, 99.3),
        _bar(ed + timedelta(days=1), 84.5),
    ]
    amc = OutcomeService.outcome_from_bars(bars, ed, timing="Post Market")
    assert amc["pre_earnings_close"] == pytest.approx(99.3)
    assert amc["post_earnings_close"] == pytest.approx(84.5)
