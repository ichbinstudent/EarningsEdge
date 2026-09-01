"""Gating tests for remaining professionalization phases (desk, loop, signals)."""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from earnings_edge.alpaca_bridge import (
    MAX_DEBIT_VS_MID, StrategyBridge, debit_within_mid_cap, size_veto_reason,
)
from earnings_edge.inbox import assemble_inbox, inbox_keyboard, render_inbox
from earnings_edge.live_signals import calendar_funnel_reasons, calendar_row_reason
from framework.alerts import AlertDeduper
from framework.backup import backup_db
from framework.execution.reconcile import Reconciler, classify_assignments
from framework.health import health_ready
from framework.jobs import run_job
from framework.revision import code_sha, started_at_iso
from framework.scan_retry import next_retry, record_retry, should_chain_proposals, should_retry_scan
from sqlalchemy import text as sa_text
from earnings_edge.db import configure, engine as db_engine
from earnings_edge.bot_views import monitor_view, status_view


NOW = datetime(2026, 8, 15, 18, 0, tzinfo=timezone.utc)


def test_inbox_lists_four_live_kinds_and_expires_stale():
    stale = (NOW - timedelta(hours=10)).isoformat()
    fresh = (NOW - timedelta(hours=1)).isoformat()
    inbox = assemble_inbox(
        entries=[
            {"id": 1, "ticker": "FRESH", "side": "CALENDAR", "strategy": "calendar_call_ml",
             "created_at": fresh},
            {"id": 2, "ticker": "STALE", "side": "CALENDAR", "strategy": "calendar_call_ml",
             "created_at": stale},
        ],
        exits=[{"id": 9, "ticker": "META", "rule": "time", "strategy": "ff_ladder",
                "created_at": fresh, "reason": "event day"}],
        orphans=[{"symbol": "ATLO260918C00030000", "ticker": "ATLO", "ts": fresh,
                  "detail": "at broker, not local"}],
        jobs=[{"id": 3, "job_name": "scan", "error": "dns", "finished_at": fresh}],
        now=NOW, ttl_hours=8,
    )
    kinds = {i.kind for i in inbox.live}
    assert {"entry", "exit", "orphan", "job"} <= kinds
    assert any(i.ticker == "STALE" and i.expired for i in inbox.items)
    assert any(i.ticker == "FRESH" and not i.expired and i.kind == "entry" for i in inbox.items)
    text = render_inbox(inbox)
    assert "FRESH" in text and "STALE" in text and "EXPIRED" in text
    assert "ATLO" in text and "scan" in text and "META" in text
    grouped = inbox.grouped()
    assert "expired" in grouped and grouped["expired"][0].ticker == "STALE"
    cbs = [b.callback_data for row in inbox_keyboard(inbox) for b in row]
    assert "in_ex_1" in cbs and "in_sk_1" in cbs
    assert "in_cl_9" in cbs and "in_ad_ATLO260918C00030000" in cbs
    assert cbs[-1] == "desk_pd"
    assert "in_ex_2" not in cbs  # stale entry has no actions


def test_health_ready_vs_no_lock_and_stale_scan():
    ok = health_ready(
        lock_held=True,
        last_equity_ts=NOW.isoformat(),
        last_scan_ts=NOW.isoformat(),
        clock_ok=True, now=NOW, market_open=True, weekday=True,
    )
    assert ok["ready"] is True
    no_lock = health_ready(
        lock_held=False,
        last_equity_ts=NOW.isoformat(),
        last_scan_ts=NOW.isoformat(),
        clock_ok=True, now=NOW, market_open=True, weekday=True,
    )
    assert no_lock["ready"] is False and "no lock" in no_lock["reasons"]
    stale = health_ready(
        lock_held=True,
        last_equity_ts=NOW.isoformat(),
        last_scan_ts=(NOW - timedelta(hours=30)).isoformat(),
        clock_ok=True, now=NOW, market_open=True, weekday=True,
    )
    assert stale["ready"] is False and "scan stale" in stale["reasons"]


def test_status_view_has_required_fields(tmp_path):
    configure(tmp_path / "s.db")
    text = status_view( market_open=True, pending_proposals=1, pending_exits=0,
        last_scan_ts=NOW.isoformat(), last_equity_ts=NOW.isoformat(),
        reconcile_summary="broker=2 matched=1 orphans=1",
        broker_ok=True, broker_count=3, orphan_count=1,
        sha="abc1234", started_at=NOW.isoformat(),
    )
    assert "<b>Last scan:</b>" in text
    assert "<b>Last equity snapshot:</b>" in text or "<b>Equity:</b>" in text
    assert "<b>Last reconcile:</b>" in text and "orphans=1" in text
    assert "<b>Broker:</b> reachable" in text
    assert "vs broker 3" in text
    assert "<b>Orphans: 1</b>" in text
    assert "<b>Rev:</b> <code>abc1234</code>" in text
    assert "<b>Started:</b>" in text
    mon = monitor_view( tick=0, last_scan_ts=NOW.isoformat(),
                       last_equity_ts=NOW.isoformat(),
                       reconcile_summary="broker=2 matched=1 orphans=1",
                       broker_ok=True, broker_count=3, orphan_count=1,
                       sha="abc1234", started_at=NOW.isoformat())
    assert "orphans 1" in mon and "Last scan:" in mon and "reachable" in mon
    assert "Last reconcile:" in mon and "orphans=1" in mon
    assert "Last equity snapshot:" in mon


def test_job_runs_records_skip(tmp_path):
    configure(tmp_path / "j.db")
    out = run_job("equity_snapshot", lambda: {"skipped": "market closed"})
    assert out["skipped"] == "market closed"
    with db_engine.get_session() as s:
        row = s.execute(sa_text("SELECT success, stats_json FROM job_runs")).mappings().first()
    assert row["success"] == 1
    assert "market closed" in row["stats_json"]


def test_scan_zero_candidates_no_propose_and_one_retry():
    fail = {"success": False, "error": "No candidates", "stats": {"candidate_count": 0}}
    assert should_retry_scan(fail) is True
    assert should_chain_proposals(fail) is False
    rec = record_retry(12)
    assert rec["minutes"] == 12
    assert 10 <= rec["minutes"] <= 15
    nxt = next_retry(NOW, 12)
    assert nxt - NOW == timedelta(minutes=12)
    with pytest.raises(ValueError):
        next_retry(NOW, 5)


def test_reconcile_snapshot_orphan_and_assignment(tmp_path):
    configure(tmp_path / "r.db")
    from framework.execution.managed import record_open_positions
    record_open_positions([{"symbol": "NU260814C00014000", "side": "sell", "ratio_qty": 25,
          "option_type": "call", "strike": 14.0, "expiry": date(2026, 8, 14)}],
        "debit_size_exploit", group_id="nu-short",
        metadata={"side": "CALENDAR", "leg_side": "sell", "option_type": "call",
                  "earnings_date": "2026-08-13"},
    )
    client = MagicMock()
    client.get_positions.return_value = [
        {"symbol": "NU260814C00014000", "qty": "-25", "side": "short",
         "avg_entry_price": "0.4", "current_price": "0.1", "market_value": "-250",
         "unrealized_pl": "750"},
        {"symbol": "ATLO260918C00030000", "qty": "-1", "side": "short",
         "avg_entry_price": "1", "current_price": "4", "market_value": "-400",
         "unrealized_pl": "-300"},
        {"symbol": "NU", "qty": "-2500", "side": "short",
         "avg_entry_price": "14", "current_price": "15", "market_value": "-37500",
         "unrealized_pl": "-2500"},
    ]
    from framework.alerts import DEDUPER
    DEDUPER.reset()
    report = Reconciler(client).run()
    with db_engine.get_session() as s:
        n_snap = s.execute(sa_text("SELECT COUNT(*) AS c FROM alpaca_positions")).mappings().first()["c"]
        local = s.execute(sa_text("SELECT * FROM managed_positions WHERE status='open'")).mappings().all()
    assert n_snap >= 3
    assert "ATLO260918C00030000" in report.orphans
    assert "NU" in report.assignments
    outbox = DEDUPER.drain()
    assert any("orphan" in m.lower() and "ATLO" in m for m in outbox)
    assigned = classify_assignments(client.get_positions.return_value, local)
    assert "NU" in assigned


def test_backup_invoke(tmp_path):
    src = tmp_path / "src.db"
    dest = tmp_path / "backups"
    db_engine.configure(src)
    with db_engine.session_scope() as s:
        s.execute(sa_text("CREATE TABLE IF NOT EXISTS t (x int)"))
        s.execute(sa_text("INSERT INTO t VALUES (1)"))
    out = backup_db(src, dest, now=NOW)
    assert out.exists() and out.stat().st_size > 0
    assert out.parent == dest


def test_calendar_reasons_skip_vs_no_quote():
    import pandas as pd
    skip = {
        "ticker": "SKIPCO", "strike": 50, "near_expiry": "2026-08-20",
        "far_expiry": "2026-09-17", "net_debit_ask": 1.1, "net_debit": 1.1,
        "model_decision": "SKIP",
    }
    noq = {
        "ticker": "NOQ", "strike": None, "near_expiry": None, "far_expiry": None,
        "net_debit_ask": None, "net_debit": None, "model_decision": "TAKE",
    }
    assert calendar_row_reason(skip) == "model_skip"
    assert calendar_row_reason(noq) == "no_quote"
    counts = calendar_funnel_reasons(pd.DataFrame([skip, noq]))
    assert counts["model_skip"] == 1 and counts["no_quote"] == 1


def test_size_veto_reason_readable():
    text = size_veto_reason("short_straddle", 1884.0)
    assert "size_veto" in text and "short_straddle" in text and "1884" in text


def test_mid_cap_refuse_no_submit():
    client = MagicMock()
    client.position_symbols.return_value = set()
    client.get_account.return_value = {"equity": "100000", "buying_power": "50000"}
    # mid 1.00, proposed debit 2.00 → through cap
    def _snap(symbol):
        # near sold cheap, far expensive → positive combo mid ~2.00
        if "0731" in symbol:
            return {"latestQuote": {"bp": 0.90, "ap": 1.10}}
        return {"latestQuote": {"bp": 2.90, "ap": 3.10}}
    client.get_option_snapshot.side_effect = _snap
    from earnings_edge.trading_types import Trade
    bridge = StrategyBridge(client=client)
    trade = Trade(
        ticker="AAPL", earnings_date=date(2026, 7, 29), scan_date=date(2026, 7, 28),
        strategy="calendar_call_ml", side="CALENDAR", entry_price=3.50,
        features={
            "near_strike": 190, "far_strike": 190,
            "near_expiry": date(2026, 7, 31), "far_expiry": date(2026, 8, 28),
        },
        ml_decision="TAKE",
    )
    assert debit_within_mid_cap(2.00, 1.00) is False
    result = bridge.execute_trade(trade)
    assert result is None
    assert bridge.skip_reasons["mid_cap"] == 1
    assert "refused-vs-mid" in (bridge.last_skip_detail or "")
    client.submit_multi_leg_order.assert_not_called()


def test_alert_dedupe_once_per_window():
    d = AlertDeduper(window_s=600)
    assert d.emit("scan_fail", "boom", now=NOW) == "boom"
    assert d.emit("scan_fail", "boom again", now=NOW + timedelta(minutes=5)) is None
    assert d.emit("scan_fail", "later", now=NOW + timedelta(minutes=16)) == "later"
    assert d.drain() == ["boom", "later"]
    assert d.drain() == []


def test_monitor_text_sync_passes_desk_facts(tmp_path, monkeypatch):
    """Live /monitor gather must pass reconcile/broker/orphan like /status."""
    from bot import TradingBot

    configure(tmp_path / "mon.db")
    with db_engine.session_scope() as s:
        s.execute(
            sa_text(
                "INSERT INTO job_runs (job_name, started_at, finished_at, success, stats_json) "
                "VALUES ('reconcile', :st, :fin, 1, :stats)"
            ),
            {
                "st": NOW.isoformat(),
                "fin": NOW.isoformat(),
                "stats": json.dumps({"summary": "broker=2 matched=1 orphans=1"}),
            },
        )

    class _Client:
        def get_clock(self):
            return {"is_open": True}

        def get_positions(self):
            return [{"symbol": "ATLO260918C00030000", "qty": "-1", "side": "short"}]

    class _Bot:
        def __init__(self):
            self.approval_store = MagicMock()
            self.approval_store.list_pending.return_value = []

        def _next_events_sync(self):
            return []

        _desk_facts_sync = TradingBot._desk_facts_sync
        _monitor_text_sync = TradingBot._monitor_text_sync

    monkeypatch.setattr("bot.create_client", lambda: _Client())
    text = TradingBot._monitor_text_sync(_Bot(), 0)
    assert "Last reconcile:" in text and "orphans=1" in text
    assert "reachable" in text
    assert "orphans 1" in text
    assert "<b>Last scan:</b> never" in text


def test_collect_desk_facts_feeds_monitor(tmp_path):
    from earnings_edge.bot_views import collect_desk_facts, desk_view_kwargs, monitor_view
    from framework.alerts import DEDUPER, emit_clock_failure, is_alpaca_401
    from earnings_edge.alpaca_trading import AlpacaAuthError

    configure(tmp_path / "desk.db")
    with db_engine.session_scope() as s:
        s.execute(
            sa_text(
                "INSERT INTO equity_snapshots (ts, equity, buying_power, portfolio_value, source) "
                "VALUES (:ts, 98000, 40000, 98000, 'alpaca')"
            ),
            {"ts": NOW.isoformat()},
        )
        s.execute(
            sa_text(
                "INSERT INTO job_runs (job_name, started_at, finished_at, success, stats_json) "
                "VALUES ('reconcile', :st, :fin, 1, :stats)"
            ),
            {
                "st": NOW.isoformat(),
                "fin": NOW.isoformat(),
                "stats": json.dumps({"summary": "broker=2 matched=1 orphans=1"}),
            },
        )

    facts = collect_desk_facts(
        get_clock=lambda: {"is_open": True},
        get_positions=lambda: [
            {"symbol": "ATLO260918C00030000", "qty": "-1", "side": "short"},
        ],
    )
    assert facts["market_open"] is True
    assert facts["broker_ok"] is True
    assert facts["broker_count"] == 1
    assert facts["orphan_count"] == 1
    assert facts["reconcile_summary"] == "broker=2 matched=1 orphans=1"
    assert facts["last_equity_ts"]
    mon = monitor_view( tick=0, **desk_view_kwargs(facts))
    assert "Last reconcile:" in mon and "orphans=1" in mon
    assert "reachable" in mon and "orphans 1" in mon
    assert "Last equity snapshot:" in mon

    DEDUPER.reset()
    assert is_alpaca_401(AlpacaAuthError(401, "Invalid API keys"))
    dns = emit_clock_failure(RuntimeError("Name or service not known"))
    assert dns is not None and "Clock check failed" in dns
    assert DEDUPER.drain() == [dns]
    DEDUPER.reset()
    msg = emit_clock_failure(AlpacaAuthError(401, "Invalid API keys"))
    assert msg and "401" in msg
    assert DEDUPER.drain() == [msg]


def test_scan_fail_emit_lands_in_outbox():
    from framework.alerts import DEDUPER
    DEDUPER.reset()
    msg = DEDUPER.emit("scan_fail", "scan Earnings Calendar failed: No candidates")
    assert msg is not None
    assert DEDUPER.drain() == [msg]


def test_reconcile_missing_emits_missing_alert(tmp_path):
    from framework.alerts import DEDUPER
    from framework.execution.managed import record_open_positions
    configure(tmp_path / "miss.db")
    record_open_positions([{"symbol": "GONE260828C00100000", "side": "sell", "ratio_qty": 1,
          "option_type": "call", "strike": 100.0, "expiry": date(2026, 8, 28)}],
        "ff_ladder", group_id="gone",
        metadata={"side": "CALENDAR", "leg_side": "sell", "option_type": "call"},
    )
    client = MagicMock()
    client.get_positions.return_value = []
    DEDUPER.reset()
    report = Reconciler(client).run()
    assert "GONE260828C00100000" in report.closed_externally
    assert any("missing" in m.lower() and "GONE" in m for m in DEDUPER.drain())


def test_flush_alerts_pushes_scan_fail_to_approval_chat():
    import asyncio
    from bot import TradingBot
    from framework.alerts import DEDUPER
    DEDUPER.reset()
    DEDUPER.emit("scan_fail", "scan Earnings Calendar failed: No candidates")
    pushed = []

    class _Stub:
        async def _push_risk_alert(self, text):
            pushed.append(text)
        _flush_alerts = TradingBot._flush_alerts

    asyncio.run(_Stub()._flush_alerts())
    assert pushed == ["scan Earnings Calendar failed: No candidates"]
    assert DEDUPER.drain() == []


def test_kill_switch_trip_emits(tmp_path):
    from framework.alerts import DEDUPER
    from framework.risk.killswitch import KillSwitch
    configure(tmp_path / "ks.db")
    DEDUPER.reset()
    KillSwitch().trip("manual halt via bot", "operator")
    outbox = DEDUPER.drain()
    assert any("KILL SWITCH TRIPPED" in m and "operator" in m for m in outbox)


def test_sha_and_started_at_exist():
    assert isinstance(code_sha(), str) and len(code_sha()) >= 4
    assert "2026" in started_at_iso() or "T" in started_at_iso()


def test_main_keyboard_has_expected_desk_keys():
    from bot import MAIN_KB
    flat = [c for row in MAIN_KB for c in row]
    assert flat == [
        "🖥 Status", "💼 Positions",
        "📡 Signals", "📥 Pending",
        "🎯 Picks", "📐 Designer",
        "⚙️ Jobs", "🛠 Settings",
    ]
