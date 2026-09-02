"""Tests for repository functions added during the ORM migration."""
import json
from datetime import date, timedelta

import pytest
from sqlalchemy import text

from earnings_edge.db import engine as db_engine
from earnings_edge.db.repositories import (
    adopted_positions_insert,
    adopted_positions_symbols,
    alpaca_positions_insert,
    alpaca_positions_list,
    data_catalog_latest,
    data_catalog_query,
    data_catalog_upsert,
    equity_snapshots_day_start,
    equity_snapshots_insert,
    equity_snapshots_latest,
    exit_proposals_get,
    exit_proposals_insert,
    exit_proposals_list_pending,
    exit_proposals_mark,
    insert_scan_run,
    insert_snapshot,
    job_runs_failed,
    job_runs_finish,
    job_runs_list,
    job_runs_start,
    managed_positions_close,
    managed_positions_list,
    managed_positions_open,
    model_registry_get_active,
    model_registry_promote,
    model_registry_register,
    pending_trades_get,
    pending_trades_insert,
    pending_trades_list_pending,
    pending_trades_mark_decided,
    pending_trades_update_card,
    proposal_funnel_insert,
    record_snapshot_outcome_failure,
    risk_events_insert,
    risk_events_list,
    risk_state_get,
    risk_state_set_halted,
    scan_runs_latest_success,
    snapshots_max_scan_date,
    snapshots_optionable_universe,
    snapshots_status_counts,
    strategy_state_clear_enabled,
    strategy_state_enabled_overrides,
    strategy_state_get,
    strategy_state_insert_ignore,
    strategy_state_set_enabled,
    strategy_state_set_execution_mode,
    strategy_state_upsert,
    table_exists,
    trade_events_insert,
    trade_events_list,
)


@pytest.fixture(autouse=True)
def fresh_engine(tmp_path):
    eng = db_engine.configure(tmp_path / "t.db")
    yield eng
    db_engine.configure(tmp_path / "reset.db")


def test_record_snapshot_outcome_failure_bumps_then_marks_unavailable():
    sid = insert_snapshot({
        "ticker": "AAPL",
        "earnings_date": "2026-01-01",
        "scan_date": "2026-01-01",
    })
    record_snapshot_outcome_failure(sid, 2)
    with db_engine.session_scope() as s:
        row = s.execute(
            text(
                "SELECT outcome_fetched_at, outcome_attempt_count "
                "FROM snapshots WHERE id = :id"
            ),
            {"id": sid},
        ).mappings().one()
        assert row["outcome_fetched_at"] is None
        assert row["outcome_attempt_count"] == 1

    record_snapshot_outcome_failure(sid, 2)
    with db_engine.session_scope() as s:
        row = s.execute(
            text(
                "SELECT outcome_fetched_at, outcome_attempt_count "
                "FROM snapshots WHERE id = :id"
            ),
            {"id": sid},
        ).mappings().one()
        assert row["outcome_fetched_at"] == "unavailable"
        assert row["outcome_attempt_count"] == 2


def test_snapshots_optionable_universe_prefers_upcoming():
    today = date.today()
    insert_snapshot({
        "ticker": "AAA",
        "earnings_date": (today + timedelta(days=3)).isoformat(),
        "scan_date": today.isoformat(),
        "has_options": 1,
    })
    insert_snapshot({
        "ticker": "ZZZ",
        "earnings_date": (today - timedelta(days=30)).isoformat(),
        "scan_date": today.isoformat(),
        "has_options": 1,
    })
    tickers = snapshots_optionable_universe(10)
    assert tickers[0] == "AAA"
    assert "ZZZ" in tickers


def _insert_pending(ticker="AAPL", side="CALENDAR", score=0.5, strategy="calendar_call_ml"):
    return pending_trades_insert(
        created_at="2026-08-23T12:00:00+00:00",
        strategy=strategy,
        ticker=ticker,
        side=side,
        trade_json="{}",
        card_text="card",
        model_score=score,
    )


def test_pending_trades_insert_get_and_dedupe():
    pid = _insert_pending()
    assert pid is not None
    row = pending_trades_get(pid)
    assert row["status"] == "pending"
    assert row["ticker"] == "AAPL"
    assert row["model_score"] == pytest.approx(0.5)
    assert pending_trades_insert(
        created_at="2026-08-23T12:01:00+00:00",
        strategy="calendar_call_ml",
        ticker="AAPL",
        side="CALENDAR",
        trade_json="{}",
        card_text="dup",
        model_score=0.9,
    ) is None
    pending_trades_mark_decided(pid, "rejected", decided_by=1)
    assert _insert_pending() is not None


def test_pending_trades_list_pending_score_desc():
    _insert_pending("LOW", score=0.2)
    _insert_pending("HIGH", score=0.9)
    _insert_pending("NONE", score=None)
    rows = pending_trades_list_pending()
    assert [r["ticker"] for r in rows] == ["HIGH", "LOW", "NONE"]


def test_pending_trades_update_card_and_mark_decided():
    pid = _insert_pending()
    pending_trades_update_card(pid, "new card")
    pending_trades_mark_decided(
        pid, "executed",
        order_json='{"order_id": "x"}',
        note="ok",
        decided_by=42,
    )
    row = pending_trades_get(pid)
    assert row["card_text"] == "new card"
    assert row["status"] == "executed"
    assert row["order_json"] == '{"order_id": "x"}'
    assert row["note"] == "ok"
    assert row["decided_by"] == 42
    assert row["decided_at"] is not None
    assert pending_trades_get(999) is None
    assert pending_trades_list_pending() == []


def test_proposal_funnel_insert():
    rid = proposal_funnel_insert(
        created_at="2026-08-23T12:00:00+00:00",
        strategies='["calendar_call_ml"]',
        counts='{"calendar_call_ml": {"proposals_created": 1}}',
        proposals_total=1,
    )
    assert rid > 0
    with db_engine.session_scope() as s:
        row = s.execute(
            text(
                "SELECT strategies, counts, proposals_total "
                "FROM proposal_funnel ORDER BY id DESC LIMIT 1"
            )
        ).mappings().one()
        assert json.loads(row["counts"])["calendar_call_ml"]["proposals_created"] == 1
        assert row["proposals_total"] == 1


def test_exit_proposals_crud_and_dedupe():
    pid = exit_proposals_insert(
        group_id="g1",
        strategy="calendar_call_ml",
        ticker="AAPL",
        rule="time",
        reason="near expiry",
        card_text="exit?",
    )
    assert pid is not None
    assert exit_proposals_insert(
        group_id="g1", strategy="calendar_call_ml", ticker="AAPL", rule="time",
    ) is None
    row = exit_proposals_get(pid)
    assert row["status"] == "pending"
    assert row["group_id"] == "g1"
    pending = exit_proposals_list_pending()
    assert len(pending) == 1
    assert pending[0]["id"] == pid
    exit_proposals_mark(pid, "snoozed", snoozed_until="2026-08-24", decided_by=7)
    row = exit_proposals_get(pid)
    assert row["status"] == "snoozed"
    assert row["snoozed_until"] == "2026-08-24"
    assert row["decided_by"] == 7
    assert exit_proposals_list_pending() == []
    assert exit_proposals_get(999) is None


def test_managed_positions_open_list_close():
    n = managed_positions_open(
        [
            {"symbol": "AAPL260731C00190000", "side": "sell", "ratio_qty": 1,
             "option_type": "call", "strike": 190.0, "expiry": date(2026, 7, 31)},
            {"symbol": "AAPL260828C00190000", "side": "buy", "ratio_qty": 1,
             "option_type": "call", "strike": 190.0, "expiry": date(2026, 8, 28)},
        ],
        "calendar_call_ml",
        "ord-1",
        order_id="ord-1",
        entry_price=1.85,
        metadata={"side": "CALENDAR", "credit": False, "earnings_date": "2026-07-29"},
        exit_by=date(2026, 7, 31),
    )
    assert n == 2
    rows = managed_positions_list()
    assert len(rows) == 2
    assert all(r["status"] == "open" for r in rows)
    assert all(r["exit_by"] == "2026-07-31" for r in rows)
    assert "metadata" in rows[0]
    meta = json.loads(rows[0]["metadata"])
    assert meta["earnings_date"] == "2026-07-29"
    assert meta["leg_side"] in ("sell", "buy")
    assert managed_positions_list(strategy="other") == []
    closed = managed_positions_close("ord-1", exit_price=2.10)
    assert closed == 2
    assert managed_positions_list() == []


def test_snapshots_max_scan_date_and_scan_runs_latest_success():
    assert snapshots_max_scan_date() is None
    assert scan_runs_latest_success() is None
    insert_snapshot({
        "ticker": "AAPL",
        "earnings_date": "2026-08-20",
        "scan_date": "2026-08-19",
    })
    insert_snapshot({
        "ticker": "MSFT",
        "earnings_date": "2026-08-21",
        "scan_date": "2026-08-21",
    })
    assert snapshots_max_scan_date() == "2026-08-21"
    insert_scan_run({
        "scan_timestamp": "2026-08-20T10:00:00",
        "scanner_name": "cal",
        "trigger_type": "cron",
        "success": 0,
    })
    insert_scan_run({
        "scan_timestamp": "2026-08-21T10:00:00",
        "scanner_name": "cal",
        "trigger_type": "cron",
        "success": 1,
    })
    assert scan_runs_latest_success() == "2026-08-21T10:00:00"


def test_risk_state_get_and_set_halted():
    row = risk_state_get()
    assert row["halted"] == 0
    assert row["reason"] is None
    risk_state_set_halted(
        True, reason="daily loss", tripped_at="2026-08-23T12:00:00+00:00", tripped_by="rm",
    )
    row = risk_state_get()
    assert row["halted"] == 1
    assert row["reason"] == "daily loss"
    assert row["tripped_by"] == "rm"
    risk_state_set_halted(False)
    row = risk_state_get()
    assert row["halted"] == 0
    assert row["reason"] is None


def test_risk_events_insert_and_list():
    risk_events_insert("entry", "AAPL cost=100.00", strategy="s1", ts="2026-08-23T10:00:00+00:00")
    risk_events_insert("veto", "too big", strategy="s1")
    rows = risk_events_list(limit=10)
    assert len(rows) == 2
    assert rows[0]["event_type"] == "veto"
    entries = risk_events_list(event_type="entry", strategy="s1", since="2026-08-23")
    assert len(entries) == 1
    assert "cost=100" in entries[0]["detail"]


def test_equity_snapshots_insert_latest_day_start():
    equity_snapshots_insert(
        ts="2026-08-23T13:00:00+00:00",
        equity=100_000, buying_power=80_000, portfolio_value=100_000,
    )
    equity_snapshots_insert(
        ts="2026-08-23T14:00:00+00:00",
        equity=101_000, buying_power=81_000, portfolio_value=101_000,
    )
    latest = equity_snapshots_latest()
    assert latest["equity"] == 101_000
    assert latest["buying_power"] == 81_000
    assert equity_snapshots_day_start(date(2026, 8, 23)) == 100_000
    assert equity_snapshots_day_start(date(2026, 8, 24)) is None


def test_strategy_state_upsert_enabled_and_lifecycle():
    assert strategy_state_get("s1") is None
    strategy_state_upsert("s1", lifecycle="live", updated_by="op")
    assert strategy_state_get("s1")["lifecycle"] == "live"
    strategy_state_set_enabled("s1", False, updated_by="op")
    row = strategy_state_get("s1")
    assert row["lifecycle"] == "live"
    assert row["enabled"] == 0
    assert strategy_state_enabled_overrides() == {"s1": False}
    strategy_state_clear_enabled("s1", updated_by="op")
    assert strategy_state_get("s1")["enabled"] is None
    strategy_state_set_execution_mode("s1", "auto", updated_by="op")
    assert strategy_state_get("s1")["execution_mode"] == "auto"
    n = strategy_state_insert_ignore("s1", "paper")
    assert n == 0
    n = strategy_state_insert_ignore("s2", "probation")
    assert n == 1
    assert strategy_state_get("s2")["lifecycle"] == "probation"


def test_job_runs_start_finish_list():
    rid = job_runs_start("equity_snapshot", started_at="2026-08-23T13:00:00+00:00")
    assert rid > 0
    job_runs_finish(rid, success=1, stats_json='{"n": 1}')
    rows = job_runs_list(name="equity_snapshot")
    assert len(rows) == 1
    assert rows[0]["success"] == 1
    assert '"n": 1' in rows[0]["stats_json"]
    assert rows[0]["finished_at"]
    bad = job_runs_start("bad")
    job_runs_finish(bad, success=0, error="boom")
    fails = job_runs_failed()
    assert fails[0]["job_name"] == "bad"


def test_data_catalog_upsert_query_latest():
    data_catalog_upsert(
        "options_chain", symbol="AAPL", as_of_date="2026-07-20",
        source="alpaca", available_at="2026-07-20T20:00:00+00:00",
    )
    data_catalog_upsert(
        "options_chain", symbol="AAPL", as_of_date="2026-07-21",
        source="alpaca", available_at="2026-07-21T20:00:00+00:00",
    )
    data_catalog_upsert(
        "chain_snapshot", symbol="AAPL", as_of_date="2026-07-20",
        source="lse", available_at="2026-07-20T21:00:00+00:00", pit_safe=False,
    )
    assert data_catalog_query(
        "options_chain", "2026-07-21T13:00:00+00:00", symbol="AAPL",
    ) == ["2026-07-20"]
    assert data_catalog_query(
        "options_chain", "2026-07-22T13:00:00+00:00", symbol="AAPL",
    ) == ["2026-07-20", "2026-07-21"]
    latest = data_catalog_latest("options_chain", "AAPL")
    assert latest["as_of_date"] == "2026-07-21"
    assert data_catalog_query(
        "chain_snapshot", "2026-07-21T00:00:00+00:00", symbol="AAPL", pit_only=True,
    ) == []
    assert data_catalog_query(
        "chain_snapshot", "2026-07-21T00:00:00+00:00", symbol="AAPL", pit_only=False,
    ) == ["2026-07-20"]


def test_model_registry_register_promote_active(tmp_path):
    p1 = tmp_path / "m1.joblib"
    p1.write_bytes(b"v1")
    rid = model_registry_register("cal", str(p1), "abc", trained_at="2026-08-01T00:00:00+00:00")
    assert rid > 0
    assert model_registry_register("cal", str(p1), "abc") == rid
    model_registry_register("cal", str(p1), "def", trained_at="2026-08-02T00:00:00+00:00")
    model_registry_promote("cal", "def", promoted_at="2026-08-03T00:00:00+00:00")
    active = model_registry_get_active("cal")
    assert active["sha256"] == "def"
    model_registry_promote("cal", "abc", promoted_at="2026-08-04T00:00:00+00:00")
    assert model_registry_get_active("cal")["sha256"] == "abc"
    assert model_registry_get_active("missing") is None


def test_trade_events_adopted_alpaca_and_table_exists():
    trade_events_insert("orphan_found", symbol="XYZ", strategy="unmanaged", detail="no local")
    rows = trade_events_list(event_type="orphan_found")
    assert len(rows) == 1 and rows[0]["symbol"] == "XYZ"
    adopted_positions_insert("XYZ", "2026-08-23T12:00:00+00:00")
    adopted_positions_insert("XYZ")  # ignore dup
    assert adopted_positions_symbols() == {"XYZ"}
    alpaca_positions_insert(
        ts="2026-08-23T12:00:00+00:00", symbol="XYZ", qty=1, side="long",
        strategy="unmanaged", managed=0,
    )
    pos = alpaca_positions_list()
    assert len(pos) == 1 and pos[0]["symbol"] == "XYZ"
    assert table_exists("trade_events")
    assert table_exists("no_such_table") is False
    counts = snapshots_status_counts()
    assert counts["total"] == 0
    assert counts["pending"] == 0
