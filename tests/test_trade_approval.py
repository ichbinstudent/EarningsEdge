"""Tests for the human-in-the-loop trade approval flow."""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from sqlalchemy import text

from earnings_edge.alpaca_bridge import BridgeConfig, StrategyBridge
from earnings_edge.db import engine as db_engine
from earnings_edge.trading_types import StrategyResult, Trade
from earnings_edge.trade_approval import (
    PROPOSAL_TTL_HOURS,
    PendingTradeStore,
    build_ff_proposals,
    build_proposals,
    execute_proposal,
    ff_candidate_from_trade,
    reject_proposal,
    trade_from_json,
    trade_to_json,
)


def _age_proposal(store: PendingTradeStore, proposal_id: int, created_at: str) -> None:
    store._ensure_engine()
    with db_engine.session_scope() as s:
        s.execute(
            text("UPDATE pending_trades SET created_at = :ts WHERE id = :id"),
            {"ts": created_at, "id": proposal_id},
        )


def _trade(ticker="AAPL", score=0.61, decision="TAKE", side="CALENDAR"):
    return Trade(
        ticker=ticker,
        earnings_date=date(2026, 7, 29),
        scan_date=date(2026, 7, 28),
        strategy="calendar_call_ml",
        side=side,
        entry_price=1.85,
        features={
            "near_strike": 190.0, "far_strike": 190.0,
            "near_expiry": date(2026, 7, 31), "far_expiry": date(2026, 8, 28),
        },
        model_score=score,
        ml_decision=decision,
        notes="test trade",
    )


@pytest.fixture
def store(tmp_path):
    return PendingTradeStore(str(tmp_path / "test.db"))


@pytest.fixture
def mock_bridge():
    client = MagicMock()
    client.position_symbols.return_value = set()
    bridge = StrategyBridge(client=client, config=BridgeConfig(dry_run=False))
    return bridge


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def test_trade_json_roundtrip():
    t = _trade()
    t2 = trade_from_json(trade_to_json(t))
    assert t2.ticker == t.ticker
    assert t2.earnings_date == t.earnings_date
    assert t2.scan_date == t.scan_date
    assert t2.strategy == t.strategy
    assert t2.side == t.side
    assert t2.entry_price == t.entry_price
    assert t2.model_score == t.model_score
    assert t2.ml_decision == t.ml_decision
    # feature dates survive as ISO strings — the bridge's _parse_date handles them
    assert t2.features["near_strike"] == 190.0


# ---------------------------------------------------------------------------
# Store lifecycle
# ---------------------------------------------------------------------------

def test_store_add_get_mark(store):
    pid = store.add(_trade(), "card")
    assert pid is not None
    row = store.get(pid)
    assert row["status"] == "pending"
    assert row["ticker"] == "AAPL"
    assert row["model_score"] == 0.61

    store.mark(pid, "executed", order_json={"order_id": "x"}, decided_by=42)
    row = store.get(pid)
    assert row["status"] == "executed"
    assert row["decided_by"] == 42
    assert row["decided_at"] is not None


def test_store_dedupes_pending(store):
    pid1 = store.add(_trade(), "card")
    pid2 = store.add(_trade(), "card")
    assert pid1 is not None
    assert pid2 is None  # identical pending proposal — no double-asking
    # after a decision, a fresh proposal is allowed again
    store.mark(pid1, "rejected")
    assert store.add(_trade(), "card") is not None


def test_store_list_pending(store):
    store.add(_trade("AAPL", score=0.5), "a")
    store.add(_trade("MSFT", score=0.9), "b")
    rows = store.list_pending()
    assert len(rows) == 2
    assert rows[0]["ticker"] == "MSFT"  # sorted by score desc


# ---------------------------------------------------------------------------
# build_proposals
# ---------------------------------------------------------------------------

def _fake_strategy(trades):
    strat = MagicMock()
    strat.run.return_value = StrategyResult(name="calendar_call_ml", trades=trades)
    return strat


def _source(mapping):
    """trade_source injection: strategy name -> Trade list."""
    return lambda name: mapping.get(name, [])


def test_build_proposals_filters_and_ranks(store, mock_bridge):
    trades = [
        _trade("LOW", score=0.2),
        _trade("HIGH", score=0.9),
        _trade("SKIPROW", score=0.99, decision="SKIP"),
    ]
    rows = build_proposals(store, strategies=["calendar_call_ml"],
                           bridge=mock_bridge,
                           trade_source=_source({"calendar_call_ml": trades}))
    tickers = [r["ticker"] for r in rows]
    assert "SKIPROW" not in tickers, "SKIP trades must never be proposed"
    assert tickers == ["HIGH", "LOW"], "proposals ranked by model score"
    assert all(r["status"] == "pending" for r in rows)
    assert "#" in rows[0]["card_text"]


def test_build_proposals_respects_max(store, mock_bridge):
    trades = [_trade(f"T{i}", score=0.5 + i * 0.01) for i in range(10)]
    rows = build_proposals(store, strategies=["calendar_call_ml"],
                           max_proposals=3, bridge=mock_bridge,
                           trade_source=_source({"calendar_call_ml": trades}))
    assert len(rows) == 3


def test_build_proposals_skips_when_position_exists(store):
    client = MagicMock()
    client.position_symbols.return_value = {"AAPL260731C00190000"}  # near leg
    bridge = StrategyBridge(client=client, config=BridgeConfig(dry_run=False))
    rows = build_proposals(store, strategies=["calendar_call_ml"],
                           bridge=bridge,
                           trade_source=_source({"calendar_call_ml": [_trade()]}))
    assert rows == []


def test_build_proposals_funnel_counters(store, mock_bridge):
    from earnings_edge import trade_approval

    trades = [_trade("A", score=0.5), _trade("B", score=0.6, decision="SKIP")]
    rows = build_proposals(store, strategies=["calendar_call_ml"],
                           bridge=mock_bridge,
                           trade_source=_source({"calendar_call_ml": trades}))
    assert len(rows) == 1
    f = trade_approval.LAST_FUNNEL
    stage = f["strategies"]["calendar_call_ml"]
    assert stage["decision_pass"] == 2       # both trades returned by source
    assert stage["legs_ok"] == 1             # SKIP filtered before legs
    assert stage["dte_ok"] == 1
    assert stage["position_ok"] == 1
    assert stage["proposals_created"] == 1
    assert f["proposals"] == 1
    # persisted for audit
    store._ensure_engine()
    with db_engine.session_scope() as s:
        row = s.execute(
            text(
                "SELECT strategies, counts, proposals_total FROM proposal_funnel "
                "ORDER BY id DESC LIMIT 1"
            )
        ).one()
    import json as _json
    assert _json.loads(row[1])["calendar_call_ml"]["proposals_created"] == 1
    assert row[2] == 1


def test_build_proposals_skips_unmapped_strategies(store, mock_bridge):
    # earnings_quality has no live mapping — must be skipped without error
    rows = build_proposals(store, strategies=["earnings_quality"],
                           bridge=mock_bridge, trade_source=_source({}))
    assert rows == []


# ---------------------------------------------------------------------------
# execute_proposal guards
# ---------------------------------------------------------------------------

def test_execute_not_found(store):
    assert execute_proposal(store, 999)["ok"] is False


def test_execute_rejects_non_pending(store):
    pid = store.add(_trade(), "card")
    store.mark(pid, "rejected")
    result = execute_proposal(store, pid)
    assert result["ok"] is False
    assert "already rejected" in result["error"]


def test_execute_expires_stale_proposals(store, mock_bridge):
    pid = store.add(_trade(), "card")
    # age the row beyond the TTL
    stale = (datetime.now(timezone.utc) - timedelta(hours=PROPOSAL_TTL_HOURS + 1)).isoformat()
    _age_proposal(store, pid, stale)

    result = execute_proposal(store, pid, bridge=mock_bridge)
    assert result["ok"] is False
    assert "expired" in result["error"]
    assert store.get(pid)["status"] == "expired"
    mock_bridge.client.submit_multi_leg_order.assert_not_called()


def test_execute_success_marks_executed(store):
    pid = store.add(_trade(), "card")
    client = MagicMock()
    client.position_symbols.return_value = set()
    client.get_option_snapshot.return_value = {}
    client.submit_multi_leg_order.return_value = {
        "id": "ord-1", "status": "filled", "filled_qty": 1, "filled_avg_price": 1.85,
        "legs": [],
    }
    bridge = StrategyBridge(client=client, config=BridgeConfig(dry_run=False))
    result = execute_proposal(store, pid, bridge=bridge, decided_by=7)
    assert result["ok"] is True
    assert result["order_id"] == "ord-1"
    row = store.get(pid)
    assert row["status"] == "executed"
    assert row["decided_by"] == 7


def test_execute_bridge_rejection_marks_error(store):
    pid = store.add(_trade(), "card")
    client = MagicMock()
    client.position_symbols.return_value = set()
    bridge = StrategyBridge(client=client, config=BridgeConfig(dry_run=False))
    bridge.execute_trade = MagicMock(return_value=None)
    result = execute_proposal(store, pid, bridge=bridge)
    assert result["ok"] is False
    assert store.get(pid)["status"] == "error"


def test_reject_proposal(store):
    pid = store.add(_trade(), "card")
    assert reject_proposal(store, pid, decided_by=1)["ok"] is True
    assert store.get(pid)["status"] == "rejected"
    assert reject_proposal(store, pid)["ok"] is False  # no double-decide


# ---------------------------------------------------------------------------
# run_auto_trade TAKE filter (bug: SKIP rows were being submitted)
# ---------------------------------------------------------------------------

def test_run_auto_trade_skips_non_take_trades():
    from earnings_edge.alpaca_bridge import run_auto_trade

    skip_trade = _trade("SKIPROW", decision="SKIP")
    take_trade = _trade("TAKEROW", decision="TAKE")
    strat = MagicMock()
    strat.run.return_value = StrategyResult(name="calendar_call_ml", trades=[skip_trade, take_trade])

    client = MagicMock()
    client.position_symbols.return_value = set()
    client.buying_power.return_value = 50000.0

    with patch("earnings_edge.alpaca_bridge.create_client", return_value=client), \
         patch("earnings_edge.alpaca_bridge._resolve_strategy", return_value=strat), \
         patch("earnings_edge.alpaca_bridge.DataBundle") as mock_bundle, \
         patch.object(StrategyBridge, "execute_trade", return_value=None) as mock_exec:
        mock_bundle.from_db.return_value = MagicMock()
        run_auto_trade(strategies=["calendar_call_ml"])

    executed = [c.args[0].ticker for c in mock_exec.call_args_list]
    assert executed == ["TAKEROW"], "SKIP trades must never reach the bridge"


# ---------------------------------------------------------------------------
# Market-closed guard (production path — no injected bridge)
# ---------------------------------------------------------------------------

def test_execute_refuses_when_market_closed(store):
    pid = store.add(_trade(), "card")
    client = MagicMock()
    client.get_clock.return_value = {"is_open": False}
    with patch("earnings_edge.trade_approval.create_client", return_value=client):
        result = execute_proposal(store, pid)
    assert result["ok"] is False
    assert "closed" in result["error"]
    # not consumed — the operator can still confirm next session
    assert store.get(pid)["status"] == "pending"


def test_execute_refuses_when_clock_check_fails(store):
    pid = store.add(_trade(), "card")
    client = MagicMock()
    client.get_clock.side_effect = RuntimeError("network down")
    with patch("earnings_edge.trade_approval.create_client", return_value=client):
        result = execute_proposal(store, pid)
    assert result["ok"] is False
    assert "clock check failed" in result["error"]
    assert store.get(pid)["status"] == "pending"


# ---------------------------------------------------------------------------
# FF ladder proposals — same store, same confirm path as every strategy
# ---------------------------------------------------------------------------

def _ff_candidate(ticker="AAPL"):
    from earnings_edge.fwd_factor_ladder import CalendarCandidate
    return CalendarCandidate(
        ticker=ticker, earnings_date=date.today().isoformat(),
        spot=190.0, strike=190.0,
        near_symbol=f"{ticker}260731C00190000", far_symbol=f"{ticker}260828C00190000",
        near_expiry="2026-07-31", far_expiry="2026-08-28",
        near_bid=5.0, near_ask=5.2, far_bid=7.0, far_ask=7.2,
        sigma_fwd=0.45, hist_rms_move=0.05, tau_days=28,
        d_start=1.90, d_cap=2.00, mid_debit=1.85,
    )


class _FakeRunner:
    def __init__(self, lid=11):
        self.lid = lid
        self.armed = []
        self.events = []

    def arm(self, cand, armed_by=None):
        self.armed.append((cand, armed_by))
        return self.lid

    def drain_events(self):
        ev, self.events = self.events, []
        return ev


def test_build_ff_proposals_persists_cards(store):
    rows = build_ff_proposals(store, [_ff_candidate("AAPL"), _ff_candidate("MSFT")])
    assert len(rows) == 2
    assert all(r["strategy"] == "ff_ladder" for r in rows)
    assert "ff_ladder" in rows[0]["card_text"]
    assert f"#{rows[0]['id']}" in rows[0]["card_text"]
    # candidate round-trips through trade_json
    cand = ff_candidate_from_trade(trade_from_json(rows[0]["trade_json"]))
    assert cand.ticker == rows[0]["ticker"]
    assert cand.d_cap == 2.00


def test_build_ff_proposals_dedupes(store):
    assert len(build_ff_proposals(store, [_ff_candidate()])) == 1
    assert build_ff_proposals(store, [_ff_candidate()]) == []


def _during_ff_window(created_at: str) -> datetime:
    """14:00 ET on the proposal's created date — inside 14:00–15:45, same day."""
    import pytz
    eastern = pytz.timezone("US/Eastern")
    created = datetime.fromisoformat(created_at)
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    created_et = created.astimezone(eastern)
    return created_et.replace(hour=14, minute=0, second=0, microsecond=0).astimezone(timezone.utc)


def test_execute_ff_arms_via_runner(store):
    rows = build_ff_proposals(store, [_ff_candidate()])
    runner = _FakeRunner()
    result = execute_proposal(
        store, rows[0]["id"], ff_runner=runner, decided_by=5,
        now=_during_ff_window(rows[0]["created_at"]),
    )
    assert result["ok"] is True
    assert result["ladder_id"] == 11
    assert runner.armed and runner.armed[0][1] == 5
    row = store.get(rows[0]["id"])
    assert row["status"] == "executed"
    import json as _json
    assert _json.loads(row["order_json"])["ladder_id"] == 11


def test_execute_ff_refusal_marks_error(store):
    rows = build_ff_proposals(store, [_ff_candidate()])
    runner = _FakeRunner(lid=None)
    runner.events = ["⛔ FF arm refused: AAPL — kill switch is halted"]
    result = execute_proposal(
        store, rows[0]["id"], ff_runner=runner,
        now=_during_ff_window(rows[0]["created_at"]),
    )
    assert result["ok"] is False
    assert "kill switch" in result["error"]
    assert store.get(rows[0]["id"])["status"] == "error"


def test_execute_ff_stale_candidate_expires(store):
    rows = build_ff_proposals(store, [_ff_candidate()])
    yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    _age_proposal(store, rows[0]["id"], yesterday)
    runner = _FakeRunner()
    result = execute_proposal(store, rows[0]["id"], ff_runner=runner)
    assert result["ok"] is False
    # either guard may fire first (8h TTL or same-ET-day staleness) — both
    # expire the row and neither arms the ladder
    assert "expired" in result["error"] or "stale" in result["error"]
    assert runner.armed == []
    assert store.get(rows[0]["id"])["status"] == "expired"
