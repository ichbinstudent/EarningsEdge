"""Tests for the exit engine: rules, ExitManager, bot-visible flows."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from unittest.mock import MagicMock

import pytest

from framework.core.calendar import TradingCalendar
from framework.core.config import StrategyConfig
from framework.core.registry import StrategyRegistry
from framework.execution.managed import (
    close_positions, open_groups, record_open_positions,
)
from framework.execution.order_manager import OrderManager
from framework.positions.exits import (
    LegPos, MarketView, PositionGroup, ProfitTargetExit, ScheduledExit, StopLossExit,
    TimeExit, build_exit_rules, pnl_pct, remaining_close_plan, structure_value,
)
from framework.positions.manager import ExitManager
from framework.risk.killswitch import KillSwitch
from sqlalchemy import text
from earnings_edge.db import engine as db_engine


@pytest.fixture
def conn(tmp_path):
    db_engine.configure(tmp_path / "fw.db")


TODAY = date(2026, 7, 27)  # Monday


def _debit_group(entry=1.85, opened="2026-07-24T14:00:00+00:00"):
    return PositionGroup(
        group_id="g1", strategy="calendar_call_ml",
        legs=[
            LegPos("AAPL260731C00190000", "sell", 1, "call", 190.0, date(2026, 7, 31)),
            LegPos("AAPL260828C00190000", "buy", 1, "call", 190.0, date(2026, 8, 28)),
        ],
        entry_price=entry, opened_at=opened, credit=False,
        event_date=date(2026, 7, 29),
    )


def _credit_group(entry=2.50):
    return PositionGroup(
        group_id="g2", strategy="short_straddle",
        legs=[
            LegPos("AAPL260731C00190000", "sell", 1, "call", 190.0, date(2026, 7, 31)),
            LegPos("AAPL260731P00190000", "sell", 1, "put", 190.0, date(2026, 7, 31)),
        ],
        entry_price=entry, opened_at="2026-07-24T14:00:00+00:00", credit=True,
        event_date=date(2026, 7, 29),
    )


def _snaps(prices: dict[str, float], width: float = 0.1) -> dict[str, dict]:
    return {
        sym: {"latestQuote": {"bp": px - width / 2, "ap": px + width / 2}}
        for sym, px in prices.items()
    }


# ── rule math ---------------------------------------------------------------

def test_structure_value_net_mid():
    legs = _debit_group().legs
    snaps = _snaps({"AAPL260731C00190000": 1.0, "AAPL260828C00190000": 4.0})
    # sell near (−1.0) + buy far (+4.0) = 3.0
    assert structure_value(legs, snaps) == pytest.approx(3.0)
    assert structure_value(legs, {"AAPL260731C00190000": {}}) is None  # missing quote


def test_pnl_pct_debit_and_credit():
    d = _debit_group(entry=2.0)
    assert pnl_pct(d, 3.0) == pytest.approx(0.50)     # value 3 vs paid 2
    assert pnl_pct(d, 0.5) == pytest.approx(-0.75)
    c = _credit_group(entry=2.0)
    assert pnl_pct(c, -1.0) == pytest.approx(0.50)    # liability shrank to 1
    assert pnl_pct(c, -3.0) == pytest.approx(-0.50)   # liability grew
    assert pnl_pct(_debit_group(entry=0.0), 1.0) is None


def test_profit_target_and_stop_loss():
    g = _debit_group(entry=2.0)
    pt, sl = ProfitTargetExit(0.5), StopLossExit(0.75)
    m_win = MarketView(value_now=3.2, today=TODAY, sessions_since_open=1)
    m_loss = MarketView(value_now=0.4, today=TODAY, sessions_since_open=1)
    m_flat = MarketView(value_now=2.0, today=TODAY, sessions_since_open=1)
    assert pt.evaluate(g, m_win).auto is True
    assert sl.evaluate(g, m_loss) is not None
    assert pt.evaluate(g, m_flat) is None and sl.evaluate(g, m_flat) is None
    assert pt.evaluate(g, MarketView(None, TODAY, 1)) is None  # no quote → no signal


def test_time_exit_sessions_and_event():
    g = _debit_group()
    t3 = TimeExit(days_after_entry=3)
    assert t3.evaluate(g, MarketView(None, TODAY, sessions_since_open=3)) is not None
    assert t3.evaluate(g, MarketView(None, TODAY, sessions_since_open=2)) is None
    t1 = TimeExit(days_before_event=1)
    assert t1.evaluate(g, MarketView(None, TODAY, 0, sessions_until_event=1)) is not None
    assert t1.evaluate(g, MarketView(None, TODAY, 0, sessions_until_event=5)) is None
    assert t1.evaluate(g, MarketView(None, TODAY, 0, sessions_until_event=0)) is not None
    post = TimeExit(days_after_event=0)
    sig = post.evaluate(g, MarketView(None, TODAY, 0, sessions_until_event=0))
    assert sig is not None and sig.auto is True
    assert post.evaluate(g, MarketView(None, TODAY, 0, sessions_until_event=2)) is None


def test_build_exit_rules_from_config():
    rules = build_exit_rules([
        {"rule": "time", "days_after_entry": 3},
        {"rule": "profit_target", "pct": 0.5},
        {"rule": "stop_loss", "pct": 0.75},
        {"rule": "time", "days_before_event": 1},
        {"rule": "time", "days_after_event": 0},
        {"rule": "scheduled"},
    ])
    assert [type(r).__name__ for r in rules] == [
        "TimeExit", "ProfitTargetExit", "StopLossExit", "TimeExit", "TimeExit", "ScheduledExit"]
    assert rules[4].days_after_event == 0


# ── ScheduledExit: structural, entry-computed deadline (not a TOML day-count) ─

def _calendar_group(exit_by=None):
    return PositionGroup(
        group_id="g3", strategy="debit_size_exploit",
        legs=[
            LegPos("TPR260814C00131000", "sell", 9, "call", 131.0, date(2026, 8, 14)),
            LegPos("TPR260911C00130000", "buy", 9, "call", 130.0, date(2026, 9, 11)),
        ],
        entry_price=3.17, opened_at="2026-08-13T15:46:03+00:00",
        exit_by=exit_by,
    )


def test_scheduled_exit_fires_on_deadline_within_close_window():
    g = _calendar_group(exit_by=date(2026, 8, 14))
    rule = ScheduledExit(minutes_before_close=90)
    fires = MarketView(None, date(2026, 8, 14), 1, minutes_to_close=60)
    signal = rule.evaluate(g, fires)
    assert signal is not None and signal.auto is True and signal.rule == "scheduled"


def test_scheduled_exit_does_not_fire_before_deadline_date():
    g = _calendar_group(exit_by=date(2026, 8, 14))
    rule = ScheduledExit(minutes_before_close=90)
    # deadline is tomorrow — even with little time left in TODAY's session
    too_early = MarketView(None, date(2026, 8, 13), 1, minutes_to_close=30)
    assert rule.evaluate(g, too_early) is None


def test_scheduled_exit_does_not_fire_outside_close_window():
    g = _calendar_group(exit_by=date(2026, 8, 14))
    rule = ScheduledExit(minutes_before_close=90)
    # right day, but still 4 hours from close — not yet within the window
    early_in_day = MarketView(None, date(2026, 8, 14), 1, minutes_to_close=240)
    assert rule.evaluate(g, early_in_day) is None


def test_scheduled_exit_fires_on_a_later_day_too_not_only_the_exact_date():
    g = _calendar_group(exit_by=date(2026, 8, 14))
    rule = ScheduledExit()
    later = MarketView(None, date(2026, 8, 17), 1, minutes_to_close=60)
    assert rule.evaluate(g, later) is not None


def test_scheduled_exit_no_op_without_exit_by_or_clock():
    g_no_deadline = _calendar_group(exit_by=None)
    rule = ScheduledExit()
    assert rule.evaluate(g_no_deadline, MarketView(None, TODAY, 1, minutes_to_close=10)) is None

    g = _calendar_group(exit_by=date(2026, 8, 14))
    # market closed / clock unreadable → minutes_to_close is None → no signal,
    # never a guess
    assert rule.evaluate(g, MarketView(None, date(2026, 8, 14), 1, minutes_to_close=None)) is None


# ── ExitManager --------------------------------------------------------------

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


def _registry_with_exits():
    cfg = StrategyConfig(
        name="calendar_call_ml", exits=[
            {"rule": "time", "days_after_entry": 3},
            {"rule": "profit_target", "pct": 0.5},
            {"rule": "stop_loss", "pct": 0.75},
        ])
    return StrategyRegistry(configs={"calendar_call_ml": cfg})


def _seed_group(conn, entry=1.85, opened="2026-07-24T14:00:00+00:00", group_id="g1"):
    legs = [
        {"symbol": "AAPL260731C00190000", "side": "sell", "ratio_qty": 1,
         "option_type": "call", "strike": 190.0, "expiry": date(2026, 7, 31)},
        {"symbol": "AAPL260828C00190000", "side": "buy", "ratio_qty": 1,
         "option_type": "call", "strike": 190.0, "expiry": date(2026, 8, 28)},
    ]
    record_open_positions(legs, "calendar_call_ml", group_id=group_id,
                          entry_price=entry,
                          metadata={"side": "CALENDAR", "credit": False,
                                    "earnings_date": "2026-07-29"})
    with db_engine.session_scope() as s:
        s.execute(
            text("UPDATE managed_positions SET opened_at = :opened WHERE group_id = :gid"),
            {"opened": opened, "gid": group_id},
        )


def test_manager_auto_closes_on_profit_target(conn):
    _seed_group(conn, entry=1.85)
    # value now 3.0 → pnl +62% ≥ 50%
    snaps = _snaps({"AAPL260731C00190000": 1.0, "AAPL260828C00190000": 4.0})
    client = _stub_client(snaps, fill_price=3.0)
    mgr = ExitManager(client, registry=_registry_with_exits(),
                      order_manager=OrderManager(client, poll_secs=0, sleep=lambda s: None),
                      today=TODAY)
    out = mgr.evaluate_all()
    assert len(out["auto_closed"]) == 1
    assert "profit_target" in out["auto_closed"][0]
    assert open_groups() == []  # all closed
    with db_engine.get_session() as s:
        ev = s.execute(text("SELECT * FROM trade_events WHERE event_type = 'exit_filled'")).mappings().first()
    assert ev is not None and "realized_pnl" in ev["detail"]


def test_manager_holds_when_no_signal(conn):
    _seed_group(conn, entry=1.85, opened=f"{TODAY.isoformat()}T14:00:00+00:00")
    snaps = _snaps({"AAPL260731C00190000": 1.0, "AAPL260828C00190000": 2.8})  # ~entry
    client = _stub_client(snaps)
    mgr = ExitManager(client, registry=_registry_with_exits(),
                      order_manager=OrderManager(client, poll_secs=0, sleep=lambda s: None),
                      today=TODAY)
    out = mgr.evaluate_all()
    assert out["held"] == 1 and not out["auto_closed"] and not out["proposed"]


def test_manager_time_exit_proposes_and_dedupes(conn):
    _seed_group(conn, opened="2026-07-21T14:00:00+00:00")  # ≥3 sessions old
    snaps = _snaps({"AAPL260731C00190000": 1.0, "AAPL260828C00190000": 2.8})
    client = _stub_client(snaps)
    mgr = ExitManager(client, registry=_registry_with_exits(),
                      order_manager=OrderManager(client, poll_secs=0, sleep=lambda s: None),
                      today=TODAY)
    out = mgr.evaluate_all()
    assert len(out["proposed"]) == 1
    out2 = mgr.evaluate_all()  # second pass: deduped
    assert not out2["proposed"]
    rows = mgr.pending_exit_proposals()
    assert len(rows) == 1 and rows[0]["rule"] == "time"


def test_manager_decide_exit_close_and_snooze(conn):
    _seed_group(conn, opened="2026-07-21T14:00:00+00:00")
    snaps = _snaps({"AAPL260731C00190000": 1.0, "AAPL260828C00190000": 2.8})
    client = _stub_client(snaps, fill_price=1.8)
    mgr = ExitManager(client, registry=_registry_with_exits(),
                      order_manager=OrderManager(client, poll_secs=0, sleep=lambda s: None),
                      today=TODAY)
    mgr.evaluate_all()
    pid = mgr.pending_exit_proposals()[0]["id"]

    res = mgr.decide_exit(pid, close=False, decided_by=7)
    assert res["ok"] and res["status"] == "snoozed"
    assert mgr.decide_exit(pid, close=True)["ok"] is False  # no longer pending

    mgr2 = ExitManager(client, registry=_registry_with_exits(),
                       order_manager=OrderManager(client, poll_secs=0, sleep=lambda s: None),
                       today=TODAY + timedelta(days=1))
    mgr2.evaluate_all()  # new proposal (old one snoozed, group still open)
    pid2 = mgr2.pending_exit_proposals()[0]["id"]
    res2 = mgr2.decide_exit(pid2, close=True, decided_by=7)
    assert res2["ok"] is True
    assert open_groups() == []


def test_kill_switch_does_not_block_exits(conn):
    _seed_group(conn, entry=1.85)
    KillSwitch().trip("test halt", by="test")
    snaps = _snaps({"AAPL260731C00190000": 1.0, "AAPL260828C00190000": 4.0})
    client = _stub_client(snaps, fill_price=3.0)
    mgr = ExitManager(client, registry=_registry_with_exits(),
                      order_manager=OrderManager(client, poll_secs=0, sleep=lambda s: None),
                      today=TODAY)
    out = mgr.evaluate_all()
    assert len(out["auto_closed"]) == 1  # exits work while halted


def test_manager_event_day_time_exit(conn):
    # event today → sessions_until_event = 0 → days_before_event=1 fires
    _seed_group(conn, opened="2026-07-27T13:00:00+00:00")
    with db_engine.session_scope() as s:
        s.execute(text(
            "UPDATE managed_positions SET metadata = json_set(metadata, '$.earnings_date', '2026-07-27') "
            "WHERE group_id = 'g1'"
        ))
    cfg = StrategyConfig(name="calendar_call_ml",
                         exits=[{"rule": "time", "days_before_event": 1}])
    reg = StrategyRegistry(configs={"calendar_call_ml": cfg})
    snaps = _snaps({"AAPL260731C00190000": 1.0, "AAPL260828C00190000": 2.8})
    client = _stub_client(snaps)
    mgr = ExitManager(client, registry=reg,
                      order_manager=OrderManager(client, poll_secs=0, sleep=lambda s: None),
                      today=TODAY)
    out = mgr.evaluate_all()
    assert len(out["proposed"]) == 1


def test_exit_by_round_trips_through_managed_positions(conn):
    legs = [
        {"symbol": "TPR260814C00131000", "side": "sell", "ratio_qty": 9,
         "option_type": "call", "strike": 131.0, "expiry": date(2026, 8, 14)},
        {"symbol": "TPR260911C00130000", "side": "buy", "ratio_qty": 9,
         "option_type": "call", "strike": 130.0, "expiry": date(2026, 9, 11)},
    ]
    record_open_positions(legs, "debit_size_exploit", group_id="g4",
                          entry_price=3.17, exit_by=date(2026, 8, 14),
                          metadata={"side": "CALENDAR"})
    groups = {g.group_id: g for g in open_groups()}
    assert groups["g4"].exit_by == date(2026, 8, 14)


def test_exit_by_none_when_not_passed(conn):
    _seed_group(conn)  # no exit_by kwarg — single-expiry-style call
    groups = {g.group_id: g for g in open_groups()}
    assert groups["g1"].exit_by is None


# ── ExitManager._minutes_to_close ---------------------------------------------

def test_minutes_to_close_computed_when_market_open(conn):
    client = MagicMock()
    client.get_clock.return_value = {
        "is_open": True,
        "timestamp": "2026-08-14T14:30:00-04:00",
        "next_close": "2026-08-14T16:00:00-04:00",
    }
    mgr = ExitManager(client)
    assert mgr._minutes_to_close() == 90


def test_minutes_to_close_none_when_market_closed(conn):
    client = MagicMock()
    client.get_clock.return_value = {"is_open": False}
    mgr = ExitManager(client)
    assert mgr._minutes_to_close() is None


def test_minutes_to_close_none_on_clock_fetch_failure(conn):
    client = MagicMock()
    client.get_clock.side_effect = ConnectionError("down")
    mgr = ExitManager(client)
    assert mgr._minutes_to_close() is None


def test_manager_auto_closes_calendar_on_scheduled_deadline(conn):
    """End-to-end: a debit_size_exploit-style calendar with exit_by today,
    evaluated with 60min left in the session, auto-closes via ScheduledExit
    — the actual bug that motivated this: a fixed days_after_entry TOML
    rule doesn't align with the near leg's real expiry."""
    legs = [
        {"symbol": "TPR260814C00131000", "side": "sell", "ratio_qty": 9,
         "option_type": "call", "strike": 131.0, "expiry": date(2026, 8, 14)},
        {"symbol": "TPR260911C00130000", "side": "buy", "ratio_qty": 9,
         "option_type": "call", "strike": 130.0, "expiry": date(2026, 9, 11)},
    ]
    record_open_positions(legs, "debit_size_exploit", group_id="tpr1",
                          entry_price=3.17, exit_by=date(2026, 8, 14),
                          metadata={"side": "CALENDAR", "credit": False})
    cfg = StrategyConfig(name="debit_size_exploit",
                         exits=[{"rule": "scheduled", "minutes_before_close": 90}])
    reg = StrategyRegistry(configs={"debit_size_exploit": cfg})
    snaps = _snaps({"TPR260814C00131000": 1.0, "TPR260911C00130000": 4.5})
    client = _stub_client(snaps, fill_price=3.5)
    client.get_clock.return_value = {
        "is_open": True,
        "timestamp": "2026-08-14T14:30:00-04:00",
        "next_close": "2026-08-14T16:00:00-04:00",
    }
    mgr = ExitManager(client, registry=reg,
                      order_manager=OrderManager(client, poll_secs=0, sleep=lambda s: None),
                      today=date(2026, 8, 14))
    out = mgr.evaluate_all()
    assert len(out["auto_closed"]) == 1
    assert "scheduled" in out["auto_closed"][0]
    assert open_groups() == []


def test_sessions_since_open_skips_holiday():
    # Entry Thu Jul 2, eval Mon Jul 6: Jul 3 was a holiday → 1 session, not 3
    cal = TradingCalendar()
    sessions = len(cal.sessions_between(date(2026, 7, 2), date(2026, 7, 6))) - 1
    assert sessions == 1


def test_remaining_close_plan_combo_vs_near_missing():
    near = LegPos("TPR260814C00131000", "sell", 9, "call", 131.0, date(2026, 8, 14))
    far = LegPos("TPR260911C00130000", "buy", 9, "call", 130.0, date(2026, 9, 11))
    both = _snaps({"TPR260814C00131000": 1.0, "TPR260911C00130000": 4.5})
    combo = remaining_close_plan([near, far], both, date(2026, 8, 14))
    assert combo["mode"] == "combo"
    far_only = _snaps({"TPR260911C00130000": 4.5})
    rem = remaining_close_plan([near, far], far_only, date(2026, 8, 14))
    assert rem["mode"] == "remaining"
    assert [l.symbol for l in rem["close_legs"]] == ["TPR260911C00130000"]
    assert [l.symbol for l in rem["drop_legs"]] == ["TPR260814C00131000"]
    none = remaining_close_plan([near, far], {}, date(2026, 8, 14))
    assert none["mode"] == "expired"


def test_close_group_remaining_leg_submits_far_only(conn):
    """Near snapshot missing → combo quote is None; a far-leg close is submitted."""
    near_sym = "TPR260814C00131000"
    far_sym = "TPR260911C00130000"
    legs = [
        {"symbol": near_sym, "side": "sell", "ratio_qty": 9,
         "option_type": "call", "strike": 131.0, "expiry": date(2026, 8, 14)},
        {"symbol": far_sym, "side": "buy", "ratio_qty": 9,
         "option_type": "call", "strike": 130.0, "expiry": date(2026, 9, 11)},
    ]
    record_open_positions(legs, "debit_size_exploit", group_id="tpr-rem",
                          entry_price=3.17, exit_by=date(2026, 8, 14),
                          metadata={"side": "CALENDAR", "credit": False})
    far_only = _snaps({far_sym: 4.5})
    client = _stub_client(far_only, fill_price=4.5)
    group = open_groups()[0]
    mgr = ExitManager(
        client,
        order_manager=OrderManager(client, poll_secs=0, sleep=lambda s: None),
        today=date(2026, 8, 14),
    )
    assert structure_value(group.legs, far_only) is None
    mo = mgr.close_group(group, reason="test remaining")
    assert mo.state in ("filled", "partial")
    assert client.submit_order.called
    submitted_syms = [c.kwargs.get("symbol") or c.args[0]
                      for c in client.submit_order.call_args_list]
    assert far_sym in submitted_syms
    assert near_sym not in submitted_syms
    assert open_groups() == []


def test_remaining_leg_close_limit_is_per_share_mid(conn):
    """TPR 9-lot far mid $4.50 must submit limit ≈ $4.50, not $40.50 (mid×qty)."""
    near_sym = "TPR260814C00131000"
    far_sym = "TPR260911C00130000"
    far_mid = 4.50
    record_open_positions([
            {"symbol": near_sym, "side": "sell", "ratio_qty": 9,
             "option_type": "call", "strike": 131.0, "expiry": date(2026, 8, 14)},
            {"symbol": far_sym, "side": "buy", "ratio_qty": 9,
             "option_type": "call", "strike": 130.0, "expiry": date(2026, 9, 11)},
        ],
        "debit_size_exploit", group_id="tpr-lim",
        entry_price=3.17, exit_by=date(2026, 8, 14),
        metadata={"side": "CALENDAR", "credit": False},
    )
    far_only = _snaps({far_sym: far_mid})
    client = _stub_client(far_only, fill_price=far_mid)
    mgr = ExitManager(
        client,
        order_manager=OrderManager(client, poll_secs=0, sleep=lambda s: None),
        today=date(2026, 8, 14),
    )
    mo = mgr.close_group(open_groups()[0], reason="per-share limit")
    assert mo.state in ("filled", "partial")
    assert client.submit_order.called
    limits = [
        c.kwargs.get("limit_price") if c.kwargs else None
        for c in client.submit_order.call_args_list
    ]
    limits = [p for p in limits if p is not None]
    assert limits, "close_group must submit a priced limit"
    # LimitWalkPolicy starts at mid ± 25bps and walks at most 100bps — never mid×qty.
    assert all(abs(p - far_mid) < 0.15 for p in limits)
    assert all(abs(p - far_mid * 9) > 10 for p in limits)
    qtys = [
        c.kwargs.get("qty") if c.kwargs else (c.args[1] if c.args and len(c.args) > 1 else None)
        for c in client.submit_order.call_args_list
    ]
    assert 9 in qtys


def test_remaining_leg_exhaust_does_not_orphan_far(conn):
    """If the far close does not fill, keep the group open — do not mark
    closed just because the near expired."""
    near_sym = "TPR260814C00131000"
    far_sym = "TPR260911C00130000"
    record_open_positions([
            {"symbol": near_sym, "side": "sell", "ratio_qty": 9,
             "option_type": "call", "strike": 131.0, "expiry": date(2026, 8, 14)},
            {"symbol": far_sym, "side": "buy", "ratio_qty": 9,
             "option_type": "call", "strike": 130.0, "expiry": date(2026, 9, 11)},
        ],
        "debit_size_exploit", group_id="tpr-exh",
        entry_price=3.17, exit_by=date(2026, 8, 14),
        metadata={"side": "CALENDAR", "credit": False},
    )
    far_only = _snaps({far_sym: 4.50})
    client = _stub_client(far_only, fill_price=4.50)
    client.get_order.side_effect = lambda oid: {
        "id": oid, "status": "new", "filled_qty": 0, "filled_avg_price": None,
    }
    mgr = ExitManager(
        client,
        order_manager=OrderManager(client, poll_secs=0, sleep=lambda s: None),
        today=date(2026, 8, 14),
    )
    from framework.alerts import DEDUPER
    DEDUPER.reset()
    mo = mgr.close_group(open_groups()[0], reason="exhaust")
    assert mo.state == "exhausted"
    still = open_groups()
    assert len(still) == 1
    assert {l.symbol for l in still[0].legs} == {near_sym, far_sym}
    outbox = DEDUPER.drain()
    assert any("Remaining-leg close exhausted" in m and "TPR" in m for m in outbox)


def test_ff_ladder_exits_are_scheduled_pt_sl():
    """ff_ladder.toml uses scheduled near-expiry close + PT/SL, not event-day time."""
    from framework.core.config import load_strategy_configs
    cfgs = load_strategy_configs()
    ff = cfgs["ff_ladder"]
    rules = build_exit_rules(ff.exits)
    assert any(getattr(r, "minutes_before_close", None) == 90 for r in rules)
    assert not any(getattr(r, "days_after_event", None) == 0 for r in rules)
    assert not any(getattr(r, "days_before_event", None) == 1 for r in rules)
