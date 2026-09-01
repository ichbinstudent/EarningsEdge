"""Tests for the framework execution layer: order manager, reconcile, lifecycle."""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock

import pytest

from framework.execution.lifecycle import LifecycleManager
from framework.execution.managed import open_positions, record_open_positions
from framework.execution.order_manager import (
    LimitWalkPolicy, MarketPolicy, MidPricePolicy, OrderManager,
)
from framework.execution.reconcile import Reconciler
from sqlalchemy import text
from earnings_edge.db import engine as db_engine


@pytest.fixture
def conn(tmp_path):
    db_engine.configure(tmp_path / "fw.db")


# ── Stub broker ----------------------------------------------------------

class StubBroker:
    """Alpaca-shaped stub; fills on a configurable submit number."""

    def __init__(self, fill_on_submit: int = 1):
        self.fill_on_submit = fill_on_submit
        self.submit_count = 0
        self.seq = 0
        self.orders: dict[str, dict] = {}
        self.canceled: list[str] = []

    def _submit(self, qty, order_type, limit_price):
        self.seq += 1
        self.submit_count += 1
        oid = f"o{self.seq}"
        filled = self.submit_count >= self.fill_on_submit
        self.orders[oid] = {
            "id": oid,
            "status": "filled" if filled else "new",
            "filled_qty": qty if filled else 0,
            "filled_avg_price": limit_price if filled else None,
        }
        return self.orders[oid]

    def submit_order(self, symbol, qty, side, order_type, limit_price, time_in_force, client_order_id):
        return self._submit(qty, order_type, limit_price)

    def submit_multi_leg_order(self, legs, qty, order_type, limit_price, time_in_force, client_order_id):
        return self._submit(qty, order_type, limit_price)

    def get_order(self, order_id):
        return self.orders[order_id]

    def cancel_order(self, order_id):
        self.orders[order_id]["status"] = "canceled"
        self.canceled.append(order_id)
        return self.orders[order_id]


LEGS = [{"symbol": "AAPL260815C00100000", "side": "buy", "ratio_qty": 1}]


def _manager(broker):
    return OrderManager(broker, poll_secs=0, sleep=lambda s: None)


# ── Pricing policies -------------------------------------------------------

def test_limit_walk_buy_walks_up_from_mid():
    prices = LimitWalkPolicy(steps=3, step_improve_bps=0, final_improve_bps=100).walk(10.0, "buy")
    assert prices == [10.0, 10.05, 10.1]


def test_limit_walk_sell_walks_down_from_mid():
    prices = LimitWalkPolicy(steps=3, step_improve_bps=0, final_improve_bps=100).walk(10.0, "sell")
    assert prices == [10.0, 9.95, 9.9]


def test_mid_and_market_policies():
    assert MidPricePolicy().walk(10.0, "buy") == [10.0]
    assert MarketPolicy().walk(10.0, "buy") == [None]


# ── Order manager -----------------------------------------------------------

def test_execute_fills_first_rung():
    broker = StubBroker(fill_on_submit=1)
    mo = _manager(broker).execute(LEGS, 1, MidPricePolicy(), lambda: 10.0)
    assert mo.state == "filled" and mo.filled_qty == 1
    assert mo.filled_avg_price == 10.0
    assert broker.canceled == []  # no cancel needed


def test_execute_walks_then_fills():
    broker = StubBroker(fill_on_submit=3)
    policy = LimitWalkPolicy(steps=3, step_improve_bps=0, final_improve_bps=100)
    mo = _manager(broker).execute(LEGS, 1, policy, lambda: 10.0)
    assert mo.state == "filled"
    assert mo.rungs_used == 3 and mo.filled_avg_price == 10.1
    assert len(broker.canceled) == 2  # first two rungs canceled before replace


def test_execute_exhausted_cancels_and_reports():
    broker = StubBroker(fill_on_submit=99)
    mo = _manager(broker).execute(LEGS, 1, MidPricePolicy(), lambda: 10.0)
    assert mo.state == "exhausted"
    assert len(broker.canceled) == 1


def test_execute_no_quote_is_error():
    broker = StubBroker()
    mo = _manager(broker).execute(LEGS, 1, MidPricePolicy(), lambda: None)
    assert mo.state == "error" and broker.submit_count == 0


def test_execute_market_policy_submits_market():
    broker = StubBroker(fill_on_submit=1)
    mo = _manager(broker).execute(LEGS, 1, MarketPolicy(), lambda: 10.0)
    assert mo.state == "filled"
    order = broker.orders[mo.order_ids[0]]
    assert order["filled_avg_price"] is None  # market: no limit price passed


# ── Reconcile --------------------------------------------------------------

def _broker_positions(*rows):
    client = MagicMock()
    client.get_positions.return_value = [
        {"symbol": s, "qty": q, "side": "long", "avg_entry_price": e,
         "current_price": c, "market_value": mv, "unrealized_pl": pl}
        for s, q, e, c, mv, pl in rows
    ]
    return client


def test_reconcile_matched_position(conn):
    record_open_positions([{"symbol": "SYM1", "ratio_qty": 1}], "s1", group_id="g1")
    rec = Reconciler(_broker_positions(("SYM1", 1, 5.0, 6.0, 600, 100)))
    report = rec.run()
    assert report.matched == 1 and not report.orphans and not report.closed_externally
    with db_engine.get_session() as s:
        row = s.execute(text("SELECT * FROM alpaca_positions WHERE symbol = 'SYM1'")).mappings().first()
    assert row["managed"] == 1 and row["strategy"] == "s1"


def test_reconcile_orphan_position(conn):
    # Baseline already established → a NEW unknown position alerts as orphan
    from earnings_edge.db import adopted_positions_insert
    adopted_positions_insert("BASE", "2026-07-25")
    record_open_positions([{"symbol": "BASE", "ratio_qty": 1}], "s1", group_id="g0")
    rec = Reconciler(_broker_positions(("ORPH", 2, 1.0, 1.5, 300, 100)))
    report = rec.run()
    assert report.orphans == ["ORPH"]
    with db_engine.get_session() as s:
        ev = s.execute(text("SELECT * FROM trade_events WHERE event_type = 'orphan_found'")).mappings().first()
        row = s.execute(text("SELECT * FROM alpaca_positions WHERE symbol = 'ORPH'")).mappings().first()
    assert ev["symbol"] == "ORPH"
    assert row["managed"] == 0 and row["strategy"] == "unmanaged"


def test_reconcile_closed_externally(conn):
    record_open_positions([{"symbol": "GONE", "ratio_qty": 1}], "s1", group_id="g1")
    rec = Reconciler(_broker_positions())
    report = rec.run()
    assert report.closed_externally == ["GONE"]
    with db_engine.get_session() as s:
        row = s.execute(text("SELECT status FROM managed_positions WHERE symbol = 'GONE'")).mappings().first()
    assert row["status"] == "closed"
    assert open_positions() == []


def test_reconcile_broker_failure_does_not_close_anything(conn):
    record_open_positions([{"symbol": "KEEP", "ratio_qty": 1}], "s1", group_id="g1")
    client = MagicMock()
    client.get_positions.side_effect = ConnectionError("broker down")
    report = Reconciler(client).run()
    assert report.errors
    assert open_positions() != []  # untouched


def test_backfill_exit_by_and_local_mark(tmp_path):
    from datetime import date
    from framework.execution.managed import (
        backfill_exit_by, mark_group_closed, open_groups, record_open_positions,
    )
    db_engine.configure(tmp_path / "bf.db")

    record_open_positions([
            {"symbol": "NU260814C00014000", "side": "sell", "ratio_qty": 25,
             "option_type": "call", "strike": 14.0, "expiry": date(2026, 8, 14)},
            {"symbol": "NU260911C00014000", "side": "buy", "ratio_qty": 25,
             "option_type": "call", "strike": 14.0, "expiry": date(2026, 9, 11)},
        ],
        "debit_size_exploit", group_id="nu1", entry_price=0.27,
        metadata={"side": "CALENDAR", "earnings_date": "2026-08-13"},
    )
    assert open_groups()[0].exit_by is None
    assert backfill_exit_by() == 1
    assert open_groups()[0].exit_by == date(2026, 8, 14)
    n = mark_group_closed("nu1", "test flatten", ticker="NU", strategy="debit_size_exploit")
    assert n == 2
    assert open_groups() == []


# ── Lifecycle --------------------------------------------------------------

def test_lifecycle_defaults_to_paper(conn):
    assert LifecycleManager().state("s1") == "paper"


def test_lifecycle_promote_demote(conn):
    lm = LifecycleManager()
    lm.set_state("s1", "probation", by="test")
    assert lm.state("s1") == "probation"
    assert lm.size_multiplier("s1") == 0.5
    lm.set_state("s1", "live", by="test")
    assert lm.size_multiplier("s1") == 1.0
    with pytest.raises(ValueError):
        lm.set_state("s1", "bogus")
    with db_engine.get_session() as s:
        events = s.execute(
            text("SELECT event_type FROM risk_events WHERE strategy = 's1' ORDER BY id")
        ).mappings().all()
    assert [e["event_type"] for e in events] == ["promote", "promote"]


def test_lifecycle_eligibility():
    good = {"closed_trades": 25, "win_rate": 0.55, "max_drawdown_pct": 0.05}
    assert LifecycleManager.eligible_for_promotion(good)
    assert not LifecycleManager.eligible_for_promotion({**good, "closed_trades": 5})
    assert not LifecycleManager.eligible_for_promotion({**good, "win_rate": 0.3})
    assert not LifecycleManager.eligible_for_promotion({**good, "max_drawdown_pct": 0.5})


# ── Managed positions --------------------------------------------------------

def test_record_and_query_open_positions(conn):
    legs = [
        {"symbol": "A", "side": "sell", "ratio_qty": 1, "option_type": "call",
         "strike": 100.0, "expiry": date(2026, 8, 1)},
        {"symbol": "B", "side": "buy", "ratio_qty": 1, "option_type": "call",
         "strike": 110.0, "expiry": date(2026, 9, 1)},
    ]
    n = record_open_positions(legs, "s1", group_id="g1",
                              order_id="o1", entry_price=2.5,
                              metadata={"side": "CALENDAR"})
    assert n == 2
    rows = open_positions()
    assert len(rows) == 2
    import json
    meta = json.loads(rows[0]["metadata"])
    assert meta["leg_side"] == "sell" and meta["strike"] == 100.0
    assert open_positions(strategy="other") == []


# ── Reconcile orphan baseline (Part C) ----------------------------------------

def test_reconcile_baseline_adoption_silences_preexisting(conn):
    """First reconcile ever: existing broker positions adopted, zero orphan alerts."""
    rec = Reconciler(_broker_positions(
        ("OLD1", 1, 5.0, 6.0, 600, 100), ("OLD2", 2, 1.0, 1.5, 300, 100)))
    report = rec.run()
    assert report.orphans == []  # adopted, not alerted
    with db_engine.get_session() as s:
        adopted = {r["symbol"] for r in s.execute(text("SELECT symbol FROM adopted_positions")).mappings()}
        ev = s.execute(text("SELECT * FROM risk_events WHERE event_type = 'baseline_adopted'")).mappings().first()
        rows = s.execute(text("SELECT managed FROM alpaca_positions")).mappings().all()
    assert adopted == {"OLD1", "OLD2"}
    assert ev is not None
    assert all(r["managed"] == 0 for r in rows)


def test_reconcile_new_orphan_alerts_after_baseline(conn):
    from earnings_edge.db import adopted_positions_insert
    adopted_positions_insert("OLD1", "2026-07-25")
    rec = Reconciler(_broker_positions(
        ("OLD1", 1, 5.0, 6.0, 600, 100), ("NEW1", 1, 2.0, 2.5, 250, 50)))
    report = rec.run()
    assert report.orphans == ["NEW1"]  # baseline symbol quiet, new one alerts
    with db_engine.get_session() as s:
        ev = s.execute(text("SELECT * FROM trade_events WHERE event_type = 'orphan_found'")).mappings().first()
    assert ev["symbol"] == "NEW1"
