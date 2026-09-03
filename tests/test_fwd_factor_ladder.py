"""Tests for the FF ladder: pair selection, candidate build, runner mechanics."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict
from datetime import date, datetime, timedelta

import pytest

from earnings_edge.fwd_factor import ET, LadderSpec, occ_symbol
from earnings_edge.fwd_factor_ladder import (
    CalendarCandidate,
    LadderRunner,
    _pick_pair,
    build_candidate,
    hist_rms_move,
)
from earnings_edge.option_math import black_scholes_price

TODAY = date(2026, 7, 27)  # Monday
EARNINGS = TODAY + timedelta(days=1)
T1_EXP = TODAY + timedelta(days=45)
T2_EXP = TODAY + timedelta(days=73)
# Frozen clock for LadderRunner — arm() guards on "today ET" against stale
# earnings events; without injection the fixtures time-bomb once real time
# passes EARNINGS.
FROZEN_NOW = datetime(2026, 7, 27, 14, 0, tzinfo=ET)
SPOT = 100.0
IV1, IV2 = 0.42, 0.38

NEAR_SYM = occ_symbol("TEST", T1_EXP, SPOT)
FAR_SYM = occ_symbol("TEST", T2_EXP, SPOT)


def _bs(iv, T):
    return black_scholes_price(SPOT, SPOT, T, 0.045, iv, "call")


class FakeAlpaca:
    """Records orders; serves canned quotes."""

    def __init__(self):
        self.orders: dict[str, dict] = {}
        self.cancelled: list[str] = []
        self._seq = 0
        mid1, mid2 = _bs(IV1, 45 / 365), _bs(IV2, 73 / 365)
        self.chain = {
            NEAR_SYM: {"bid": mid1 - 0.05, "ask": mid1 + 0.05, "quote_time": "t"},
            FAR_SYM: {"bid": mid2 - 0.05, "ask": mid2 + 0.05, "quote_time": "t"},
        }
        self._fill_next = False
        self.buying_power = 1_000_000.0
        self.submit_error: Exception | None = None

    def get_account(self):
        return {"buying_power": str(self.buying_power)}

    def get_clock(self):
        return {"is_open": True}

    def get_stock_latest_trade(self, symbol):
        return SPOT

    def get_options_chain_snapshots(self, underlying, page_limit=250):
        return dict(self.chain)

    def get_option_snapshots_bulk(self, *symbols):
        return {
            s: {"latestQuote": {"bp": self.chain[s]["bid"], "ap": self.chain[s]["ask"]}}
            for s in symbols if s in self.chain
        }

    def submit_multi_leg_order(self, legs, order_type="market", time_in_force="day",
                               limit_price=None, client_order_id=None, qty=1):
        if self.submit_error is not None:
            err, self.submit_error = self.submit_error, None
            raise err
        self._seq += 1
        oid = f"order-{self._seq}"
        status = "filled" if self._fill_next else "accepted"
        self._fill_next = False
        self.orders[oid] = {
            "id": oid, "status": status, "limit_price": str(limit_price),
            "filled_avg_price": str(limit_price), "legs": legs, "qty": str(qty),
        }
        return self.orders[oid]

    def get_order(self, order_id):
        return self.orders[order_id]

    def cancel_order(self, order_id):
        self.cancelled.append(order_id)
        self.orders[order_id]["status"] = "canceled"
        return {}


@pytest.fixture
def conn(tmp_path):
    from earnings_edge.db import engine as db_engine
    p = tmp_path / "ff.db"
    db_engine.configure(p)
    c = sqlite3.connect(str(p), timeout=30)
    c.row_factory = sqlite3.Row
    for i, mv in enumerate((3.5, 4.0, 4.5, 3.8, 4.2)):
        c.execute(
            "INSERT INTO snapshots (ticker, earnings_date, scan_date, timing, "
            "actual_move_pct, outcome_fetched_at) "
            "VALUES ('TEST', ?, '2026-01-14', 'Post Market', ?, '2026-01-16')",
            (f"2026-0{i + 1}-15", mv),
        )
    c.commit()
    yield c
    c.close()


# ── selection + candidate build ----------------------------------------------

def test_pick_pair_selects_first_expiry_on_or_after_event():
    # event = today; first listed expiry is 10 DTE. T2 closest to T1+30.
    chain = dict(FakeAlpaca().chain)  # has 45 DTE + 73 DTE at ATM
    for d in (10, 32, 61, 95):
        sym = occ_symbol("TEST", TODAY + timedelta(days=d), SPOT)
        chain[sym] = {"bid": 1, "ask": 1}
    t1, t2 = _pick_pair(chain, SPOT, TODAY, event_date=TODAY)
    assert t1["expiry"] == TODAY + timedelta(days=10)
    # T1+30 = 40 DTE; 32 is 22d after T1, 45 is 35d after T1 → 45 wins
    assert t2["expiry"] == TODAY + timedelta(days=45)
    assert t1["strike"] == SPOT


def test_pick_pair_tenor_selects_within_window():
    from earnings_edge.fwd_factor_ladder import _pick_pair_tenor
    chain = dict(FakeAlpaca().chain)  # has 45 DTE + 73 DTE at ATM
    for d in (10, 32, 58, 61, 95):
        sym = occ_symbol("TEST", TODAY + timedelta(days=d), SPOT)
        chain[sym] = {"bid": 1, "ask": 1}
    
    # closest to 45 in [30, 60] -> 45 wins (already in FakeAlpaca chain)
    t1, t2 = _pick_pair_tenor(chain, SPOT, TODAY)
    assert t1["expiry"] == TODAY + timedelta(days=45)
    # T2 approx 30d after T1 (73 DTE is 28d after T1)
    assert t2["expiry"] == TODAY + timedelta(days=73)

    # Empty window -> None
    t1_none, t2_none = _pick_pair_tenor(chain, SPOT, TODAY, t1_min_days=100, t1_max_days=200)
    assert t1_none is None



def test_hist_rms_move(conn):
    rms, n = hist_rms_move(ticker="TEST")
    assert n == 5
    assert rms == pytest.approx(0.04, abs=0.002)


def test_build_candidate_happy_path(conn):
    cand = build_candidate(FakeAlpaca(), "TEST", EARNINGS, today=TODAY)
    assert cand.skip_reason is None, cand.skip_reason
    assert cand.near_symbol == NEAR_SYM and cand.far_symbol == FAR_SYM
    assert cand.sigma_fwd == pytest.approx(0.305, abs=0.01)
    assert cand.tau_days == 2
    # market near IV (0.42) is far above the 20% threshold → mid well below cap
    assert 0 < cand.mid_debit < cand.d_cap
    assert cand.d_start < cand.d_cap


def test_build_candidate_rejects_low_price(conn):
    al = FakeAlpaca()
    al.get_stock_latest_trade = lambda s: 2.50
    cand = build_candidate(al, "TEST", EARNINGS, today=TODAY)
    assert cand.skip_reason and "price" in cand.skip_reason


def test_build_candidate_rejects_thin_history(tmp_path):
    from earnings_edge.db import engine as db_engine
    db_engine.configure(tmp_path / "thin.db")
    c = sqlite3.connect(str(tmp_path / "thin.db"))
    c.execute(
        "INSERT INTO snapshots (ticker, earnings_date, scan_date, actual_move_pct, outcome_fetched_at) "
        "VALUES ('TEST', '2026-01-01', '2026-01-01', 5.0, '2026-01-01')"
    )
    c.commit()
    c.close()
    cand = build_candidate(FakeAlpaca(), "TEST", EARNINGS, today=TODAY)
    assert "hist events" in cand.skip_reason


# ── runner mechanics -----------------------------------------------------------

def _runner(conn):
    al = FakeAlpaca()
    runner = LadderRunner(al, spec=LadderSpec(), now_fn=lambda: FROZEN_NOW)
    cand = build_candidate(al, "TEST", EARNINGS, today=TODAY)
    return runner, al, cand


def test_arm_and_first_rung_places_order(conn):
    runner, al, cand = _runner(conn)
    lid = runner.arm(cand, armed_by=42)
    now = datetime(2026, 7, 27, 14, 0, tzinfo=ET)
    runner.step(now)
    assert len(al.orders) == 1
    order = next(iter(al.orders.values()))
    # rung 0 → limit == d_start (recomputed, close to original)
    assert abs(float(order["limit_price"]) - cand.d_start) < 0.05
    assert order["legs"][0]["side"] == "buy" and order["legs"][1]["side"] == "sell"
    row = conn.execute("SELECT status, order_id FROM ff_ladders WHERE id=?", (lid,)).fetchone()
    assert row[0] == "armed" and row[1] == order["id"]


def test_reprice_concedes_one_tick(conn):
    runner, al, cand = _runner(conn)
    runner.arm(cand)
    runner.step(datetime(2026, 7, 27, 14, 0, tzinfo=ET))
    first_id = next(iter(al.orders))
    runner.step(datetime(2026, 7, 27, 14, 15, tzinfo=ET))
    assert first_id in al.cancelled            # old order replaced
    assert len(al.orders) == 2
    new_order = al.orders[[k for k in al.orders if k != first_id][0]]
    assert float(new_order["limit_price"]) == pytest.approx(
        min(cand.d_start + 0.01, cand.d_cap), abs=0.05)


def test_fill_marks_ladder_filled(conn):
    runner, al, cand = _runner(conn)
    lid = runner.arm(cand)
    al._fill_next = True
    runner.step(datetime(2026, 7, 27, 14, 0, tzinfo=ET))
    status = conn.execute("SELECT status FROM ff_ladders WHERE id=?", (lid,)).fetchone()[0]
    assert status == "filled"
    assert any("filled" in e for e in runner.drain_events())


def test_runaway_mid_disarms(conn):
    runner, al, cand = _runner(conn)
    lid = runner.arm(cand)
    runner.step(datetime(2026, 7, 27, 14, 0, tzinfo=ET))
    # market runs away: far leg dumps, combo mid far above cap
    al.chain[FAR_SYM]["bid"] = 1.0
    al.chain[FAR_SYM]["ask"] = 1.1
    runner.step(datetime(2026, 7, 27, 14, 15, tzinfo=ET))
    status = conn.execute("SELECT status FROM ff_ladders WHERE id=?", (lid,)).fetchone()[0]
    assert status == "disarmed"
    assert any("disarmed" in e for e in runner.drain_events())


def test_no_order_outside_window(conn):
    runner, al, cand = _runner(conn)
    runner.arm(cand)
    runner.step(datetime(2026, 7, 27, 11, 0, tzinfo=ET))  # before 14:00 ET
    assert len(al.orders) == 0


# ── hardening: data issues, funds, kill switch, staleness --------------------

def test_invalid_quotes_hold_silently(conn):
    runner, al, cand = _runner(conn)
    lid = runner.arm(cand)
    al.chain[NEAR_SYM]["bid"] = 0.0           # crossed/degenerate quote
    runner.step(datetime(2026, 7, 27, 14, 0, tzinfo=ET))
    assert len(al.orders) == 0                # no order placed on bad data
    status = conn.execute("SELECT status FROM ff_ladders WHERE id=?", (lid,)).fetchone()[0]
    assert status == "armed"                  # held, not killed
    assert runner.drain_events() == []        # silent — no Telegram spam


def test_stale_quote_timestamp_holds(conn):
    runner, al, cand = _runner(conn)
    runner.arm(cand)
    old = "2026-07-27T12:00:00Z"              # 2h before the 14:00 ET step
    al.get_option_snapshots_bulk = lambda *s: {
        sym: {"latestQuote": {"bp": al.chain[sym]["bid"], "ap": al.chain[sym]["ask"], "t": old}}
        for sym in s if sym in al.chain
    }
    runner.step(datetime(2026, 7, 27, 14, 0, tzinfo=ET))
    assert len(al.orders) == 0


def test_terminal_submit_error_disarms(conn):
    from earnings_edge.alpaca_trading import AlpacaError
    runner, al, cand = _runner(conn)
    lid = runner.arm(cand)
    al.submit_error = AlpacaError(403, "insufficient buying power")
    runner.step(datetime(2026, 7, 27, 14, 0, tzinfo=ET))
    status = conn.execute("SELECT status FROM ff_ladders WHERE id=?", (lid,)).fetchone()[0]
    assert status == "disarmed"
    assert any("broker rejected" in e for e in runner.drain_events())


def test_transient_submit_error_keeps_armed(conn):
    import requests
    runner, al, cand = _runner(conn)
    lid = runner.arm(cand)
    al.submit_error = requests.ConnectionError("network down")
    runner.step(datetime(2026, 7, 27, 14, 0, tzinfo=ET))
    status = conn.execute("SELECT status FROM ff_ladders WHERE id=?", (lid,)).fetchone()[0]
    assert status == "armed"                  # retried next step, not killed


def test_low_buying_power_blocks_placement(conn):
    runner, al, cand = _runner(conn)
    lid = runner.arm(cand)
    al.buying_power = 10.0                    # way below worst-case cost
    runner.step(datetime(2026, 7, 27, 14, 0, tzinfo=ET))
    assert len(al.orders) == 0
    events = runner.drain_events()
    assert any("buying power" in e for e in events)
    # second step: no duplicate warning
    runner.step(datetime(2026, 7, 27, 14, 15, tzinfo=ET))
    assert runner.drain_events() == []


def test_arm_refused_on_low_buying_power(conn):
    runner, al, cand = _runner(conn)
    al.buying_power = 1.0
    lid = runner.arm(cand)
    assert lid is None
    assert any("buying power" in e for e in runner.drain_events())
    assert conn.execute("SELECT COUNT(*) FROM ff_ladders").fetchone()[0] == 0


def test_arm_dedupe_per_ticker(conn):
    runner, al, cand = _runner(conn)
    lid1 = runner.arm(cand)
    lid2 = runner.arm(cand)
    assert lid1 is not None and lid2 is None
    assert conn.execute("SELECT COUNT(*) FROM ff_ladders WHERE status='armed'").fetchone()[0] == 1


def test_kill_switch_disarms_all(conn):
    runner, al, cand = _runner(conn)
    lid = runner.arm(cand)
    from framework.risk.killswitch import KillSwitch
    KillSwitch().trip("test halt", by="test")
    runner.step(datetime(2026, 7, 27, 14, 0, tzinfo=ET))
    status = conn.execute("SELECT status FROM ff_ladders WHERE id=?", (lid,)).fetchone()[0]
    assert status == "disarmed"
    assert any("Kill switch" in e for e in runner.drain_events())


def test_spot_drift_disarms(conn):
    runner, al, cand = _runner(conn)
    lid = runner.arm(cand)
    al.get_stock_latest_trade = lambda s: SPOT * 1.05   # +5% — strike no longer ATM
    runner.step(datetime(2026, 7, 27, 14, 0, tzinfo=ET))
    status = conn.execute("SELECT status FROM ff_ladders WHERE id=?", (lid,)).fetchone()[0]
    assert status == "disarmed"
    assert any("drifted" in e for e in runner.drain_events())


def test_stale_ladder_from_yesterday_expires(conn):
    runner, al, _ = _runner(conn)
    # candidate whose earnings event is in the past relative to the FROZEN
    # clock (arm() reads "today ET" from now_fn, which the fixture froze)
    frozen_past = TODAY - timedelta(days=1)
    stale_cand = build_candidate(al, "TEST", frozen_past, today=TODAY)
    lid = runner.arm(stale_cand)
    assert lid is None  # arm refused for a past event
    # and even if one exists in the DB (e.g. armed yesterday, bot restart),
    # the step guard expires it without trading
    past = CalendarCandidate(**{**asdict(stale_cand), "skip_reason": None})
    conn.execute(
        "INSERT INTO ff_ladders (ticker, candidate_json, status) VALUES (?, ?, 'armed')",
        ("TEST", json.dumps(asdict(past))),
    )
    conn.commit()
    runner.step(datetime(2026, 7, 27, 14, 0, tzinfo=ET))
    status = conn.execute("SELECT status FROM ff_ladders ORDER BY id DESC LIMIT 1").fetchone()[0]
    assert status == "expired"
    assert len(al.orders) == 0                # never traded a stale candidate
    assert any("expired" in e for e in runner.drain_events())


def test_expired_ladder_books_fill_if_order_filled(conn):
    """A resting order that filled after we decided to expire must be booked."""
    runner, al, _ = _runner(conn)
    frozen_past = TODAY - timedelta(days=1)
    stale_cand = build_candidate(al, "TEST", frozen_past, today=TODAY)
    past = CalendarCandidate(**{**asdict(stale_cand), "skip_reason": None})
    al.orders["order-fill"] = {
        "id": "order-fill", "status": "filled", "filled_qty": "1",
        "filled_avg_price": "1.25", "legs": [],
    }
    conn.execute(
        "INSERT INTO ff_ladders (ticker, candidate_json, status, order_id) "
        "VALUES (?, ?, 'armed', ?)",
        ("TEST", json.dumps(asdict(past)), "order-fill"),
    )
    conn.commit()
    runner.step(datetime(2026, 7, 27, 14, 0, tzinfo=ET))
    status = conn.execute("SELECT status FROM ff_ladders ORDER BY id DESC LIMIT 1").fetchone()[0]
    assert status == "filled"
    assert any("filled" in e for e in runner.drain_events())


def test_conn_factory_fresh_connections(conn, tmp_path):
    """Session-per-op: stepping from another thread must not raise."""
    import threading
    from earnings_edge.db import engine as db_engine
    db = tmp_path / "t.db"
    db_engine.configure(db)
    shared = sqlite3.connect(str(db))
    for mv in (3.5, 4.0, 4.5, 3.8, 4.2):
        shared.execute(
            "INSERT INTO snapshots (ticker, earnings_date, scan_date, actual_move_pct, outcome_fetched_at) "
            "VALUES ('TEST', '2026-01-15', '2026-01-14', ?, '2026-01-16')",
            (mv,),
        )
    shared.commit()
    shared.close()
    al = FakeAlpaca()
    runner = LadderRunner(al, db_path=db, spec=LadderSpec(), now_fn=lambda: FROZEN_NOW)
    cand = build_candidate(al, "TEST", EARNINGS, today=TODAY)
    lid = runner.arm(cand)
    assert lid is not None
    t = threading.Thread(target=lambda: runner.step(datetime(2026, 7, 27, 14, 0, tzinfo=ET)))
    t.start(); t.join(timeout=30)
    assert not t.is_alive()
    rows = sqlite3.connect(str(db)).execute("SELECT status FROM ff_ladders").fetchall()
    assert rows[0][0] == "armed"
