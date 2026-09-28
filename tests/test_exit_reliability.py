"""Exit-reliability regressions (2026-09 incident: calendars never closed,
COST's near leg was assigned into -100 shares).

The fake broker fills realistically: a single-leg buy fills only at/above
the ask, a sell only at/below the bid, and combo (multi-leg) orders never
fill — the wide-book situation where the old 1%-off-mid walk stalled.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from unittest.mock import MagicMock

import pytest

from earnings_edge.db import engine as db_engine
from earnings_edge.db import managed_positions_list
from earnings_edge.fwd_factor_ladder import CalendarCandidate, LadderRunner
from framework.core.config import StrategyConfig
from framework.core.registry import StrategyRegistry
from framework.execution.managed import open_groups, record_open_positions
from framework.execution.order_manager import NaturalWalkPolicy, OrderManager
from framework.positions.manager import ExitManager, _sessions_after_reaction

TODAY = date(2026, 9, 28)  # Monday
NEAR = "COST260925C00900000"  # expired last Friday
NEAR_LIVE = "COST260928C00900000"  # expires today
FAR = "COST261023C00900000"


class Broker:
    def __init__(self, quotes: dict[str, tuple[float, float]], positions=(), unfillable=(), last=924.20):
        self.quotes = quotes
        self.positions = list(positions)
        self.unfillable = set(unfillable)
        self.last = last
        self.orders: dict[str, dict] = {}
        self.submitted: list[tuple[str, str, float, float]] = []  # (symbol, side, qty, limit)
        self._n = 0

    # data
    def get_option_snapshots_bulk(self, *symbols):
        return {
            s: {"latestQuote": {"bp": self.quotes[s][0], "ap": self.quotes[s][1]}}
            for s in symbols
            if s in self.quotes
        }

    def get_positions(self):
        return list(self.positions)

    def get_stock_latest_trade(self, ticker):
        return self.last

    def get_clock(self):
        now = datetime.now(UTC)
        return {
            "is_open": True,
            "timestamp": now.isoformat(),
            "next_close": (now + timedelta(hours=3)).isoformat(),
        }

    # orders
    def _new(self, status, **fields):
        self._n += 1
        oid = f"o{self._n}"
        self.orders[oid] = {"id": oid, "status": status, **fields}
        return self.orders[oid]

    def submit_multi_leg_order(self, legs, qty, order_type, limit_price, time_in_force, client_order_id):
        self.submitted.append(("MLEG", "combo", qty, limit_price))
        return self._new("accepted", filled_qty=0, limit_price=limit_price)

    def submit_order(self, symbol, qty, side, order_type, limit_price, time_in_force, client_order_id):
        self.submitted.append((symbol, side, qty, limit_price))
        bid, ask = self.quotes.get(symbol, (self.last, self.last))
        ok = symbol not in self.unfillable and (limit_price >= ask if side == "buy" else limit_price <= bid)
        if ok:
            return self._new("filled", filled_qty=qty, filled_avg_price=limit_price)
        return self._new("accepted", filled_qty=0, limit_price=limit_price)

    def get_order(self, oid):
        return self.orders[oid]

    def cancel_order(self, oid):
        self.orders[oid]["status"] = "canceled"
        return {}


def _seed(near=NEAR_LIVE, near_expiry=TODAY, exit_by=TODAY, strategy="ff_ladder", timing="Post Market"):
    record_open_positions(
        [
            {
                "symbol": near,
                "side": "sell",
                "ratio_qty": 1,
                "option_type": "call",
                "strike": 900.0,
                "expiry": near_expiry,
            },
            {
                "symbol": FAR,
                "side": "buy",
                "ratio_qty": 1,
                "option_type": "call",
                "strike": 900.0,
                "expiry": date(2026, 10, 23),
            },
        ],
        strategy,
        group_id="g-cost",
        entry_price=19.80,
        metadata={"side": "CALENDAR", "credit": False, "earnings_date": "2026-09-24", "timing": timing},
        exit_by=exit_by,
    )


def _mgr(broker, exits=None):
    cfg = StrategyConfig(name="ff_ladder", exits=exits or [])
    return ExitManager(
        broker,
        registry=StrategyRegistry(configs={"ff_ladder": cfg}),
        order_manager=OrderManager(broker, poll_secs=0, sleep=lambda s: None),
        today=TODAY,
    )


@pytest.fixture(autouse=True)
def db(tmp_path):
    db_engine.configure(tmp_path / "exits.db")
    from framework.alerts import DEDUPER

    DEDUPER.reset()


# ── bookkeeping ───────────────────────────────────────────────────────────


def test_ladder_fill_books_exit_by_and_the_proposing_strategy():
    """_record_fill booked FF calendars without exit_by, so ScheduledExit
    never fired; arb fills were booked as ff_ladder."""
    cand = CalendarCandidate(
        ticker="COST", earnings_date="2026-09-24", spot=905.0, strike=900.0,
        near_symbol=NEAR, far_symbol=FAR, near_expiry="2026-09-25", far_expiry="2026-10-23",
        near_bid=6, near_ask=6.2, far_bid=25, far_ask=26, sigma_fwd=0.2, hist_rms_move=0.05,
        tau_days=1, d_start=19.7, d_cap=19.9, mid_debit=9.96, strategy_override="forward_factor_arb",
        timing="Post Market",
    )  # fmt: skip
    LadderRunner(MagicMock())._record_fill(cand, {"id": "fill1", "filled_avg_price": "19.80"})
    rows = managed_positions_list()
    assert {r["exit_by"] for r in rows} == {"2026-09-25"}
    assert {r["strategy"] for r in rows} == {"forward_factor_arb"}
    assert open_groups()[0].timing == "Post Market"


def test_exit_eval_backfills_missing_exit_by():
    _seed(exit_by=None)
    assert open_groups()[0].exit_by is None
    _mgr(Broker({NEAR_LIVE: (60.0, 62.0), FAR: (70.0, 72.0)})).evaluate_all()
    assert open_groups()[0].exit_by == TODAY


# ── reaction session ─────────────────────────────────────────────────────


def test_reaction_session_depends_on_timing():
    from framework.core.calendar import get_calendar

    cal = get_calendar()
    thu, fri, mon = date(2026, 9, 24), date(2026, 9, 25), date(2026, 9, 28)
    assert _sessions_after_reaction(cal, thu, "Pre Market", thu) == 0  # BMO reacts same day
    assert _sessions_after_reaction(cal, thu, "Post Market", thu) == -1  # AMC: not yet
    assert _sessions_after_reaction(cal, thu, "Post Market", fri) == 0
    assert _sessions_after_reaction(cal, thu, None, fri) == 0  # unknown → like AMC
    assert _sessions_after_reaction(cal, fri, "Post Market", mon) == 0  # over the weekend
    assert _sessions_after_reaction(cal, thu, "Post Market", mon) == 1


# ── natural walk ──────────────────────────────────────────────────────────


def test_natural_walk_reaches_the_far_side():
    p = NaturalWalkPolicy(5.0, 6.0, steps=3)
    assert p.walk(5.5, "buy") == [5.5, 5.75, 6.0]
    assert p.walk(5.5, "sell") == [5.5, 5.25, 5.0]
    assert NaturalWalkPolicy(0.0, 0.10).walk(0.05, "sell")[-1] == 0.01  # never a zero/negative limit


# ── leg-out on the deadline ───────────────────────────────────────────────


def test_expiry_day_close_legs_out_short_first_when_combo_stalls():
    """COST 2026-09-25: the combo close never filled on a wide book, the
    near leg expired in the money and was assigned."""
    _seed()
    broker = Broker(
        {NEAR_LIVE: (60.0, 64.0), FAR: (70.0, 76.0)},
        positions=[{"symbol": NEAR_LIVE, "qty": "-1"}, {"symbol": FAR, "qty": "1"}],
    )
    mo = _mgr(broker).close_group(open_groups()[0], reason="scheduled", leg_out=True)
    assert mo.state == "filled"
    singles = [s for s in broker.submitted if s[0] != "MLEG"]
    assert singles[0][:2] == (NEAR_LIVE, "buy")  # the assignment risk goes first
    assert (NEAR_LIVE, "buy", 1, 64.0) in singles  # reached the ask
    assert (FAR, "sell", 1, 70.0) in singles  # reached the bid
    assert open_groups() == []


def test_long_hedge_kept_when_short_cannot_be_bought_back():
    _seed()
    broker = Broker(
        {NEAR_LIVE: (60.0, 64.0), FAR: (70.0, 76.0)},
        positions=[{"symbol": NEAR_LIVE, "qty": "-1"}, {"symbol": FAR, "qty": "1"}],
        unfillable={NEAR_LIVE},
    )
    mo = _mgr(broker).close_group(open_groups()[0], reason="scheduled", leg_out=True)
    assert mo.state == "exhausted"
    assert not [s for s in broker.submitted if s[0] == FAR]  # never sold the hedge
    assert {leg.symbol for leg in open_groups()[0].legs} == {NEAR_LIVE, FAR}
    from framework.alerts import DEDUPER

    assert any("urgent close incomplete" in m for m in DEDUPER.drain())


def test_leg_out_never_buys_a_short_the_broker_no_longer_holds():
    """Buying back an already-assigned short would OPEN a long call."""
    _seed()
    broker = Broker({NEAR_LIVE: (60.0, 64.0), FAR: (70.0, 76.0)}, positions=[{"symbol": FAR, "qty": "1"}])
    _mgr(broker).close_group(open_groups()[0], reason="scheduled", leg_out=True)
    assert not [s for s in broker.submitted if s[0] == NEAR_LIVE]
    assert open_groups() == []


def test_without_leg_out_a_wide_book_is_not_crossed():
    _seed(exit_by=TODAY + timedelta(days=5))
    broker = Broker({NEAR_LIVE: (60.0, 64.0), FAR: (70.0, 76.0)})
    mo = _mgr(broker).close_group(open_groups()[0], reason="profit target")
    assert mo.state != "filled"
    assert all(s[0] == "MLEG" for s in broker.submitted)


# ── assignment ────────────────────────────────────────────────────────────


def test_assigned_shares_are_covered_and_the_calendar_closed():
    """Monday after an ITM near-leg assignment: -100 COST shares + the long far."""
    _seed(near=NEAR, near_expiry=date(2026, 9, 25), exit_by=date(2026, 9, 25))
    broker = Broker(
        {FAR: (32.0, 33.6)},
        positions=[{"symbol": "COST", "qty": "-100"}, {"symbol": FAR, "qty": "1"}],
    )
    out = _mgr(broker).evaluate_all()
    assert broker.submitted[0][:3] == ("COST", "buy", 100)
    assert (FAR, "sell", 1, 32.0) in broker.submitted
    assert not [s for s in broker.submitted if s[0] == NEAR]
    assert open_groups() == []
    assert any("ASSIGNMENT" in m for m in out["auto_closed"])


@pytest.mark.parametrize(
    "positions",
    [
        [{"symbol": "COST", "qty": "50"}, {"symbol": FAR, "qty": "1"}],  # long stock: not from a short call
        [{"symbol": "KVUE", "qty": "-112"}, {"symbol": FAR, "qty": "1"}],  # another system's book
    ],
)
def test_foreign_or_long_stock_is_left_alone(positions):
    _seed(exit_by=TODAY + timedelta(days=5))
    broker = Broker({NEAR_LIVE: (60.0, 64.0), FAR: (70.0, 76.0)}, positions=positions)
    _mgr(broker).evaluate_all()
    assert not [s for s in broker.submitted if s[0] in ("COST", "KVUE")]


def test_assignment_cover_is_capped_at_the_contracts_held():
    _seed(near=NEAR, near_expiry=date(2026, 9, 25), exit_by=date(2026, 9, 25))
    broker = Broker(
        {FAR: (32.0, 33.6)},
        positions=[{"symbol": "COST", "qty": "-300"}, {"symbol": FAR, "qty": "1"}],
    )
    _mgr(broker).evaluate_all()
    assert broker.submitted[0][:3] == ("COST", "buy", 100)
