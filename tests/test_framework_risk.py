"""Tests for the framework risk layer: sizers, kill switch, risk manager, equity."""

from __future__ import annotations

from datetime import date, datetime, timezone
from unittest.mock import MagicMock

import pytest

from framework.risk.equity import daily_pnl, day_start_equity, latest_equity, snapshot_equity
from framework.risk.killswitch import KillSwitch
from framework.risk.manager import RiskLimits, RiskManager
from framework.risk.sizing import (
    FixedDollarSizer, PercentOfPortfolioSizer, SizeContext, VolTargetSizer, build_sizer,
)
from sqlalchemy import text
from earnings_edge.db import engine as db_engine


@pytest.fixture
def conn(tmp_path):
    db_engine.configure(tmp_path / "fw.db")


CTX = SizeContext(equity=100_000, buying_power=80_000,
                  price_per_unit=250.0, max_loss_per_unit=250.0)


# ── Sizers ---------------------------------------------------------------

def test_fixed_dollar_sizer():
    assert FixedDollarSizer(1000).quantity(CTX) == 4
    assert FixedDollarSizer(100).quantity(CTX) == 0  # budget < price → veto


def test_pct_portfolio_sizer():
    assert PercentOfPortfolioSizer(0.05).quantity(CTX) == 20  # 5k / 250
    with pytest.raises(ValueError):
        PercentOfPortfolioSizer(1.5)


def test_vol_target_sizer():
    assert VolTargetSizer(0.02).quantity(CTX) == 8  # 2k risk / 250
    naked = SizeContext(equity=100_000, buying_power=80_000,
                        price_per_unit=500.0, max_loss_per_unit=None)
    assert VolTargetSizer(0.02).quantity(naked) == 4  # falls back to price


def test_vol_target_qty_at_least_one_when_stress_fits_budget():
    """2x expected-move stress under the 1% / $100k budget → qty ≥ 1."""
    stress = 800.0  # 2 * $4 EM * 100
    ctx = SizeContext(equity=100_000, buying_power=80_000,
                      price_per_unit=stress, max_loss_per_unit=stress)
    assert VolTargetSizer(0.01).quantity(ctx) >= 1


def test_build_sizer_from_config():
    s = build_sizer("pct_portfolio", {"pct": 0.05})
    assert isinstance(s, PercentOfPortfolioSizer)
    with pytest.raises(ValueError):
        build_sizer("nope", {})


# ── Kill switch ------------------------------------------------------------

def test_killswitch_trip_resume_persists(conn):
    ks = KillSwitch()
    assert ks.is_halted() is False
    ks.trip("test halt", by="tester")
    assert ks.is_halted() is True
    assert KillSwitch().is_halted() is True  # survives re-read
    status = ks.status()
    assert status["reason"] == "test halt" and status["tripped_by"] == "tester"
    ks.resume(by="tester")
    assert ks.is_halted() is False


# ── Risk manager ------------------------------------------------------------

def test_check_trade_approves_within_limits(conn):
    rm = RiskManager()
    d = rm.check_trade("s1", "AAPL", est_cost=1000, equity=100_000, buying_power=50_000)
    assert d.approved and d.qty_multiplier == 1.0


def test_check_trade_vetoes_when_halted(conn):
    rm = RiskManager()
    rm.killswitch.trip("manual", by="test")
    d = rm.check_trade("s1", "AAPL", est_cost=100, equity=100_000, buying_power=50_000)
    assert not d.approved and "kill switch" in d.reason


def test_check_trade_per_trade_cap(conn):
    rm = RiskManager(RiskLimits(max_pct_per_trade=0.10))
    d = rm.check_trade("s1", "AAPL", est_cost=6000, equity=100_000, buying_power=50_000)
    assert not d.approved and "buying power" in d.reason


def test_check_trade_min_buying_power(conn):
    rm = RiskManager(RiskLimits(min_buying_power=10_000))
    d = rm.check_trade("s1", "AAPL", est_cost=100, equity=100_000, buying_power=5000)
    assert not d.approved


def test_check_trade_underlying_cap(conn):
    rm = RiskManager(RiskLimits(max_pct_per_underlying=0.25))
    d = rm.check_trade("s1", "AAPL", est_cost=6000, equity=100_000,
                       buying_power=500_000, underlying_exposure=20_000)
    assert not d.approved and "underlying" in d.reason


def test_check_trade_strategy_daily_budget(conn):
    rm = RiskManager(RiskLimits(max_pct_per_strategy_day=0.30))
    rm.record_entry("s1", "AAPL", 28_000)
    d = rm.check_trade("s1", "MSFT", est_cost=5000, equity=100_000, buying_power=500_000)
    assert not d.approved and "strategy daily spend" in d.reason
    # other strategies unaffected
    d2 = rm.check_trade("s2", "MSFT", est_cost=5000, equity=100_000, buying_power=500_000)
    assert d2.approved


def test_check_trade_probation_multiplier(conn):
    rm = RiskManager()
    d = rm.check_trade("s1", "AAPL", est_cost=100, equity=100_000,
                       buying_power=50_000, lifecycle="probation")
    assert d.approved and d.qty_multiplier == 0.5


def test_check_trade_paper_lifecycle_rejected_for_execution(conn):
    rm = RiskManager()
    d = rm.check_trade("s1", "AAPL", est_cost=100, equity=100_000,
                       buying_power=50_000, lifecycle="unknown")
    assert not d.approved


def test_vetoes_are_audited(conn):
    rm = RiskManager()
    rm.check_trade("s1", "AAPL", est_cost=6000, equity=100_000, buying_power=50_000)
    with db_engine.get_session() as s:
        rows = s.execute(text("SELECT * FROM risk_events WHERE event_type = 'veto'")).mappings().all()
    assert len(rows) == 1 and rows[0]["strategy"] == "s1"


def test_daily_loss_trips_killswitch(conn):
    rm = RiskManager(RiskLimits(daily_loss_limit_pct=0.05))
    today = datetime.now(timezone.utc).date().isoformat()
    with db_engine.session_scope() as s:
        s.execute(
            text(
                "INSERT INTO equity_snapshots (ts, equity, buying_power, portfolio_value) "
                "VALUES (:ts, :eq, :bp, :pv)"
            ),
            {"ts": today + "T13:00:00+00:00", "eq": 100_000, "bp": 90_000, "pv": 100_000},
        )
    assert rm.check_daily_loss(96_000) is False  # -4% < 5%
    assert rm.check_daily_loss(94_500) is True   # -5.5% → trip
    assert rm.killswitch.is_halted()


def test_rejection_streak_trips_killswitch(conn):
    rm = RiskManager(RiskLimits(max_consecutive_rejections=3))
    rm.record_entry("s1", "AAPL", 100)  # break candidate: entry between rejections
    rm.record_broker_rejection("s1", "r1")
    rm.record_broker_rejection("s1", "r2")
    assert rm.killswitch.is_halted() is False
    rm.record_broker_rejection("s1", "r3")
    # entry broke the streak (entry is most recent non-rejection before r3? no:
    # order is entry, r1, r2, r3 → last 3 are r1,r2,r3 all rejections → trip)
    assert rm.killswitch.is_halted()


def test_resume_resets_rejection_streak(conn):
    """/resume must actually clear the consecutive-rejection counter, not
    just the halted flag — a single stray rejection right after resuming
    should not immediately re-trip the kill switch."""
    rm = RiskManager(RiskLimits(max_consecutive_rejections=3))
    rm.record_broker_rejection("s1", "r1")
    rm.record_broker_rejection("s1", "r2")
    rm.record_broker_rejection("s1", "r3")
    assert rm.killswitch.is_halted()

    rm.killswitch.resume("operator")
    assert rm.killswitch.is_halted() is False

    rm.record_broker_rejection("s1", "r4")
    assert rm.killswitch.is_halted() is False


def test_gcd_reject_does_not_increment_kill_switch_streak(conn):
    from framework.risk.manager import is_gcd_reject
    assert is_gcd_reject("leg ratio quantities should be relatively prime: GCD[11 11] = 11")
    rm = RiskManager(RiskLimits(max_consecutive_rejections=3))
    detail = "leg ratio quantities should be relatively prime: GCD[11 11] = 11"
    assert rm.record_broker_rejection("s1", detail) == 0
    assert rm.record_broker_rejection("s1", detail) == 0
    assert rm.record_broker_rejection("s1", detail) == 0
    assert rm.killswitch.is_halted() is False
    with db_engine.get_session() as s:
        kinds = [r["event_type"] for r in s.execute(text("SELECT event_type FROM risk_events")).mappings().all()]
    assert kinds == ["gcd_reject", "gcd_reject", "gcd_reject"]


# ── Equity -------------------------------------------------------------------

def _stub_client(equity=100_000, bp=80_000):
    c = MagicMock()
    c.get_account.return_value = {
        "equity": str(equity), "buying_power": str(bp), "portfolio_value": str(equity),
    }
    return c


def test_snapshot_and_daily_pnl(conn):
    snapshot_equity(_stub_client(100_000))
    snapshot_equity(_stub_client(101_500))
    latest = latest_equity()
    assert latest["equity"] == 101_500
    assert day_start_equity() == 100_000
    assert daily_pnl(101_500) == 1500


def test_daily_pnl_none_without_baseline(conn):
    assert daily_pnl(100_000) is None
