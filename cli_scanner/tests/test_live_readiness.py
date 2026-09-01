"""Gating tests for Alpaca live-readiness: mode switch, last-look, sizers, chain cache."""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from earnings_edge.alpaca_bridge import (
    LAST_LOOK_MAX_DEBIT_PCT_OF_SPOT,
    MAX_DEBIT_VS_MID,
    last_look_veto,
)
from earnings_edge.alpaca_mode import (
    alpaca_live_enabled,
    broker_label,
    force_approval_on_live,
    live_max_qty,
    resolve_credentials,
)
from earnings_edge.chain_cache import captured_hour, default_underlyings, row_for_contract
from sqlalchemy import text
from earnings_edge.db import insert_options_chain_rows
from framework.risk.manager import RiskManager
from framework.risk.sizing import FixedDollarSizer, SizeContext
from earnings_edge.db import configure, engine as db_engine


def test_mode_defaults_paper(monkeypatch):
    monkeypatch.delenv("ALPACA_LIVE", raising=False)
    assert alpaca_live_enabled() is False
    assert broker_label() == "paper"
    assert force_approval_on_live() is False
    assert live_max_qty(25) == 25
    key, secret, paper = resolve_credentials(paper=None)
    assert paper is True


def test_mode_live_fail_closed(monkeypatch):
    monkeypatch.setenv("ALPACA_LIVE", "1")
    monkeypatch.setenv("APCA_LIVE_API_KEY_ID", "live-k")
    monkeypatch.setenv("APCA_LIVE_API_SECRET_KEY", "live-s")
    monkeypatch.setenv("APCA_API_KEY_ID", "paper-k")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "paper-s")
    assert alpaca_live_enabled() is True
    assert broker_label() == "live"
    assert force_approval_on_live() is True
    assert live_max_qty(25) == 1
    key, secret, paper = resolve_credentials(paper=None)
    assert paper is False and key == "live-k"
    # explicit paper=True still wins (tests)
    _, _, paper2 = resolve_credentials(paper=True)
    assert paper2 is True
    monkeypatch.setenv("ALPACA_LIVE_ALLOW_AUTO", "1")
    assert force_approval_on_live() is False
    monkeypatch.setenv("ALPACA_LIVE_MAX_QTY", "2")
    assert live_max_qty(25) == 2


def test_paper_lifecycle_vetoed_on_live_broker(tmp_path):
    configure(tmp_path / "fw.db")
    rm = RiskManager()
    d = rm.check_trade(
        "calendar_call_ml", "AAPL", est_cost=100,
        equity=100_000, buying_power=50_000,
        lifecycle="paper", live_broker=True,
    )
    assert not d.approved
    assert "live broker" in d.reason


def test_live_entries_fail_closed_without_day_start(tmp_path):
    configure(tmp_path / "fw.db")
    rm = RiskManager()
    d = rm.check_trade(
        "calendar_call_ml", "AAPL", est_cost=100,
        equity=100_000, buying_power=50_000,
        lifecycle="probation", live_broker=True,
    )
    assert not d.approved
    assert "day-start" in d.reason


def test_fixed_dollar_clamps_to_pct_of_equity():
    sizer = FixedDollarSizer(2000.0, max_pct_of_equity=0.05)
    small = SizeContext(equity=7_000, buying_power=5_000, price_per_unit=250.0)
    # 5% of 7k = 350 → 1 contract, not 2000/250 = 8
    assert sizer.quantity(small) == 1
    big = SizeContext(equity=100_000, buying_power=80_000, price_per_unit=250.0)
    assert sizer.quantity(big) == 8  # budget 2000 still binds


def test_last_look_vetoes_wide_spread_and_fat_debit():
    legs = [
        {"symbol": "AAPL260828C00200000", "side": "sell"},
        {"symbol": "AAPL260918C00200000", "side": "buy"},
    ]
    tight = {
        "AAPL260828C00200000": {"latestQuote": {"bp": 2.0, "ap": 2.1}},
        "AAPL260918C00200000": {"latestQuote": {"bp": 4.0, "ap": 4.1}},
    }
    # net mid = -2.05 + 4.05 = 2.00
    assert last_look_veto(legs, tight, spot=100.0, proposed_debit=2.0) is None
    wide = {
        "AAPL260828C00200000": {"latestQuote": {"bp": 1.0, "ap": 3.0}},
        "AAPL260918C00200000": {"latestQuote": {"bp": 3.0, "ap": 5.0}},
    }
    reason = last_look_veto(legs, wide, spot=100.0)
    assert reason and "spread" in reason
    fat = last_look_veto(legs, tight, spot=10.0, proposed_debit=2.0)
    assert fat and "spot" in fat
    through = last_look_veto(
        legs, tight, spot=100.0,
        proposed_debit=2.0 * MAX_DEBIT_VS_MID + 0.5,
    )
    assert through and "mid" in through
    assert LAST_LOOK_MAX_DEBIT_PCT_OF_SPOT == 0.15


def test_hourly_chain_allows_two_hours(tmp_path):
    configure(tmp_path / "ml.db")
    now1 = datetime(2026, 8, 22, 14, 0, tzinfo=timezone.utc)
    now2 = datetime(2026, 8, 22, 15, 0, tzinfo=timezone.utc)
    snap = {"dailyBar": {"c": 1.5, "o": 1.4, "h": 1.6, "l": 1.3, "n": 1, "v": 10, "vw": 1.5},
            "latestQuote": {"bp": 1.4, "ap": 1.6, "bs": 1, "as": 1}}
    r1 = row_for_contract("run1", "AAPL", "AAPL260828C00200000", snap, now=now1)
    r2 = row_for_contract("run2", "AAPL", "AAPL260828C00200000", snap, now=now2)
    assert r1["captured_hour"] == "2026-08-22T14"
    assert r2["captured_hour"] == "2026-08-22T15"
    assert insert_options_chain_rows([r1]) == 1
    assert insert_options_chain_rows([r2]) == 1
    assert insert_options_chain_rows([r1]) == 0  # same hour ignored
    with db_engine.get_session() as s:
        n = s.execute(text("SELECT COUNT(*) FROM options_chain")).scalar()
    assert n == 2


def test_default_underlyings_prefer_upcoming(tmp_path):
    configure(tmp_path / "ml.db")
    with db_engine.session_scope() as s:
        s.execute(text(
            "INSERT INTO snapshots (ticker, earnings_date, scan_date, has_options) "
            "VALUES ('AAA', date('now','+3 days'), date('now'), 1)"
        ))
        s.execute(text(
            "INSERT INTO snapshots (ticker, earnings_date, scan_date, has_options) "
            "VALUES ('ZZZ', date('now','-30 days'), date('now'), 1)"
        ))
    tickers = default_underlyings(max_tickers=10)
    assert tickers[0] == "AAA"


def test_preflight_live_requires_flag(monkeypatch):
    import scripts.preflight as preflight
    monkeypatch.setenv("ALPACA_LIVE", "1")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "t")
    monkeypatch.setenv("APCA_API_KEY_ID", "k")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "s")
    monkeypatch.setenv("LSE_API_KEY", "l")
    client = MagicMock()
    client.paper = True
    client.base_url = "https://paper-api.alpaca.markets/v2"
    client.get_account.return_value = {"equity": "100000", "buying_power": "50000", "status": "ACTIVE"}
    client.get_clock.return_value = {"is_open": True}
    ks = MagicMock()
    ks.status.return_value = {"halted": False, "reason": None}
    with patch("earnings_edge.alpaca_trading.create_client", return_value=client), \
         patch("earnings_edge.market_data_provider.LSEProvider") as lse, \
         patch("framework.risk.killswitch.KillSwitch", return_value=ks), \
         patch("requests.get") as tg:
        lse.return_value.healthy.return_value = True
        tg.return_value.status_code = 200
        preflight.RESULTS.clear()
        assert preflight.main([]) == 1
        assert any("live confirmation" in n for n, _, _ in preflight.RESULTS)
