"""Tests for the live signal layer (scan session -> executable Trades)."""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from earnings_edge import live_signals
from earnings_edge.alpaca_bridge import BridgeConfig, StrategyBridge
from earnings_edge.trade_approval import PendingTradeStore, build_proposals


def _iso(dt: datetime) -> str:
    return dt.isoformat()


@pytest.fixture
def scan_db(tmp_path):
    """Tmp DB with the earnings_edge schema + a fresh scan session."""
    import sqlite3
    from earnings_edge.db import engine as db_engine

    path = tmp_path / "test.db"
    db_engine.configure(path)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    now = datetime.now(timezone.utc)
    earn = (now + timedelta(days=1)).date()
    near = earn + timedelta(days=3)
    far = earn + timedelta(days=31)
    fresh = _iso(now)
    stale = _iso(now - timedelta(hours=48))

    cols = (
        "scan_timestamp, ticker, earnings_date, tier, passed, price, strike, "
        "near_expiry, far_expiry, net_debit, net_debit_ask, net_debit_mid, "
        "iv_rv_ratio, expected_move_pct, expected_move_dollars, "
        "model_expected_return, model_decision"
    )

    def ins(ts, ticker, ed, tier, passed, price, strike, n, f, debit, ask, mid,
            iv_rv, em_pct, em_usd, score, decision):
        conn.execute(
            f"INSERT INTO scanner_scan_outputs ({cols}) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (ts, ticker, str(ed), tier, passed, price, strike, str(n), str(f),
             debit, ask, mid, iv_rv, em_pct, em_usd, score, decision),
        )

    # fresh session
    ins(fresh, "TAKECO", earn, 1, 1, 100.0, 100.0, near, far, 1.50, 1.80, 1.60,
        1.10, 5.0, 5.0, 0.25, "TAKE")
    ins(fresh, "SKIPCO", earn, 2, 1, 50.0, 50.0, near, far, 2.00, 2.40, 2.10,
        1.05, 4.0, 2.0, -0.10, "SKIP")
    ins(fresh, "CHEAP", earn, 1, 1, 40.0, 40.0, near, far, 0.90, 1.10, 1.00,
        1.08, 5.5, 2.2, None, "SKIP")  # cheap debit: 1.10/40 = 2.75% <= 3%
    ins(fresh, "EXPENSIVE", earn, 1, 1, 200.0, 200.0, near, far, 8.00, 9.00, 8.50,
        1.06, 4.5, 9.0, None, "SKIP")  # 9/200 = 4.5% > 3%
    ins(fresh, "RICHVOL", earn, 1, 1, 80.0, 80.0, near, far, 3.00, 3.50, 3.20,
        1.60, 8.0, 6.4, None, "SKIP")   # iv_rv 1.6, EM 8% -> straddle candidate
    ins(fresh, "MIDVOL", earn, 1, 1, 60.0, 60.0, near, far, 2.00, 2.50, 2.20,
        1.30, 7.0, 4.2, None, "SKIP")   # iv_rv 1.3: short_straddle only
    ins(fresh, "PASTCO", earn - timedelta(days=10), 1, 1, 30.0, 30.0, near, far,
        1.00, 1.20, 1.10, 1.20, 6.0, 1.8, 0.30, "TAKE")  # earnings in the past
    # duplicate ticker in same session: passed=0 row must lose the dedupe
    ins(fresh, "TAKECO", earn, 3, 0, 100.0, 100.0, near, far, 9.99, 9.99, 9.99,
        1.01, 1.0, 1.0, None, "SKIP")
    # stale session: must never leak into the frame
    ins(stale, "STALE", earn, 1, 1, 70.0, 70.0, near, far, 1.00, 1.20, 1.10,
        1.50, 8.0, 5.6, 0.40, "TAKE")

    conn.execute(
        "INSERT INTO live_calendar_candidates "
        "(scan_timestamp, ticker, earnings_date, straddle_price) VALUES (?,?,?,?)",
        (fresh, "RICHVOL", str(earn), 6.10),
    )
    conn.commit()
    conn.close()
    return path


def test_latest_scan_frame_fresh_session_only(scan_db):
    df = live_signals.latest_scan_frame(scan_db)
    tickers = set(df["ticker"])
    assert "STALE" not in tickers, "older sessions must not leak in"
    assert "PASTCO" not in tickers, "past earnings must be dropped"
    assert len(df) == 6  # 7 fresh rows minus the TAKECO duplicate
    # dedupe kept the passed=1 row (debit 1.80), not the passed=0 row (9.99)
    row = df[df["ticker"] == "TAKECO"].iloc[0]
    assert row["net_debit_ask"] == 1.80


def test_latest_scan_frame_stale_returns_empty(scan_db):
    df = live_signals.latest_scan_frame(scan_db, max_age_hours=1)
    # session is fresh (<1h) so this is NOT empty... force staleness instead:
    assert not df.empty
    df = live_signals.latest_scan_frame(scan_db, max_age_hours=0)
    assert df.empty


def _bridge():
    client = MagicMock()
    client.position_symbols.return_value = set()
    return StrategyBridge(client=client, config=BridgeConfig(dry_run=False))


def test_calendar_call_ml_mapping(scan_db):
    df = live_signals.latest_scan_frame(scan_db)
    trades = live_signals.build_live_trades(df, "calendar_call_ml")
    assert [t.ticker for t in trades] == ["TAKECO"]  # PASTCO dropped upstream
    t = trades[0]
    assert t.side == "CALENDAR"
    assert t.entry_price == 1.80  # combo ASK, not mid/debit
    assert t.model_score == 0.25
    # legs must build through the real bridge (real strikes + expiries)
    legs = _bridge()._build_legs(t)
    assert len(legs) == 2
    sell, buy = legs
    assert sell["side"] == "sell" and buy["side"] == "buy"
    assert sell["strike"] == buy["strike"] == 100.0
    assert sell["expiry"] < buy["expiry"]


def test_debit_size_exploit_unmapped(scan_db):
    df = live_signals.latest_scan_frame(scan_db)
    assert live_signals.build_live_trades(df, "debit_size_exploit") == []


def test_vol_risk_premium_mapping(scan_db):
    df = live_signals.latest_scan_frame(scan_db)
    trades = live_signals.build_live_trades(df, "vol_risk_premium")
    assert [t.ticker for t in trades] == ["RICHVOL"]  # MIDVOL iv_rv 1.3 < 1.4
    t = trades[0]
    assert t.side == "SHORT_STRADDLE"
    assert t.entry_price == 6.10  # straddle_price from live_calendar_candidates
    assert t.features["expected_move_dollars"] == 6.4
    legs = _bridge()._build_legs(t)
    assert {l["option_type"] for l in legs} == {"call", "put"}
    assert all(l["side"] == "sell" for l in legs)


def test_short_straddle_mapping(scan_db):
    df = live_signals.latest_scan_frame(scan_db)
    trades = live_signals.build_live_trades(df, "short_straddle")
    assert {t.ticker for t in trades} == {"RICHVOL", "MIDVOL"}  # gate 1.2


def test_unknown_strategy_returns_empty(scan_db):
    df = live_signals.latest_scan_frame(scan_db)
    assert live_signals.build_live_trades(df, "earnings_quality") == []
    assert live_signals.build_live_trades(df, "long_straddle") == []


def test_build_proposals_end_to_end_on_scan_db(scan_db):
    """Default trade source (no injection): fixture scan -> real proposals."""
    import earnings_edge.trade_approval as ta

    store = PendingTradeStore(str(scan_db))  # same tmp DB
    rows = build_proposals(store, bridge=_bridge(), db_path=str(scan_db))
    by_strat = {}
    for r in rows:
        by_strat.setdefault(r["strategy"], []).append(r["ticker"])
    assert by_strat.get("calendar_call_ml") == ["TAKECO"]
    assert "debit_size_exploit" not in by_strat
    # Live v1 TOML disables undefined-risk shorts; they must not produce cards.
    assert "short_straddle" not in by_strat
    assert "vol_risk_premium" not in by_strat
    f = ta.LAST_FUNNEL
    assert "calendar_call_ml" in f["strategies"]
    assert "debit_size_exploit" not in f["strategies"]
    assert "short_straddle" not in f["strategies"]
    assert f["strategies"]["calendar_call_ml"]["rows_scanned"] == 6
    assert f["proposals"] == len(rows)
    # cards contain real OCC legs
    card = next(r["card_text"] for r in rows if r["strategy"] == "calendar_call_ml")
    assert "TAKECO" in card and "C00100000" in card
