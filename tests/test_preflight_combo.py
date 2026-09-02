"""Tests for preflight_combo (proposal-time Alpaca book check)."""
from datetime import date
from unittest.mock import MagicMock

import pytest

from earnings_edge.alpaca_bridge import (
    BridgeConfig,
    StrategyBridge,
    preflight_combo,
)
from earnings_edge.trading_types import Trade


def _trade(ticker="AAPL", entry=1.85):
    return Trade(
        ticker=ticker,
        earnings_date=date(2026, 7, 29),
        scan_date=date(2026, 7, 28),
        strategy="calendar_call_ml",
        side="CALENDAR",
        entry_price=entry,
        features={
            "near_strike": 190.0, "far_strike": 190.0,
            "near_expiry": date(2026, 7, 31), "far_expiry": date(2026, 8, 28),
        },
        model_score=0.61,
        ml_decision="TAKE",
        notes="test",
    )


def _legs():
    return [
        {"symbol": "AAPL260731C00190000", "side": "buy", "ratio_qty": 1},
        {"symbol": "AAPL260828C00190000", "side": "sell", "ratio_qty": 1},
    ]


def _snap(bid, ask):
    return {"latestQuote": {"bp": bid, "ap": ask}}


def _bridge(snaps=None, raises=None):
    client = MagicMock()
    if raises:
        client.get_option_snapshots_bulk.side_effect = raises
    else:
        client.get_option_snapshots_bulk.return_value = snaps or {}
    return StrategyBridge(client=client, config=BridgeConfig())


def test_preflight_passes_tight_book():
    # net mid 0.50, net spread 0.16 -> 32% of mid: passes the 40% gate
    b = _bridge({
        "AAPL260731C00190000": _snap(5.00, 5.08),
        "AAPL260828C00190000": _snap(4.50, 4.58),
    })
    veto, mid = preflight_combo(b, _trade(), _legs())
    assert veto is None
    assert mid == pytest.approx(0.50)


def test_preflight_missing_symbol_vetoes():
    # PL case: strike exists on LSEG, absent on Alpaca
    b = _bridge({"AAPL260731C00190000": _snap(0.90, 0.94)})
    veto, mid = preflight_combo(b, _trade(), _legs())
    assert veto == "preflight: AAPL260828C00190000 not on Alpaca"
    assert mid is None


def test_preflight_one_sided_book_vetoes():
    b = _bridge({
        "AAPL260731C00190000": _snap(0.90, 0.94),
        "AAPL260828C00190000": _snap(0.0, 0.84),  # no bid
    })
    veto, mid = preflight_combo(b, _trade(), _legs())
    assert veto == "preflight: AAPL260828C00190000 no two-sided book"
    assert mid is None


def test_preflight_wide_spread_vetoes():
    # HPE case: spread 0.49 vs mid 1.21
    b = _bridge({
        "AAPL260731C00190000": _snap(1.30, 1.79),
        "AAPL260828C00190000": _snap(0.80, 0.81),
    })
    veto, mid = preflight_combo(b, _trade(), _legs())
    assert veto is not None and "spread" in veto
    assert mid is not None


def test_preflight_api_failure_vetoes():
    b = _bridge(raises=RuntimeError("422"))
    veto, mid = preflight_combo(b, _trade(), _legs())
    assert veto is not None
    assert mid is None


def test_preflight_pass_mid_is_combo_net():
    # buy near @ mid 5.04, sell far @ mid 4.54 -> net debit mid 0.50
    b = _bridge({
        "AAPL260731C00190000": _snap(5.00, 5.08),
        "AAPL260828C00190000": _snap(4.50, 4.58),
    })
    veto, mid = preflight_combo(b, _trade(), _legs())
    assert veto is None
    assert mid == pytest.approx(0.50)
