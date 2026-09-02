"""Tests for the realism enrichment of strategy backtest results."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from earnings_edge.backtest.enrich import enrich_result
from earnings_edge.backtest.realism import ibkr_commission
from earnings_edge.trading_types import StrategyResult, Trade

DAY0 = date(2026, 1, 5)


def _trade(pnl: float, *, premiums: list[float], margin: float,
           scan_offset: int = 0) -> Trade:
    return Trade(
        ticker="XYZ",
        earnings_date=DAY0 + timedelta(days=scan_offset + 2),
        scan_date=DAY0 + timedelta(days=scan_offset),
        strategy="iron_condor_real",
        side="IRON_CONDOR",
        entry_price=1.0,
        exit_price=0.0,
        pnl=pnl,
        pnl_pct=pnl,
        features={"leg_premiums": premiums, "margin_dollars": margin},
    )


def _result(trades: list[Trade]) -> StrategyResult:
    return StrategyResult("iron_condor_real", trades, {"total": len(trades)})


def test_enrich_deducts_ibkr_commissions_per_leg():
    # two fills at $1.20 (>= $0.10 tier -> $0.65 each), one at $0.04 (-> $0.25)
    t = _trade(2.0, premiums=[1.20, 1.20, 0.04], margin=400.0)
    out = enrich_result(_result([t]))
    expected_comm = ibkr_commission(1.20) * 2 + ibkr_commission(0.04)
    assert out.summary["commissions_total"] == pytest.approx(expected_comm)
    # gross = 2.0 * 100 = $200 on one contract
    assert out.summary["gross_total_pnl_dollars"] == pytest.approx(200.0)
    assert out.summary["net_total_pnl_dollars"] == pytest.approx(200.0 - expected_comm)


def test_enrich_contract_scaling():
    t = _trade(1.0, premiums=[1.0, 1.0], margin=400.0)
    out = enrich_result(_result([t]), contracts=3)
    assert out.summary["gross_total_pnl_dollars"] == pytest.approx(300.0)
    assert out.summary["commissions_total"] == pytest.approx(ibkr_commission(1.0, 3) * 2)


def test_enrich_return_on_margin_and_net_win_rate():
    trades = [
        _trade(1.0, premiums=[1.0], margin=500.0, scan_offset=0),   # net = 100 - 0.65
        _trade(-0.5, premiums=[1.0], margin=500.0, scan_offset=1),  # net = -50 - 0.65
    ]
    out = enrich_result(_result(trades))
    assert out.summary["net_win_rate"] == pytest.approx(0.5)
    roms = [(100 - 0.65) / 500, (-50 - 0.65) / 500]
    assert out.summary["avg_return_on_margin"] == pytest.approx(sum(roms) / 2)


def test_enrich_train_test_split_is_chronological():
    trades = [_trade(1.0 if i < 7 else -1.0, premiums=[1.0], margin=100.0,
                     scan_offset=i) for i in range(10)]
    out = enrich_result(_result(trades))
    train = out.summary["train_stats"]
    test = out.summary["test_stats"]
    assert train["count"] == 7
    assert test["count"] == 3
    assert train["mean"] > 0
    assert test["mean"] < 0


def test_enrich_missing_features_skips_margin_and_commissions():
    t = Trade("XYZ", DAY0, DAY0, "s", "SPREAD", entry_price=1.0, pnl=1.0, pnl_pct=1.0)
    out = enrich_result(_result([t]))
    # without leg_premiums there is nothing to tier commissions on
    assert "commissions_total" not in out.summary
    assert "avg_return_on_margin" not in out.summary
    # dollar P&L metrics are still reported
    assert out.summary["gross_total_pnl_dollars"] == pytest.approx(100.0)


def test_enrich_empty_result():
    out = enrich_result(StrategyResult("empty", [], {"total": 0}))
    assert out.summary.get("net_total_pnl_dollars", 0.0) == 0.0
