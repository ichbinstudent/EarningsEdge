"""Tests for per-strategy signal subscriptions, routing, funnel line, setups."""
from __future__ import annotations

from earnings_edge.subscriptions import (
    SIGNAL_STRATEGIES,
    StrategySubscriptions,
    funnel_line,
    route_proposals,
)
from earnings_edge.bot_views import SETUP_STRATEGIES, setup_card, setup_menu_text


def test_default_is_subscribed(tmp_path):
    subs = StrategySubscriptions(str(tmp_path / "subs.json"))
    assert subs.is_subscribed("calendar_call_ml", 42) is True
    assert subs.recipients("calendar_call_ml", {1, 2, 3}) == {1, 2, 3}


def test_opt_out_persists_roundtrip(tmp_path):
    path = str(tmp_path / "subs.json")
    subs = StrategySubscriptions(path)
    subs.set_subscribed("vol_risk_premium", 42, False)
    assert subs.is_subscribed("vol_risk_premium", 42) is False
    assert subs.is_subscribed("vol_risk_premium", 7) is True
    assert subs.is_subscribed("calendar_call_ml", 42) is True

    subs2 = StrategySubscriptions(path)  # reload from disk
    assert subs2.is_subscribed("vol_risk_premium", 42) is False
    assert subs2.recipients("vol_risk_premium", {42, 7}) == {7}

    subs2.set_subscribed("vol_risk_premium", 42, True)  # re-subscribe
    subs3 = StrategySubscriptions(path)
    assert subs3.is_subscribed("vol_risk_premium", 42) is True


def test_unknown_strategy_rejected(tmp_path):
    subs = StrategySubscriptions(str(tmp_path / "subs.json"))
    import pytest
    with pytest.raises(KeyError):
        subs.set_subscribed("not_a_strategy", 1, False)


def test_route_proposals_respects_opt_outs(tmp_path):
    subs = StrategySubscriptions(str(tmp_path / "subs.json"))
    subs.set_subscribed("short_straddle", 2, False)
    proposals = [
        {"id": 1, "strategy": "calendar_call_ml", "ticker": "A"},
        {"id": 2, "strategy": "short_straddle", "ticker": "B"},
    ]
    routed = route_proposals(proposals, universe={1, 2}, subs=subs)
    assert [r["id"] for r in routed[1]] == [1, 2]
    assert [r["id"] for r in routed[2]] == [1]  # muted short_straddle


def test_route_proposals_override_gets_everything(tmp_path):
    subs = StrategySubscriptions(str(tmp_path / "subs.json"))
    subs.set_subscribed("short_straddle", 99, False)
    proposals = [{"id": 1, "strategy": "short_straddle", "ticker": "B"}]
    routed = route_proposals(proposals, universe={99}, subs=subs, override_chat=99)
    assert [r["id"] for r in routed[99]] == [1]  # override bypasses opt-outs


def test_route_proposals_all_muted_drops(tmp_path):
    subs = StrategySubscriptions(str(tmp_path / "subs.json"))
    subs.set_subscribed("calendar_call_ml", 1, False)
    subs.set_subscribed("calendar_call_ml", 2, False)
    proposals = [{"id": 1, "strategy": "calendar_call_ml", "ticker": "A"}]
    assert route_proposals(proposals, universe={1, 2}, subs=subs) == {}


def test_funnel_line():
    funnel = {
        "proposals": 3,
        "strategies": {
            "calendar_call_ml": {"rows_scanned": 10, "decision_pass": 4,
                                 "legs_ok": 4, "dte_ok": 3, "position_ok": 3,
                                 "proposals_created": 2},
            "vol_risk_premium": {"rows_scanned": 10, "decision_pass": 2,
                                 "legs_ok": 2, "dte_ok": 2, "position_ok": 2,
                                 "proposals_created": 1},
        },
    }
    line = funnel_line(funnel)
    assert line == "funnel: 10 scanned -> 6 decision -> 6 legs -> 5 dte -> 5 no-pos -> 3 proposals"
    assert funnel_line(None) == ""
    assert funnel_line({}) == ""
    assert funnel_line({"strategies": {}}) == ""


def test_setup_cards_contain_structure_and_toml_exits():
    for name in SETUP_STRATEGIES:
        card = setup_card(name)
        assert name in card
        assert "Exits (from TOML):" in card
    cal = setup_card("calendar_call_ml")
    assert "SELL 1x" in cal and "BUY  1x" in cal
    assert "profit target +50%" in cal and "stop loss -75%" in cal
    assert "structural deadline" in cal
    vrp = setup_card("vol_risk_premium")
    assert "SELL 1x" in vrp and "UNDEFINED RISK" in vrp
    assert "time exit 2d after entry" in vrp
    ff = setup_card("ff_ladder")
    assert "auto-close" in ff
    arb = setup_card("forward_factor_arb")
    assert "forward factor" in arb.lower()
    assert "TRADE SETUPS" in setup_menu_text()


def test_setup_card_unknown():
    assert "No setup card" in setup_card("nope")


def test_signal_strategies_are_live_mapped():
    # /signals only offers strategies that actually push cards: the
    # proposal-flow strategies (live_signals) plus ff_ladder, whose live
    # path is the LadderRunner arm-card flow rather than live_signals.
    from earnings_edge.live_signals import LIVE_STRATEGIES
    assert set(SIGNAL_STRATEGIES) <= set(LIVE_STRATEGIES) | {"ff_ladder", "forward_factor_arb"}


def test_ff_ladder_opt_out_routes(tmp_path):
    subs = StrategySubscriptions(str(tmp_path / "subs.json"))
    subs.set_subscribed("ff_ladder", 2, False)
    rows = [{"strategy": "ff_ladder", "key": 1}]
    routed = route_proposals(rows, universe={1, 2}, subs=subs)
    assert [r["key"] for r in routed[1]] == [1]
    assert 2 not in routed
