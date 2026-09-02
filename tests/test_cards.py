"""Tests for earnings_edge.cards: the shared HTML card/grouping renderer
used by trade_approval (entries), framework.positions.manager (exits), and
bot.py's batched Telegram pushes."""
from __future__ import annotations

from earnings_edge import cards


def test_esc_escapes_html_specials():
    assert cards.esc("<script>&\"'") == "&lt;script&gt;&amp;\"'"


def test_esc_leaves_underscores_alone():
    # The whole reason cards moved off Markdown: strategy names and OCC
    # symbols are full of underscores, which Telegram's Markdown parser
    # treats as formatting and silently drops the message over.
    assert cards.esc("calendar_call_ml") == "calendar_call_ml"
    assert cards.esc("AAPL260731C00190000") == "AAPL260731C00190000"


def test_bold_and_code_wrap_and_escape():
    assert cards.bold("a&b") == "<b>a&amp;b</b>"
    assert cards.code("a<b>") == "<code>a&lt;b&gt;</code>"


def test_header_combines_emoji_and_bold_title():
    assert cards.header("📋", "Trade Proposal #1") == "📋 <b>Trade Proposal #1</b>"


def test_card_frame_assembles_all_sections():
    text = cards.card_frame("📋", "Title", "Subtitle", ["line1", "line2"], "Footer")
    assert text == "📋 <b>Title</b>\nSubtitle\nline1\nline2\nFooter"


def test_card_frame_omits_empty_subtitle_and_footer():
    text = cards.card_frame("📋", "Title", "", ["line1"])
    assert text == "📋 <b>Title</b>\nline1"


def test_group_by_strategy_preserves_first_seen_order():
    rows = [
        {"strategy": "b", "ticker": "T1"},
        {"strategy": "a", "ticker": "T2"},
        {"strategy": "b", "ticker": "T3"},
    ]
    groups = cards.group_by_strategy(rows)
    assert list(groups.keys()) == ["b", "a"]
    assert [r["ticker"] for r in groups["b"]] == ["T1", "T3"]
    assert [r["ticker"] for r in groups["a"]] == ["T2"]


def test_entry_summary_line_includes_score_when_present():
    row = {"ticker": "AAPL", "side": "CALENDAR", "model_score": 0.842}
    line = cards.entry_summary_line(row)
    assert "AAPL" in line and "CALENDAR" in line and "0.842" in line


def test_entry_summary_line_omits_score_when_absent():
    row = {"ticker": "AAPL", "side": "CALENDAR", "model_score": None}
    assert "ML" not in cards.entry_summary_line(row)


def test_exit_summary_line_includes_rule_and_reason():
    row = {"ticker": "MSFT", "rule": "profit_target", "reason": "+22%"}
    line = cards.exit_summary_line(row)
    assert "MSFT" in line and "profit_target" in line and "+22%" in line


def test_group_message_entry_header_and_lines():
    rows = [
        {"strategy": "calendar_call_ml", "ticker": "AAPL", "side": "CALENDAR", "model_score": 0.9},
        {"strategy": "calendar_call_ml", "ticker": "MSFT", "side": "CALENDAR", "model_score": 0.5},
    ]
    text = cards.group_message("calendar_call_ml", rows, "entry")
    assert "calendar_call_ml" in text
    assert "2 proposals pending" in text
    assert "AAPL" in text and "MSFT" in text


def test_group_message_singular_noun():
    rows = [{"strategy": "s1", "ticker": "AAPL", "side": "CALENDAR", "model_score": None}]
    text = cards.group_message("s1", rows, "entry")
    assert "1 proposal pending" in text


def test_group_message_exit_kind():
    rows = [{"strategy": "vol_risk_premium", "ticker": "AAPL", "rule": "stop_loss", "reason": "-40%"}]
    text = cards.group_message("vol_risk_premium", rows, "exit")
    assert "1 exit pending" in text
    assert "stop_loss" in text


def test_batch_overview_totals_and_per_strategy_counts():
    text = cards.batch_overview({"calendar_call_ml": 2, "ff_ladder": 1}, "entry")
    assert "3 proposals pending" in text
    assert "calendar_call_ml: 2" in text
    assert "ff_ladder: 1" in text


def test_batch_overview_appends_escaped_extra():
    text = cards.batch_overview({"s1": 1}, "entry", extra="note <danger>")
    assert "note &lt;danger&gt;" in text
