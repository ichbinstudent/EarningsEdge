"""Tests for the /settings surface added to bot.py: risk/halt/resume,
strategy lifecycle stepping, and the settings/lifecycle keyboard builders.

These call TradingBot's instance methods against a lightweight stand-in
that only implements _fw() (a framework DB connection factory) — the real
TradingBot.__init__ touches the default framework DB path and a Telegram
token, which isn't needed to exercise this pure/DB logic.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bot import MAIN_KB, TradingBot
from earnings_edge.db import engine as db_engine


class _FakeBot:
    def __init__(self, db_path):
        self._db_path = db_path


@pytest.fixture
def bot(tmp_path):
    p = tmp_path / "fw.db"
    db_engine.configure(p)
    return _FakeBot(p)


def test_risk_status_reports_armed_by_default(bot):
    text = TradingBot._risk_status_sync(bot)
    assert "armed (not halted)" in text


def test_halt_then_resume_round_trip(bot):
    TradingBot._halt_sync(bot, "test-operator")
    text = TradingBot._risk_status_sync(bot)
    assert "HALTED" in text and "test-operator" in text

    TradingBot._resume_sync(bot, "test-operator")
    text = TradingBot._risk_status_sync(bot)
    assert "armed (not halted)" in text


def test_lifecycle_step_promotes_and_clamps_at_live(bot):
    cur, new = TradingBot._lifecycle_step_sync(bot, "s1", True, "test")
    assert (cur, new) == ("paper", "probation")
    cur, new = TradingBot._lifecycle_step_sync(bot, "s1", True, "test")
    assert (cur, new) == ("probation", "live")
    cur, new = TradingBot._lifecycle_step_sync(bot, "s1", True, "test")
    assert (cur, new) == ("live", "live")  # already at the top: clamps, no-op


def test_lifecycle_step_demotes_and_clamps_at_paper(bot):
    cur, new = TradingBot._lifecycle_step_sync(bot, "s2", False, "test")
    assert (cur, new) == ("paper", "paper")  # already at the bottom: clamps


def test_lifecycle_menu_data_includes_stepped_strategy(bot):
    TradingBot._lifecycle_step_sync(bot, "s1", True, "test")
    names, states = TradingBot._lifecycle_menu_data_sync(bot)
    assert "s1" in names
    assert states["s1"] == "probation"


def test_settings_kb_has_expected_actions():
    kb = TradingBot._settings_kb()
    flat = [b.callback_data for row in kb.inline_keyboard for b in row]
    assert flat == ["set_risk", "set_halt", "set_resume", "set_lifecycle",
                    "set_restart", "set_home"]


def test_lifecycle_kb_has_promote_demote_per_strategy_and_back_row():
    kb = TradingBot._lifecycle_kb(["s1", "s2"], {"s1": "probation"})
    assert len(kb.inline_keyboard) == 3  # s1 row, s2 row, back row
    s1_row = [b.callback_data for b in kb.inline_keyboard[0]]
    assert s1_row == ["sig_noop", "set_promote_s1", "set_demote_s1"]
    assert kb.inline_keyboard[0][0].text == "s1 (probation)"
    assert kb.inline_keyboard[1][0].text == "s2 (paper)"  # default when unset
    assert kb.inline_keyboard[-1][0].callback_data == "set_back"


def test_set_home_sends_a_fresh_message_with_the_main_keyboard(bot):
    query = MagicMock()
    query.message.reply_text = AsyncMock()
    asyncio.run(TradingBot._handle_settings_callback(bot, query, uid=1, data="set_home"))
    query.message.reply_text.assert_awaited_once()
    args, kwargs = query.message.reply_text.await_args
    assert "Main Menu" in args[0]
    rows = [[btn.text for btn in row] for row in kwargs["reply_markup"].keyboard]
    # _main_reply_kb() appends an optional "🖥 Open desk" row when a Mini App URL is configured.
    if rows and rows[-1] == ["🖥 Open desk"]:
        rows = rows[:-1]
    assert rows == MAIN_KB
