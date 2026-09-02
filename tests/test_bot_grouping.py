"""Tests for TradingBot's per-strategy batch-keyboard helpers (bot.py).

These are pure/staticmethod-only: building a keyboard from rows, and
reading proposal ids back off a clicked message's own keyboard. No
Telegram Application/token is instantiated — TradingBot.__init__ is never
called, only its @staticmethods.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bot import TradingBot


class _FakeButton:
    def __init__(self, callback_data):
        self.callback_data = callback_data


class _FakeMarkup:
    def __init__(self, rows):
        self.inline_keyboard = rows


class _FakeMessage:
    def __init__(self, reply_markup):
        self.reply_markup = reply_markup


class _FakeQuery:
    def __init__(self, reply_markup):
        self.message = _FakeMessage(reply_markup)


def _kb_from_ids(ids, prefix_pair):
    exec_prefix, skip_prefix = prefix_pair
    rows = [[_FakeButton(f"{exec_prefix}{i}"), _FakeButton(f"{skip_prefix}{i}")] for i in ids]
    return _FakeMarkup(rows)


def test_row_kb_entry_uses_execute_skip_prefixes():
    kb = TradingBot._row_kb({"id": 7}, "entry")
    flat = [b.callback_data for row in kb.inline_keyboard for b in row]
    assert flat == ["pt_exec_7", "pt_skip_7"]


def test_row_kb_exit_uses_close_snooze_prefixes():
    kb = TradingBot._row_kb({"id": 9}, "exit")
    flat = [b.callback_data for row in kb.inline_keyboard for b in row]
    assert flat == ["ex_close_9", "ex_skip_9"]


def test_grouped_kb_one_row_per_ticker_plus_bulk_row():
    rows = [{"id": 1, "ticker": "AAPL"}, {"id": 2, "ticker": "MSFT"}]
    kb = TradingBot._grouped_kb(rows, "entry")
    assert len(kb.inline_keyboard) == 3  # 2 ticker rows + 1 bulk row
    assert kb.inline_keyboard[0][0].callback_data == "pt_exec_1"
    assert kb.inline_keyboard[1][0].callback_data == "pt_exec_2"
    assert kb.inline_keyboard[2][0].callback_data == "pt_exec_grp"


def test_grouped_kb_single_row_has_no_bulk_row():
    rows = [{"id": 1, "ticker": "AAPL"}]
    kb = TradingBot._grouped_kb(rows, "entry")
    assert len(kb.inline_keyboard) == 1


def test_grouped_kb_exit_bulk_callback():
    rows = [{"id": 1, "ticker": "AAPL"}, {"id": 2, "ticker": "MSFT"}]
    kb = TradingBot._grouped_kb(rows, "exit")
    assert kb.inline_keyboard[-1][0].callback_data == "ex_close_grp"


def test_kb_ids_reads_ids_back_off_the_keyboard():
    markup = _kb_from_ids([101, 102, 103], ("pt_exec_", "pt_skip_"))
    query = _FakeQuery(markup)
    assert TradingBot._kb_ids(query, ("pt_exec_", "pt_skip_")) == [101, 102, 103]


def test_kb_ids_ignores_bulk_and_unrelated_buttons():
    rows = [
        [_FakeButton("pt_exec_5"), _FakeButton("pt_skip_5")],
        [_FakeButton("pt_exec_grp")],
        [_FakeButton("sig_on_ff_ladder")],
    ]
    query = _FakeQuery(_FakeMarkup(rows))
    assert TradingBot._kb_ids(query, ("pt_exec_", "pt_skip_")) == [5]


def test_kb_ids_empty_when_no_markup():
    query = _FakeQuery(None)
    assert TradingBot._kb_ids(query, ("pt_exec_", "pt_skip_")) == []


def test_kb_ids_deduplicates_ids_seen_via_both_prefixes():
    rows = [[_FakeButton("pt_exec_5"), _FakeButton("pt_skip_5")]]
    query = _FakeQuery(_FakeMarkup(rows))
    assert TradingBot._kb_ids(query, ("pt_exec_", "pt_skip_")) == [5]
