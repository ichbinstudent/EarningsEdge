"""Phase 0 desk safety: operator lock, instance lock, broker-truth book."""
from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock

import pytest

from earnings_edge.bot_views import positions_view
from earnings_edge.ops_auth import (
    AUTH_UNCONFIGURED, auth_message, is_operator, operator_chat_ids, operators_configured,
)
from framework.execution.managed import record_open_positions
from framework.ops import InstanceLock, SecretRedactFilter
from framework.positions.book import classify_book
from framework.positions.book_actions import adopt_orphan, ignore_orphan, mark_missing_closed
from sqlalchemy import text
from earnings_edge.db import configure, engine as db_engine


def test_operator_lock_fail_closed_when_unset():
    env = {}
    assert operator_chat_ids(env) == []
    assert operators_configured(env) is False
    assert is_operator(256565866, env) is False
    assert auth_message(env) == AUTH_UNCONFIGURED


def test_operator_lock_only_listed_chat():
    env = {"TELEGRAM_APPROVAL_CHAT_ID": "256565866"}
    assert operator_chat_ids(env) == [256565866]
    assert is_operator(256565866, env) is True
    assert is_operator(1, env) is False
    assert is_operator(None, env) is False


def test_operator_lock_csv_ids():
    env = {"TELEGRAM_APPROVAL_CHAT_IDS": "1, 2, x, 3"}
    assert operator_chat_ids(env) == [1, 2, 3]
    assert is_operator(2, env)


def test_secret_redact_filter():
    filt = SecretRedactFilter()
    rec = type("R", (), {})()
    rec.getMessage = lambda: "POST https://api.telegram.org/bot0000000000:AA-FAKE-TOKEN-FOR-TESTS/getUpdates"
    rec.msg = rec.getMessage()
    rec.args = ()
    assert filt.filter(rec) is True
    assert "bot<redacted>" in rec.msg
    assert "AA-FAKE-TOKEN-FOR-TESTS" not in rec.msg
    assert "0000000000" not in rec.msg


def test_instance_lock_second_process_refused(tmp_path):
    path = tmp_path / "bot.lock"
    a = InstanceLock(path).acquire()
    with pytest.raises(RuntimeError, match="another trading-bot"):
        InstanceLock(path).acquire()
    a.release()
    # after release, a new holder can take it
    b = InstanceLock(path).acquire()
    b.release()


def test_classify_book_three_buckets(tmp_path):
    configure(tmp_path / "fw.db")
    record_open_positions([
            {"symbol": "META260828C00595000", "side": "sell", "ratio_qty": 1,
             "option_type": "call", "strike": 595.0, "expiry": date(2026, 8, 28)},
            {"symbol": "META260918C00595000", "side": "buy", "ratio_qty": 1,
             "option_type": "call", "strike": 595.0, "expiry": date(2026, 9, 18)},
        ],
        "ff_ladder", group_id="g-meta", entry_price=10.2,
        metadata={"side": "CALENDAR", "earnings_date": "2026-07-29"},
    )
    record_open_positions([{"symbol": "GONE260828C00100000", "side": "sell", "ratio_qty": 1,
          "option_type": "call", "strike": 100.0, "expiry": date(2026, 8, 28)}],
        "ff_ladder", group_id="g-gone", entry_price=1.0,
        metadata={"side": "CALENDAR", "earnings_date": "2026-07-29"},
    )
    from framework.execution.managed import open_groups
    broker = [
        {"symbol": "META260828C00595000", "qty": "-1", "side": "short",
         "unrealized_pl": "10", "current_price": "20", "avg_entry_price": "32"},
        {"symbol": "META260918C00595000", "qty": "1", "side": "long",
         "unrealized_pl": "-5", "current_price": "30", "avg_entry_price": "42"},
        {"symbol": "ATLO260918C00030000", "qty": "-1", "side": "short",
         "unrealized_pl": "-300", "current_price": "4", "avg_entry_price": "1"},
        {"symbol": "NU", "qty": "-2500", "side": "short",
         "unrealized_pl": "-3000", "current_price": "15", "avg_entry_price": "14"},
    ]
    book = classify_book(open_groups(), broker)
    assert {i.symbol for i in book.managed} == {
        "META260828C00595000", "META260918C00595000"}
    assert {i.symbol for i in book.orphan} == {"ATLO260918C00030000", "NU"}
    assert {i.symbol for i in book.missing} == {"GONE260828C00100000"}
    nu = next(i for i in book.orphan if i.symbol == "NU")
    assert nu.ticker == "NU" and nu.qty == -2500
    text = positions_view( broker_positions=broker)
    assert "ORPHAN" in text and "ATLO" in text and "NU" in text
    assert "MISSING" in text and "GONE" in text
    assert "MANAGED" in text and "META" in text


def test_positions_view_empty_local_still_legacy(tmp_path):
    configure(tmp_path / "fw.db")
    assert "No open managed positions" in positions_view()
    assert "No positions at broker or locally" in positions_view( broker_positions=[])


def test_book_action_banner_and_panel_refresh(tmp_path):
    from earnings_edge.bot_views import (
        book_action_banner, build_positions_panel, positions_keyboard,
    )
    from framework.positions.book import classify_book
    from framework.execution.managed import open_groups

    assert "Adopted ATLO" in book_action_banner(
        "adopt", {"ok": True, "group_id": "adopt-ATLO"}, "ATLO260918C00030000")
    assert "⚠️" in book_action_banner("adopt", {"ok": False, "error": "gone"})

    configure(tmp_path / "fw.db")
    broker = [
        {"symbol": "ATLO260918C00030000", "qty": "-1", "side": "short",
         "unrealized_pl": "-300", "current_price": "4", "avg_entry_price": "1"},
    ]
    text, rows = build_positions_panel(broker_positions=broker)
    assert "ORPHAN" in text and "ATLO" in text
    callbacks = [b.callback_data for row in rows for b in row]
    assert "bk_ad_ATLO260918C00030000" in callbacks
    assert callbacks[-1] == "bk_rf"
    book = classify_book(open_groups(), broker)
    kb = positions_keyboard(book)
    assert kb[-1][0].callback_data == "bk_rf"

    ign = ignore_orphan("DAL", by="pre")
    assert ign["ok"]
    text_ig, _ = build_positions_panel(
        broker_positions=broker + [
            {"symbol": "DAL", "qty": "10", "side": "long",
             "avg_entry_price": "1", "current_price": "1", "unrealized_pl": "0"},
        ])
    assert "DAL" not in text_ig

    rec = adopt_orphan(broker[0], by="test")
    assert rec["ok"]
    text2, rows2 = build_positions_panel(
        broker_positions=broker, banner="✅ Adopted ATLO260918C00030000 — now on the managed book.")
    assert text2.startswith("✅ Adopted ATLO")
    assert "MANAGED" in text2
    assert "ORPHAN" not in text2
    cb2 = [b.callback_data for row in rows2 for b in row]
    assert "bk_ad_ATLO260918C00030000" not in cb2
    assert "bk_rf" in cb2


def test_adopt_callback_edits_positions_panel(tmp_path, monkeypatch):
    """Adopt must rewrite the same Positions message — not leave a stale book."""
    import asyncio
    from unittest.mock import AsyncMock, MagicMock
    from bot import TradingBot

    path = tmp_path / "fw.db"
    configure(path)
    broker = [{
        "symbol": "ATLO260918C00030000", "qty": "-1", "side": "short",
        "avg_entry_price": "1", "current_price": "4", "unrealized_pl": "-300",
    }]

    class _Client:
        def get_positions(self):
            return list(broker)

    class _Bot:
        def _risk_authorized(self, uid):
            return True

        async def _flush_alerts(self):
            return None

        _positions_panel_sync = TradingBot._positions_panel_sync
        _refresh_positions_query = TradingBot._refresh_positions_query
        _edit_panel = TradingBot._edit_panel
        _handle_book_callback = TradingBot._handle_book_callback

    monkeypatch.setattr("bot.create_client", lambda: _Client())
    query = MagicMock()
    query.edit_message_text = AsyncMock()
    query.message.reply_text = AsyncMock()
    asyncio.run(TradingBot._handle_book_callback(
        _Bot(), query, uid=1, data="bk_ad_ATLO260918C00030000"))
    query.edit_message_text.assert_awaited()
    text = query.edit_message_text.await_args.args[0]
    assert "Adopted ATLO260918C00030000" in text
    assert "MANAGED" in text
    markup = query.edit_message_text.await_args.kwargs.get("reply_markup")
    assert markup is not None
    cbs = [b.callback_data for row in markup.inline_keyboard for b in row]
    assert "bk_rf" in cbs
    assert "bk_ad_ATLO260918C00030000" not in cbs


def test_inbox_skip_rewrites_same_panel(tmp_path):
    """Skip from Pending must edit the inbox, not send a new card flood."""
    import asyncio
    from unittest.mock import AsyncMock, MagicMock
    from bot import TradingBot
    from earnings_edge.trade_approval import PendingTradeStore
    from earnings_edge.trading_types import Trade

    path = tmp_path / "fw.db"
    configure(path)
    store = PendingTradeStore(str(tmp_path / "pending.db"))
    trade = Trade(
        ticker="FRESH", earnings_date=date(2026, 8, 20), scan_date=date(2026, 8, 15),
        strategy="calendar_call_ml", side="CALENDAR", entry_price=1.5,
        features={}, ml_decision="TAKE",
    )
    pid = store.add(trade, "card")

    class _Bot:
        approval_store = store

        def _risk_authorized(self, uid):
            return True

        _pending_panel_sync = TradingBot._pending_panel_sync
        _refresh_pending_query = TradingBot._refresh_pending_query
        _edit_panel = TradingBot._edit_panel
        _handle_inbox_callback = TradingBot._handle_inbox_callback
        _entry_skip_outcome = TradingBot._entry_skip_outcome

    query = MagicMock()
    query.edit_message_text = AsyncMock()
    query.message.reply_text = AsyncMock()
    asyncio.run(TradingBot._handle_inbox_callback(
        _Bot(), query, uid=1, data=f"in_sk_{pid}"))
    query.edit_message_text.assert_awaited()
    text = query.edit_message_text.await_args.args[0]
    assert "Skipped" in text or "Pending inbox" in text
    markup = query.edit_message_text.await_args.kwargs.get("reply_markup")
    cbs = [b.callback_data for row in markup.inline_keyboard for b in row]
    assert f"in_ex_{pid}" not in cbs
    assert "desk_pd" in cbs


def test_adopt_ignore_mark(tmp_path):
    configure(tmp_path / "fw.db")
    rec = adopt_orphan({
        "symbol": "ATLO260918C00030000", "qty": "-1", "side": "short",
        "avg_entry_price": "1",
    }, by="test")
    assert rec["ok"]
    from framework.execution.managed import open_groups
    groups = open_groups()
    assert any(g.ticker == "ATLO" for g in groups)
    ign = ignore_orphan("DAL", by="test")
    assert ign["ok"]
    with db_engine.get_session() as s:
        assert s.execute(text("SELECT symbol FROM adopted_positions WHERE symbol='DAL'")).first()
    record_open_positions([{"symbol": "MISS260101C00100000", "side": "buy", "ratio_qty": 1,
          "option_type": "call", "strike": 10, "expiry": date(2026, 1, 1)}],
        "unmanaged", group_id="g-miss",
    )
    marked = mark_missing_closed("g-miss", by="test")
    assert marked["ok"]
    assert all(g.group_id != "g-miss" for g in open_groups())
