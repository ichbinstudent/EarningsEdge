import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import ContextTypes

from bot import TradingBot


async def _async_false(*args, **kwargs):
    """Awaitable False — what send_rich_html returns when Telegram rejects."""
    return False


from earnings_edge.rich_msg import (
    edit_rich_html,
    equity_rich_view,
    jobs_rich_view,
    orders_rich_view,
    send_rich_html,
)


@pytest.fixture
def mock_bot(tmp_path):
    from earnings_edge.db import configure

    configure(tmp_path / "fw.db")
    bot = TradingBot("dummy_token")
    bot.application = MagicMock()
    bot.application.bot = MagicMock()
    bot.application.bot.base_url = "http://fake.api"
    bot._send_panel = AsyncMock()
    return bot


# --- Tests for send_rich_html and edit_rich_html ---


@patch("httpx.AsyncClient")
def test_send_rich_html_success(mock_httpx):
    mock_client_instance = AsyncMock()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_client_instance.post.return_value = mock_response
    mock_httpx.return_value.__aenter__.return_value = mock_client_instance

    bot = MagicMock()
    bot.base_url = "http://fake.api"

    markup = InlineKeyboardMarkup([[InlineKeyboardButton("Test", callback_data="test")]])

    success = asyncio.run(send_rich_html(bot, 12345, "<h1>Hello</h1>", reply_markup=markup))

    assert success is True
    mock_client_instance.post.assert_called_once()
    url, kwargs = mock_client_instance.post.call_args
    assert url[0] == "http://fake.api/sendRichMessage"
    assert "json" in kwargs
    payload = kwargs["json"]
    assert payload["chat_id"] == 12345
    assert payload["rich_message"]["html"] == "<h1>Hello</h1>"
    assert "inline_keyboard" in payload["reply_markup"]


@patch("httpx.AsyncClient")
def test_send_rich_html_failure(mock_httpx):
    mock_client_instance = AsyncMock()
    mock_response = MagicMock()
    mock_response.status_code = 400
    mock_client_instance.post.return_value = mock_response
    mock_httpx.return_value.__aenter__.return_value = mock_client_instance

    bot = MagicMock()
    bot.base_url = "http://fake.api"

    success = asyncio.run(send_rich_html(bot, 12345, "<h1>Hello</h1>"))
    assert success is False


@patch("httpx.AsyncClient")
def test_edit_rich_html_success(mock_httpx):
    mock_client_instance = AsyncMock()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_client_instance.post.return_value = mock_response
    mock_httpx.return_value.__aenter__.return_value = mock_client_instance

    bot = MagicMock()
    bot.base_url = "http://fake.api"

    success = asyncio.run(edit_rich_html(bot, 12345, 67890, "<h1>Edit</h1>"))

    assert success is True
    mock_client_instance.post.assert_called_once()
    url, kwargs = mock_client_instance.post.call_args
    assert url[0] == "http://fake.api/editMessageText"
    assert "json" in kwargs
    payload = kwargs["json"]
    assert payload["chat_id"] == 12345
    assert payload["message_id"] == 67890
    assert payload["rich_message"]["html"] == "<h1>Edit</h1>"


@patch("httpx.AsyncClient")
def test_edit_rich_html_failure(mock_httpx):
    mock_client_instance = AsyncMock()
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_client_instance.post.return_value = mock_response
    mock_httpx.return_value.__aenter__.return_value = mock_client_instance

    bot = MagicMock()
    bot.base_url = "http://fake.api"

    success = asyncio.run(edit_rich_html(bot, 12345, 67890, "<h1>Edit</h1>"))
    assert success is False


# --- Tests for rich views ---


def test_orders_rich_view(seeded_db):
    html = orders_rich_view()
    assert "<h3>ORDERS</h3>" in html
    assert "<table bordered striped compact>" in html
    assert "<th>Event</th>" in html
    assert "<th>Price</th>" in html
    assert "<th>Detail</th>" not in html
    assert "09-04 08:00" in html
    assert "buy_to_open" in html
    assert "momentum" in html
    assert "AAPL" in html
    assert "150.50" in html
    # Detail now renders as a full-width paragraph, not a table cell
    assert "Bought 100 shares &lt;foo&gt;" in html
    assert "&lt;foo&gt;" in html  # HTML escaping


def test_orders_rich_view_empty(seeded_db):
    from sqlalchemy import text

    from earnings_edge.db.engine import get_engine

    with get_engine().begin() as conn:
        conn.execute(text("DELETE FROM trade_events"))
    html = orders_rich_view()
    assert "<h3>ORDERS</h3>" in html
    assert "<i>No trade events yet.</i>" in html
    assert "<table>" not in html


def test_jobs_rich_view(seeded_db):
    html = jobs_rich_view()
    assert "<h3>JOB RUNS</h3>" in html
    assert "<table bordered striped compact>" in html
    assert "✓" in html
    assert "sync" in html
    assert "2026-09-04 08:00:00" in html
    assert "a=1 b=2" in html


def test_jobs_rich_view_empty(seeded_db):
    from sqlalchemy import text

    from earnings_edge.db.engine import get_engine

    with get_engine().begin() as conn:
        conn.execute(text("DELETE FROM job_runs"))
    html = jobs_rich_view()
    assert "<h3>JOB RUNS</h3>" in html
    assert "<i>No job runs found.</i>" in html


@patch("framework.risk.equity.latest_equity")
@patch("framework.risk.equity.day_start_equity")
@patch("framework.risk.equity.daily_pnl")
def test_equity_rich_view(mock_pnl, mock_start, mock_latest, seeded_db):
    mock_latest.return_value = {"equity": 10000.50, "buying_power": 5000.0, "ts": "2026-09-04"}
    mock_start.return_value = 9900.0
    mock_pnl.return_value = 100.50

    html = equity_rich_view()
    assert "<h3>EQUITY</h3>" in html
    assert "Latest: $10,000.50" in html
    assert "BP: $5,000.00" in html
    assert "Day-start: $9,900.00" in html
    assert "Day-PnL: $100.50" in html
    assert "<table bordered striped compact>" in html
    assert "2026-09-03" in html
    assert "$9,950.00" in html


@patch("framework.risk.equity.latest_equity")
def test_equity_rich_view_empty(mock_latest, seeded_db):
    mock_latest.return_value = None
    html = equity_rich_view()
    assert "<i>No equity data available.</i>" in html


@patch("framework.risk.equity.latest_equity")
@patch("framework.risk.equity.day_start_equity")
@patch("framework.risk.equity.daily_pnl")
def test_equity_rich_view_no_day_start(mock_pnl, mock_start, mock_latest, seeded_db):
    """day_start_equity()/daily_pnl() can be None outside market hours (live 09-04)."""
    mock_latest.return_value = {"equity": 10000.50, "buying_power": 5000.0, "ts": "2026-09-04"}
    mock_start.return_value = None
    mock_pnl.return_value = None
    html = equity_rich_view()
    assert "Day-start" not in html
    assert "$10,000.50" in html
    assert "$9,950.00" in html


# --- Tests for Handler fallbacks ---


@patch("earnings_edge.rich_msg.send_rich_html")
@patch("earnings_edge.rich_msg.orders_rich_view")
def test_cmd_orders_fallback(mock_view, mock_send, mock_bot):
    mock_send.side_effect = _async_false
    mock_view.return_value = "<html>mock</html>"
    mock_bot._orders_text_sync = MagicMock(return_value="Plain text orders")

    update = MagicMock(spec=Update)
    update.effective_chat.id = 12345
    ctx = MagicMock(spec=ContextTypes.DEFAULT_TYPE)

    import earnings_edge.handlers

    asyncio.run(earnings_edge.handlers.cmd_orders(mock_bot, update, ctx))

    mock_bot._send_panel.assert_called_once()
    assert mock_bot._send_panel.call_args[0][1] == "Plain text orders"


@patch("earnings_edge.rich_msg.send_rich_html")
@patch("earnings_edge.rich_msg.jobs_rich_view")
def test_cmd_jobs_fallback(mock_view, mock_send, mock_bot):
    mock_send.side_effect = _async_false
    mock_view.return_value = "<html>mock</html>"
    mock_bot._jobs_text_sync = MagicMock(return_value="Plain text jobs")

    update = MagicMock(spec=Update)
    update.effective_chat.id = 12345
    ctx = MagicMock(spec=ContextTypes.DEFAULT_TYPE)

    import earnings_edge.handlers

    asyncio.run(earnings_edge.handlers.cmd_jobs(mock_bot, update, ctx))

    mock_bot._send_panel.assert_called_once()
    assert mock_bot._send_panel.call_args[0][1] == "Plain text jobs"


@patch("earnings_edge.rich_msg.send_rich_html")
def test_cmd_status_fallback(mock_send, mock_bot):
    mock_send.side_effect = _async_false
    mock_bot._status_rich_sync = MagicMock(return_value="<html>mock</html>")
    mock_bot._status_text_sync = MagicMock(return_value="Plain text status")

    update = MagicMock(spec=Update)
    update.effective_chat.id = 12345
    ctx = MagicMock(spec=ContextTypes.DEFAULT_TYPE)

    import earnings_edge.handlers

    asyncio.run(earnings_edge.handlers.cmd_status(mock_bot, update, ctx))

    mock_bot._send_panel.assert_called_once()
    assert mock_bot._send_panel.call_args[0][1] == "Plain text status"


@patch("earnings_edge.rich_msg.send_rich_html")
def test_cmd_positions_fallback(mock_send, mock_bot):
    mock_send.side_effect = _async_false
    mock_bot._positions_panel_sync_rich = MagicMock(
        return_value=("Plain text positions", [], "<html>mock</html>")
    )

    update = MagicMock(spec=Update)
    update.effective_chat.id = 12345
    ctx = MagicMock(spec=ContextTypes.DEFAULT_TYPE)

    import earnings_edge.handlers

    asyncio.run(earnings_edge.handlers.cmd_positions(mock_bot, update, ctx))

    mock_bot._send_panel.assert_called_once()
    assert mock_bot._send_panel.call_args[0][1] == "Plain text positions"


@patch("earnings_edge.rich_msg.send_rich_html")
def test_cmd_pending_fallback(mock_send, mock_bot):
    mock_send.side_effect = _async_false
    mock_bot._pending_panel_sync_rich = MagicMock(
        return_value=("Plain text pending", [], "<html>mock</html>")
    )

    update = MagicMock(spec=Update)
    update.effective_chat.id = 12345
    ctx = MagicMock(spec=ContextTypes.DEFAULT_TYPE)

    import earnings_edge.handlers

    asyncio.run(earnings_edge.handlers.cmd_pending(mock_bot, update, ctx))

    mock_bot._send_panel.assert_called_once()
    assert mock_bot._send_panel.call_args[0][1] == "Plain text pending"


@patch("earnings_edge.rich_msg.send_rich_html")
def test_cmd_strategies_fallback(mock_send, mock_bot):
    mock_send.side_effect = _async_false
    mock_bot._strategies_panel_sync_rich = MagicMock(
        return_value=("Plain text strats", [], "<html>mock</html>")
    )

    update = MagicMock(spec=Update)
    update.effective_chat.id = 12345
    ctx = MagicMock(spec=ContextTypes.DEFAULT_TYPE)

    import earnings_edge.handlers

    asyncio.run(earnings_edge.handlers.cmd_strategies(mock_bot, update, ctx))

    mock_bot._send_panel.assert_called_once()
    assert mock_bot._send_panel.call_args[0][1] == "Plain text strats"


def test_positions_rich_view():
    from earnings_edge.rich_msg import positions_rich_view
    from framework.positions.book import Book

    class MockLeg:
        def __init__(self):
            self.side = "sell"
            self.qty = 1.0
            self.symbol = "O:AAPL240119C00150000"
            self.option_type = "call"
            self.strike = 150.0
            self.expiry = None

    class MockGroup:
        def __init__(self):
            self.strategy = "test_strat"
            self.ticker = "AAPL"
            self.credit = True
            self.entry_price = 1.50
            self.opened_at = "2026-09-01T12:00:00Z"
            self.legs = [MockLeg()]

    class MockItem:
        def __init__(self):
            self.strategy = "strat1"
            self.ticker = "MSFT"
            self.qty = 100
            self.upl = 50.0
            self.current_price = 300.0
            self.event_date = None
            self.expiry = None
            self.symbol = "O:MSFT_test"

    with patch("framework.execution.managed.open_groups") as mock_groups:
        mock_groups.return_value = [MockGroup()]
        # Without broker positions
        html = positions_rich_view()
        assert "OPEN POSITIONS (1 groups) — local book only" in html
        assert "<td>test_strat</td>" in html
        assert "O:AAPL240119C00150000" in html

        # With broker positions, but Mock classify_book
        with patch("framework.positions.book.classify_book") as mock_classify:
            mock_book = Book(managed=[MockItem()], orphan=[MockItem()], missing=[MockItem()])
            mock_classify.return_value = mock_book

            html = positions_rich_view(broker_positions=[])
            assert "BOOK: broker=2 local=2 orphans=1 missing=1" in html
            assert "MANAGED (matched)" in html
            assert "ORPHAN (at broker, not local)" in html
            assert "MISSING (local open, not at broker)" in html
            assert "<td>MSFT</td>" in html
            assert "<td>strat1</td>" in html
            assert "<td>$+50</td>" in html
            assert "O:MSFT_test" in html


def test_pending_rich_view():
    from earnings_edge.inbox import Inbox, InboxItem
    from earnings_edge.rich_msg import pending_rich_view

    inbox = Inbox(
        items=[
            InboxItem(
                kind="entry",
                item_id="1",
                ticker="AAPL",
                strategy="strat_1",
                detail="buy",
                created_at="2026-09-01T12:00:00",
            ),
            InboxItem(
                kind="exit",
                item_id="2",
                ticker="MSFT",
                strategy="strat_2",
                detail="sell",
                created_at="2026-09-02T12:00:00",
            ),
        ]
    )
    html = pending_rich_view(inbox, banner="Test Banner")
    assert "<h3>Test Banner</h3>" in html
    assert "<h4>Entries</h4>" in html
    assert "<td><b>AAPL</b></td>" in html
    assert "<td>entry</td>" in html
    assert "<td>strat_1</td>" in html

    empty_inbox = Inbox(items=[])
    html = pending_rich_view(empty_inbox)
    assert "Nothing pending." in html


def test_strategies_rich_view():
    from earnings_edge.rich_msg import strategies_rich_view

    class MockRegistry:
        configs = ["strat1", "strat2"]

        def is_enabled(self, name):
            return True

        def get(self, name):
            return None

    with (
        patch("framework.execution.lifecycle.LifecycleManager.all_states") as mock_states,
        patch("framework.risk.manager.RiskManager.strategy_spend_today") as mock_spend,
    ):
        mock_states.return_value = {"strat1": "paper"}
        mock_spend.return_value = 100.0

        html, buttons = strategies_rich_view(MockRegistry())
        assert "STRATEGIES" in html
        assert "strat1" in html
        assert "strat2" in html
        assert "$100" in html
        assert len(buttons) == 2


def test_status_rich_view():
    from earnings_edge.rich_msg import status_rich_view

    with (
        patch("framework.risk.equity.latest_equity") as mock_eq,
        patch("framework.risk.killswitch.KillSwitch.status") as mock_ks,
        patch("framework.execution.managed.open_groups") as mock_groups,
        patch("earnings_edge.db.strategy_state_list") as mock_states,
    ):
        mock_eq.return_value = {"equity": 10000, "buying_power": 5000, "ts": "2026-09-01"}
        mock_ks.return_value = {"halted": False}
        mock_groups.return_value = []
        mock_states.return_value = [{"name": "strat1", "lifecycle": "live", "enabled": 1}]

        html = status_rich_view(market_open=True, broker_ok=True, sha="abcdef", started_at="2026-09-01T12:00")
        assert "SYSTEM STATUS" in html
        assert "🟢 open" in html
        assert "reachable" in html
        assert "abcdef" in html
        assert "🟢 armed" in html
        assert "10,000" in html
        assert "strat1" in html
        assert "live" in html


@patch("httpx.AsyncClient")
def test_send_rich_html_positions_keyboard(mock_httpx):
    mock_client_instance = AsyncMock()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_client_instance.post.return_value = mock_response
    mock_httpx.return_value.__aenter__.return_value = mock_client_instance

    from telegram import InlineKeyboardMarkup

    from earnings_edge.bot_views import positions_keyboard
    from framework.positions.book import Book

    class MockItem:
        def __init__(self):
            self.group_id = "test_grp"
            self.ticker = "AAPL"
            self.symbol = "O:AAPL240119C00150000"
            self.upl = 0
            self.current_price = 0

    mock_book = Book(managed=[MockItem()], orphan=[MockItem()], missing=[MockItem()])
    rows = positions_keyboard(mock_book)
    markup = InlineKeyboardMarkup(rows)

    bot = MagicMock()
    bot.base_url = "http://fake.api"

    success = asyncio.run(send_rich_html(bot, 12345, "<h1>Hello</h1>", reply_markup=markup))

    assert success is True
    mock_client_instance.post.assert_called_once()
    payload = mock_client_instance.post.call_args[1]["json"]
    assert "inline_keyboard" in payload["reply_markup"]


@patch("httpx.AsyncClient")
def test_send_rich_html_pending_keyboard(mock_httpx):
    mock_client_instance = AsyncMock()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_client_instance.post.return_value = mock_response
    mock_httpx.return_value.__aenter__.return_value = mock_client_instance

    from telegram import InlineKeyboardMarkup

    from earnings_edge.inbox import Inbox, InboxItem, inbox_keyboard

    inbox = Inbox(
        items=[
            InboxItem(kind="entry", item_id="1", ticker="AAPL"),
            InboxItem(kind="exit", item_id="2", ticker="MSFT"),
            InboxItem(kind="orphan", item_id="3", ticker="TSLA"),
            InboxItem(kind="assignment", item_id="4", ticker="GOOG"),
        ]
    )
    rows = inbox_keyboard(inbox)
    markup = InlineKeyboardMarkup(rows)

    bot = MagicMock()
    bot.base_url = "http://fake.api"

    success = asyncio.run(send_rich_html(bot, 12345, "<h1>Hello</h1>", reply_markup=markup))

    assert success is True
    mock_client_instance.post.assert_called_once()
    payload = mock_client_instance.post.call_args[1]["json"]
    assert "inline_keyboard" in payload["reply_markup"]
