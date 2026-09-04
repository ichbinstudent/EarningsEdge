import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from telegram import Update, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import ContextTypes

from bot import TradingBot
from earnings_edge.rich_msg import (
    send_rich_html,
    edit_rich_html,
    orders_rich_view,
    jobs_rich_view,
    equity_rich_view,
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
    from earnings_edge.db.engine import get_engine
    from sqlalchemy import text
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
    from earnings_edge.db.engine import get_engine
    from sqlalchemy import text
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
    mock_send.return_value = False
    mock_view.return_value = "<html>mock</html>"
    mock_bot._orders_text_sync = MagicMock(return_value="Plain text orders")

    update = MagicMock(spec=Update)
    update.effective_chat.id = 12345
    ctx = MagicMock(spec=ContextTypes.DEFAULT_TYPE)

    import earnings_edge.handlers
    asyncio.run(earnings_edge.handlers.cmd_orders(mock_bot, update, ctx))

    mock_bot._send_panel.assert_called_once()
    assert mock_bot._send_panel.call_args[0][1] == "Plain text orders"
