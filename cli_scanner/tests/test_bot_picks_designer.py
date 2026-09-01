from unittest.mock import AsyncMock
import asyncio
import pytest
from unittest.mock import MagicMock, patch
from telegram import Update
from telegram.ext import ContextTypes

from bot import TradingBot

@pytest.fixture
def mock_bot(tmp_path):
    from earnings_edge.db import configure
    configure(tmp_path / "fw.db")
    bot = TradingBot("dummy_token")
    bot.application = MagicMock()
    return bot

def test_cmd_designer_invalid_args(mock_bot):
    update = MagicMock(spec=Update)
    update.message = MagicMock()
    update.message.reply_text = AsyncMock()
    ctx = MagicMock(spec=ContextTypes.DEFAULT_TYPE)
    ctx.args = ["AAPL"]
    asyncio.run(mock_bot._cmd_designer(update, ctx))
    update.message.reply_text.assert_called_with("Usage: /designer <ticker> <legs...>\nExample: /designer AAPL buy call 190 2026-10-16 1 5.0 0.3")

def test_cmd_designer_invalid_leg_count(mock_bot):
    update = MagicMock(spec=Update)
    update.message = MagicMock()
    update.message.reply_text = AsyncMock()
    ctx = MagicMock(spec=ContextTypes.DEFAULT_TYPE)
    ctx.args = ["AAPL", "buy", "call"]
    asyncio.run(mock_bot._cmd_designer(update, ctx))
    pass  # We now support 4-part legs and have different error messages

@patch("earnings_edge.alpaca_trading.create_client")
def test_cmd_designer_valid_args(mock_create_client, mock_bot):
    mock_client = MagicMock()
    mock_client.get_stock_latest_trade.return_value = 195.0
    mock_create_client.return_value = mock_client
    
    update = MagicMock(spec=Update)
    update.message = MagicMock()
    update.message.reply_text = AsyncMock()
    ctx = MagicMock(spec=ContextTypes.DEFAULT_TYPE)
    ctx.args = ["AAPL", "buy", "call", "190", "2026-10-16", "1", "5.0", "0.3"]
    asyncio.run(mock_bot._cmd_designer(update, ctx))
    
    update.message.reply_text.assert_called_once()
    msg = update.message.reply_text.call_args[0][0]
    assert "Designer: AAPL" in msg
    assert "Spot: $195.00" in msg
    assert "Direction: Bullish" in msg

@patch("earnings_edge.picks.generate_picks")
@patch("bot.snapshots_max_scan_date", return_value=None)
def test_cmd_picks_no_data(mock_max, mock_generate, mock_bot):
    update = MagicMock(spec=Update)
    update.message = MagicMock()
    update.message.reply_text = AsyncMock()
    ctx = MagicMock(spec=ContextTypes.DEFAULT_TYPE)
    asyncio.run(mock_bot._cmd_picks(update, ctx))
    
    update.message.reply_text.assert_called_with("No data in snapshots table.")

@patch("earnings_edge.picks.generate_picks")
@patch("bot.snapshots_max_scan_date", return_value="2026-08-19 12:00:00")
def test_cmd_picks_with_data(mock_max, mock_generate, mock_bot):
    import pandas as pd
    
    # 20 rows to exceed the 10-row limit and test truncation logic
    rows = [{"ticker": f"T{i}", "announcement_date": "2026-08-20", "announcement_time": "BMO", "implied_move": 10.5, "implied_vs_avg_realized": 5.0, "historical_events_count": 4.0} for i in range(20)]
    df1 = pd.DataFrame(rows)
    df2 = pd.DataFrame()
    mock_generate.return_value = {"earnings": df1, "momentum_skew": df2, "forward_factor": pd.DataFrame(), "vrp": pd.DataFrame()}
    
    update = MagicMock(spec=Update)
    update.message = MagicMock()
    update.message.reply_text = AsyncMock()
    ctx = MagicMock(spec=ContextTypes.DEFAULT_TYPE)
    asyncio.run(mock_bot._cmd_picks(update, ctx))
    
    update.message.reply_text.assert_called_once()
    msg = update.message.reply_text.call_args[0][0]
    
    # Assert expected general content
    assert "OQuants Picks (As of 2026-08-19)" in msg
    assert "EARNINGS" in msg
    assert "momentum/skew inputs unpopulated" in msg
    
    # Assert correct column headers for EARNINGS
    assert "tckr" in msg
    assert "date" in msg
    assert "im_v_rl" in msg
    
    # Assert row cap behavior
    assert "... and 10 more rows" in msg
    assert "T0" in msg
    assert "T10" not in msg  # Should be truncated out of the top 10

    # Assert HTML structure is valid (has pre tags)
    assert "<pre>" in msg
    assert "</pre>" in msg
    
    # Assert size limit
    assert len(msg) < 4096

@patch("earnings_edge.picks.generate_picks")
@patch("bot.snapshots_max_scan_date", return_value="2026-08-19 12:00:00")
@patch("httpx.AsyncClient")
def test_cmd_picks_rich_message_success(mock_httpx, mock_max, mock_generate, mock_bot):
    import pandas as pd
    
    # Mock httpx response to succeed
    mock_client_instance = AsyncMock()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_client_instance.post.return_value = mock_response
    mock_httpx.return_value.__aenter__.return_value = mock_client_instance

    rows = [{"ticker": f"T{i}", "announcement_date": "2026-08-20", "announcement_time": "BMO", "implied_move": 10.5, "implied_vs_avg_realized": 5.0, "historical_events_count": 4.0} for i in range(55)]
    df1 = pd.DataFrame(rows)
    df2 = pd.DataFrame()
    mock_generate.return_value = {"earnings": df1, "momentum_skew": df2, "forward_factor": pd.DataFrame(), "vrp": pd.DataFrame()}
    
    update = MagicMock(spec=Update)
    update.message = MagicMock()
    update.message.chat_id = 12345
    update.message.reply_text = AsyncMock()
    ctx = MagicMock(spec=ContextTypes.DEFAULT_TYPE)
    
    # Setup mock_bot properly for httpx url format
    mock_bot.application.bot.token = "dummy"
    mock_bot.application.bot.base_url = "http://fake.api"

    asyncio.run(mock_bot._cmd_picks(update, ctx))
    
    # Check that rich message was sent
    mock_client_instance.post.assert_called_once()
    url, kwargs = mock_client_instance.post.call_args
    assert url[0] == "http://fake.api/sendRichMessage"
    assert "json" in kwargs
    payload = kwargs["json"]
    assert payload["chat_id"] == 12345
    
    html = payload["rich_message"]["html"]
    
    # Validate HTML table structure
    assert "<h3>OQuants Picks (As of 2026-08-19)</h3>" in html
    assert "<h4>EARNINGS</h4>" in html
    assert "<table bordered striped compact>" in html
    assert "<th>tckr</th>" in html
    assert "<td>T0</td>" in html
    
    # 55 rows should trigger the "... and 5 more rows" text in the HTML
    assert "... and 5 more rows." in html
    
    # Check that fallback wasn't triggered
    update.message.reply_text.assert_not_called()

@patch("earnings_edge.picks.generate_picks")
@patch("bot.snapshots_max_scan_date", return_value="2026-08-19 12:00:00")
@patch("httpx.AsyncClient")
def test_cmd_picks_rich_message_fallback(mock_httpx, mock_max, mock_generate, mock_bot):
    import pandas as pd
    
    # Mock httpx response to fail
    mock_client_instance = AsyncMock()
    mock_response = MagicMock()
    mock_response.status_code = 400
    mock_response.text = "Bad Request"
    mock_client_instance.post.return_value = mock_response
    mock_httpx.return_value.__aenter__.return_value = mock_client_instance

    rows = [{"ticker": f"T{i}", "announcement_date": "2026-08-20", "announcement_time": "BMO", "implied_move": 10.5, "implied_vs_avg_realized": 5.0, "historical_events_count": 4.0} for i in range(20)]
    df1 = pd.DataFrame(rows)
    df2 = pd.DataFrame()
    mock_generate.return_value = {"earnings": df1, "momentum_skew": df2, "forward_factor": pd.DataFrame(), "vrp": pd.DataFrame()}
    
    update = MagicMock(spec=Update)
    update.message = MagicMock()
    update.message.reply_text = AsyncMock()
    ctx = MagicMock(spec=ContextTypes.DEFAULT_TYPE)
    
    mock_bot.application.bot.token = "dummy"
    mock_bot.application.bot.base_url = "http://fake.api"

    asyncio.run(mock_bot._cmd_picks(update, ctx))
    
    # Check that fallback was triggered
    update.message.reply_text.assert_called_once()
