"""Tests for historical-move backfill with Polygon fallback."""

from datetime import date, datetime, timedelta, timezone
from unittest.mock import MagicMock, patch
import pytest
import pandas as pd
from sqlalchemy import text

from earnings_edge.db import engine as db_engine


def create_test_db():
    """Re-point the engine at a fresh temp DB (per-test isolation, like the
    old in-memory :memory: database). Returns None — verifications go
    through db_engine.session_scope()."""
    import tempfile
    from pathlib import Path

    tmp = Path(tempfile.mkdtemp(prefix="hist_backfill_test_"))
    db_engine.configure(tmp / "hist_backfill.db")
    return None


def make_bars(ticker, ed, close_pre=100.0, close_post=105.0):
    """Create Polygon-shaped bar data."""
    bars = []
    for i in range(-7, 4):
        bar_date = ed + timedelta(days=i)
        ts = int(datetime.combine(bar_date, datetime.min.time()).timestamp() * 1000)
        if i < 0:
            close = close_pre
        elif i == 0:
            close = close_post
        else:
            close = close_post * (1 + i * 0.01)
        bars.append({
            "t": ts,
            "o": close * 0.98,
            "h": close * 1.02,
            "l": close * 0.97,
            "c": close,
            "v": 1000000
        })
    return bars


def test_lse_has_data_polygon_not_called():
    """When LSE returns data, Polygon should never be called."""
    from earnings_edge.fwd_factor_ladder import ensure_hist_moves
    
    conn = create_test_db()
    ticker = "AAPL"
    ed = date(2026, 7, 1)
    
    # Mock yfinance to return one earnings date
    mock_df = pd.DataFrame(index=pd.to_datetime([ed - timedelta(days=10)]))
    mock_ticker = MagicMock()
    mock_ticker.get_earnings_dates = MagicMock(return_value=mock_df)
    
    # Mock LSE to return bars
    mock_lse = MagicMock()
    mock_lse.daily_bars = MagicMock(return_value=make_bars(ticker, ed - timedelta(days=10)))
    
    # Mock Polygon (should never be called)
    mock_polygon = MagicMock()
    mock_polygon.get_daily_bars = MagicMock()
    
    with patch("earnings_edge.fwd_factor_ladder._lse_bars_client", return_value=mock_lse):
        with patch("earnings_edge.fwd_factor_ladder._polygon_bars_client", return_value=mock_polygon):
            with patch("yfinance.Ticker", return_value=mock_ticker):
                result = ensure_hist_moves(ticker, today=ed)
    
    # Verify Polygon was never called
    mock_polygon.get_daily_bars.assert_not_called()
    
    # Verify LSE was called
    mock_lse.daily_bars.assert_called_once()
    
    # Verify row was inserted
    with db_engine.session_scope() as s:
        rows = [dict(r) for r in s.execute(
            text("SELECT * FROM snapshots WHERE ticker=:t"), {"t": ticker}).mappings().all()]
    assert len(rows) == 1
    assert rows[0]["actual_move_pct"] == 5.0  # actual_move_pct
    assert result == 1


def test_lse_returns_empty_polygon_used():
    """When LSE returns empty list, Polygon should be used."""
    from earnings_edge.fwd_factor_ladder import ensure_hist_moves
    
    conn = create_test_db()
    ticker = "INTL"
    ed = date(2026, 7, 1)
    
    # Mock yfinance to return one earnings date
    mock_df = pd.DataFrame(index=pd.to_datetime([ed - timedelta(days=10)]))
    mock_ticker = MagicMock()
    mock_ticker.get_earnings_dates = MagicMock(return_value=mock_df)
    
    # Mock LSE to return empty (foreign ticker)
    mock_lse = MagicMock()
    mock_lse.daily_bars = MagicMock(return_value=[])
    
    # Mock Polygon to return bars
    mock_polygon = MagicMock()
    mock_polygon.get_daily_bars = MagicMock(return_value=make_bars(ticker, ed - timedelta(days=10)))
    
    with patch("earnings_edge.fwd_factor_ladder._lse_bars_client", return_value=mock_lse):
        with patch("earnings_edge.fwd_factor_ladder._polygon_bars_client", return_value=mock_polygon):
            with patch("yfinance.Ticker", return_value=mock_ticker):
                result = ensure_hist_moves(ticker, today=ed)
    
    # Verify both were called (LSE first, then Polygon)
    mock_lse.daily_bars.assert_called_once()
    mock_polygon.get_daily_bars.assert_called_once()
    
    # Verify Polygon call used ISO date strings
    call_args = mock_polygon.get_daily_bars.call_args[0]
    assert call_args[0] == ticker
    assert isinstance(call_args[1], str)  # ISO date string
    assert isinstance(call_args[2], str)  # ISO date string
    
    # Verify row was inserted with Backfill timing
    with db_engine.session_scope() as s:
        rows = [dict(r) for r in s.execute(
            text("SELECT * FROM snapshots WHERE ticker=:t"), {"t": ticker}).mappings().all()]
    assert len(rows) == 1
    assert rows[0]["timing"] == "Backfill"  # timing
    assert rows[0]["actual_move_pct"] == 5.0  # actual_move_pct
    assert result == 1


def test_both_sources_fail_returns_existing_count():
    """When both LSE and Polygon fail, returns pre-existing count without exception."""
    from earnings_edge.fwd_factor_ladder import ensure_hist_moves
    
    conn = create_test_db()
    ticker = "FAIL"
    ed = date(2026, 7, 1)
    
    # Pre-existing row
    with db_engine.session_scope() as s:
        s.execute(text("INSERT INTO snapshots (ticker, earnings_date, scan_date, timing, actual_move_pct, outcome_fetched_at) "
                       "VALUES (:t, :ed, :sc, 'Backfill', :m, :f)"),
                  {"t": ticker, "ed": "2026-06-01", "sc": "2026-06-02", "m": 3.5, "f": "2026-06-02"})
    
    # Mock yfinance to return one earnings date
    mock_df = pd.DataFrame(index=pd.to_datetime([ed - timedelta(days=10)]))
    mock_ticker = MagicMock()
    mock_ticker.get_earnings_dates = MagicMock(return_value=mock_df)
    
    # Mock LSE to raise exception
    mock_lse = MagicMock()
    mock_lse.daily_bars = MagicMock(side_effect=Exception("LSE error"))
    
    # Mock Polygon to raise exception
    mock_polygon = MagicMock()
    mock_polygon.get_daily_bars = MagicMock(side_effect=Exception("Polygon error"))
    
    with patch("earnings_edge.fwd_factor_ladder._lse_bars_client", return_value=mock_lse):
        with patch("earnings_edge.fwd_factor_ladder._polygon_bars_client", return_value=mock_polygon):
            with patch("yfinance.Ticker", return_value=mock_ticker):
                result = ensure_hist_moves(ticker, today=ed)
    
    # Should return the pre-existing count
    assert result == 1
    
    # No new rows should be inserted
    with db_engine.session_scope() as s:
        rows = [dict(r) for r in s.execute(
            text("SELECT * FROM snapshots WHERE ticker=:t"), {"t": ticker}).mappings().all()]
    assert len(rows) == 1


def test_early_exit_when_min_events_satisfied():
    """Stop fetching once have + written >= min_events."""
    from earnings_edge.fwd_factor_ladder import ensure_hist_moves
    
    conn = create_test_db()
    ticker = "MSFT"
    ed = date(2026, 7, 1)
    
    # Pre-existing row
    with db_engine.session_scope() as s:
        s.execute(text("INSERT INTO snapshots (ticker, earnings_date, scan_date, timing, actual_move_pct, outcome_fetched_at) "
                       "VALUES (:t, :ed, :sc, 'Backfill', :m, :f)"),
                  {"t": ticker, "ed": "2026-03-01", "sc": "2026-03-02", "m": 2.5, "f": "2026-03-02"})
    
    # Mock yfinance to return 4 earnings dates
    dates = [ed - timedelta(days=i*90) for i in range(1, 5)]
    mock_df = pd.DataFrame(index=pd.to_datetime(dates))
    mock_ticker = MagicMock()
    mock_ticker.get_earnings_dates = MagicMock(return_value=mock_df)
    
    # Mock LSE to return bars
    mock_lse = MagicMock()
    mock_lse.daily_bars = MagicMock(side_effect=[
        make_bars(ticker, dates[0]),
        make_bars(ticker, dates[1]),
        make_bars(ticker, dates[2]),  # Should not be called
        make_bars(ticker, dates[3]),  # Should not be called
    ])
    
    with patch("earnings_edge.fwd_factor_ladder._lse_bars_client", return_value=mock_lse):
        with patch("earnings_edge.fwd_factor_ladder._polygon_bars_client", return_value=None):
            with patch("yfinance.Ticker", return_value=mock_ticker):
                # min_events=3, already have 1, so should stop after 2 more
                result = ensure_hist_moves(ticker, today=ed, min_events=3)
    
    # Should have fetched only 2 dates (early exit after reaching 3 total)
    assert mock_lse.daily_bars.call_count == 2
    assert result == 3
    
    # Should have 3 total rows (1 pre-existing + 2 new)
    with db_engine.session_scope() as s:
        rows = [dict(r) for r in s.execute(
            text("SELECT * FROM snapshots WHERE ticker=:t"), {"t": ticker}).mappings().all()]
    assert len(rows) == 3


def test_existing_good_outcome_not_overwritten():
    """Existing rows with good outcomes should not be overwritten."""
    from earnings_edge.fwd_factor_ladder import ensure_hist_moves
    
    conn = create_test_db()
    ticker = "GOOGL"
    ed = date(2026, 7, 1)
    earnings_date = (ed - timedelta(days=10)).isoformat()
    
    # Pre-existing row with good outcome
    with db_engine.session_scope() as s:
        s.execute(text("INSERT INTO snapshots (ticker, earnings_date, scan_date, timing, actual_move_pct, outcome_fetched_at, pre_earnings_close) "
                       "VALUES (:t, :ed, :sc, 'Backfill', :m, :f, :p)"),
                  {"t": ticker, "ed": earnings_date, "sc": "2026-06-15", "m": 4.5, "f": "2026-06-15", "p": 150.0})
    
    # Mock yfinance to return same earnings date
    mock_df = pd.DataFrame(index=pd.to_datetime([ed - timedelta(days=10)]))
    mock_ticker = MagicMock()
    mock_ticker.get_earnings_dates = MagicMock(return_value=mock_df)
    
    # Mock LSE to return bars (different values)
    mock_lse = MagicMock()
    mock_lse.daily_bars = MagicMock(return_value=make_bars(ticker, ed - timedelta(days=10), 200, 210))
    
    with patch("earnings_edge.fwd_factor_ladder._lse_bars_client", return_value=mock_lse):
        with patch("earnings_edge.fwd_factor_ladder._polygon_bars_client", return_value=None):
            with patch("yfinance.Ticker", return_value=mock_ticker):
                result = ensure_hist_moves(ticker, today=ed)
    
    # Should return 1 (the pre-existing row)
    assert result == 1
    
    # Row should not be modified
    with db_engine.session_scope() as s:
        rows = [dict(r) for r in s.execute(
            text("SELECT * FROM snapshots WHERE ticker=:t"), {"t": ticker}).mappings().all()]
    assert len(rows) == 1
    assert rows[0]["actual_move_pct"] == 4.5  # actual_move_pct unchanged
    assert rows[0]["pre_earnings_close"] == 150.0  # pre_earnings_close unchanged


def test_lse_raises_exception_polygon_fallback():
    """When LSE raises an exception, Polygon should be used."""
    from earnings_edge.fwd_factor_ladder import ensure_hist_moves
    
    conn = create_test_db()
    ticker = "ERROR"
    ed = date(2026, 7, 1)
    
    # Mock yfinance to return one earnings date
    mock_df = pd.DataFrame(index=pd.to_datetime([ed - timedelta(days=10)]))
    mock_ticker = MagicMock()
    mock_ticker.get_earnings_dates = MagicMock(return_value=mock_df)
    
    # Mock LSE to raise exception
    mock_lse = MagicMock()
    mock_lse.daily_bars = MagicMock(side_effect=Exception("LSE network error"))
    
    # Mock Polygon to return bars
    mock_polygon = MagicMock()
    mock_polygon.get_daily_bars = MagicMock(return_value=make_bars(ticker, ed - timedelta(days=10)))
    
    with patch("earnings_edge.fwd_factor_ladder._lse_bars_client", return_value=mock_lse):
        with patch("earnings_edge.fwd_factor_ladder._polygon_bars_client", return_value=mock_polygon):
            with patch("yfinance.Ticker", return_value=mock_ticker):
                result = ensure_hist_moves(ticker, today=ed)
    
    # Verify both were called
    mock_lse.daily_bars.assert_called_once()
    mock_polygon.get_daily_bars.assert_called_once()
    
    # Verify row was inserted
    with db_engine.session_scope() as s:
        rows = [dict(r) for r in s.execute(
            text("SELECT * FROM snapshots WHERE ticker=:t"), {"t": ticker}).mappings().all()]
    assert len(rows) == 1
    assert rows[0]["actual_move_pct"] == 5.0  # actual_move_pct
    assert result == 1


def test_no_api_keys_returns_existing():
    """When neither LSE nor Polygon API keys are configured, returns existing count."""
    from earnings_edge.fwd_factor_ladder import ensure_hist_moves
    
    conn = create_test_db()
    ticker = "NOAPI"
    ed = date(2026, 7, 1)
    
    # Pre-existing row
    with db_engine.session_scope() as s:
        s.execute(text("INSERT INTO snapshots (ticker, earnings_date, scan_date, timing, actual_move_pct, outcome_fetched_at) "
                       "VALUES (:t, :ed, :sc, 'Backfill', :m, :f)"),
                  {"t": ticker, "ed": "2026-06-01", "sc": "2026-06-02", "m": 3.5, "f": "2026-06-02"})
    
    # Mock clients to return None (no API keys)
    with patch("earnings_edge.fwd_factor_ladder._lse_bars_client", return_value=None):
        with patch("earnings_edge.fwd_factor_ladder._polygon_bars_client", return_value=None):
            result = ensure_hist_moves(ticker, today=ed)
    
    # Should return the pre-existing count
    assert result == 1
    
    # No new rows should be inserted
    with db_engine.session_scope() as s:
        rows = [dict(r) for r in s.execute(
            text("SELECT * FROM snapshots WHERE ticker=:t"), {"t": ticker}).mappings().all()]
    assert len(rows) == 1


def test_update_existing_row_with_null_outcome():
    """Existing rows with NULL outcome should be updated."""
    from earnings_edge.fwd_factor_ladder import ensure_hist_moves
    
    conn = create_test_db()
    ticker = "UPDATE"
    ed = date(2026, 7, 1)
    earnings_date = (ed - timedelta(days=10)).isoformat()
    
    # Pre-existing row with NULL outcome
    with db_engine.session_scope() as s:
        s.execute(text(
            "INSERT INTO snapshots (ticker, earnings_date, scan_date, timing, actual_move_pct, outcome_fetched_at) "
            "VALUES (:t, :ed, :sc, 'Backfill', NULL, NULL)"),
            {"t": ticker, "ed": earnings_date, "sc": "2026-06-15"})
    
    # Mock yfinance to return same earnings date
    mock_df = pd.DataFrame(index=pd.to_datetime([ed - timedelta(days=10)]))
    mock_ticker = MagicMock()
    mock_ticker.get_earnings_dates = MagicMock(return_value=mock_df)
    
    # Mock Polygon to return bars (LSE unavailable)
    mock_polygon = MagicMock()
    mock_polygon.get_daily_bars = MagicMock(return_value=make_bars(ticker, ed - timedelta(days=10)))
    
    with patch("earnings_edge.fwd_factor_ladder._lse_bars_client", return_value=None):
        with patch("earnings_edge.fwd_factor_ladder._polygon_bars_client", return_value=mock_polygon):
            with patch("yfinance.Ticker", return_value=mock_ticker):
                result = ensure_hist_moves(ticker, today=ed)
    
    # Should return 1
    assert result == 1
    
    # Row should be updated
    with db_engine.session_scope() as s:
        rows = [dict(r) for r in s.execute(
            text("SELECT * FROM snapshots WHERE ticker=:t"), {"t": ticker}).mappings().all()]
    assert len(rows) == 1
    assert rows[0]["actual_move_pct"] == 5.0  # actual_move_pct now set
    assert rows[0]["outcome_fetched_at"] is not None  # outcome_fetched_at now set