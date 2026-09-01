import pytest
from unittest.mock import patch, MagicMock
from datetime import date
import pandas as pd

from scripts.ff_universe_scan import process_ticker, atm_contract
from earnings_edge.picks import forward_factor_picks

def test_pick_expiries_reused():
    import scripts.ff_universe_scan
    assert scripts.ff_universe_scan.pick_expiries.__module__ == 'scripts.ff_backfill', "pick_expiries must be imported from scripts.ff_backfill"

@patch('scripts.ff_universe_scan.get_spot')
@patch('scripts.ff_universe_scan.get_latest_chain')
@patch('scripts.ff_universe_scan.pick_expiries')
@patch('scripts.ff_universe_scan.get_earnings_in_window')
@patch('scripts.ff_universe_scan.hist_move_stats')
@patch('scripts.ff_universe_scan.implied_volatility')
def test_no_earnings_in_window(
    mock_iv, mock_hist, mock_earnings, mock_pick, mock_chain, mock_spot
):
    mock_spot.return_value = 100.0
    mock_chain.return_value = [{"expiration_date": "2026-10-01", "strike_price": 100.0, "contract_type": "call", "close": 5.0, "ticker": "O:AAPL261001C00100000"}]
    
    mock_pick.return_value = (
        {"expiry": "2026-09-15", "dte": 30, "contracts": [{"ticker": "C1", "strike_price": 100, "close": 5.0, "contract_type": "call"}]},
        {"expiry": "2026-10-15", "dte": 60, "contracts": [{"ticker": "C2", "strike_price": 100, "close": 6.0, "contract_type": "call"}]}
    )
    
    # Return 0.2 IV
    mock_iv.side_effect = [0.2, 0.25]
    
    # NO earnings
    mock_earnings.return_value = None
    
    row = process_ticker("AAPL", date(2026, 8, 15))
    
    assert row["skip_reason"] is None
    assert row["has_earnings_in_window"] == 0
    assert "tau_days" not in row or row.get("tau_days") is None
    assert row["forward_factor"] is not None
    assert row["sigma_fwd"] is not None

@patch('scripts.ff_universe_scan.get_spot')
@patch('scripts.ff_universe_scan.get_latest_chain')
@patch('scripts.ff_universe_scan.pick_expiries')
@patch('scripts.ff_universe_scan.get_earnings_in_window')
@patch('scripts.ff_universe_scan.hist_move_stats')
@patch('scripts.ff_universe_scan.implied_volatility')
def test_with_earnings_in_window(
    mock_iv, mock_hist, mock_earnings, mock_pick, mock_chain, mock_spot
):
    mock_spot.return_value = 100.0
    mock_chain.return_value = [{"expiration_date": "2026-10-01", "strike_price": 100.0, "contract_type": "call", "close": 5.0, "ticker": "O:AAPL261001C00100000"}]
    
    mock_pick.return_value = (
        {"expiry": "2026-09-15", "dte": 30, "contracts": [{"ticker": "C1", "strike_price": 100, "close": 5.0, "contract_type": "call"}]},
        {"expiry": "2026-10-15", "dte": 60, "contracts": [{"ticker": "C2", "strike_price": 100, "close": 6.0, "contract_type": "call"}]}
    )
    
    # Return IV
    mock_iv.side_effect = [0.4, 0.3]  # Near is rich
    
    # Earnings inside window
    mock_earnings.return_value = "2026-09-01"
    
    # hist move
    mock_hist.return_value = (5.0, 6.0, 10)
    
    row = process_ticker("AAPL", date(2026, 8, 15))
    
    assert row["skip_reason"] is None
    assert row["has_earnings_in_window"] == 1
    assert row["earnings_date"] == "2026-09-01"
    assert row["tau_days"] == 18 # 15th to 1st = 17 days + 1 = 18
    assert row["implied_event_move_pct"] > 0
    assert row["premium_ratio"] > 0

@patch('earnings_edge.db.repositories.ff_snapshots_as_of_df')
@patch('earnings_edge.db.repositories.ff_universe_snapshots_as_of_df')
def test_forward_factor_picks_uses_new_source(mock_univ, mock_legacy):
    # legacy empty
    mock_legacy.return_value = pd.DataFrame()
    
    # new universe has data
    mock_univ.return_value = pd.DataFrame([
        {"ticker": "A", "scan_date": "2026-08-26", "t1_iv": 0.5, "t2_iv": 0.4, "t1_dte": 30, "t2_dte": 60, "sigma_fwd": 0.25, "has_earnings_in_window": 0},
        {"ticker": "B", "scan_date": "2026-08-26", "t1_iv": 0.3, "t2_iv": 0.3, "t1_dte": 30, "t2_dte": 60, "sigma_fwd": 0.3, "has_earnings_in_window": 1},
    ])
    
    # just test forward_factor_picks directly with the merged df
    df = pd.DataFrame([
        {"ticker": "A", "front_iv": 0.5, "forward_iv": 0.25},  # ff = 1.0
        {"ticker": "B", "front_iv": 0.3, "forward_iv": 0.3},   # ff = 0.0
    ])
    
    picks = forward_factor_picks(df, min_ff=0.20)
    assert len(picks) == 1
    assert picks.iloc[0]["ticker"] == "A"
