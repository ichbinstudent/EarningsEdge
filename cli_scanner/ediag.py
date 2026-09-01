#!/usr/bin/env python3
"""Quick diagnostic: check DB state for today's paper trade run."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from earnings_edge.trading_types import DataBundle
from earnings_edge.backtest.calendar import get_strategy
from earnings_edge.backtest.positional import POSITIONAL_STRATEGIES
from datetime import date, timedelta
import pandas as pd

b = DataBundle.from_db()
df = b.get_df()
print(f'DB rows: {len(df)}')

today = pd.Timestamp('2026-07-13')
future_mask = pd.to_datetime(df['earnings_date']) >= today
print(f'Future earnings rows: {future_mask.sum()}')

strategies = ['calendar_call_ml', 'debit_size_exploit', 'short_straddle', 'vol_risk_premium', 'earnings_quality']
for s in strategies:
    try:
        if s in POSITIONAL_STRATEGIES:
            strat = POSITIONAL_STRATEGIES[s]()
        else:
            strat = get_strategy(s)
        result = strat.run(b)
        print(f'{s}: {len(result.trades)} signals')
    except Exception as e:
        print(f'{s}: ERROR - {e}')
