"""Backtest-only strategy namespace.

Everything in this package evaluates HISTORICAL data (snapshots,
calendar_call_trades, outcomes) and must NEVER feed live execution.
Live trade generation is earnings_edge/live_signals.py, driven by the
TOML strategies in strategies/*.toml. Shared types (Trade, DataBundle,
StrategyResult) live in earnings_edge/trading_types.py.
"""
from earnings_edge.backtest.calendar import (
    CalendarCallHighConviction,
    CalendarCallNoML,
    CalendarCallStrategy,
    DebitSizeExploit,
    EarningsQualityStrategy,
    IVRVMeanReversion,
    ShortStraddleStrategy,
    StockDriftStrategy,
    TermStructureSteepener,
    get_strategy,
    list_strategies,
    register,
)
from earnings_edge.backtest.positional import (
    POSITIONAL_STRATEGIES,
    DirectionalCall,
    DirectionalPut,
    LongStraddle,
    ShortStraddle,
    VolRiskPremium,
    run_positional,
)

__all__ = [
    "POSITIONAL_STRATEGIES",
    "CalendarCallHighConviction",
    "CalendarCallNoML",
    "CalendarCallStrategy",
    "DebitSizeExploit",
    "DirectionalCall",
    "DirectionalPut",
    "EarningsQualityStrategy",
    "IVRVMeanReversion",
    "LongStraddle",
    "ShortStraddle",
    "ShortStraddleStrategy",
    "StockDriftStrategy",
    "TermStructureSteepener",
    "VolRiskPremium",
    "get_strategy",
    "list_strategies",
    "register",
    "run_positional",
]
