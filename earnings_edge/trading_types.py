"""
Shared trading abstractions — the common currency of backtest AND live paths.

These types are strategy-agnostic: backtest strategies (earnings_edge/backtest/),
the live signal layer (live_signals.py), the approval flow (trade_approval.py),
and the execution bridge (alpaca_bridge.py) all exchange Trade / StrategyResult /
DataBundle objects. Keep this module free of strategy logic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, List, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Trade abstraction
# ---------------------------------------------------------------------------

@dataclass
class Trade:
    """A single strategy trade to be backtested or scored live."""
    ticker: str
    earnings_date: date
    scan_date: date
    strategy: str
    side: str                 # LONG, SHORT, SPREAD, CALENDAR
    entry_price: float        # raw cost basis (debit or mid-price or spread)
    exit_price: float = 0.0   # 0 = not yet closed
    pnl: float = 0.0          # absolute PnL (dollars for options, percent for stock)
    pnl_pct: float = 0.0      # return_on_debit for options, simple return for stock
    features: Dict[str, Any] = field(default_factory=dict)
    model_score: Optional[float] = None
    ml_decision: str = "SKIP"
    notes: str = ""

    def is_winner(self) -> bool:
        return self.pnl > 0


# ---------------------------------------------------------------------------
# DataBundle — everything a strategy needs (no DB/network calls)
# ---------------------------------------------------------------------------

@dataclass
class DataBundle:
    snapshots: pd.DataFrame
    calendar_trades: pd.DataFrame
    live_candidates: pd.DataFrame
    scan_outputs: pd.DataFrame
    options_chain: pd.DataFrame = field(default_factory=pd.DataFrame)

    @classmethod
    def from_db(cls, db_path: str | None = None) -> DataBundle:
        from earnings_edge.db import configure, get_engine

        if db_path is not None:
            configure(db_path)
        engine = get_engine()

        snapshots = pd.read_sql("SELECT * FROM snapshots", engine)
        calendar_trades = pd.read_sql("SELECT * FROM calendar_call_trades", engine)
        live_candidates = pd.read_sql("SELECT * FROM live_calendar_candidates", engine)
        scan_outputs = pd.read_sql("SELECT * FROM scanner_scan_outputs", engine)
        options_chain = pd.read_sql("SELECT * FROM options_chain", engine)

        return cls(
            snapshots=snapshots,
            calendar_trades=calendar_trades,
            live_candidates=live_candidates,
            scan_outputs=scan_outputs,
            options_chain=options_chain,
        )


# ---------------------------------------------------------------------------
# Result of one strategy backtest
# ---------------------------------------------------------------------------

@dataclass
class StrategyResult:
    name: str
    trades: List[Trade]
    summary: Dict[str, Any] = field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame()
        rows = []
        for t in self.trades:
            rows.append({
                "ticker": t.ticker,
                "earnings_date": t.earnings_date,
                "scan_date": t.scan_date,
                "strategy": t.strategy,
                "side": t.side,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "pnl": t.pnl,
                "pnl_pct": t.pnl_pct,
                "model_score": t.model_score,
                "ml_decision": t.ml_decision,
                "notes": t.notes,
            })
        return pd.DataFrame(rows)
