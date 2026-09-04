"""Equity snapshots and daily PnL for the risk layer.

A scheduled job (bot) calls ``snapshot_equity`` during market hours; the risk
manager uses the first snapshot of the day vs. current equity for the daily
loss limit, and %-based sizers read the latest equity.
"""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime

from earnings_edge.db import (
    equity_snapshots_day_start,
    equity_snapshots_insert,
    equity_snapshots_latest,
)

logger = logging.getLogger("framework.risk.equity")


def _utcnow() -> str:
    return datetime.now(UTC).isoformat()


def snapshot_equity(client, source: str = "alpaca") -> dict:
    """Fetch account equity from the broker and persist a snapshot row."""
    acct = client.get_account()
    row = {
        "ts": _utcnow(),
        "equity": float(acct.get("equity", 0) or 0),
        "buying_power": float(acct.get("buying_power", 0) or 0),
        "portfolio_value": float(acct.get("portfolio_value", 0) or 0),
        "source": source,
    }
    equity_snapshots_insert(
        ts=row["ts"],
        equity=row["equity"],
        buying_power=row["buying_power"],
        portfolio_value=row["portfolio_value"],
        source=row["source"],
    )
    return row


def latest_equity() -> dict | None:
    return equity_snapshots_latest()


def day_start_equity(on: date | None = None) -> float | None:
    """First snapshot of the given UTC day (daily-loss baseline)."""
    return equity_snapshots_day_start(on)


def daily_pnl(equity_now: float, on: date | None = None) -> float | None:
    """Current equity minus day-start equity (None when no baseline)."""
    start = day_start_equity(on)
    if start is None or start <= 0:
        return None
    return equity_now - start
