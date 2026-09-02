"""Point-in-time data catalog.

Every dataset ingest records *when the data became available to us*
(``available_at``) alongside *what date it describes* (``as_of_date``). The
backtest engine filters ``available_at <= decision_time`` so simulations
cannot see data before it existed — the lookahead-bias guard.

``pit_safe`` marks sources that can answer true as-of queries. E.g. Polygon's
contracts endpoint supports ``as_of`` (safe); an LSE chain is a current
snapshot (unsafe for historical chain reconstruction — flagged at record
time by the collector).
"""

from __future__ import annotations

import logging
from typing import Optional

from earnings_edge.db import data_catalog_latest, data_catalog_query, data_catalog_upsert

logger = logging.getLogger("framework.data.catalog")


def record(
    dataset: str,
    symbol: Optional[str] = None,
    as_of_date: Optional[str] = None,
    source: str = "unknown",
    available_at: Optional[str] = None,
    pit_safe: bool = True,
) -> None:
    """Record that ``dataset`` rows for ``symbol``/``as_of_date`` are available."""
    data_catalog_upsert(
        dataset,
        symbol=symbol,
        as_of_date=as_of_date,
        source=source,
        available_at=available_at,
        pit_safe=pit_safe,
    )


def available_as_of(
    dataset: str,
    decision_time: str,
    symbol: Optional[str] = None,
    as_of_start: Optional[str] = None,
    as_of_end: Optional[str] = None,
    pit_only: bool = True,
) -> list[str]:
    """Distinct as_of_dates knowable at ``decision_time`` (PIT-safe sources)."""
    return data_catalog_query(
        dataset,
        decision_time,
        symbol=symbol,
        as_of_start=as_of_start,
        as_of_end=as_of_end,
        pit_only=pit_only,
    )


def latest_availability(dataset: str, symbol: Optional[str] = None) -> Optional[dict]:
    """Most recent catalog row for a dataset (freshness check)."""
    return data_catalog_latest(dataset, symbol)
