"""Tradegate BSX public index JSON (bid/ask snapshots, no auth).

The exchange website refreshes ``/json/indizes-{ISIN}.json`` for each listed
index. That payload has a live two-sided book per constituent; last-trade is
not included, so callers should use mid.
"""

from __future__ import annotations

import logging
from typing import Optional

import requests

logger = logging.getLogger(__name__)

TRADEGATE_JSON = "https://www.tradegatebsx.com/json/indizes-{isin}.json"

# Liquid Tradegate books we can pull in a handful of requests.
DEFAULT_INDEX_ISINS = (
    "DE000A1EXRV0",  # DAX
    "DE000A1EXRW8",  # MDAX
    "DE000A1EXRY4",  # TecDAX
    "DE000A1EXRX6",  # SDAX
    "EU0009658145",  # EURO STOXX 50
    "US0000000002",  # US top titles
)


class TradegateCollector:
    """Fetch Tradegate bid/ask snapshots for configured index universes."""

    def __init__(
        self,
        index_isins: Optional[tuple[str, ...]] = None,
        session: Optional[requests.Session] = None,
    ):
        self.index_isins = tuple(
            i.strip() for i in (index_isins or DEFAULT_INDEX_ISINS) if i.strip()
        )
        self._session = session or requests.Session()

    def fetch_index_quotes(self) -> list[dict]:
        """Return raw constituent dicts (isin/name/bid/ask/...). Deduped by ISIN."""
        by_isin: dict[str, dict] = {}
        for isin in self.index_isins:
            url = TRADEGATE_JSON.format(isin=isin)
            try:
                res = self._session.get(
                    url,
                    timeout=15,
                    headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"},
                )
            except Exception as exc:
                logger.warning("Tradegate fetch %s failed: %s", isin, exc)
                continue
            if res.status_code != 200:
                logger.warning("Tradegate %s -> %s", isin, res.status_code)
                continue
            try:
                payload = res.json()
            except ValueError:
                logger.warning("Tradegate %s non-JSON", isin)
                continue
            rows = payload.get("indizes") or []
            ts = None
            date_block = payload.get("date") or {}
            if isinstance(date_block, dict):
                ts = date_block.get("timestamp")
            for row in rows:
                if not isinstance(row, dict):
                    continue
                key = str(row.get("isin") or "").strip()
                if not key:
                    continue
                out = dict(row)
                out["_index_isin"] = isin
                if ts is not None:
                    out["_timestamp"] = ts
                by_isin[key] = out
        return list(by_isin.values())
