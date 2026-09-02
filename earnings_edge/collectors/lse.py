"""London Strategic Edge collector for historical backfills.

Wraps the official ``lse-data`` client (REST vault) with retry/circuit-breaker
semantics from BaseCollector. Shapes responses to mirror the PolygonClient used
by the backfill scripts (``polygon_*_backfill.py``) so either source drops into
the same pipeline via a ``--source`` flag.

Caveats vs Polygon:
- The option "chain" is a current vault snapshot: it includes recently expired
  contracts but cannot be queried as-of a historical date. Contract coverage
  further back than a few weeks is not guaranteed.
- No historical greeks: ``option_close`` returns contract 1m-bar closes from
  which IV must be solved locally (same as the Polygon path).
- Plan limits: 200 calls/min, 5000 rows/request, 5 export jobs/hour
  (``history()`` bulk Parquet exports — not used by these methods).

Key: ``LSE_API_KEY`` env var.
"""

from __future__ import annotations

import logging
import time
from datetime import date, datetime, timedelta
from typing import Any, Optional

import requests

from .base import BaseCollector
from ..settings import get_settings

logger = logging.getLogger("earnings_edge.collectors.lse")

LSE_API_BASE = "https://api.londonstrategicedge.com"


class LSECollector(BaseCollector):
    """LSE vault client with PolygonClient-compatible method shapes."""

    def __init__(self, api_key: Optional[str] = None, client=None, sleep: float = 0.35):
        super().__init__(
            name="lse",
            max_retries=3,
            base_delay=2.0,
            circuit_threshold=5,
        )
        self.api_key = api_key if api_key is not None else get_settings().lse_api_key
        self._client = client  # injectable; lazily constructed from the key
        self.sleep = sleep  # pacing between calls (200/min plan cap)
        self._chain_cache: dict[str, list[dict]] = {}

    @property
    def client(self):
        if self._client is None:
            if not self.api_key:
                raise ValueError("LSE_API_KEY not set")
            from lse import LSE  # lazy import: optional dependency
            self._client = LSE(api_key=self.api_key)
        return self._client

    def _call(self, fn):
        result = self.with_retry(fn)
        if self.sleep > 0:
            time.sleep(self.sleep)
        return result

    # -- PolygonClient-compatible surface ------------------------------------

    def daily_bars(self, ticker: str, start: date, end: date, limit: int = 5000) -> list[dict]:
        """Daily OHLCV bars, Polygon-shaped keys (o/h/l/c/v/t).

        A 404 "no candle data" is a deterministic per-ticker response (OTC /
        foreign / delisted names the vault does not cover), not a service
        failure: swallow it to [] so it neither burns retries nor trips the
        circuit breaker for the tickers behind it in a batch run.
        """
        def _fetch():
            try:
                return self.client.candles(
                    ticker, "1d", start=start.isoformat(), end=end.isoformat(),
                    limit=limit, order="asc",
                )
            except Exception as exc:
                msg = str(exc)
                if "[404]" in msg and "no candle data" in msg:
                    return []
                raise

        rows = self._call(_fetch) or []
        return [
            {
                "o": r["open"], "h": r["high"], "l": r["low"], "c": r["close"],
                "v": r.get("volume") or 0,
                "t": int(datetime.fromisoformat(r["timestamp"].replace("Z", "+00:00")).timestamp() * 1000),
            }
            for r in rows
        ]

    def option_contracts(
        self,
        underlying: str,
        as_of: Optional[date] = None,
        expiry_gte: Optional[date] = None,
        expiry_lte: Optional[date] = None,
        contract_type: Optional[str] = None,
    ) -> list[dict]:
        """Contracts from the vault chain snapshot, Polygon-shaped keys.

        ``as_of`` is accepted for interface compatibility but cannot be
        honoured — the chain is a current snapshot (see module docstring).
        """
        rows = self._chain(underlying)
        out = []
        for r in rows:
            expiry = r.get("expiry")
            if not expiry:
                continue
            if expiry_gte and expiry < expiry_gte.isoformat():
                continue
            if expiry_lte and expiry > expiry_lte.isoformat():
                continue
            if contract_type and r.get("contract_type") != contract_type:
                continue
            out.append({
                "ticker": r.get("ticker", ""),
                "strike_price": r.get("strike"),
                "expiration_date": expiry,
                "contract_type": r.get("contract_type"),
            })
        return out

    def option_close(self, contract_ticker: str, as_of: date, lookback_days: int = 4) -> Optional[float]:
        """Most recent 1m-bar close on/before ``as_of`` (looks back a few days)."""
        rows = self._call(lambda: self.client.option_candles(
            contract_ticker,
            start=(as_of - timedelta(days=lookback_days)).isoformat(),
            end=(as_of + timedelta(days=1)).isoformat(),
            order="asc", limit=5000,
        )) or []
        cutoff = as_of + timedelta(days=1)
        closes = [
            float(r["close"]) for r in rows
            if r.get("close") is not None
            and datetime.fromisoformat(
                (r.get("minute") or r.get("timestamp")).replace("Z", "+00:00")
            ).date() < cutoff
        ]
        return closes[-1] if closes else None

    # -- extras ----------------------------------------------------------------

    def usage(self) -> dict:
        """Plan allowance snapshot (calls/min, export budget, byte caps)."""
        if not self.api_key:
            return {}
        resp = requests.get(
            f"{LSE_API_BASE}/vault/usage",
            headers={"x-api-key": self.api_key}, timeout=15,
        )
        resp.raise_for_status()
        return resp.json()

    # -- internals -------------------------------------------------------------

    def _chain(self, underlying: str) -> list[dict]:
        if underlying not in self._chain_cache:
            rows = self._call(lambda: self.client.options(underlying, limit=5000)) or []
            self._chain_cache[underlying] = rows
            while len(self._chain_cache) > 32:
                self._chain_cache.pop(next(iter(self._chain_cache)))
        return self._chain_cache[underlying]
