"""Resilient market-data provider chain: LSE primary, Yahoo fallback.

Motivation: this host's IP gets rate-limited/blocked by Yahoo Finance for days
at a time (observed 2026-07-21 → 07-24: fc.yahoo.com TCP-refused, chart
endpoints 429). When that happens the whole pricing layer (analyzer, validator,
live calendar quotes) collapses even though other backends keep working.

Design:
- ``LSEProvider``     — London Strategic Edge vault via the official
  ``lse-data`` client (key from ``LSE_API_KEY``). Chains carry native
  IV/greeks/volume but no open interest and no bid/ask (bid/ask collapse to
  last_price, which can be stale per-contract). ``oi_available=False``.
- ``YahooProvider``   — thin wrapper over yfinance using the shared curl_cffi
  session (which honours ``YFINANCE_PROXY`` when set).
- ``PolygonProvider`` — re-implements the yfinance-shaped surface
  (history / options_expiries / option_chain) on Polygon endpoints with
  per-endpoint-class adaptive rate limiting. Option quotes are EOD closes
  (no snapshot/greeks entitlement), so IV and delta are computed locally
  via Black-Scholes and bid/ask collapse to the close. Open interest is NOT
  available — chains carry ``oi_available=False`` so callers can skip OI gates.
  NOT part of the default live chain (see below) — reserved for the
  historical backfill/backtest scripts (``polygon_backfill.py`` and
  friends), which use it directly rather than through this module.
- ``ResilientProvider`` — auto mode: health-checks providers in priority
  order, latches to the first working backend, fails over mid-run on errors,
  and periodically re-probes higher-priority providers so a recovered
  connection is picked up again.

Select via ``EARNINGS_PRICE_PROVIDER`` = auto | lse | yahoo | polygon
(default auto = LSE→Yahoo, no Polygon; LSE is skipped when no key is
configured). ``polygon`` mode is available as an explicit opt-in, and
``ResilientProvider(polygon=...)`` still accepts one directly (tests,
one-off scripts) — it's just no longer auto-added to the live chain.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import requests

from .option_math import black_scholes_delta, implied_volatility
from .settings import get_settings

logger = logging.getLogger("earnings_edge.market_data_provider")

POLYGON_BASE = "https://api.polygon.io"
YAHOO_HEALTH_URL = "https://query2.finance.yahoo.com/v8/finance/chart/SPY?range=1d&interval=1d"
RISK_FREE_RATE = 0.04

# yfinance history period → calendar days of range to fetch from Polygon
_PERIOD_DAYS = {"1d": 7, "5d": 10, "1mo": 35, "3mo": 100, "6mo": 200, "1y": 370}

_CHAIN_COLUMNS = [
    "contractSymbol", "strike", "bid", "ask", "lastPrice",
    "impliedVolatility", "openInterest", "volume", "delta", "inTheMoney",
]


@dataclass
class OptionChainData:
    """yfinance-shaped option chain (calls/puts DataFrames)."""

    calls: pd.DataFrame
    puts: pd.DataFrame
    oi_available: bool = True
    source: str = "yahoo"


class _AimdRateLimiter:
    """Additive-increase/multiplicative-decrease minimum-interval limiter."""

    def __init__(self, start: float, minimum: float, maximum: float):
        self._interval = start
        self._min = minimum
        self._max = maximum
        self._next_ok = 0.0
        self._lock = threading.Lock()

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait = self._next_ok - now
            if wait > 0:
                time.sleep(wait)
            self._next_ok = max(time.monotonic(), self._next_ok) + self._interval

    def success(self) -> None:
        with self._lock:
            self._interval = max(self._min, self._interval * 0.9)

    def throttled(self) -> None:
        with self._lock:
            self._interval = min(self._max, self._interval * 2)


class YahooProvider:
    """yfinance backend (uses the shared curl_cffi session, proxy-aware)."""

    name = "yahoo"
    max_expiries_hint: Optional[int] = None  # full chains are cheap on Yahoo

    def __init__(self, session=None):
        if session is None:
            from .config import session as default_session
            session = default_session
        self._session = session

    def _ticker(self, ticker: str):
        import yfinance as yf
        return yf.Ticker(ticker, session=self._session)

    def healthy(self, timeout: float = 6.0) -> bool:
        """Probe the Yahoo chart endpoint through the configured session."""
        try:
            resp = self._session.get(YAHOO_HEALTH_URL, timeout=timeout)
            return getattr(resp, "status_code", 0) == 200
        except Exception as exc:
            logger.info("Yahoo health check failed: %s", exc)
            return False

    def history(self, ticker: str, period: str = "1d") -> pd.DataFrame:
        df = self._ticker(ticker).history(period=period)
        # Yahoo occasionally emits an all-NaN placeholder row for the most
        # recent session — it poisons rolling vol windows (Yang-Zhang).
        if not df.empty:
            df = df.dropna(subset=[c for c in ("Open", "High", "Low", "Close") if c in df.columns])
        return df

    def options_expiries(self, ticker: str) -> list[str]:
        return list(self._ticker(ticker).options or [])

    def option_chain(self, ticker: str, expiry: str) -> OptionChainData:
        chain = self._ticker(ticker).option_chain(expiry)
        return OptionChainData(calls=chain.calls, puts=chain.puts,
                               oi_available=True, source=self.name)


class PolygonProvider:
    """Polygon.io backend with a yfinance-shaped surface.

    Entitlements on the current plan: stock aggs (fast), options aggs
    (~5 req/min sustained), options contracts reference. NO snapshot/greeks —
    IV/delta are computed locally from EOD closes via Black-Scholes.
    """

    name = "polygon"
    max_expiries_hint: Optional[int] = 3  # keep options-class calls bounded

    def __init__(self, api_key: Optional[str] = None, http: Optional[requests.Session] = None):
        self._key = api_key if api_key is not None else get_settings().polygon_api_key
        self._http = http or requests.Session()
        self._limiters = {
            "stock": _AimdRateLimiter(0.3, 0.15, 10.0),
            "options": _AimdRateLimiter(12.0, 10.0, 60.0),
            "reference": _AimdRateLimiter(1.0, 0.3, 30.0),
        }
        self._contracts_cache: dict[str, list[dict]] = {}
        self._grouped_cache: dict[str, dict[str, dict]] = {}
        if not self._key:
            logger.warning("POLYGON_API_KEY not set — Polygon provider will fail")

    # -- HTTP plumbing -----------------------------------------------------

    def _get(self, path: str, params: Optional[dict] = None, kind: str = "stock") -> dict:
        if not self._key:
            raise ValueError("POLYGON_API_KEY not set")
        params = dict(params or {})
        params["apiKey"] = self._key
        limiter = self._limiters[kind]
        last_exc: Optional[Exception] = None
        for attempt in range(4):
            limiter.acquire()
            try:
                resp = self._http.get(f"{POLYGON_BASE}{path}", params=params, timeout=20)
                if resp.status_code == 429:
                    limiter.throttled()
                    logger.info("Polygon 429 on %s (attempt %d)", path, attempt + 1)
                    continue
                resp.raise_for_status()
                limiter.success()
                return resp.json()
            except Exception as exc:  # includes HTTPError
                last_exc = exc
                if isinstance(exc, requests.HTTPError) and exc.response is not None \
                        and exc.response.status_code in (403, 404):
                    raise
                limiter.throttled()
        raise ValueError(f"Polygon GET {path} failed after retries: {last_exc}")

    # -- stock data ----------------------------------------------------------

    def _grouped_daily(self, day: str) -> dict[str, dict]:
        """All US stock bars for one day in a single call (cached)."""
        if day not in self._grouped_cache:
            data = self._get(
                f"/v2/aggs/grouped/locale/us/market/stocks/{day}",
                {"adjusted": "true"}, kind="stock",
            )
            self._grouped_cache[day] = {
                r["T"]: r for r in data.get("results", [])
            }
            # keep cache bounded
            while len(self._grouped_cache) > 4:
                self._grouped_cache.pop(next(iter(self._grouped_cache)))
        return self._grouped_cache[day]

    def _latest_bar(self, ticker: str) -> Optional[dict]:
        """Most recent daily bar via grouped daily, falling back to /prev."""
        for back in range(0, 7):
            day = date.today() - timedelta(days=back)
            if day.weekday() >= 5:  # market closed Sat/Sun (grouped 403s)
                continue
            try:
                bars = self._grouped_daily(day.isoformat())
            except Exception as exc:
                logger.info("grouped daily %s failed: %s", day, exc)
                continue
            if ticker in bars:
                return bars[ticker]
            if bars:  # market was open, ticker just not in it
                return None
        # fallback: previous-close endpoint
        try:
            data = self._get(f"/v2/aggs/ticker/{ticker}/prev", {"adjusted": "true"})
            results = data.get("results", [])
            return results[0] if results else None
        except Exception as exc:
            logger.info("prev bar for %s failed: %s", ticker, exc)
            return None

    def history(self, ticker: str, period: str = "1d") -> pd.DataFrame:
        """yfinance-shaped OHLCV DataFrame (empty when no data)."""
        if period == "1d":
            bar = self._latest_bar(ticker)
            if not bar:
                return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
            return self._bars_to_df([bar])

        days = _PERIOD_DAYS.get(period, 100)
        to_day = date.today()
        from_day = to_day - timedelta(days=days)
        try:
            data = self._get(
                f"/v2/aggs/ticker/{ticker}/range/1/day/{from_day}/{to_day}",
                {"adjusted": "true", "sort": "asc", "limit": 5000},
                kind="stock",
            )
        except Exception as exc:
            logger.info("history(%s, %s) failed: %s", ticker, period, exc)
            return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
        return self._bars_to_df(data.get("results", []))

    @staticmethod
    def _bars_to_df(bars: list[dict]) -> pd.DataFrame:
        if not bars:
            return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
        df = pd.DataFrame({
            "Open": [b["o"] for b in bars],
            "High": [b["h"] for b in bars],
            "Low": [b["l"] for b in bars],
            "Close": [b["c"] for b in bars],
            "Volume": [b["v"] for b in bars],
        }, index=pd.to_datetime([b["t"] for b in bars], unit="ms").normalize())
        df.index.name = "Date"
        return df

    # -- options data --------------------------------------------------------

    def _contracts(self, ticker: str) -> list[dict]:
        if ticker in self._contracts_cache:
            return self._contracts_cache[ticker]
        contracts: list[dict] = []
        params = {
            "underlying_ticker": ticker,
            "expired": "false",
            "limit": 1000,
        }
        data = self._get("/v3/reference/options/contracts", params, kind="reference")
        contracts.extend(data.get("results", []))
        next_url = data.get("next_url")
        while next_url:
            # next_url already contains the cursor; path+params split
            path, _, query = next_url.partition("?")
            path = path.replace(POLYGON_BASE, "")
            page_params = dict(p.split("=", 1) for p in query.split("&") if "=" in p)
            page_params.pop("apiKey", None)
            data = self._get(path, page_params, kind="reference")
            contracts.extend(data.get("results", []))
            next_url = data.get("next_url")
        self._contracts_cache[ticker] = contracts
        return contracts

    def options_expiries(self, ticker: str) -> list[str]:
        try:
            today = date.today().isoformat()
            expiries = {
                c["expiration_date"]
                for c in self._contracts(ticker)
                if c.get("expiration_date") and c["expiration_date"] >= today
            }
            return sorted(expiries)
        except Exception as exc:
            logger.info("options_expiries(%s) failed: %s", ticker, exc)
            return []

    def _option_close(self, contract_ticker: str) -> Optional[dict]:
        """Most recent daily bar for one option contract."""
        to_day = date.today()
        from_day = to_day - timedelta(days=14)
        data = self._get(
            f"/v2/aggs/ticker/{contract_ticker}/range/1/day/{from_day}/{to_day}",
            {"adjusted": "true", "sort": "desc", "limit": 1},
            kind="options",
        )
        results = data.get("results", [])
        return results[0] if results else None

    def option_chain(self, ticker: str, expiry: str) -> OptionChainData:
        """Sparse ATM chain for one expiry.

        Only the single strike closest to spot per option type is quoted
        (options-aggs rate budget is ~5 req/min). bid/ask/lastPrice are the
        EOD close; IV and delta are solved locally. OI is unavailable.
        """
        bar = self._latest_bar(ticker)
        if not bar:
            raise ValueError(f"No spot price for {ticker}")
        spot = float(bar["c"])

        today = date.today()
        T = max((datetime.strptime(expiry, "%Y-%m-%d").date() - today).days, 1) / 365.0

        contracts = [
            c for c in self._contracts(ticker)
            if c.get("expiration_date") == expiry
        ]
        if not contracts:
            raise ValueError(f"No contracts for {ticker} {expiry}")

        frames = {}
        parity_iv: dict[str, float] = {}
        for ctype in ("call", "put"):
            typed = [c for c in contracts if c.get("contract_type") == ctype]
            if not typed:
                frames[ctype] = pd.DataFrame(columns=_CHAIN_COLUMNS)
                continue
            best = min(typed, key=lambda c: abs(float(c["strike_price"]) - spot))
            strike = float(best["strike_price"])
            opt_bar = self._option_close(best["ticker"])
            close = float(opt_bar["c"]) if opt_bar else np.nan
            volume = float(opt_bar["v"]) if opt_bar else 0.0
            iv = np.nan
            delta = np.nan
            if close and close > 0:
                iv = implied_volatility(close, spot, strike, T, RISK_FREE_RATE, ctype)
                if not np.isfinite(iv):
                    # Solver miss on one side — put-call parity says the IV of
                    # the opposite type at the same strike is a sound stand-in.
                    other = "put" if ctype == "call" else "call"
                    iv = parity_iv.get(other, np.nan)
                else:
                    parity_iv[ctype] = iv
                if np.isfinite(iv):
                    delta = black_scholes_delta(spot, strike, T, RISK_FREE_RATE, iv, ctype)
            frames[ctype] = pd.DataFrame([{
                "contractSymbol": best["ticker"],
                "strike": strike,
                "bid": close,
                "ask": close,
                "lastPrice": close,
                "impliedVolatility": iv,
                "openInterest": 0,
                "volume": volume,
                "delta": delta,
                "inTheMoney": (spot > strike) if ctype == "call" else (spot < strike),
            }], columns=_CHAIN_COLUMNS)

        return OptionChainData(calls=frames["call"], puts=frames["put"],
                               oi_available=False, source=self.name)


class LSEProvider:
    """London Strategic Edge backend with a yfinance-shaped surface.

    Uses the official ``lse-data`` client (REST vault). Chains carry native
    IV/greeks/volume but NO open interest and no bid/ask — bid/ask collapse
    to ``last_price`` (which can be stale per-contract: it reflects the last
    trade, refreshed at LSE's snapshot cadence). ``oi_available=False`` so
    callers skip OI gates. Plan limits observed: 200 calls/min, 5000 rows/req,
    2 concurrent connections (``vault_concurrency``).

    The scanner's worker pool runs up to 8 tickers in parallel
    (services/scan_service.py), all sharing this one provider instance. The
    ``_AimdRateLimiter`` only paces call STARTS (min interval between
    acquires) — it does not cap how many calls are in flight at once. Without
    a separate concurrency gate, 8 workers fire past the vault's 2-connection
    ceiling, the excess get rejected/timeout, and every rejection calls
    ``throttled()`` on the ONE SHARED limiter — doubling its interval (toward
    the 5s ceiling) for every worker, not just the one that failed. That
    failure/backoff spiral, not the per-minute pacing, is what was making
    scans crawl. ``_concurrency`` below caps actual in-flight LSE calls at
    the plan's real limit so workers queue instead of colliding.
    """

    name = "lse"
    max_expiries_hint: Optional[int] = None  # whole chain is one call
    _CHAIN_TTL_SECS = 900.0  # re-fetch a ticker's chain at most every 15 min
    _VAULT_CONCURRENCY = 2  # observed plan limit — see class docstring

    def __init__(self, api_key: Optional[str] = None, client=None):
        self._key = api_key if api_key is not None else get_settings().lse_api_key
        self._client = client  # injectable; lazily constructed from the key
        self._limiter = _AimdRateLimiter(0.4, 0.31, 5.0)  # stay under 200/min
        self._concurrency = threading.Semaphore(self._VAULT_CONCURRENCY)
        self._chain_cache: dict[str, tuple[float, list[dict]]] = {}

    @property
    def client(self):
        if self._client is None:
            from lse import LSE  # lazy import: optional dependency
            self._client = LSE(api_key=self._key)
        return self._client

    # HTTP statuses the vault returns for a well-formed, promptly-answered
    # request that just has no data for this specific query — NOT a load
    # signal. A scan universe is mostly small-caps/OTC/delisted tickers the
    # vault's catalog was never going to carry, so treating these as
    # throttling pins the shared limiter at its 5s ceiling almost
    # immediately and keeps it there for the whole run.
    _NON_OVERLOAD_STATUSES = {400, 404}

    def _call(self, fn):
        self._limiter.acquire()
        with self._concurrency:
            try:
                result = fn()
            except Exception as exc:
                if getattr(exc, "status", None) in self._NON_OVERLOAD_STATUSES:
                    self._limiter.success()
                else:
                    self._limiter.throttled()
                raise
        self._limiter.success()
        return result

    def healthy(self, timeout: float = 6.0) -> bool:
        if not self._key:
            return False
        try:
            start = (date.today() - timedelta(days=10)).isoformat()
            rows = self._call(lambda: self.client.candles("SPY", "1d", start=start, limit=5))
            return bool(rows)
        except Exception as exc:
            logger.info("LSE health check failed: %s", exc)
            return False

    # -- stock data ----------------------------------------------------------

    def history(self, ticker: str, period: str = "1d") -> pd.DataFrame:
        """yfinance-shaped OHLCV DataFrame (empty when no data)."""
        days = _PERIOD_DAYS.get(period, 100)
        start = (date.today() - timedelta(days=days)).isoformat()
        try:
            rows = self._call(lambda: self.client.candles(
                ticker, "1d", start=start, order="asc", limit=5000))
        except Exception as exc:
            logger.info("LSE history(%s, %s) failed: %s", ticker, period, exc)
            return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
        df = self._rows_to_df(rows)
        if period == "1d" and not df.empty:
            df = df.tail(1)  # match Yahoo/Polygon single-session semantics
        return df

    @staticmethod
    def _rows_to_df(rows: list[dict]) -> pd.DataFrame:
        if not rows:
            return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
        df = pd.DataFrame({
            "Open": [float(r["open"]) for r in rows],
            "High": [float(r["high"]) for r in rows],
            "Low": [float(r["low"]) for r in rows],
            "Close": [float(r["close"]) for r in rows],
            "Volume": [float(r.get("volume") or 0) for r in rows],
        }, index=pd.to_datetime([r["timestamp"] for r in rows], utc=True).tz_convert(None).normalize())
        df.index.name = "Date"
        return df

    # -- options data --------------------------------------------------------

    def _chain(self, ticker: str) -> list[dict]:
        cached = self._chain_cache.get(ticker)
        if cached and (time.monotonic() - cached[0]) < self._CHAIN_TTL_SECS:
            return cached[1]
        rows = self._call(lambda: self.client.options(ticker, limit=5000))
        self._chain_cache[ticker] = (time.monotonic(), rows or [])
        while len(self._chain_cache) > 32:
            self._chain_cache.pop(next(iter(self._chain_cache)))
        return self._chain_cache[ticker][1]

    def options_expiries(self, ticker: str) -> list[str]:
        today = date.today().isoformat()
        expiries = {
            r["expiry"] for r in self._chain(ticker)
            if r.get("expiry") and r["expiry"] >= today
        }
        return sorted(expiries)

    def option_chain(self, ticker: str, expiry: str) -> OptionChainData:
        rows = [
            r for r in self._chain(ticker)
            if r.get("expiry") == expiry and r.get("strike") is not None
        ]
        if not rows:
            raise ValueError(f"No LSE contracts for {ticker} {expiry}")

        frames = {}
        for ctype in ("call", "put"):
            typed = [r for r in rows if r.get("contract_type") == ctype]
            records = []
            for r in sorted(typed, key=lambda x: float(x["strike"])):
                strike = float(r["strike"])
                last = r.get("last_price")
                last = float(last) if last is not None else np.nan
                spot = r.get("underlying_price")
                records.append({
                    "contractSymbol": r.get("ticker", ""),
                    "strike": strike,
                    "bid": last,
                    "ask": last,
                    "lastPrice": last,
                    "impliedVolatility": r.get("iv") if r.get("iv") is not None else np.nan,
                    "openInterest": 0,
                    "volume": float(r.get("volume_today") or 0),
                    "delta": r.get("delta") if r.get("delta") is not None else np.nan,
                    "inTheMoney": (spot > strike) if ctype == "call" else (spot < strike)
                    if spot is not None else False,
                })
            frames[ctype] = pd.DataFrame(records, columns=_CHAIN_COLUMNS)

        return OptionChainData(calls=frames["call"], puts=frames["put"],
                               oi_available=False, source=self.name)


class ResilientProvider:
    """Failover wrapper over an ordered provider chain: LSE → Yahoo.

    Starts on the first healthy provider (providers without a ``healthy()``
    method are always eligible as last resort), latches to it, advances down
    the chain on errors, and periodically re-probes higher-priority providers
    so a recovered connection is picked up again. LSE is only included when
    explicitly passed or an ``LSE_API_KEY`` is configured.

    Polygon is deliberately NOT auto-added here — it's reserved for the
    backtest/backfill scripts, which call it directly. Pass ``polygon=``
    explicitly (tests, one-off scripts) to include it in the chain anyway.
    """

    name = "resilient"

    def __init__(
        self,
        yahoo: Optional[YahooProvider] = None,
        polygon: Optional[PolygonProvider] = None,
        lse: Optional[LSEProvider] = None,
        recheck_calls: int = 60,
    ):
        if lse is None and get_settings().lse_api_key:
            lse = LSEProvider()
        self._yahoo = yahoo or YahooProvider()
        self._polygon = polygon
        self._order = [p for p in (lse, self._yahoo, self._polygon) if p is not None]
        self._recheck_calls = recheck_calls
        self._lock = threading.Lock()
        self._call_count = 0
        self._active = self._first_healthy()
        if self._active is not self._order[0]:
            logger.warning("Market data provider: %s unhealthy — starting on %s",
                           self._order[0].name, self._active.name)
        else:
            logger.info("Market data provider: using %s", self._active.name)

    def _first_healthy(self):
        for p in self._order[:-1]:
            probe = getattr(p, "healthy", None)
            if probe is None or probe():
                return p
        return self._order[-1]

    @property
    def active_name(self) -> str:
        return self._active.name

    @property
    def max_expiries_hint(self) -> Optional[int]:
        return self._active.max_expiries_hint

    def _maybe_recheck(self) -> None:
        idx = self._order.index(self._active)
        if idx == 0 or self._call_count % self._recheck_calls != 0:
            return
        for p in self._order[:idx]:
            probe = getattr(p, "healthy", None)
            if probe is None or probe():
                logger.warning("Market data provider: %s recovered — switching back", p.name)
                self._active = p
                return

    def _dispatch(self, method: str, *args):
        with self._lock:
            self._call_count += 1
            self._maybe_recheck()
            start = self._order.index(self._active)
        last_exc: Optional[Exception] = None
        for idx in range(start, len(self._order)):
            provider = self._order[idx]
            try:
                return getattr(provider, method)(*args)
            except Exception as exc:
                last_exc = exc
                if idx >= len(self._order) - 1:
                    break
                nxt = self._order[idx + 1]
                logger.warning("%s %s failed (%s) — switching to %s",
                               provider.name, method, exc, nxt.name)
                with self._lock:
                    if self._active is provider:
                        self._active = nxt
        raise last_exc

    def history(self, ticker: str, period: str = "1d") -> pd.DataFrame:
        return self._dispatch("history", ticker, period)

    def options_expiries(self, ticker: str) -> list[str]:
        return self._dispatch("options_expiries", ticker)

    def option_chain(self, ticker: str, expiry: str) -> OptionChainData:
        return self._dispatch("option_chain", ticker, expiry)


# ── Singleton -------------------------------------------------------------

_provider = None
_provider_lock = threading.Lock()


def get_provider():
    """Global market-data provider, configured via EARNINGS_PRICE_PROVIDER."""
    global _provider
    with _provider_lock:
        if _provider is None:
            mode = get_settings().price_provider
            if mode == "yahoo":
                _provider = YahooProvider()
            elif mode == "polygon":
                _provider = PolygonProvider()
            elif mode == "lse":
                _provider = LSEProvider()
            else:
                _provider = ResilientProvider()
        return _provider


def reset_provider() -> None:
    """Drop the singleton (tests / config reload)."""
    global _provider
    with _provider_lock:
        _provider = None
