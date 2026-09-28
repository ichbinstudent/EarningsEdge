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
  historical backfill/backtest scripts (``scripts/polygon_backfill.py`` and
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
import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta

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
    "contractSymbol",
    "strike",
    "bid",
    "ask",
    "lastPrice",
    "impliedVolatility",
    "openInterest",
    "volume",
    "delta",
    "inTheMoney",
]


class DataUnavailable(ValueError):
    """The backend answered but holds no usable data for *this* query.

    Ticker not in the vault's catalog, no live expiries, unknown expiry, only
    stale quotes. This is a per-ticker gap, not an outage: ResilientProvider
    serves the call from the next backend without moving its latch, so one
    uncovered small-cap no longer flips a whole scan off LSE. Subclasses
    ValueError so pre-existing ``except ValueError`` callers keep working.
    """


# Statuses meaning "well-formed request, nothing here" rather than overload.
_DATA_MISS_STATUSES = {400, 404}


_OSI_ROOT = re.compile(r"^([A-Z][A-Z0-9.]{0,9})\d{6}[CP]\d{8}$")


def normalize_symbol(sym: str) -> str:
    """Upper-case, punctuation-free symbol (BRK.B / BRK-B / BRKB compare equal)."""
    return re.sub(r"[^A-Z0-9]", "", str(sym).upper())


def lse_row_underlying(row: dict) -> str | None:
    """Normalized underlying of an LSE option row, or None when unknowable.

    Uses the row's ``underlying`` field, else the OSI root of its ticker.
    """
    und = row.get("underlying")
    if und:
        return normalize_symbol(und)
    m = _OSI_ROOT.match(str(row.get("ticker") or "").upper())
    return normalize_symbol(m.group(1)) if m else None


def is_data_miss(exc: BaseException) -> bool:
    """True for per-query data gaps; False for outages (timeouts, 5xx, 429, auth)."""
    return isinstance(exc, DataUnavailable) or getattr(exc, "status", None) in _DATA_MISS_STATUSES


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
    max_expiries_hint: int | None = None  # full chains are cheap on Yahoo

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
        return OptionChainData(calls=chain.calls, puts=chain.puts, oi_available=True, source=self.name)


class PolygonProvider:
    """Polygon.io backend with a yfinance-shaped surface.

    Entitlements on the current plan: stock aggs (fast), options aggs
    (~5 req/min sustained), options contracts reference. NO snapshot/greeks —
    IV/delta are computed locally from EOD closes via Black-Scholes.
    """

    name = "polygon"
    max_expiries_hint: int | None = 3  # keep options-class calls bounded

    def __init__(self, api_key: str | None = None, http: requests.Session | None = None):
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

    def _get(self, path: str, params: dict | None = None, kind: str = "stock") -> dict:
        if not self._key:
            raise ValueError("POLYGON_API_KEY not set")
        params = dict(params or {})
        params["apiKey"] = self._key
        limiter = self._limiters[kind]
        last_exc: Exception | None = None
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
                if (
                    isinstance(exc, requests.HTTPError)
                    and exc.response is not None
                    and exc.response.status_code in (403, 404)
                ):
                    raise
                limiter.throttled()
        raise ValueError(f"Polygon GET {path} failed after retries: {last_exc}")

    # -- stock data ----------------------------------------------------------

    def _grouped_daily(self, day: str) -> dict[str, dict]:
        """All US stock bars for one day in a single call (cached)."""
        if day not in self._grouped_cache:
            data = self._get(
                f"/v2/aggs/grouped/locale/us/market/stocks/{day}",
                {"adjusted": "true"},
                kind="stock",
            )
            self._grouped_cache[day] = {r["T"]: r for r in data.get("results", [])}
            # keep cache bounded
            while len(self._grouped_cache) > 4:
                self._grouped_cache.pop(next(iter(self._grouped_cache)))
        return self._grouped_cache[day]

    def _latest_bar(self, ticker: str) -> dict | None:
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
        df = pd.DataFrame(
            {
                "Open": [b["o"] for b in bars],
                "High": [b["h"] for b in bars],
                "Low": [b["l"] for b in bars],
                "Close": [b["c"] for b in bars],
                "Volume": [b["v"] for b in bars],
            },
            index=pd.to_datetime([b["t"] for b in bars], unit="ms").normalize(),
        )
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

    def _option_close(self, contract_ticker: str) -> dict | None:
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

        contracts = [c for c in self._contracts(ticker) if c.get("expiration_date") == expiry]
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
            frames[ctype] = pd.DataFrame(
                [
                    {
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
                    }
                ],
                columns=_CHAIN_COLUMNS,
            )

        return OptionChainData(calls=frames["call"], puts=frames["put"], oi_available=False, source=self.name)


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

    Data-integrity guards (each raises ``DataUnavailable`` so the resilient
    chain serves that ticker from Yahoo instead):

    - the client resolves an unknown symbol by fuzzy company-name match, so a
      ticker missing from the catalog can come back as ANOTHER company's
      chain; rows whose underlying differs from the request are dropped;
    - a contract whose last update is older than ``_MAX_QUOTE_AGE_DAYS`` has
      its price/IV blanked — a days-old last trade is not a quote;
    - transient failures (transport, 429, 5xx) are retried before the error
      escapes, so one blip does not latch the whole scan onto Yahoo.
    """

    name = "lse"
    max_expiries_hint: int | None = None  # whole chain is one call
    _CHAIN_TTL_SECS = 900.0  # re-fetch a ticker's chain at most every 15 min
    _HISTORY_TTL_SECS = 300.0  # one candles call serves 1d/1mo/3mo per ticker
    _HISTORY_MIN_DAYS = 100  # fetch window covering 1d/5d/1mo/3mo in one call
    _CACHE_MAX_TICKERS = 64
    _VAULT_CONCURRENCY = 2  # observed plan limit — see class docstring
    _REST_TIMEOUT_SECS = 30.0  # client default is 60s: a hung call pins a vault slot
    _MAX_ATTEMPTS = 3
    _RETRY_DELAY_SECS = 1.0
    _TRANSIENT_STATUSES = {0, 429, 500, 502, 503, 504}  # 0 = no HTTP response
    _MAX_QUOTE_AGE_DAYS = 5  # covers a weekend + holiday; older = no quote

    def __init__(self, api_key: str | None = None, client=None):
        self._key = api_key if api_key is not None else get_settings().lse_api_key
        self._client = client  # injectable; lazily constructed from the key
        self._limiter = _AimdRateLimiter(0.4, 0.31, 5.0)  # stay under 200/min
        self._concurrency = threading.Semaphore(self._VAULT_CONCURRENCY)
        self._cache_lock = threading.Lock()
        self._chain_cache: OrderedDict[str, tuple[float, list[dict]]] = OrderedDict()
        self._history_cache: OrderedDict[str, tuple[float, int, pd.DataFrame]] = OrderedDict()
        self._retry_delay = self._RETRY_DELAY_SECS

    @property
    def client(self):
        if self._client is None:
            from lse import LSE  # lazy import: optional dependency

            self._client = LSE(api_key=self._key, timeout=self._REST_TIMEOUT_SECS)
        return self._client

    # HTTP statuses the vault returns for a well-formed, promptly-answered
    # request that just has no data for this specific query — NOT a load
    # signal. A scan universe is mostly small-caps/OTC/delisted tickers the
    # vault's catalog was never going to carry, so treating these as
    # throttling pins the shared limiter at its 5s ceiling almost
    # immediately and keeps it there for the whole run.
    _NON_OVERLOAD_STATUSES = _DATA_MISS_STATUSES

    def _call(self, fn):
        """One paced, concurrency-gated attempt (no retry)."""
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

    def _is_transient(self, exc: Exception) -> bool:
        status = getattr(exc, "status", None)
        if status is not None:
            return status in self._TRANSIENT_STATUSES
        return isinstance(exc, (ConnectionError, TimeoutError, OSError))

    def _request(self, fn):
        """``_call`` with bounded retry on transient failures.

        The concurrency slot is released between attempts (``_call`` scopes
        it), so a backing-off worker never blocks the other scan workers.
        """
        for attempt in range(1, self._MAX_ATTEMPTS + 1):
            try:
                return self._call(fn)
            except Exception as exc:
                if attempt >= self._MAX_ATTEMPTS or not self._is_transient(exc):
                    raise
                delay = self._retry_delay * 2 ** (attempt - 1)
                logger.info("LSE transient error (%s) — retry %d in %.1fs", exc, attempt, delay)
                if delay > 0:
                    time.sleep(delay)
        raise RuntimeError("unreachable")  # pragma: no cover

    def healthy(self, timeout: float = 6.0) -> bool:
        """Candles AND at least one live, recently-updated option contract.

        The vault's options catalog can freeze while candles keep flowing
        (observed 2026-07-01: every chain row carried a July expiry with
        dte=1 forever). A candles-only probe latches ResilientProvider onto
        a provider whose chains are all dead, so every options_expiries
        call silently returns [] and the scan funnels everything to
        no_quote. Requiring one live, fresh expiry catches that state.
        """
        if not self._key:
            return False
        try:
            start = (date.today() - timedelta(days=10)).isoformat()
            rows = self._call(lambda: self.client.candles("SPY", "1d", start=start, limit=5))
            if not rows:
                return False
            today = date.today().isoformat()
            opts = self._call(lambda: self.client.options("SPY", limit=50))
            return any(
                r.get("expiry") and r["expiry"] >= today and not self._is_stale(r) for r in (opts or [])
            )
        except Exception as exc:
            logger.info("LSE health check failed: %s", exc)
            return False

    # -- stock data ----------------------------------------------------------

    def history(self, ticker: str, period: str = "1d") -> pd.DataFrame:
        """yfinance-shaped OHLCV DataFrame.

        Raises when the vault has nothing for the ticker: the small-cap
        universe 404s almost everywhere on LSE, and an empty-DataFrame
        return hides that from ResilientProvider (no failover to Yahoo,
        candidates die as no_price/no_quote). Empty DataFrames remain
        valid only for genuinely empty successful responses (rows == []).

        One candles call per ticker per ``_HISTORY_TTL_SECS`` serves every
        period up to ``_HISTORY_MIN_DAYS`` (the validator + analyzer ask for
        1d, 1mo and 3mo of the same ticker back to back): 1 call instead of
        3-4 against the 200/min plan budget.
        """
        days = _PERIOD_DAYS.get(period, 100)
        df = self._cached_history(ticker, days)
        if df is None:
            window = max(days, self._HISTORY_MIN_DAYS)
            start = (date.today() - timedelta(days=window)).isoformat()
            rows = self._request(
                lambda: self.client.candles(ticker, "1d", start=start, order="asc", limit=5000)
            )
            df = self._rows_to_df(rows)
            with self._cache_lock:
                self._history_cache[ticker] = (time.monotonic(), window, df)
                self._history_cache.move_to_end(ticker)
                while len(self._history_cache) > self._CACHE_MAX_TICKERS:
                    self._history_cache.popitem(last=False)
        if not df.empty:
            cutoff = pd.Timestamp(date.today() - timedelta(days=days))
            df = df[df.index >= cutoff]
        if period == "1d" and not df.empty:
            df = df.tail(1)  # match Yahoo/Polygon single-session semantics
        return df.copy()

    def _cached_history(self, ticker: str, days: int) -> pd.DataFrame | None:
        with self._cache_lock:
            hit = self._history_cache.get(ticker)
        if hit is None:
            return None
        fetched_at, window, df = hit
        if window < days or (time.monotonic() - fetched_at) >= self._HISTORY_TTL_SECS:
            return None
        return df

    @staticmethod
    def _rows_to_df(rows: list[dict]) -> pd.DataFrame:
        if not rows:
            return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
        df = pd.DataFrame(
            {
                "Open": [float(r["open"]) for r in rows],
                "High": [float(r["high"]) for r in rows],
                "Low": [float(r["low"]) for r in rows],
                "Close": [float(r["close"]) for r in rows],
                "Volume": [float(r.get("volume") or 0) for r in rows],
            },
            index=pd.to_datetime([r["timestamp"] for r in rows], utc=True).tz_convert(None).normalize(),
        )
        df.index.name = "Date"
        return df

    # -- options data --------------------------------------------------------

    def _is_stale(self, row: dict) -> bool:
        raw = row.get("last_trade_at") or row.get("updated_at")
        if not raw:
            return False  # no timestamp: can't tell, keep
        try:
            ts = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        except ValueError:
            return False
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
        return datetime.now(UTC) - ts > timedelta(days=self._MAX_QUOTE_AGE_DAYS)

    def _chain(self, ticker: str) -> list[dict]:
        with self._cache_lock:
            cached = self._chain_cache.get(ticker)
        if cached and (time.monotonic() - cached[0]) < self._CHAIN_TTL_SECS:
            return cached[1]
        rows = self._request(lambda: self.client.options(ticker, limit=5000)) or []
        want = normalize_symbol(ticker)
        kept = [r for r in rows if lse_row_underlying(r) in (None, want)]
        if rows and not kept:
            got = sorted({u for r in rows if (u := lse_row_underlying(r))})[:3]
            logger.warning(
                "LSE returned another underlying's chain for %s (%s) — ignoring (name-match resolution)",
                ticker,
                ",".join(got),
            )
        with self._cache_lock:
            self._chain_cache[ticker] = (time.monotonic(), kept)
            self._chain_cache.move_to_end(ticker)
            while len(self._chain_cache) > self._CACHE_MAX_TICKERS:
                self._chain_cache.popitem(last=False)
        return kept

    def options_expiries(self, ticker: str) -> list[str]:
        today = date.today().isoformat()
        rows = self._chain(ticker)
        expiries = sorted(
            {r["expiry"] for r in rows if r.get("expiry") and r["expiry"] >= today and not self._is_stale(r)}
        )
        if not expiries:
            # Vault catalog gap (ticker not carried, or the options feed
            # froze like 2026-07-01). An empty list must NOT return
            # normally: ResilientProvider only advances on exceptions, so
            # a silent [] here pins the scan to LSE and kills the ticker
            # as no_quote before option_chain (which does raise) is ever
            # consulted. Raise so the chain can try Yahoo.
            raise DataUnavailable(f"No live LSE expiries for {ticker} (catalog gap or stale feed)")
        return expiries

    def option_chain(self, ticker: str, expiry: str) -> OptionChainData:
        rows = [r for r in self._chain(ticker) if r.get("expiry") == expiry and r.get("strike") is not None]
        if not rows:
            raise DataUnavailable(f"No LSE contracts for {ticker} {expiry}")
        if all(self._is_stale(r) for r in rows):
            raise DataUnavailable(f"Only stale LSE quotes for {ticker} {expiry}")

        frames = {}
        for ctype in ("call", "put"):
            typed = [r for r in rows if r.get("contract_type") == ctype]
            records = []
            for r in sorted(typed, key=lambda x: float(x["strike"])):
                strike = float(r["strike"])
                stale = self._is_stale(r)
                last = r.get("last_price")
                last = float(last) if last is not None and not stale else np.nan
                iv = r.get("iv") if r.get("iv") is not None and not stale else np.nan
                spot_raw = r.get("underlying_price")
                spot = float(spot_raw) if spot_raw is not None else None
                records.append(
                    {
                        "contractSymbol": r.get("ticker", ""),
                        "strike": strike,
                        "bid": last,
                        "ask": last,
                        "lastPrice": last,
                        "impliedVolatility": iv,
                        "openInterest": 0,
                        "volume": float(r.get("volume_today") or 0),
                        "delta": r.get("delta") if r.get("delta") is not None else np.nan,
                        "inTheMoney": ((spot > strike) if ctype == "call" else (spot < strike))
                        if spot is not None
                        else False,
                    }
                )
            frames[ctype] = pd.DataFrame(records, columns=_CHAIN_COLUMNS)

        return OptionChainData(calls=frames["call"], puts=frames["put"], oi_available=False, source=self.name)


class ResilientProvider:
    """Failover wrapper over an ordered provider chain: LSE → Yahoo.

    Starts on the first healthy provider (providers without a ``healthy()``
    method are always eligible as last resort), latches to it, advances down
    the chain on errors, and periodically re-probes higher-priority providers
    so a recovered connection is picked up again. LSE is only included when
    explicitly passed or an ``LSE_API_KEY`` is configured.

    Two kinds of failure are handled differently:

    - **outage** (timeout, 5xx, 429, auth, anything unclassified): the
      latch moves to the next provider for every later call, as before;
    - **data miss** (``is_data_miss``: DataUnavailable or a 400/404 — the
      ticker simply isn't carried): only *this* call falls through, and the
      ticker is pinned to the provider that served it for
      ``affinity_ttl_secs`` so its remaining calls (every expiry's chain) go
      straight there and come from one consistent source. The latch stays,
      so the rest of the universe keeps using the primary.

    Polygon is deliberately NOT auto-added here — it's reserved for the
    backtest/backfill scripts, which call it directly. Pass ``polygon=``
    explicitly (tests, one-off scripts) to include it in the chain anyway.
    """

    name = "resilient"

    def __init__(
        self,
        yahoo: YahooProvider | None = None,
        polygon: PolygonProvider | None = None,
        lse: LSEProvider | None = None,
        recheck_calls: int = 60,
        affinity_ttl_secs: float = 1800.0,
    ):
        if lse is None and get_settings().lse_api_key:
            lse = LSEProvider()
        self._yahoo = yahoo or YahooProvider()
        self._polygon = polygon
        self._order = [p for p in (lse, self._yahoo, self._polygon) if p is not None]
        self._recheck_calls = recheck_calls
        self._affinity_ttl = affinity_ttl_secs
        self._affinity: OrderedDict[tuple[str, str], tuple[int, float]] = OrderedDict()
        self._lock = threading.Lock()
        self._probing = False
        self._call_count = 0
        self._active = self._first_healthy()
        if self._active is not self._order[0]:
            logger.warning(
                "Market data provider: %s unhealthy — starting on %s", self._order[0].name, self._active.name
            )
        else:
            logger.info("Market data provider: using %s", self._active.name)

    @staticmethod
    def _probe(provider) -> bool:
        probe = getattr(provider, "healthy", None)
        if probe is None:
            return True
        try:
            return bool(probe())
        except Exception as exc:
            logger.info("%s health probe raised: %s", provider.name, exc)
            return False

    def _first_healthy(self):
        for p in self._order[:-1]:
            if self._probe(p):
                return p
        return self._order[-1]

    @property
    def active_name(self) -> str:
        return self._active.name

    @property
    def max_expiries_hint(self) -> int | None:
        return self._active.max_expiries_hint

    def _maybe_recheck(self) -> None:
        """Every ``recheck_calls`` calls, re-probe the providers above the latch.

        Probes run OUTSIDE the lock (they are network calls — holding the
        lock stalled every scan worker for the probe's duration), and only
        one thread probes at a time.
        """
        with self._lock:
            idx = self._order.index(self._active)
            if idx == 0 or self._probing or self._call_count % self._recheck_calls != 0:
                return
            self._probing = True
            candidates = self._order[:idx]
        try:
            for p in candidates:
                if self._probe(p):
                    with self._lock:
                        if self._order.index(p) < self._order.index(self._active):
                            logger.warning("Market data provider: %s recovered — switching back", p.name)
                            self._active = p
                    return
        finally:
            with self._lock:
                self._probing = False

    @staticmethod
    def _family(method: str) -> str:
        return "history" if method == "history" else "options"

    def _start_index(self, method: str, ticker) -> int:
        """Latch index, or a later provider this ticker is pinned to. Caller holds the lock."""
        start = self._order.index(self._active)
        if ticker is None:
            return start
        key = (str(ticker), self._family(method))
        pin = self._affinity.get(key)
        if pin is None:
            return start
        idx, expires = pin
        if time.monotonic() >= expires:
            self._affinity.pop(key, None)
            return start
        return max(start, idx)

    def _pin(self, method: str, ticker, idx: int) -> None:
        if ticker is None:
            return
        key = (str(ticker), self._family(method))
        with self._lock:
            self._affinity[key] = (idx, time.monotonic() + self._affinity_ttl)
            self._affinity.move_to_end(key)
            while len(self._affinity) > 2048:
                self._affinity.popitem(last=False)

    def _dispatch(self, method: str, *args):
        ticker = args[0] if args else None
        with self._lock:
            self._call_count += 1
        self._maybe_recheck()
        with self._lock:
            start = self._start_index(method, ticker)
        last_exc: Exception | None = None
        missed = False
        for idx in range(start, len(self._order)):
            provider = self._order[idx]
            try:
                result = getattr(provider, method)(*args)
            except Exception as exc:
                last_exc = exc
                if idx >= len(self._order) - 1:
                    break
                nxt = self._order[idx + 1]
                if is_data_miss(exc):
                    missed = True
                    logger.info(
                        "%s has no %s data for %s (%s) — serving it from %s",
                        provider.name,
                        method,
                        ticker,
                        exc,
                        nxt.name,
                    )
                    continue
                logger.warning("%s %s failed (%s) — switching to %s", provider.name, method, exc, nxt.name)
                with self._lock:
                    if self._active is provider:
                        self._active = nxt
                continue
            if missed:
                self._pin(method, ticker, idx)
            return result
        if last_exc is not None:
            raise last_exc
        raise RuntimeError(f"provider chain failed with no exception for {method}")

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
