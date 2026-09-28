"""Hourly Alpaca options-chain cache.

Pulls live chain snapshots (data.alpaca.markets) for a wide earnings
universe and persists one row per contract per hour into ``options_chain``.
Shared by ``scripts/collect_options_snapshot.py`` and the bot scheduler.
"""

from __future__ import annotations

import logging
import math
import time
from datetime import UTC, datetime

from earnings_edge.db import insert_options_chain_rows, snapshots_optionable_universe

logger = logging.getLogger("earnings_edge.chain_cache")

DEFAULT_MAX_TICKERS = 400
HOURLY_MAX_TICKERS = 250
# Page cap per underlying (x chain_snapshot page size). Large chains run to a
# few thousand contracts; the cap only guards against a token loop.
MAX_CHAIN_PAGES = 20


def captured_hour(now: datetime | None = None) -> str:
    now = now or datetime.now(UTC)
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)
    return now.strftime("%Y-%m-%dT%H")


def default_underlyings(max_tickers: int = DEFAULT_MAX_TICKERS) -> list[str]:
    """Upcoming earnings first, then recently-optionable names."""
    return snapshots_optionable_universe(max_tickers)


def _finite(value) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _positive(value) -> float | None:
    v = _finite(value)
    return v if v is not None and v > 0 else None


def _valid_mid(bid, ask) -> float | None:
    b, a = _finite(bid), _finite(ask)
    if b is None or a is None or b <= 0 or a < b:
        return None
    return (b + a) / 2


def fetch_chain(client, underlying: str, max_pages: int = MAX_CHAIN_PAGES) -> tuple[dict, int]:
    """All contract snapshots for one underlying -> ({symbol: snap}, api_calls).

    The snapshots endpoint is paginated. Reading only the first page kept
    just the front of the chain (pages run in symbol order, nearest expiry
    first), so the >= 21 DTE expiries the daily signals read and the 30-90
    DTE legs forward-factor needs were often never stored.
    """
    contracts: dict = {}
    token = None
    calls = 0
    for _ in range(max_pages):
        snap, token = client.chain_snapshot(underlying, page_token=token)
        calls += 1
        contracts.update((snap or {}).get("snapshots") or {})
        if not token:
            break
    else:
        logger.warning("chain cache: %s truncated at %d pages", underlying, max_pages)
    return contracts, calls


def row_for_contract(
    run_id: str, underlying: str, contract_ticker: str, snap: dict, *, now: datetime | None = None
) -> dict:
    from earnings_edge.fwd_factor import occ_parse

    now = now or datetime.now(UTC)
    bar = snap.get("dailyBar") or {}
    q = snap.get("latestQuote") or {}
    greeks = snap.get("greeks") or {}
    bid, ask = q.get("bp"), q.get("ap")
    # A zero bid or crossed book has no meaningful mid: (0 + ask) / 2 solved to
    # a confident-looking but fake IV downstream. Leave it null so consumers
    # fall back to the bar close (or skip the contract).
    midpoint = _valid_mid(bid, ask)
    expiry_str, strike_val, contract_type = "", None, ""
    try:
        parsed = occ_parse(contract_ticker)
        expiry_str = parsed["expiry"].isoformat()
        strike_val = parsed["strike"]
        contract_type = parsed["option_type"]
    except (ValueError, IndexError, KeyError):
        pass
    hour = captured_hour(now)
    return {
        "collector_run_id": run_id,
        "ticker": underlying,
        "scan_date": now.strftime("%Y-%m-%d"),
        "contract_ticker": contract_ticker,
        "underlying": underlying,
        "expiry": expiry_str,
        "strike": strike_val,
        "contract_type": contract_type,
        "style": "american",
        "bid": bid,
        "ask": ask,
        "bid_size": q.get("bs"),
        "ask_size": q.get("as"),
        "midpoint": midpoint,
        "close": bar.get("c"),
        "open_price": bar.get("o"),
        "high": bar.get("h"),
        "low": bar.get("l"),
        "trade_count": bar.get("n"),
        "volume": bar.get("v"),
        "vwap": bar.get("vw"),
        # Filled when the feed carries them (Alpaca includes IV/greeks on some
        # data tiers); otherwise null and solved locally from the mid.
        "implied_volatility": _positive(snap.get("impliedVolatility")),
        "delta": _finite(greeks.get("delta")),
        "gamma": _finite(greeks.get("gamma")),
        "theta": _finite(greeks.get("theta")),
        "vega": _finite(greeks.get("vega")),
        "captured_at": now.isoformat(),
        "captured_hour": hour,
    }


def collect(
    client, underlyings, *, run_id: str | None = None, dry_run: bool = False, sleep_s: float = 0.22
) -> dict:
    """Pull chain for each underlying. Returns stats dict."""
    run_id = run_id or datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    now = datetime.now(UTC)
    inserted = 0
    api_calls = 0
    empty = 0
    for i, und in enumerate(underlyings):
        if i > 0 and sleep_s:
            time.sleep(sleep_s)
        contracts, calls = fetch_chain(client, und)
        api_calls += calls
        if not contracts:
            empty += 1
            continue
        rows = [row_for_contract(run_id, und, ct, s, now=now) for ct, s in contracts.items()]
        if dry_run:
            inserted += len(rows)
            continue
        inserted += insert_options_chain_rows(rows)
    return {
        "run_id": run_id,
        "underlyings": len(underlyings),
        "inserted": inserted,
        "api_calls": api_calls,
        "empty": empty,
        "captured_hour": captured_hour(now),
    }


def run_hourly(max_tickers: int = HOURLY_MAX_TICKERS, dry_run: bool = False) -> dict:
    """Bot/job entry: resolve universe from DB, pull Alpaca, persist."""
    import os

    from earnings_edge.collectors.alpaca_options import AlpacaOptionsClient

    key = os.environ.get("APCA_API_KEY_ID", "")
    secret = os.environ.get("APCA_API_SECRET_KEY", "")
    if not key or not secret:
        raise RuntimeError("APCA_API_KEY_ID / APCA_API_SECRET_KEY required for chain cache")
    client = AlpacaOptionsClient(api_key=key, api_secret=secret)
    tickers = default_underlyings(max_tickers)
    if not tickers:
        logger.warning("chain cache: no underlyings")
        return {"inserted": 0, "underlyings": 0, "note": "no underlyings"}
    logger.info("chain cache: %d underlyings (cap %d)", len(tickers), max_tickers)
    return collect(client, tickers, dry_run=dry_run)
