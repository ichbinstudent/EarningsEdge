"""Hourly Alpaca options-chain cache.

Pulls live chain snapshots (data.alpaca.markets) for a wide earnings
universe and persists one row per contract per hour into ``options_chain``.
Shared by ``scripts/collect_options_snapshot.py`` and the bot scheduler.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Optional

from earnings_edge.db import insert_options_chain_rows, snapshots_optionable_universe

logger = logging.getLogger("earnings_edge.chain_cache")

DEFAULT_MAX_TICKERS = 400
HOURLY_MAX_TICKERS = 250


def captured_hour(now: Optional[datetime] = None) -> str:
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    return now.strftime("%Y-%m-%dT%H")


def default_underlyings(max_tickers: int = DEFAULT_MAX_TICKERS) -> list[str]:
    """Upcoming earnings first, then recently-optionable names."""
    return snapshots_optionable_universe(max_tickers)


def row_for_contract(run_id: str, underlying: str, contract_ticker: str,
                     snap: dict, *, now: Optional[datetime] = None) -> dict:
    from earnings_edge.fwd_factor import occ_parse

    now = now or datetime.now(timezone.utc)
    bar = snap.get("dailyBar") or {}
    q = snap.get("latestQuote") or {}
    bid, ask = q.get("bp"), q.get("ap")
    midpoint = ((bid + ask) / 2) if (bid is not None and ask is not None) else None
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
        "implied_volatility": None,
        "delta": None,
        "gamma": None,
        "theta": None,
        "vega": None,
        "captured_at": now.isoformat(),
        "captured_hour": hour,
    }


def collect(client, underlyings, *, run_id: Optional[str] = None,
            dry_run: bool = False, sleep_s: float = 0.22) -> dict:
    """Pull chain for each underlying. Returns stats dict."""
    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    now = datetime.now(timezone.utc)
    inserted = 0
    api_calls = 0
    empty = 0
    for i, und in enumerate(underlyings):
        if i > 0 and sleep_s:
            time.sleep(sleep_s)
        snap, _ = client.chain_snapshot(und)
        api_calls += 1
        contracts = (snap or {}).get("snapshots") or {}
        if not contracts:
            empty += 1
            continue
        rows = [
            row_for_contract(run_id, und, ct, s, now=now)
            for ct, s in contracts.items()
        ]
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
