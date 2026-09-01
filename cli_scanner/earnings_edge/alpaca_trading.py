"""Alpaca Paper Trading Client Wrapper.

Provides:
- AlpacaTradingClient — wraps trading API calls for option chains, orders, positions, and account info
- Bridge: Strategy TAKE list → Alpaca order submission
- PositionManager: track fills, open positions, exits, PnL
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Optional
from urllib.parse import urljoin

import requests

logger = logging.getLogger(__name__)

# Alpaca paper API base URLs
PAPER_BASE = "https://paper-api.alpaca.markets/v2"
DATA_BASE = "https://data.alpaca.markets/v1beta1"
BROKER_BASE = "https://broker-api.alpaca.markets/v1"

DEFAULT_HEADERS = {
    "APCA-API-KEY-ID": "",
    "APCA-API-SECRET-KEY": "",
}


class AlpacaError(RuntimeError):
    """Alpaca API error."""

    def __init__(self, status_code: int, message: str):
        super().__init__(f"[{status_code}] {message}")
        self.status_code = status_code
        self.message = message


class AlpacaAuthError(AlpacaError):
    pass


class AlpacaNotFoundError(AlpacaError):
    pass


class AlpacaTradingClient:
    """Paper trading client for Alpaca.

    Supports:
    - Account info & portfolio status
    - Option chain retrieval (from options/snapshots)
    - Multi-leg option orders (complex option spreads)
    - Position tracking & order management
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        paper: bool = True,
    ):
        self.paper = bool(paper)
        self.api_key = api_key or os.environ.get("APCA_API_KEY_ID", "")
        self.api_secret = api_secret or os.environ.get("APCA_API_SECRET_KEY", "")
        self.base_url = PAPER_BASE if self.paper else "https://api.alpaca.markets/v2"
        self.data_url = DATA_BASE
        self.session = requests.Session()
        self.session.headers.update({
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.api_secret,
            "Content-Type": "application/json",
        })
        logger.debug("AlpacaTradingClient initialized (paper=%s)", paper)

    def _request(
        self,
        method: str,
        path: str,
        base: Optional[str] = None,
        params: Optional[dict] = None,
        json_body: Optional[dict] = None,
        retry: int = 2,
    ) -> dict | list:
        """Make an API request with basic retry + error handling."""
        url = (base or self.base_url).rstrip("/") + "/" + path.lstrip("/")
        for attempt in range(retry + 1):
            try:
                resp = self.session.request(method, url, params=params, json=json_body, timeout=30)
                # 401/404 are deterministic — retrying them just burns rate budget
                if resp.status_code == 401:
                    from framework.alerts import DEDUPER
                    DEDUPER.emit(
                        "alpaca_401",
                        "🛑 Alpaca 401 — invalid API keys. "
                        "Broker calls will fail until keys are fixed.",
                    )
                    raise AlpacaAuthError(401, "Invalid API keys")
                if resp.status_code == 404:
                    raise AlpacaNotFoundError(404, f"Not found: {path}")
                if resp.status_code == 429:
                    wait = 2 ** attempt
                    logger.warning("Rate limited, waiting %ds", wait)
                    time.sleep(wait)
                    continue
                if resp.status_code >= 400:
                    try:
                        detail = resp.json().get("message", resp.text)
                    except Exception:
                        detail = resp.text
                    raise AlpacaError(resp.status_code, detail)
                if resp.status_code == 204:
                    return {}
                return resp.json()
            except (AlpacaAuthError, AlpacaNotFoundError):
                raise
            except requests.RequestException as e:
                if attempt == retry:
                    raise
                logger.warning("Request failed: %s (retry %d)", e, attempt + 1)
                time.sleep(2 ** attempt)
        return {}

    # ──────────────────────── Account & Portfolio ────────────────────────

    def get_account(self) -> dict:
        """Return account status (buying power, equity, etc.)."""
        return self._request("GET", "/account")

    def get_clock(self) -> dict:
        """Market clock: {is_open, next_open, next_close, timestamp}."""
        return self._request("GET", "/clock")

    def get_positions(self) -> list[dict]:
        """Return all open positions."""
        return self._request("GET", "/positions")

    def get_position(self, symbol: str) -> Optional[dict]:
        """Get position for a specific symbol. Returns None if not found."""
        try:
            return self._request("GET", f"/positions/{symbol}")
        except AlpacaNotFoundError:
            return None

    def close_position(self, symbol: str, qty: Optional[int] = None) -> dict:
        """Close a position (option or stock). qty=None closes all."""
        params = {"percentage": "100"} if qty is None else {"quantity": str(qty)}
        return self._request("DELETE", f"/positions/{symbol}", params=params)

    def close_all_positions(self) -> list[dict]:
        """Close ALL positions."""
        return self._request("DELETE", "/positions")

    # ──────────────────────── Option Chains ────────────────────────────────

    def get_option_contracts(
        self,
        underlying_symbol: str,
        expiration_date_gte: Optional[str] = None,
        expiration_date_lte: Optional[str] = None,
        strike_price_gte: Optional[float] = None,
        strike_price_lte: Optional[float] = None,
        style: Optional[str] = None,
        status: str = "active",
        limit: int = 200,
        page_token: Optional[str] = None,
    ) -> dict:
        """Get option contracts (metadata) for an underlying.

        Returns raw Alpaca contracts response with pagination.
        """
        params = {
            "underlying_symbols": underlying_symbol,
            "status": status,
            "limit": min(limit, 100),
        }
        if expiration_date_gte:
            params["expiration_date_gte"] = expiration_date_gte
        if expiration_date_lte:
            params["expiration_date_lte"] = expiration_date_lte
        if strike_price_gte:
            params["strike_price_gte"] = str(strike_price_gte)
        if strike_price_lte:
            params["strike_price_lte"] = str(strike_price_lte)
        if style:
            params["style"] = style
        if page_token:
            params["page_token"] = page_token

        result = self._request("GET", "/options/contracts", params=params)
        return result

    def get_option_snapshot(self, symbol: str) -> dict:
        """Get full snapshot (greeks, bid/ask, etc.) for a single option contract."""
        res = self._request("GET", "/options/snapshots", params={"symbols": symbol}, base=self.data_url)
        return res.get("snapshots", {}).get(symbol, {})

    def get_option_snapshots_bulk(self, *symbols: str) -> dict[str, dict]:
        """Get snapshots for multiple option contracts."""
        if not symbols:
            return {}
        symbol_str = ",".join(symbols)
        result = self._request("GET", "/options/snapshots", params={"symbols": symbol_str}, base=self.data_url)
        return result.get("snapshots", {})

    def get_options_chain_snapshots(self, underlying: str, page_limit: int = 250) -> dict[str, dict]:
        """All option snapshots for one underlying (paginated).

        Returns {contract_symbol: {"bid": float, "ask": float, "quote_time": str}}.
        Quotes carry latestQuote only on this feed (no greeks/IV).
        """
        out: dict[str, dict] = {}
        params: dict = {"limit": page_limit}
        while True:
            result = self._request("GET", f"/options/snapshots/{underlying}",
                                   params=params, base=self.data_url)
            for sym, snap in (result.get("snapshots") or {}).items():
                q = snap.get("latestQuote") or {}
                bp, ap = q.get("bp"), q.get("ap")
                if bp is None or ap is None:
                    continue
                out[sym] = {"bid": float(bp), "ask": float(ap), "quote_time": q.get("t", "")}
            token = result.get("next_page_token")
            if not token:
                return out
            params["page_token"] = token

    def get_stock_latest_trade(self, symbol: str) -> Optional[float]:
        """Latest stock trade price (data API)."""
        result = self._request(
            "GET", f"/stocks/{symbol}/trades/latest",
            base="https://data.alpaca.markets/v2",
        )
        trade = result.get("trade") or {}
        price = trade.get("p")
        return float(price) if price is not None else None

    def get_option_chain_full(
        self,
        underlying: str,
        expiration_date_gte: Optional[str] = None,
        expiration_date_lte: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict]:
        """Convenience: full option chain with snapshots merged.

        Flattens the contracts + snapshots into a list of enriched contracts.
        """
        # Step 1: Get contracts
        resp = self.get_option_contracts(
            underlying,
            expiration_date_gte=expiration_date_gte,
            expiration_date_lte=expiration_date_lte,
            limit=limit,
        )
        contracts = resp.get("option_contracts", [])
        next_token = resp.get("next_page_token")
        while next_token and len(contracts) < 500:
            resp = self.get_option_contracts(
                underlying,
                expiration_date_gte=expiration_date_gte,
                expiration_date_lte=expiration_date_lte,
                limit=limit,
                page_token=next_token,
            )
            contracts.extend(resp.get("option_contracts", []))
            next_token = resp.get("next_page_token")

        # Step 2: Get snapshots in batches
        ticker_to_contract = {c["symbol"]: c for c in contracts}
        tickers = list(ticker_to_contract.keys())
        snapshots: dict[str, dict] = {}
        for i in range(0, len(tickers), 100):
            batch = tickers[i : i + 100]
            snaps = self.get_option_snapshots_bulk(*batch)
            snapshots.update(snaps)

        # Step 3: Merge
        enriched = []
        for ticker, contract in ticker_to_contract.items():
            snap = snapshots.get(ticker, {})
            merged = {
                **contract,
                "latest_quote": snap.get("latestQuote", {}),
                "latest_trade": snap.get("latestTrade", {}),
                "implied_volatility": snap.get("impliedVolatility"),
                "greeks": snap.get("greeks", {}),
                "bid_price": snap.get("latestQuote", {}).get("bp"),
                "ask_price": snap.get("latestQuote", {}).get("ap"),
                "bid_size": snap.get("latestQuote", {}).get("bs"),
                "ask_size": snap.get("latestQuote", {}).get("as"),
                "volume": snap.get("latestTrade", {}).get("v"),
                "vwap": snap.get("latestTrade", {}).get("vw"),
            }
            enriched.append(merged)

        return enriched

    # ──────────────────────── Option Bars ────────────────────────────────

    def get_option_bars(
        self,
        symbols: list[str],
        start: str,
        end: str,
        *,
        timeframe: str = "1D",
    ) -> dict[str, list[dict]]:
        """Get OHLCV bars for one or more option contracts."""
        if not symbols:
            return {}
        params = {
            "symbols": ",".join(symbols),
            "timeframe": timeframe,
            "start": start,
            "end": end,
            "limit": 1000,
        }
        result = self._request("GET", "/options/bars", base=self.data_url, params=params)
        return result.get("bars", {})

    # ──────────────────────── Order Management ─────────────────────────────

    def submit_order(
        self,
        symbol: str,
        qty: int,
        side: str,
        order_type: str = "market",
        time_in_force: str = "day",
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        client_order_id: Optional[str] = None,
        extended_hours: bool = False,
    ) -> dict:
        """Submit a single-leg option order."""
        body: dict = {
            "symbol": symbol,
            "qty": str(qty),
            "side": side,
            "type": order_type,
            "time_in_force": time_in_force,
        }
        if limit_price is not None:
            body["limit_price"] = str(round(limit_price, 2))
        if stop_price is not None:
            body["stop_price"] = str(round(stop_price, 2))
        if client_order_id:
            body["client_order_id"] = client_order_id
        if extended_hours:
            body["extended_hours"] = True

        order = self._request("POST", "/orders", json_body=body)
        logger.info("Order submitted: %s %s %s @%s → %s", side, qty, symbol, order_type, order.get("id", "?"))
        return order

    def submit_multi_leg_order(
        self,
        legs: list[dict],
        order_type: str = "market",
        time_in_force: str = "day",
        limit_price: Optional[float] = None,
        client_order_id: Optional[str] = None,
        qty: int = 1,
    ) -> dict:
        """Submit a multi-leg option order (complex spread).

        legs: list of {"symbol": str, "ratio_qty": int, "side": "buy"/"sell"}

        Alpaca supports up to 4 legs for complex option spreads.
        Verified 2026-07-25: order_class must be "mleg" and top-level qty
        is required; limit_price is the NET debit/credit for the combo.
        """
        if len(legs) < 2:
            raise ValueError("Multi-leg order requires at least 2 legs")
        if len(legs) > 4:
            raise ValueError("Alpaca supports max 4 legs per multi-leg order")

        order_class = "mleg"
        body: dict = {
            "type": order_type,
            "time_in_force": time_in_force,
            "order_class": order_class,
            "qty": str(qty),
            "legs": [
                {
                    "symbol": leg["symbol"],
                    "ratio_qty": str(leg["ratio_qty"]),
                    "side": leg["side"],
                }
                for leg in legs
            ],
        }
        if limit_price is not None:
            body["limit_price"] = str(round(limit_price, 2))
        if client_order_id:
            body["client_order_id"] = client_order_id

        order = self._request("POST", "/orders", json_body=body)
        logger.info(
            "Multi-leg order submitted: %d legs @%s → %s",
            len(legs),
            order_type,
            order.get("id", "?"),
        )
        return order

    def get_orders(self, status: str = "open", symbols: Optional[list[str]] = None) -> list[dict]:
        """Get orders (open, closed, etc.)."""
        params: dict[str, str] = {"status": status}
        if symbols:
            params["symbols"] = ",".join(symbols)
        return self._request("GET", "/orders", params=params)

    def get_order(self, order_id: str) -> dict:
        """Get a single order."""
        return self._request("GET", f"/orders/{order_id}")

    def cancel_order(self, order_id: str) -> dict:
        """Cancel an open order."""
        return self._request("DELETE", f"/orders/{order_id}")

    def cancel_all_orders(self) -> list[dict]:
        """Cancel all open orders."""
        return self._request("DELETE", "/orders")

    # ──────────────── Convenience Methods ────────────────────────────────

    def buying_power(self) -> float:
        """Available cash for trading."""
        acct = self.get_account()
        return float(acct.get("buying_power", 0))

    def portfolio_value(self) -> float:
        """Total portfolio value."""
        acct = self.get_account()
        return float(acct.get("portfolio_value", 0))

    def has_position(self, symbol: str) -> bool:
        """Check if position exists for a symbol."""
        return self.get_position(symbol) is not None

    def position_symbols(self) -> set[str]:
        """All open position symbols in ONE call (batch alternative to
        per-symbol has_position, which costs an API call per symbol).
        Any failure degrades to an empty set (positions unknown = don't block
        trading; the per-order exchange checks still apply)."""
        try:
            return {str(p.get("symbol", "")) for p in self.get_positions()}
        except Exception as e:
            logger.warning("position fetch failed (treating as empty): %s", e)
            return set()

    def is_optionable(self, symbol: str) -> bool:
        """Quick check if underlying has option chain available."""
        try:
            contracts = self.get_option_contracts(symbol, limit=1)
            return len(contracts.get("option_contracts", [])) > 0
        except AlpacaError:
            return False

    def find_option_contract(
        self,
        underlying: str,
        option_type: str,
        target_strike: float,
        target_expiry: date,
        tolerance: float = 0.01,
    ) -> Optional[dict]:
        """Find nearest matching contract for type/expiry/strike.

        Walks contracts with expiry within ±2 days of target_expiry,
        returns nearest strike within tolerance.
        """
        expiry_str = target_expiry.isoformat()
        expiry_min = (target_expiry - timedelta(days=2)).isoformat()
        expiry_max = (target_expiry + timedelta(days=2)).isoformat()

        contracts_resp = self.get_option_contracts(
            underlying,
            expiration_date_gte=expiry_min,
            expiration_date_lte=expiry_max,
            limit=200,
        )
        contracts = contracts_resp.get("option_contracts", [])

        best = None
        best_dist = float("inf")
        for c in contracts:
            if c.get("type", "").lower() != option_type.lower():
                continue
            strike = float(c.get("strike_price", 0))
            dist = abs(strike - target_strike) / target_strike
            if dist < best_dist:
                best_dist = dist
                best = c

        if best and best_dist <= tolerance:
            return best
        return None


# ---------------------------------------------------------------------------
# Position Manager
# ---------------------------------------------------------------------------


class PositionManager:
    """Track option positions from strategy signal → fill → exit."""

    def __init__(self, client: AlpacaTradingClient):
        self.client = client

    def open_positions(self) -> list[dict]:
        """All open option positions."""
        positions = self.client.get_positions()
        return [p for p in positions if p.get("asset_class") == "option"]

    def option_exposure(self, underlying: str) -> float:
        """Total notional exposure for an underlying's options."""
        positions = self.open_positions()
        return sum(
            abs(float(p.get("qty", 0)) * float(p.get("market_value", 0)))
            for p in positions
            if p.get("underlying_symbol", "") == underlying
        )

    def close_all_options(self, underlying: Optional[str] = None) -> list[dict]:
        """Close all option positions (optionally filter by underlying)."""
        positions = self.open_positions()
        results = []
        for p in positions:
            if underlying and p.get("underlying_symbol", "") != underlying:
                continue
            result = self.client.close_position(p["symbol"])
            results.append(result)
        return results


# ---------------------------------------------------------------------------
# Order Result
# ---------------------------------------------------------------------------


@dataclass
class OrderResult:
    order_id: str
    client_order_id: str
    symbol: str
    strategy: str
    legs: list[dict]
    status: str
    filled_qty: int
    filled_avg_price: Optional[float]
    created_at: str
    raw: dict
    exit_by: Optional[date] = None   # structural exit deadline, set by the
                                      # caller (StrategyBridge.execute_trade)
                                      # — not derivable from Alpaca's response

    @classmethod
    def from_alpaca(cls, raw: dict, strategy: str = "") -> OrderResult:
        filled_qty = int(raw.get("filled_qty", 0))
        return cls(
            order_id=raw.get("id", ""),
            client_order_id=raw.get("client_order_id", ""),
            symbol=raw.get("symbol", ""),
            strategy=strategy,
            legs=raw.get("legs", []),
            status=raw.get("status", ""),
            filled_qty=filled_qty,
            filled_avg_price=raw.get("filled_avg_price"),
            created_at=raw.get("created_at", ""),
            raw=raw,
        )


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def create_client(
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None,
    paper: Optional[bool] = None,
) -> AlpacaTradingClient:
    """Factory: build a trading client from env vars or args.

    ``paper=None`` (the default) follows ALPACA_LIVE — unset/0 is paper.
    Explicit ``paper=True`` always stays on the paper URL (tests).
    Explicit ``paper=False`` forces live credentials.
    """
    from earnings_edge.alpaca_mode import resolve_credentials

    key, secret, is_paper = resolve_credentials(api_key, api_secret, paper)
    return AlpacaTradingClient(api_key=key, api_secret=secret, paper=is_paper)
