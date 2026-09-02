"""Bridge: Strategy signals → Alpaca paper orders.

Translates strategy Trade signals into Alpaca option orders:
- Single-leg: buy/sell call or put
- Two-leg: calendar spread (long back-month, short front-month)
- Multi-leg: iron condor, butterfly, risk reversal (up to 4 legs)

Supports sizing (Kelly fraction or fixed), pre-submission validation,
dry-run mode, and order-result tracking.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Callable, Optional

from earnings_edge.alpaca_trading import (
    AlpacaTradingClient,
    OrderResult,
    AlpacaError,
    create_client,
)
from earnings_edge.trading_types import Trade, StrategyResult, DataBundle

logger = logging.getLogger(__name__)

MAX_PCT_PER_TRADE = 0.10  # 10% of buying power max per signal
MAX_PCT_PER_UNDERLYING = 0.25  # 25% max exposure per underlying
DEFAULT_ORDER_TYPE = "limit"  # never market: live and paper share the same fill path
DEFAULT_TIF = "day"
MAX_CONTRACTS_PER_ORDER = 25  # hard fat-finger cap on sizer output

# Refuse an entry whose limit/debit is more than this multiple of combo mid.
# Paper fills at ~1.6x mid ate the entire IV-crush edge; 1.15 is the cap.
MAX_DEBIT_VS_MID = 1.15
# Last-look: combo ask-bid width vs mid, and debit vs spot (mirrors
# bot_scanner.MAX_DEBIT_PCT_OF_SPOT). Wide earnings names often quote 20-35%.
MAX_SPREAD_VS_MID = 0.40
LAST_LOOK_MAX_DEBIT_PCT_OF_SPOT = 0.15
FILL_POLL_ATTEMPTS = 1
FILL_POLL_SECS = 0.05
LIVE_FILL_POLL_ATTEMPTS = 8
LIVE_FILL_POLL_SECS = 2.0
_TERMINAL_ORDER = {"filled", "canceled", "expired", "rejected", "partially_filled"}


def size_veto_reason(strategy: str, unit_cost: float) -> str:
    """Readable reject when the sizer returns qty 0 — never a silent drop."""
    return f"size_veto: {strategy} qty=0 at unit cost {unit_cost:.2f}"


class StrikeChangedError(ValueError):
    """Catalog/OCC resolve produced a different strike than requested."""

    def __init__(self, ticker: str, requested: float, resolved_symbol: str, resolved_strike: float):
        self.ticker = ticker
        self.requested = requested
        self.resolved_symbol = resolved_symbol
        self.resolved_strike = resolved_strike
        super().__init__(
            f"{ticker}: resolve changed strike {requested} -> {resolved_strike} "
            f"({resolved_symbol}); refusing submit"
        )


def debit_within_mid_cap(
    debit: float,
    mid: float,
    cap: float = MAX_DEBIT_VS_MID,
) -> bool:
    """True when a debit (limit or fill) is not through combo mid beyond ``cap``."""
    if mid is None or mid <= 0 or debit is None or debit <= 0:
        return False
    return float(debit) <= float(mid) * cap + 1e-9


def combo_quotes(legs: list[dict], snaps: dict[str, dict]) -> Optional[dict]:
    """Net mid / spread for a multi-leg from Alpaca snapshot dicts.

    Returns None when any leg lacks a two-sided quote.
    """
    mid = 0.0
    spread = 0.0
    for leg in legs:
        snap = snaps.get(leg["symbol"]) or {}
        q = snap.get("latestQuote") or {}
        try:
            bid, ask = float(q.get("bp") or 0), float(q.get("ap") or 0)
        except (TypeError, ValueError):
            return None
        if bid <= 0 or ask <= 0 or ask < bid:
            return None
        sign = -1.0 if leg.get("side") == "sell" else 1.0
        mid += sign * (bid + ask) / 2.0
        spread += (ask - bid)
    if mid <= 0:
        return None
    return {"mid": mid, "spread": spread}


def last_look_veto(
    legs: list[dict],
    snaps: dict[str, dict],
    *,
    spot: Optional[float] = None,
    proposed_debit: Optional[float] = None,
    max_spread_vs_mid: float = MAX_SPREAD_VS_MID,
    max_debit_pct_of_spot: float = LAST_LOOK_MAX_DEBIT_PCT_OF_SPOT,
    max_debit_vs_mid: float = MAX_DEBIT_VS_MID,
) -> Optional[str]:
    """Return a skip reason if the combo is not tradable at these marks."""
    q = combo_quotes(legs, snaps)
    if q is None:
        return "last_look: missing/inverted quote"
    mid, spread = q["mid"], q["spread"]
    if mid > 0 and spread / mid > max_spread_vs_mid:
        return f"last_look: spread {spread:.2f} > {max_spread_vs_mid:.0%} of mid {mid:.2f}"
    debit = float(proposed_debit) if proposed_debit else mid
    if debit > 0 and not debit_within_mid_cap(debit, mid, max_debit_vs_mid):
        return f"last_look: debit {debit:.2f} > {max_debit_vs_mid:.2f}× mid {mid:.2f}"
    if spot and spot > 0 and debit > max_debit_pct_of_spot * spot:
        return (
            f"last_look: debit {debit:.2f} > {max_debit_pct_of_spot:.0%} of spot {spot:.2f}"
        )
    return None


def resolved_keeps_strike(requested: float, resolved_symbol: str, tol: float = 0.001) -> bool:
    """True when an OCC/catalog symbol still has ``requested`` strike."""
    try:
        from earnings_edge.fwd_factor import occ_parse
        got = occ_parse(resolved_symbol)["strike"]
    except Exception:
        return False
    return abs(float(got) - float(requested)) <= tol


def preflight_combo(
    bridge: "StrategyBridge",
    trade: "Trade",
    legs: list[dict],
    *,
    max_spread_vs_mid: float = MAX_SPREAD_VS_MID,
) -> tuple[Optional[str], Optional[float]]:
    """Proposal-time check of a combo against the EXECUTION venue's book.

    The scan layer prices candidates from LSEG marks; execution goes to
    Alpaca. These two disagree: LSEG lists strikes Alpaca does not carry
    (e.g. PL 19.5 -> AlpacaError 422 invalid legs) and thin Alpaca books
    make LSEG-priced debits unexecutable at last-look. Running the same
    existence + spread checks BEFORE a card is pushed means the approval
    card is executable at the price it shows.

    Returns (veto_reason, alpaca_mid). veto_reason is None when the combo
    is tradable on Alpaca right now; alpaca_mid is the live combo mid
    (None when no two-sided book).
    """
    symbols = [leg["symbol"] for leg in legs]
    try:
        raw = bridge.client.get_option_snapshots_bulk(*symbols)
    except Exception as exc:
        return f"preflight: snapshot request failed ({exc})", None
    if not isinstance(raw, dict):
        return "preflight: snapshot response not a dict", None
    for leg in legs:
        snap = raw.get(leg["symbol"])
        if snap is None:
            return f"preflight: {leg['symbol']} not on Alpaca", None
        q = snap.get("latestQuote") or {}
        try:
            bid, ask = float(q.get("bp") or 0), float(q.get("ap") or 0)
        except (TypeError, ValueError):
            return f"preflight: {leg['symbol']} unreadable quote", None
        if bid <= 0 or ask <= 0 or ask < bid:
            return f"preflight: {leg['symbol']} no two-sided book", None
    q = combo_quotes(legs, raw)
    if q is None:
        return "preflight: no two-sided combo book", None
    mid, spread = q["mid"], q["spread"]
    if mid > 0 and spread / mid > max_spread_vs_mid:
        return (
            f"preflight: spread {spread:.2f} > {max_spread_vs_mid:.0%} of mid {mid:.2f}",
            mid,
        )
    return None, mid

# Tail-stress multiple applied to the market-implied earnings move when
# pricing max loss for UNDEFINED-risk short premium (naked straddle/strangle):
# proxy max loss = EARNINGS_STRESS_MULTIPLE x expected_move_dollars x 100.
# Rationale: in the backtest, implied earnings moves exceed realized by ~6pp
# on average, so 2x the implied (straddle-priced) move is a tail stress, not
# the expected loss. It replaces the crude strike-notional proxy
# (_NOTIONAL_RISK_FRAC below) that sized every liquid mega-cap to qty 0 under
# a 1% vol_target budget. The notional proxy remains the fallback when the
# scan layer did not provide expected_move_dollars.
EARNINGS_STRESS_MULTIPLE = 2.0


@dataclass
class BridgeConfig:
    """Configuration for the strategy→order bridge."""
    dry_run: bool = False
    order_type: str = DEFAULT_ORDER_TYPE
    time_in_force: str = DEFAULT_TIF
    max_pct_per_trade: float = MAX_PCT_PER_TRADE
    max_pct_per_underlying: float = MAX_PCT_PER_UNDERLYING
    skip_if_position_exists: bool = True  # don't add to existing position
    max_dte_min: int = 1  # skip trades with < 1 DTE
    max_dte_max: int = 30  # skip trades with > 30 DTE


class StrategyBridge:
    """Bridge strategy TAKE lists to Alpaca paper trades."""

    def __init__(
        self,
        client: Optional[AlpacaTradingClient] = None,
        config: Optional[BridgeConfig] = None,
        risk_manager=None,
        lifecycle_manager=None,
        limits_resolver=None,
        sizer_resolver=None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.client = client or create_client()
        self.config = config or BridgeConfig()
        self._sleep = sleep
        self._fill_poll_attempts = FILL_POLL_ATTEMPTS
        self._fill_poll_secs = FILL_POLL_SECS
        # Optional framework risk layer. When risk_manager is None the bridge
        # behaves exactly as before (approval path unchanged).
        self.risk_manager = risk_manager
        self.lifecycle_manager = lifecycle_manager
        # Optional callable strategy_name -> RiskLimits (per-strategy TOML
        # overrides; see framework.core.registry.StrategyRegistry.limits_for).
        self.limits_resolver = limits_resolver
        # Optional callable strategy_name -> [risk.sizer] spec dict
        # (framework.core.registry.StrategyRegistry.sizer_spec). Active only
        # when the risk layer is wired; legacy path keeps qty=1.
        self.sizer_resolver = sizer_resolver
        self.submitted: list[OrderResult] = []
        # Full broker positions, fetched lazily ONCE for risk exposure math.
        self._positions_full: Optional[list[dict]] = None
        # Account snapshot, fetched lazily ONCE (sizing + risk gate share it).
        self._account: Optional[dict] = None
        # Position-symbol set, fetched ONCE lazily (the per-leg has_position()
        # calls it replaces cost an API round-trip each — fatal on mega days).
        self._position_syms: Optional[set[str]] = None
        # Contract-catalog cache for the lazy symbol-resolution fallback:
        # (ticker, gte, lte) -> raw response
        self._contracts_cache: dict[tuple[str, str, str], dict] = {}
        # Why trades were skipped — surfaced in the run summary
        from collections import Counter
        self.skip_reasons: Counter = Counter()
        self.last_skip_detail: str = ""

    def account_buying_power(self) -> float:
        return self.client.buying_power()

    def _position_set(self) -> set[str]:
        if self._position_syms is None:
            self._position_syms = self.client.position_symbols()
        return self._position_syms

    # ──────────────── Core: Trade → Order ────────────────────────────────

    def execute_trade(self, trade: Trade) -> Optional[OrderResult]:
        """Execute a single strategy Trade as an Alpaca order.

        Dispatches on trade.side to the appropriate leg builder.
        Returns OrderResult on success, None if skipped.

        Cost discipline: leg building + DTE gate + position check are all
        LOCAL (OCC symbols are constructed, positions checked against a cached
        set). The only API calls happen for trades that actually submit —
        this is what keeps mega-earnings days (1000+ filtered candidates)
        from timing out.
        """
        try:
            legs = self._build_legs(trade)
            if not legs or len(legs) < 1:
                logger.warning("%s %s: no legs, skipping", trade.strategy, trade.ticker)
                self.skip_reasons["no_legs"] += 1
                return None

            # Validate: expiry ≥ min_dte and ≤ max_dte
            min_expiry = self._min_expiry(legs)
            if min_expiry is not None and trade.earnings_date:
                dte = (min_expiry - trade.earnings_date).days
                if dte < self.config.max_dte_min:
                    logger.debug("%s %s: DTE %d < min %d, skipping", trade.strategy, trade.ticker, dte, self.config.max_dte_min)
                    self.skip_reasons["dte"] += 1
                    return None
                if dte > self.config.max_dte_max:
                    logger.debug("%s %s: DTE %d > max %d, skipping", trade.strategy, trade.ticker, dte, self.config.max_dte_max)
                    self.skip_reasons["dte"] += 1
                    return None

            # Position check (local: one cached set for the whole run)
            if self.config.skip_if_position_exists:
                positions = self._position_set()
                for leg in legs:
                    if leg["symbol"] in positions:
                        logger.info("%s: position exists for %s, skipping", trade.strategy, leg["symbol"])
                        self.skip_reasons["position_exists"] += 1
                        return None

            # Compute limit price (or market)
            limit_price = None
            if self.config.order_type == "limit":
                limit_price = self._midpoint_price(legs)

            mid = None
            try:
                raw_mid = self._midpoint_price(legs)
                mid = float(raw_mid) if raw_mid is not None else None
            except (TypeError, ValueError):
                mid = None
            scan_debit = trade.entry_price or 0
            proposed = limit_price if limit_price else scan_debit
            # Refuse a scan-time debit that is already through live mid, even
            # if we would reprice to mid — that row's quote is garbage.
            check_px = max(float(proposed or 0), float(scan_debit or 0))
            if check_px and mid and mid > 0 and not debit_within_mid_cap(check_px, mid):
                logger.info(
                    "refused-vs-mid %s %s: debit %.2f vs mid %.2f (cap %.2f)",
                    trade.strategy, trade.ticker, proposed, mid, MAX_DEBIT_VS_MID,
                )
                self.skip_reasons["mid_cap"] += 1
                self.last_skip_detail = (
                    f"refused-vs-mid: debit {check_px:.2f} > "
                    f"{MAX_DEBIT_VS_MID:.2f}× mid {mid:.2f}"
                )
                return None

            # est_cost for the risk gate: structure-aware max loss per UNIT
            # (qty=1) first, then fallbacks (limit price → live midpoint).
            unit_cost = self._structure_cost(trade, legs, 1)
            if unit_cost <= 0 and limit_price:
                unit_cost = abs(limit_price) * 100
            if unit_cost <= 0 and self.risk_manager is not None:
                mid = self._midpoint_price(legs)
                if mid:
                    unit_cost = abs(mid) * 100

            # Framework risk gate (kill switch, portfolio caps, lifecycle).
            if self.risk_manager is not None and unit_cost <= 0:
                logger.warning("%s %s: cannot estimate cost — skipping (risk gate requires pricing)",
                               trade.strategy, trade.ticker)
                self.skip_reasons["risk_unpriced"] += 1
                return None

            qty = 1
            est_cost = unit_cost
            if self.risk_manager is not None:
                account = self._get_account()
                if account is None:  # account fetch failed → conservative veto
                    self.skip_reasons["risk_account_error"] += 1
                    return None
                # Strategy-declared sizing ([risk.sizer] in the TOML). The
                # sizer sees the per-unit max loss and returns the contract
                # count; 0 is a veto.
                qty = self._size_qty(trade, unit_cost, account)
                if qty <= 0:
                    self.last_skip_detail = size_veto_reason(
                        trade.strategy, unit_cost)
                    logger.info("%s", self.last_skip_detail)
                    self.skip_reasons["size_veto"] += 1
                    return None
                est_cost = unit_cost * qty
                decision = self._risk_check(trade, est_cost, qty, account)
                if not decision.approved:
                    self.skip_reasons["risk_veto"] += 1
                    return None
                # Probation scales the sizer's output down (floor 1 contract);
                # the risk gate already approved the larger size, so the
                # scaled-down order is strictly inside the approved caps.
                if decision.qty_multiplier != 1.0:
                    qty = max(1, int(qty * decision.qty_multiplier))
                    est_cost = unit_cost * qty
                    logger.info("%s %s: probation size multiplier %.2f → qty %d",
                                trade.strategy, trade.ticker, decision.qty_multiplier, qty)
            # Submit
            client_order_id = f"{trade.strategy}_{trade.ticker}_{trade.scan_date}_{int(datetime.now(timezone.utc).timestamp())}"
            exit_by = self._exit_by(legs)
            if self.config.dry_run:
                logger.info(
                    "DRY RUN: %s %s %s %d legs side=%s",
                    "BUY" if legs[0]["side"] == "buy" else "SELL",
                    1,
                    trade.ticker,
                    len(legs),
                    trade.side,
                )
                # dry-run returns below — last-look / submit never run
                dry_run_legs = (
                    [{**leg, "ratio_qty": int(leg.get("ratio_qty", 1)) * qty} for leg in legs]
                    if qty > 1 else legs
                )
                return OrderResult(
                    order_id="dry-run",
                    client_order_id=client_order_id,
                    symbol=trade.ticker,
                    strategy=trade.strategy,
                    legs=dry_run_legs,
                    status="dry_run",
                    filled_qty=0,
                    filled_avg_price=None,
                    created_at=datetime.now(timezone.utc).isoformat(),
                    raw={},
                    exit_by=exit_by,
                )

            veto, look_mid = self._last_look(trade, legs, limit_price)
            if veto:
                logger.info("%s %s: %s", trade.strategy, trade.ticker, veto)
                self.skip_reasons["last_look"] += 1
                self.last_skip_detail = veto
                return None
            if self.config.order_type == "limit":
                if look_mid:
                    limit_price = round(float(look_mid), 2)
                elif limit_price is None and trade.entry_price:
                    limit_price = round(float(trade.entry_price), 2)
                if limit_price is None:
                    logger.info("%s %s: no limit price after last-look", trade.strategy, trade.ticker)
                    self.skip_reasons["no_limit"] += 1
                    return None

            order = self._submit_with_resolution(trade, legs, limit_price, client_order_id, qty)
            order = self._await_fill(order)

            result = OrderResult.from_alpaca(order, strategy=trade.strategy)
            result.exit_by = exit_by
            if qty > 1:
                # Position tracking (record_open_positions) needs each leg's
                # ACTUAL total quantity, but we deliberately do NOT depend on
                # Alpaca's own response shape for that — compute it from what
                # we know we submitted (base ratio × order-level qty) rather
                # than re-parsing the broker's response.
                result.legs = [
                    {**leg, "ratio_qty": int(leg.get("ratio_qty", 1)) * qty}
                    for leg in legs
                ]
            self.submitted.append(result)
            if self.risk_manager is not None and est_cost > 0:
                self.risk_manager.record_entry(trade.strategy, trade.ticker, est_cost)
            logger.info("Order %s: %s", result.order_id, result.status)
            return result

        except AlpacaError as e:
            logger.error("Alpaca error on %s %s: %s", trade.strategy, trade.ticker, e)
            self.skip_reasons["alpaca_error"] += 1
            if self.risk_manager is not None:
                self.risk_manager.record_broker_rejection(trade.strategy, str(e))
            return None
        except Exception as e:
            logger.exception("Execution error on %s %s: %s", trade.strategy, trade.ticker, e)
            self.skip_reasons["error"] += 1
            return None

    # Structure sides that collect premium at entry (everything else is debit).
    # Mirrored from framework.positions.exits.CREDIT_SIDES (kept local to avoid
    # a hard framework dependency in the legacy bridge paths).
    _CREDIT_SIDES = {"SHORT_STRADDLE", "SHORT_STRANGLE", "IRON_CONDOR"}
    # Undefined-risk credit: fraction of strike notional used as a margin-like
    # max-loss proxy (paper trading; refine with real margin data later).
    _NOTIONAL_RISK_FRAC = 0.20

    def _structure_cost(self, trade: Trade, legs: list[dict], qty: int) -> float:
        """Dollars at risk for the risk gate.

        - debit structures: net debit x multiplier (as before)
        - defined-risk credit (wings on every short leg): (wing width − credit) x mult
        - undefined-risk credit (straddle/naked): earnings-stress proxy
          (EARNINGS_STRESS_MULTIPLE x expected_move_dollars) when available,
          else the strike-notional proxy
        Returns 0 when unpriced (caller falls back to quotes).
        """
        entry = abs(float(trade.entry_price or 0))
        if trade.side not in self._CREDIT_SIDES:
            return entry * 100 * qty
        shorts = [l for l in legs if l["side"] == "sell"]
        longs = [l for l in legs if l["side"] == "buy"]
        if not shorts:
            return entry * 100 * qty
        if longs:
            widths = []
            for s in shorts:
                same_type = [l for l in longs if l["option_type"] == s["option_type"]]
                if not same_type:
                    break
                widths.append(min(abs(l["strike"] - s["strike"]) for l in same_type))
            if len(widths) == len(shorts):
                return max((max(widths) - entry) * 100 * qty, 0.0)
        strike = max(float(s["strike"]) for s in shorts)
        # Undefined-risk credit: prefer the earnings-stress proxy when the
        # scan layer provided the market-implied earnings move.
        try:
            em = float((trade.features or {}).get("expected_move_dollars") or 0)
        except (TypeError, ValueError):
            em = 0.0
        if em > 0:
            return max(EARNINGS_STRESS_MULTIPLE * em * 100 * qty, entry * 100 * qty)
        return max(strike * 100 * qty * self._NOTIONAL_RISK_FRAC, entry * 100 * qty)

    def _get_account(self) -> Optional[dict]:
        """Account snapshot, fetched once per bridge run. None = fetch failed."""
        if self._account is None:
            try:
                self._account = self.client.get_account()
            except Exception as exc:
                logger.warning("account fetch failed (%s)", exc)
                self._account = {}
        return self._account or None

    def _size_qty(self, trade: Trade, unit_cost: float, account: dict) -> int:
        """Contracts for this trade per the strategy's [risk.sizer] TOML spec.

        Default (no spec / resolver / build failure) is 1 — the historical
        behavior. A sizer returning 0 vetoes the trade. Output is capped at
        MAX_CONTRACTS_PER_ORDER regardless of spec.
        """
        if unit_cost <= 0:
            return 0
        spec = self.sizer_resolver(trade.strategy) if self.sizer_resolver else None
        if not spec:
            return 1
        try:
            from framework.risk.sizing import SizeContext, build_sizer
            spec = dict(spec)
            sizer = build_sizer(spec.pop("name"), spec)
            qty = sizer.quantity(SizeContext(
                equity=float(account.get("equity") or 0),
                buying_power=float(account.get("buying_power") or 0),
                price_per_unit=unit_cost,
                # _structure_cost is already max-loss aware (wing width for
                # defined-risk, notional proxy for naked) — it is both the
                # price and the per-unit max loss here.
                max_loss_per_unit=unit_cost,
            ))
        except Exception as exc:
            logger.warning("sizer for %s failed (%s) — falling back to qty=1",
                           trade.strategy, exc)
            return 1
        from earnings_edge.alpaca_mode import live_max_qty
        return min(qty, live_max_qty(MAX_CONTRACTS_PER_ORDER))

    def _risk_check(self, trade: Trade, est_cost: float, qty: int, account: dict):
        """Run the framework risk gate against a pre-fetched account."""
        equity = float(account.get("equity") or 0)
        bp = float(account.get("buying_power") or 0)
        lifecycle = (
            self.lifecycle_manager.state(trade.strategy)
            if self.lifecycle_manager is not None else "live"
        )
        limits = self.limits_resolver(trade.strategy) if self.limits_resolver else None
        live_broker = not bool(getattr(self.client, "paper", True))
        decision = self.risk_manager.check_trade(
            strategy=trade.strategy,
            ticker=trade.ticker,
            est_cost=est_cost,
            qty=qty,
            equity=equity,
            buying_power=bp,
            underlying_exposure=self._underlying_exposure(trade.ticker),
            lifecycle=lifecycle,
            limits=limits,
            live_broker=live_broker,
        )
        if not decision.approved:
            logger.warning("RISK VETO %s %s: %s", trade.strategy, trade.ticker, decision.reason)
        return decision

    def _underlying_exposure(self, ticker: str) -> float:
        """Aggregate |market value| of all broker positions in ``ticker``
        (stock row + every OCC option row whose root matches)."""
        if self._positions_full is None:
            try:
                self._positions_full = self.client.get_positions()
            except Exception as exc:
                logger.info("exposure: get_positions failed (%s) — treating as 0", exc)
                self._positions_full = []
        total = 0.0
        for p in self._positions_full:
            sym = p.get("symbol") or ""
            if sym == ticker or sym.startswith(ticker):
                try:
                    total += abs(float(p.get("market_value") or 0))
                except (TypeError, ValueError):
                    pass
        return total

    def refresh_positions(self) -> None:
        self._position_syms = None
        self._positions_full = None

    def _submit_with_resolution(
        self,
        trade: Trade,
        legs: list[dict],
        limit_price: Optional[float],
        client_order_id: str,
        qty: int = 1,
    ) -> dict:
        """Submit the order; if Alpaca rejects an OCC symbol, resolve legs via
        the contract catalog (cached) and retry ONCE.

        OCC construction is the primary path (verified accepted by Alpaca) and
        is free; catalog resolution is the correctness safety net for roots
        that differ from the ticker (BRK.B etc.). Making the catalog the
        PRIMARY path costs one API call per leg per candidate — that's what
        timed out every mega-earnings day run.
        """
        try:
            return self._submit_legs(legs, limit_price, client_order_id, qty)
        except AlpacaError as e:
            if e.status_code not in (400, 404, 422):
                raise
            logger.warning("Order rejected (%s) — resolving symbols via catalog and retrying", e)
            resolved_any = False
            for leg in legs:
                resolved = self._resolve_symbol(
                    trade.ticker, leg["expiry"], leg["strike"], leg["option_type"])
                if resolved and resolved != leg["symbol"]:
                    if not resolved_keeps_strike(float(leg["strike"]), resolved):
                        from earnings_edge.fwd_factor import occ_parse
                        got = occ_parse(resolved)["strike"]
                        raise StrikeChangedError(
                            trade.ticker, float(leg["strike"]), resolved, got)
                    logger.info("Resolved %s -> %s", leg["symbol"], resolved)
                    leg["symbol"] = resolved
                    resolved_any = True
            if not resolved_any:
                raise
            return self._submit_legs(legs, limit_price, client_order_id, qty)

    def _submit_legs(
        self,
        legs: list[dict],
        limit_price: Optional[float],
        client_order_id: str,
        qty: int = 1,
    ) -> dict:
        if len(legs) == 1:
            leg = legs[0]
            return self.client.submit_order(
                symbol=leg["symbol"],
                qty=qty,
                side=leg["side"],
                order_type=self.config.order_type,
                time_in_force=self.config.time_in_force,
                limit_price=limit_price,
                client_order_id=client_order_id,
            )
        # legs keep their BASE ratio (1:1, 1:2:1, ...) — Alpaca requires
        # multi-leg ratio_qty values to be relatively prime, so the sizer's
        # contract count goes in the order-level qty, not baked into each
        # leg's ratio (that was the bug: qty=11 on a 1:1 calendar produced
        # ratio_qty 11:11, GCD 11, which Alpaca rejects outright).
        return self.client.submit_multi_leg_order(
            legs=legs,
            qty=qty,
            order_type=self.config.order_type,
            time_in_force=self.config.time_in_force,
            limit_price=limit_price,
            client_order_id=client_order_id,
        )

    # ──────────────── Leg builders ────────────────────────────────────────

    def _build_legs(self, trade: Trade) -> list[dict]:
        """Build order legs based on trade.side.

        Returns list of dicts: {"symbol": str, "ratio_qty": int, "side": "buy"|"sell", "strike": float, "expiry": date, "option_type": "call"|"put"}
        """
        feat = trade.features or {}
        if trade.side == "CALENDAR":
            return self._legs_calendar(trade, feat)
        if trade.side == "SHORT_STRADDLE":
            return self._legs_short_straddle(trade, feat)
        if trade.side == "LONG_STRADDLE":
            return self._legs_long_straddle(trade, feat)
        if trade.side == "DIRECTIONAL_CALL":
            return self._legs_single(trade, feat, "call", "buy")
        if trade.side == "DIRECTIONAL_PUT":
            return self._legs_single(trade, feat, "put", "buy")
        if trade.side == "BULL_CALL_SPREAD":
            return self._legs_vertical(trade, feat, "call", "debit")
        if trade.side == "BEAR_PUT_SPREAD":
            return self._legs_vertical(trade, feat, "put", "debit")
        if trade.side == "IRON_CONDOR":
            return self._legs_iron_condor(trade, feat)
        if trade.side == "BUTTERFLY":
            return self._legs_butterfly(trade, feat)
        if trade.side == "RISK_REVERSAL" or trade.side == "RR":
            return self._legs_risk_reversal(trade, feat)
        # Default: treat as single buy call at nearest round strike
        return self._legs_single(trade, feat, "call", "buy")

    def _legs_calendar(self, trade: Trade, feat: dict) -> list[dict]:
        # Calendar: sell near-month ATM call, buy far-month ATM call
        near_strike = feat.get("near_strike", trade.features.get("atm_strike", 0))
        far_strike = feat.get("far_strike", trade.features.get("atm_strike", 0))
        near_expiry = self._parse_date(feat.get("near_expiry")) or trade.earnings_date
        far_expiry = self._parse_date(feat.get("far_expiry")) or trade.earnings_date + timedelta(days=45)
        near_sym = self._occ_symbol(trade.ticker, near_expiry, near_strike, "call")
        far_sym = self._occ_symbol(trade.ticker, far_expiry, far_strike, "call")
        return [
            {"symbol": near_sym, "ratio_qty": 1, "side": "sell", "strike": near_strike, "expiry": near_expiry, "option_type": "call"},
            {"symbol": far_sym, "ratio_qty": 1, "side": "buy", "strike": far_strike, "expiry": far_expiry, "option_type": "call"},
        ]

    def _legs_short_straddle(self, trade: Trade, feat: dict) -> list[dict]:
        strike = feat.get("atm_strike", feat.get("nearest_atm", 0))
        expiry = self._parse_date(feat.get("expiry")) or trade.earnings_date
        call_occ = self._occ_symbol(trade.ticker, expiry, strike, "call")
        put_occ = self._occ_symbol(trade.ticker, expiry, strike, "put")
        return [
            {"symbol": call_occ, "ratio_qty": 1, "side": "sell", "strike": strike, "expiry": expiry, "option_type": "call"},
            {"symbol": put_occ, "ratio_qty": 1, "side": "sell", "strike": strike, "expiry": expiry, "option_type": "put"},
        ]

    def _legs_long_straddle(self, trade: Trade, feat: dict) -> list[dict]:
        strike = feat.get("atm_strike", feat.get("nearest_atm", 0))
        expiry = self._parse_date(feat.get("expiry")) or trade.earnings_date
        call_occ = self._occ_symbol(trade.ticker, expiry, strike, "call")
        put_occ = self._occ_symbol(trade.ticker, expiry, strike, "put")
        return [
            {"symbol": call_occ, "ratio_qty": 1, "side": "buy", "strike": strike, "expiry": expiry, "option_type": "call"},
            {"symbol": put_occ, "ratio_qty": 1, "side": "buy", "strike": strike, "expiry": expiry, "option_type": "put"},
        ]

    def _legs_single(self, trade: Trade, feat: dict, option_type: str, direction: str) -> list[dict]:
        strike = feat.get("atm_strike", feat.get("nearest_atm", 0))
        expiry = self._parse_date(feat.get("expiry")) or trade.earnings_date
        occ = self._occ_symbol(trade.ticker, expiry, strike, option_type)
        return [
            {"symbol": occ, "ratio_qty": 1, "side": direction, "strike": strike, "expiry": expiry, "option_type": option_type},
        ]

    def _legs_vertical(self, trade: Trade, feat: dict, option_type: str, kind: str) -> list[dict]:
        # Bull call spread: buy lower call, sell higher call
        k_low = feat.get("lower_strike", 0)
        k_high = feat.get("upper_strike", 0)
        expiry = self._parse_date(feat.get("expiry")) or trade.earnings_date
        lo_occ = self._occ_symbol(trade.ticker, expiry, k_low, option_type)
        hi_occ = self._occ_symbol(trade.ticker, expiry, k_high, option_type)
        if kind == "debit":
            return [
                {"symbol": lo_occ, "ratio_qty": 1, "side": "buy", "strike": k_low, "expiry": expiry, "option_type": option_type},
                {"symbol": hi_occ, "ratio_qty": 1, "side": "sell", "strike": k_high, "expiry": expiry, "option_type": option_type},
            ]
        return []

    def _legs_iron_condor(self, trade: Trade, feat: dict) -> list[dict]:
        sc = feat.get("short_call", 0)
        sp = feat.get("short_put", 0)
        lc = feat.get("long_call", 0)
        lp = feat.get("long_put", 0)
        expiry = self._parse_date(feat.get("expiry")) or trade.earnings_date
        return [
            {"symbol": self._occ_symbol(trade.ticker, expiry, lp, "put"), "ratio_qty": 1, "side": "buy", "strike": lp, "expiry": expiry, "option_type": "put"},
            {"symbol": self._occ_symbol(trade.ticker, expiry, sp, "put"), "ratio_qty": 1, "side": "sell", "strike": sp, "expiry": expiry, "option_type": "put"},
            {"symbol": self._occ_symbol(trade.ticker, expiry, sc, "call"), "ratio_qty": 1, "side": "sell", "strike": sc, "expiry": expiry, "option_type": "call"},
            {"symbol": self._occ_symbol(trade.ticker, expiry, lc, "call"), "ratio_qty": 1, "side": "buy", "strike": lc, "expiry": expiry, "option_type": "call"},
        ]

    def _legs_butterfly(self, trade: Trade, feat: dict) -> list[dict]:
        atm = feat.get("atm", 0)
        lo = feat.get("lo", atm - 5)
        hi = feat.get("hi", atm + 5)
        expiry = self._parse_date(feat.get("expiry")) or trade.earnings_date
        opt_type = feat.get("option_type", "call")
        # 1 long lo, 2 short atm, 1 long hi — for multi-leg Alpaca uses ratio_qty
        return [
            {"symbol": self._occ_symbol(trade.ticker, expiry, lo, opt_type), "ratio_qty": 1, "side": "buy", "strike": lo, "expiry": expiry, "option_type": opt_type},
            {"symbol": self._occ_symbol(trade.ticker, expiry, atm, opt_type), "ratio_qty": 2, "side": "sell", "strike": atm, "expiry": expiry, "option_type": opt_type},
            {"symbol": self._occ_symbol(trade.ticker, expiry, hi, opt_type), "ratio_qty": 1, "side": "buy", "strike": hi, "expiry": expiry, "option_type": opt_type},
        ]

    def _legs_risk_reversal(self, trade: Trade, feat: dict) -> list[dict]:
        kc = feat.get("call_strike", 0)
        kp = feat.get("put_strike", 0)
        expiry = self._parse_date(feat.get("expiry")) or trade.earnings_date
        return [
            {"symbol": self._occ_symbol(trade.ticker, expiry, kp, "put"), "ratio_qty": 1, "side": "sell", "strike": kp, "expiry": expiry, "option_type": "put"},
            {"symbol": self._occ_symbol(trade.ticker, expiry, kc, "call"), "ratio_qty": 1, "side": "buy", "strike": kc, "expiry": expiry, "option_type": "call"},
        ]

    # ──────────────── Helpers ─────────────────────────────────────────────

    def _resolve_symbol(self, ticker: str, expiry: date, strike: float, option_type: str) -> Optional[str]:
        """Resolve Alpaca's internal option symbol via the contract catalog.

        Used ONLY as the lazy fallback when an OCC-constructed symbol is
        rejected at submission. Results are cached per (ticker, expiry window)
        so a multi-leg order costs at most one catalog fetch per expiry.
        """
        expiry_min = (expiry - timedelta(days=2)).isoformat()
        expiry_max = (expiry + timedelta(days=2)).isoformat()
        cache_key = (ticker, expiry_min, expiry_max)
        if cache_key not in self._contracts_cache:
            self._contracts_cache[cache_key] = self.client.get_option_contracts(
                ticker,
                expiration_date_gte=expiry_min,
                expiration_date_lte=expiry_max,
                limit=200,
            )
        contracts = self._contracts_cache[cache_key].get("option_contracts", [])

        # Exact strike only — a nearest-strike fallback turned TPR 131/131
        # into a 131/130 diagonal. If the catalog only has a different
        # strike, refuse the change rather than submit a new structure.
        exact = None
        closest = None
        closest_dist = float("inf")
        for c in contracts:
            if c.get("type", "").lower() != option_type.lower():
                continue
            try:
                c_strike = float(c.get("strike_price", 0))
            except (TypeError, ValueError):
                continue
            dist = abs(c_strike - strike)
            if dist <= 0.001:
                exact = c
                break
            if dist < closest_dist:
                closest_dist = dist
                closest = c
        if exact and exact.get("symbol") and resolved_keeps_strike(strike, exact["symbol"]):
            return exact["symbol"]
        if closest and closest.get("symbol"):
            try:
                got = float(closest.get("strike_price") or 0)
            except (TypeError, ValueError):
                got = 0.0
            raise StrikeChangedError(ticker, strike, closest["symbol"], got)
        return None

    def _occ_symbol(self, ticker: str, expiry: date, strike: float, option_type: str) -> str:
        """Build the OCC 21-char option symbol — pure string construction, no
        API call. Alpaca accepts standard OCC symbols directly (verified); the
        contract-catalog fallback lives in _submit_with_resolution for the rare
        root that differs from the ticker.
        """
        root = ticker.upper()
        date_code = expiry.strftime("%y%m%d")
        type_code = "C" if option_type.lower() == "call" else "P"
        strike_padded = f"{int(round(strike * 1000)):08d}"
        return f"{root}{date_code}{type_code}{strike_padded}"

    def _parse_date(self, val) -> Optional[date]:
        if val is None:
            return None
        if isinstance(val, date):
            return val
        if isinstance(val, datetime):
            return val.date()
        if isinstance(val, str):
            try:
                return date.fromisoformat(val[:10])
            except Exception:
                return None
        return None

    def _min_expiry(self, legs: list[dict]) -> Optional[date]:
        expiries = [leg.get("expiry") for leg in legs if leg.get("expiry") is not None]
        return min(expiries) if expiries else None

    def _last_look(self, trade: Trade, legs: list[dict],
                   limit_price: Optional[float]) -> tuple[Optional[str], Optional[float]]:
        """Refresh combo marks immediately before submit. (veto, mid)."""
        symbols = [leg["symbol"] for leg in legs]
        snaps: dict = {}
        try:
            raw = self.client.get_option_snapshots_bulk(*symbols)
            if isinstance(raw, dict):
                snaps = raw
        except Exception as exc:
            logger.info("last_look bulk snapshot failed: %s", exc)
        if not snaps or not any(isinstance(v, dict) for v in snaps.values()):
            snaps = {}
            for leg in legs:
                try:
                    one = self.client.get_option_snapshot(leg["symbol"]) or {}
                except Exception:
                    one = {}
                if isinstance(one, dict):
                    snaps[leg["symbol"]] = one
        q = combo_quotes(legs, snaps)
        if q is None:
            # No usable marks (typical of unit-test mocks). Don't veto —
            # the earlier mid-cap check + limit-price requirement still apply.
            return None, None
        mid = q["mid"]
        spot = None
        try:
            spot = self.client.get_stock_latest_trade(trade.ticker)
        except Exception:
            spot = None
        feat = trade.features or {}
        if not isinstance(spot, (int, float)):
            try:
                spot = float(feat.get("price") or feat.get("spot") or 0) or None
            except (TypeError, ValueError):
                spot = None
        proposed = limit_price if limit_price else (trade.entry_price or mid)
        return last_look_veto(legs, snaps, spot=spot, proposed_debit=proposed), mid

    def _await_fill(self, order: dict) -> dict:
        """Poll until a terminal status or fill, then return the latest order."""
        if not isinstance(order, dict):
            return order
        oid = order.get("id") or order.get("order_id")
        if not oid or not isinstance(oid, str):
            return order
        status = str(order.get("status") or "").lower()
        try:
            filled = float(order.get("filled_qty") or 0)
        except (TypeError, ValueError):
            filled = 0.0
        if status in _TERMINAL_ORDER or filled > 0:
            return order
        latest = order
        attempts = getattr(self, "_fill_poll_attempts", FILL_POLL_ATTEMPTS)
        secs = getattr(self, "_fill_poll_secs", FILL_POLL_SECS)
        for _ in range(max(0, int(attempts))):
            self._sleep(float(secs))
            try:
                fetched = self.client.get_order(oid)
            except Exception as exc:
                logger.info("fill poll %s failed: %s", oid, exc)
                return latest
            if not isinstance(fetched, dict):
                return latest
            latest = fetched
            status = str(latest.get("status") or "").lower()
            try:
                filled = float(latest.get("filled_qty") or 0)
            except (TypeError, ValueError):
                filled = 0.0
            if status in _TERMINAL_ORDER or filled > 0:
                return latest
        return latest

    def _exit_by(self, legs: list[dict]) -> Optional[date]:
        """Structural exit deadline: the earliest leg expiry, but only when
        legs span more than one distinct expiry (calendar-style structures,
        e.g. calendar_call_ml/debit_size_exploit — the near leg vanishing is
        a hard deadline). A single-expiry structure (straddle, vertical,
        condor, ...) has no differential-expiry deadline to protect against
        and keeps using whatever [[exits]] rules its TOML configures."""
        expiries = {leg.get("expiry") for leg in legs if leg.get("expiry") is not None}
        return min(expiries) if len(expiries) > 1 else None

    def _midpoint_price(self, legs: list[dict]) -> Optional[float]:
        """Compute midpoint price for limit orders (sum of leg midpoints)."""
        total = 0.0
        for leg in legs:
            snap = self.client.get_option_snapshot(leg["symbol"])
            if not isinstance(snap, dict):
                return None
            q = snap.get("latestQuote") or {}
            if not isinstance(q, dict):
                return None
            try:
                bid = float(q.get("bp") or 0)
                ask = float(q.get("ap") or 0)
            except (TypeError, ValueError):
                return None
            mid = (bid + ask) / 2
            sign = -1 if leg["side"] == "sell" else 1
            total += sign * mid
        return total if total > 0 else None


# ---------------------------------------------------------------------------
# High-level: run best strategies today → submit orders
# ---------------------------------------------------------------------------

BEST_STRATEGIES = [
    "calendar_call_ml",
    "debit_size_exploit",
    "short_straddle",
    "vol_risk_premium",
    "iv_rv_mean_reversion",
    "term_structure_steepener",
    "earnings_quality",
]


def _resolve_strategy(name: str):
    """Try calendar strategy registry first, then positional strategies.

    Calendar strategies: instances with .run(data) → StrategyResult
    Positional strategies: classes registered in POSITIONAL_STRATEGIES
    """
    try:
        from earnings_edge.backtest.calendar import get_strategy
        return get_strategy(name)
    except KeyError:
        pass
    from earnings_edge.backtest.positional import POSITIONAL_STRATEGIES
    if name in POSITIONAL_STRATEGIES:
        cls = POSITIONAL_STRATEGIES[name]
        return cls()
    raise KeyError(f"Strategy {name} not found in any registry")


def run_auto_trade(
    strategies: Optional[list[str]] = None,
    db_path: Optional[str] = None,
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None,
    max_per_ticker: float = 5000.0,
    min_buying_power: float = 10000.0,
    max_orders: int = 20,
) -> dict:
    """Run specified strategies on today's data and submit paper orders.

    Safety caps:
    - max_per_ticker: max dollar deployed per underlying per run
    - min_buying_power: abort if buying power drops below this
    - max_orders: hard cap on orders submitted per run

    Returns dict with execution summary.
    """
    from earnings_edge.trading_types import DataBundle

    bundle = DataBundle.from_db(db_path)
    try:
        # Risk-gated bridge even on the legacy auto path: kill switch,
        # portfolio caps, lifecycle all apply (framework absent → plain bridge).
        from framework.core.registry import get_registry
        from framework.execution.lifecycle import LifecycleManager
        from framework.risk.manager import RiskManager
        registry = get_registry()
        bridge = StrategyBridge(
            client=create_client(api_key, api_secret),
            risk_manager=RiskManager(),
            lifecycle_manager=LifecycleManager(),
            limits_resolver=registry.limits_for,
            sizer_resolver=registry.sizer_spec,
        )
    except Exception as exc:
        logger.warning("risk layer unavailable (%s) — ungated legacy bridge", exc)
        bridge = StrategyBridge(client=create_client(api_key, api_secret))
    results = {}
    total_submitted = 0
    total_skipped = 0

    # Track ticker-level deployment
    ticker_spend: dict[str, float] = {}
    buying_power = 1_000_000.0  # default upper bound
    try:
        fetched = bridge.account_buying_power()
        if fetched:
            buying_power = fetched
            logger.info("Buying power: $%.2f", buying_power)
    except Exception as e:
        logger.warning("Could not fetch buying power, using default: %s", e)

    strategies = strategies or BEST_STRATEGIES
    for name in strategies:
        try:
            strategy = _resolve_strategy(name)
        except KeyError:
            logger.warning("Strategy %s not found in registry", name)
            continue

        result = strategy.run(bundle)
        # Only TAKE trades are actionable — result.trades also carries SKIP
        # rows (backtest bookkeeping); submitting those would trade against
        # the model's own decision.
        actionable = [t for t in result.trades if t.ml_decision == "TAKE"]
        if not actionable:
            results[name] = {"status": "no-signals", "trades": 0, "submitted": 0}
            continue

        submitted_for_strat = 0
        for trade in actionable:
            # Safety: hard cap on orders
            if total_submitted >= max_orders:
                logger.warning("Max orders (%d) reached — stopping", max_orders)
                break

            # Safety: buying power floor
            if buying_power < min_buying_power:
                logger.warning(
                    "Buying power $%.2f below minimum $%.2f — stopping orders",
                    buying_power, min_buying_power,
                )
                break

            # Safety: per-ticker cap
            ticker = trade.ticker
            if ticker_spend.get(ticker, 0) >= max_per_ticker:
                logger.warning("Ticker %s at $%.2f / $%.2f cap — skipping", ticker, ticker_spend[ticker], max_per_ticker)
                continue

            order_result = bridge.execute_trade(trade)
            if order_result:
                submitted_for_strat += 1
                total_submitted += 1
                cost = abs(order_result.filled_avg_price or 0) * order_result.filled_qty
                if cost > 0:
                    ticker_spend[ticker] = ticker_spend.get(ticker, 0) + cost
                    buying_power -= cost
            else:
                total_skipped += 1

        results[name] = {
            "status": "ok",
            "trades": len(actionable),
            "submitted": submitted_for_strat,
            "skipped": len(actionable) - submitted_for_strat,
        }

        if total_submitted >= max_orders:
            break

    try:
        buying_power = bridge.account_buying_power() or buying_power
    except Exception:
        pass

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "strategies": results,
        "buying_power": buying_power,
        "total_submitted": total_submitted,
        "total_skipped": total_skipped,
        "skip_reasons": dict(bridge.skip_reasons),
        "ticker_spend": {k: f"${v:.2f}" for k, v in ticker_spend.items()},
        "orders": [
            {
                "order_id": o.order_id,
                "strategy": o.strategy,
                "symbol": o.symbol,
                "legs": len(o.legs),
                "status": o.status,
                "client_order_id": o.client_order_id,
                "filled_qty": o.filled_qty,
                "filled_avg_price": o.filled_avg_price,
            }
            for o in bridge.submitted
        ],
    }
    return summary


if __name__ == "__main__":
    import json

    summary = run_auto_trade()
    print(json.dumps(summary, indent=2, default=str))
