"""Order lifecycle manager with pricing policies.

Generalizes the forward-factor ladder's limit-chase mechanics
(``fwd_factor_ladder.LadderRunner``) into a reusable component: submit at a
passive price, poll for fills, reprice on a walk schedule toward aggressive,
cancel when exhausted. Used for entries AND exits — exits get real fills
instead of blind market orders.

The manager is broker-agnostic: it works against any client exposing the
Alpaca-shaped ``submit_multi_leg_order``/``submit_order``/``get_order``/
``cancel_order`` surface. Quote freshness is injected via ``quote_fn`` so the
caller owns data-source cost discipline.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Optional

logger = logging.getLogger("framework.execution.order_manager")


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Pricing policies ---------------------------------------------------------

class PricingPolicy:
    """Yields the sequence of limit prices to work an order at."""

    name = "base"

    def walk(self, mid: float, side: str) -> list[Optional[float]]:
        raise NotImplementedError


class MidPricePolicy(PricingPolicy):
    """One passive limit at the mid, take it or leave it."""

    name = "mid"

    def walk(self, mid: float, side: str) -> list[Optional[float]]:
        return [round(mid, 2)]


class LimitWalkPolicy(PricingPolicy):
    """Chase from the mid toward the spread in ``steps`` increments.

    ``side`` is the order side: "buy" walks UP from mid, "sell" walks DOWN.
    The final rung pays/crosses ``final_improve_bps`` through the mid, which
    for liquid underlyings behaves like a marketable limit (fill protection
    without true market-order slippage).
    """

    name = "limit_walk"

    def __init__(self, steps: int = 3, step_improve_bps: float = 25.0,
                 final_improve_bps: float = 100.0):
        if steps < 1:
            raise ValueError("steps must be >= 1")
        self.steps = steps
        self.step_improve_bps = step_improve_bps
        self.final_improve_bps = final_improve_bps

    def walk(self, mid: float, side: str) -> list[Optional[float]]:
        sign = 1.0 if side == "buy" else -1.0
        prices = []
        for i in range(self.steps):
            frac = i / max(self.steps - 1, 1)
            improve_bps = self.step_improve_bps + frac * (self.final_improve_bps - self.step_improve_bps)
            prices.append(round(mid * (1 + sign * improve_bps / 10_000), 2))
        return prices


# ── Managed order -------------------------------------------------------------

@dataclass
class ManagedOrder:
    """Result of a managed submit/chase cycle."""

    client_order_id: str
    side: str
    qty: int
    policy: str
    state: str = "working"           # working | filled | partial | canceled | exhausted | error
    order_ids: list[str] = field(default_factory=list)
    filled_qty: float = 0.0
    filled_avg_price: Optional[float] = None
    rungs_used: int = 0
    detail: str = ""


# ── Manager ---------------------------------------------------------------------

TERMINAL_STATES = {"filled", "canceled", "expired", "rejected"}


class OrderManager:
    """Submit-poll-reprice loop for single- and multi-leg orders."""

    def __init__(self, client, poll_secs: float = 15.0, sleep: Callable[[float], None] = time.sleep):
        self.client = client
        self.poll_secs = poll_secs
        self._sleep = sleep  # injectable for tests

    def execute(
        self,
        legs: list[dict],
        qty: int,
        policy: PricingPolicy,
        quote_fn: Callable[[], Optional[float]],
        side: str = "buy",
        client_order_id: Optional[str] = None,
        time_in_force: str = "day",
    ) -> ManagedOrder:
        """Work an order through the policy's price walk.

        ``legs`` are Alpaca-shaped dicts (symbol/side/ratio_qty). ``quote_fn``
        returns the current net mid for the structure (debit positive for buys).
        """
        cid = client_order_id or f"managed_{int(datetime.now(timezone.utc).timestamp())}"
        mo = ManagedOrder(client_order_id=cid, side=side, qty=qty, policy=policy.name)

        mid = quote_fn()
        if mid is None or mid <= 0:
            mo.state = "error"
            mo.detail = "no quote"
            return mo

        order_id: Optional[str] = None
        for rung, price in enumerate(policy.walk(mid, side), 1):
            mo.rungs_used = rung
            try:
                if order_id is not None:
                    self._cancel_open(order_id)
                order_id = self._submit(legs, qty, price, cid, time_in_force)
                mo.order_ids.append(order_id)
            except Exception as exc:
                logger.warning("managed order submit failed (rung %d): %s", rung, exc)
                mo.state = "error"
                mo.detail = f"submit failed: {exc}"
                return mo

            filled = self._poll_fill(order_id)
            if filled:
                mo.filled_qty, mo.filled_avg_price = filled
                mo.state = "filled" if mo.filled_qty >= qty else "partial"
                return mo

        if order_id is not None:
            self._cancel_open(order_id)
        mo.state = "exhausted"
        mo.detail = f"no fill after {mo.rungs_used} rungs"
        return mo

    # -- internals -----------------------------------------------------------

    def _submit(self, legs, qty, price, cid, tif) -> str:
        # Never market: resting limit orders at computed prices only
        # (DEFAULT_ORDER_TYPE="limit" invariant, enforced framework-side).
        if price is None:
            logger.warning("refusing market submit: no limit price for %s",
                           legs[0]["symbol"] if legs else "?")
            raise ValueError("no limit price — refusing to submit a market order")
        for attempt in range(1, 4):
            try:
                if len(legs) == 1:
                    leg = legs[0]
                    order = self.client.submit_order(
                        symbol=leg["symbol"], qty=qty * leg.get("ratio_qty", 1), side=leg["side"],
                        order_type="limit",
                        limit_price=price, time_in_force=tif, client_order_id=cid,
                    )
                else:
                    order = self.client.submit_multi_leg_order(
                        legs=legs, qty=qty,
                        order_type="limit",
                        limit_price=price, time_in_force=tif, client_order_id=cid,
                    )
                return order["id"]
            except Exception as exc:
                if attempt == 3:
                    raise
                logger.warning("submit API error (attempt %d/3): %s", attempt, exc)
                self._sleep(1.0 * (2 ** (attempt - 1)))
        raise RuntimeError("Submit failed")
    def _poll_fill(self, order_id: str) -> Optional[tuple[float, Optional[float]]]:
        """(filled_qty, avg_price) once any fill is seen, else None.
        Calculates net price from legs if the parent order is empty."""
        # Poll up to 3 times, spaced out, rather than a single wait-and-give-up
        poll_interval = max(self.poll_secs / 3.0, 1.0)
        for _ in range(3):
            self._sleep(poll_interval)
            try:
                order = self.client.get_order(order_id)
            except Exception as exc:
                logger.warning("get_order API error: %s", exc)
                continue
                
            status = (order.get("status") or "").lower()
            filled_qty = float(order.get("filled_qty") or 0)
            
            if status == "filled" or filled_qty > 0:
                avg = order.get("filled_avg_price")
                # Fallback to compute net price from legs if Alpaca omits it on the parent
                if (not avg or float(avg) == 0.0) and order.get("legs"):
                    net_amount = 0.0
                    for leg in order["legs"]:
                        l_qty = float(leg.get("filled_qty") or 0)
                        l_avg = float(leg.get("filled_avg_price") or 0)
                        if l_qty > 0:
                            # A buy leg costs money (negative cash flow), sell leg earns money
                            sign = -1.0 if leg.get("side") == "buy" else 1.0
                            # Adjust by leg ratio to get unit price
                            ratio = float(leg.get("ratio_qty") or 1.0)
                            net_amount += (l_avg * ratio) * sign
                    avg = abs(net_amount) if net_amount != 0.0 else None
                return filled_qty, float(avg) if avg is not None else None
        return None

    def _cancel_open(self, order_id: str) -> None:
        for attempt in range(1, 4):
            try:
                order = self.client.get_order(order_id)
                if (order.get("status") or "").lower() not in TERMINAL_STATES:
                    self.client.cancel_order(order_id)
                return
            except Exception as exc:
                if attempt == 3:
                    logger.info("cancel %s failed (non-fatal): %s", order_id, exc)
                    return
                self._sleep(0.5)
