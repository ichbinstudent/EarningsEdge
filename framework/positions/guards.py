"""Assignment/exercise guards for short option legs.

Flags positions at elevated early-assignment risk so the exit engine can act
(close a day early, or escalate to approval) instead of discovering stock
assignment after the fact:

- Short ITM call with an ex-dividend date before expiry (dividend capture
  assignment — the classic early-exercise trigger).
- Short ITM option of either type inside ``dte_threshold`` days of expiry.

Pure logic: legs in, risk flags out. Market data is injected.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional

_OCC_RE = re.compile(r"^([A-Z]{1,6})(\d{6})([CP])(\d{8})$")


def parse_occ(symbol: str) -> Optional[LegView]:
    """Parse an OCC option symbol (AAPL260727C00325000) into a LegView."""
    m = _OCC_RE.match(symbol)
    if not m:
        return None
    root, ymd, cp, strike = m.groups()
    expiry = datetime.strptime(ymd, "%y%m%d").date()
    return LegView(
        symbol=symbol,
        side="sell",  # caller overrides; parse is side-agnostic
        option_type="call" if cp == "C" else "put",
        strike=int(strike) / 1000.0,
        expiry=expiry,
    )


def occ_underlying(symbol: str) -> Optional[str]:
    m = _OCC_RE.match(symbol)
    return m.group(1) if m else None


@dataclass(frozen=True)
class LegView:
    """Minimal leg shape the guard needs (adapted from bridge leg dicts)."""
    symbol: str
    side: str                # "buy" | "sell"
    option_type: str         # "call" | "put"
    strike: float
    expiry: date


@dataclass(frozen=True)
class AssignmentRisk:
    symbol: str
    reason: str              # "dividend_capture" | "near_expiry_itm"
    dte: int
    dividend_date: Optional[date] = None


def _is_itm(leg: LegView, spot: float) -> bool:
    if leg.option_type == "call":
        return spot > leg.strike
    return spot < leg.strike


def check_assignment_risk(
    legs: list[LegView],
    spot: float,
    on: Optional[date] = None,
    ex_dividend_date: Optional[date] = None,
    dte_threshold: int = 3,
) -> list[AssignmentRisk]:
    """Flag short legs with elevated early-assignment risk."""
    on = on or date.today()
    risks: list[AssignmentRisk] = []
    for leg in legs:
        if leg.side != "sell":
            continue
        if not _is_itm(leg, spot):
            continue
        dte = (leg.expiry - on).days
        if (
            leg.option_type == "call"
            and ex_dividend_date is not None
            and on <= ex_dividend_date <= leg.expiry
        ):
            risks.append(AssignmentRisk(
                symbol=leg.symbol, reason="dividend_capture",
                dte=dte, dividend_date=ex_dividend_date,
            ))
        elif dte <= dte_threshold:
            risks.append(AssignmentRisk(
                symbol=leg.symbol, reason="near_expiry_itm", dte=dte,
            ))
    return risks


def leg_view_from_dict(leg: dict) -> LegView:
    """Adapt a StrategyBridge leg dict to a LegView."""
    expiry = leg["expiry"]
    if isinstance(expiry, str):
        expiry = datetime.strptime(expiry, "%Y-%m-%d").date()
    return LegView(
        symbol=leg["symbol"], side=leg["side"],
        option_type=leg["option_type"], strike=float(leg["strike"]), expiry=expiry,
    )
