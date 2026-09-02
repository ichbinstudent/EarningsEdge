"""Broker-truth book: classify local groups vs Alpaca positions.

Buckets:
- managed — symbol is open locally AND at the broker
- orphan  — at the broker, not in an open local group (ATLO/BA, assignments)
- missing — open locally, gone at the broker
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional

from .guards import occ_underlying, parse_occ


def ticker_of(symbol: str) -> str:
    return occ_underlying(symbol) or symbol


def _f(value) -> Optional[float]:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


@dataclass
class BookItem:
    bucket: str                  # managed | orphan | missing
    symbol: str
    ticker: str
    qty: float
    side: str                    # long | short | buy | sell
    strategy: Optional[str] = None
    group_id: Optional[str] = None
    event_date: Optional[date] = None
    expiry: Optional[date] = None
    upl: Optional[float] = None
    current_price: Optional[float] = None
    avg_entry: Optional[float] = None


@dataclass
class Book:
    managed: list[BookItem] = field(default_factory=list)
    orphan: list[BookItem] = field(default_factory=list)
    missing: list[BookItem] = field(default_factory=list)

    @property
    def broker_count(self) -> int:
        return len(self.managed) + len(self.orphan)

    @property
    def local_count(self) -> int:
        return len(self.managed) + len(self.missing)


def classify_book(groups, broker_positions: list[dict],
                  ignored: Optional[set[str]] = None) -> Book:
    """Diff open PositionGroups against Alpaca ``get_positions()`` rows.

    ``ignored`` is ``adopted_positions`` (operator Ignore / baseline adopt):
    those symbols stay at the broker but are not orphans on the desk.
    """
    broker_by = {p.get("symbol"): p for p in broker_positions if p.get("symbol")}
    ignored = ignored or set()
    local_syms: set[str] = set()
    book = Book()
    for g in groups:
        for leg in g.legs:
            local_syms.add(leg.symbol)
            bp = broker_by.get(leg.symbol)
            qty = float(leg.qty)
            side = "short" if leg.side == "sell" else "long"
            item = BookItem(
                bucket="managed" if bp else "missing",
                symbol=leg.symbol,
                ticker=g.ticker or ticker_of(leg.symbol),
                qty=_f(bp.get("qty")) if bp else qty,
                side=(bp.get("side") or side) if bp else side,
                strategy=g.strategy,
                group_id=g.group_id,
                event_date=g.event_date,
                expiry=leg.expiry,
                upl=_f(bp.get("unrealized_pl")) if bp else None,
                current_price=_f(bp.get("current_price")) if bp else None,
                avg_entry=_f(bp.get("avg_entry_price")) if bp else g.entry_price,
            )
            (book.managed if bp else book.missing).append(item)
    for sym, bp in broker_by.items():
        if sym in local_syms or sym in ignored:
            continue
        parsed = parse_occ(sym)
        book.orphan.append(BookItem(
            bucket="orphan",
            symbol=sym,
            ticker=ticker_of(sym),
            qty=_f(bp.get("qty")) or 0.0,
            side=bp.get("side") or "long",
            strategy="unmanaged",
            expiry=parsed.expiry if parsed else None,
            upl=_f(bp.get("unrealized_pl")),
            current_price=_f(bp.get("current_price")),
            avg_entry=_f(bp.get("avg_entry_price")),
        ))
    return book
