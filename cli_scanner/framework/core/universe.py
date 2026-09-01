"""Universe abstraction: what symbols a strategy scans.

Strategies declare a universe instead of hardcoding symbol sources (today the
earnings calendar is hardwired into the scan path). Universes are cheap to
construct and lazy — symbols are only fetched when ``symbols()`` is called.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import date
from pathlib import Path
from typing import Optional

logger = logging.getLogger("framework.core.universe")


class Universe(ABC):
    """Base class for symbol universes."""

    name: str = "base"

    @abstractmethod
    def symbols(self, on: date) -> list[str]:
        """Symbols to consider on ``on``."""

    def metadata(self, symbol: str) -> dict:
        """Optional per-symbol context (e.g. earnings timing)."""
        return {}


class StaticListUniverse(Universe):
    """Fixed symbol list from config."""

    name = "static"

    def __init__(self, symbols: list[str]):
        self._symbols = [s.upper() for s in symbols]

    def symbols(self, on: date) -> list[str]:
        return list(self._symbols)


class FileUniverse(Universe):
    """Symbols from a text file (one per line, '#' comments allowed)."""

    name = "file"

    def __init__(self, path: Path):
        self._path = Path(path)

    def symbols(self, on: date) -> list[str]:
        if not self._path.is_file():
            logger.warning("universe file %s missing", self._path)
            return []
        out = []
        for line in self._path.read_text().splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                out.append(line.upper())
        return out


class EarningsCalendarUniverse(Universe):
    """Tickers reporting earnings on a date (wraps the existing collector)."""

    name = "earnings_calendar"

    def __init__(self, collector=None):
        # Lazy default so importing this module never pulls in selenium.
        self._collector = collector

    def _get_collector(self):
        if self._collector is None:
            from earnings_edge.collectors.earnings_calendar import EarningsCalendarCollector
            self._collector = EarningsCalendarCollector()
        return self._collector

    def symbols(self, on: date) -> list[str]:
        candidates = self._get_collector().fetch(on)
        return sorted({c.ticker for c in candidates})

    def metadata(self, symbol: str) -> dict:
        return {"source": "earnings_calendar"}


def build_universe(spec: dict, collector=None) -> Universe:
    """Construct a universe from a strategy-config ``[universe]`` section."""
    utype = spec.get("type", "static")
    if utype == "static":
        return StaticListUniverse(spec.get("symbols", []))
    if utype == "file":
        return FileUniverse(Path(spec["path"]))
    if utype == "earnings_calendar":
        return EarningsCalendarUniverse(collector=collector)
    raise ValueError(f"unknown universe type {utype!r}")
