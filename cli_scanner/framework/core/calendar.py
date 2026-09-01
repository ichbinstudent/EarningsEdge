"""Exchange trading calendar (XNYS by default).

Single source of truth for market sessions, holidays and half-days, replacing
hardcoded weekday-only logic (e.g. ``earnings.scan_dates`` weekend rolls).
"""

from __future__ import annotations

import functools
from datetime import date, datetime, time as dtime
from typing import Optional

import pandas as pd

import exchange_calendars as xc

# Regular XNYS close is 16:00 ET; early close (half-days) is 13:00 ET.
_ET = "America/New_York"


class TradingCalendar:
    """Thin wrapper over exchange_calendars with date-typed helpers."""

    def __init__(self, name: str = "XNYS"):
        self.name = name
        self._cal = xc.get_calendar(name)

    # -- session dates --------------------------------------------------------

    def is_session(self, d: date) -> bool:
        return self._cal.is_session(pd.Timestamp(d))

    def next_session(self, d: date) -> date:
        """``d`` itself if a session, else the first session after it."""
        if self.is_session(d):
            return d
        future = self._cal.sessions[self._cal.sessions > pd.Timestamp(d)]
        return future[0].date()

    def next_session_after(self, d: date) -> date:
        """First session strictly after ``d``."""
        future = self._cal.sessions[self._cal.sessions > pd.Timestamp(d)]
        return future[0].date()

    def prev_session(self, d: date) -> date:
        """Last session strictly before ``d``."""
        past = self._cal.sessions[self._cal.sessions < pd.Timestamp(d)]
        return past[-1].date()

    def sessions_between(self, start: date, end: date) -> list[date]:
        return [ts.date() for ts in self._cal.sessions_in_range(pd.Timestamp(start), pd.Timestamp(end))]

    def add_sessions(self, d: date, n: int) -> date:
        """``n`` trading days after ``d`` (negative = before); 0 = ``d``'s session."""
        if n == 0:
            return self.next_session(d)
        sessions = self._cal.sessions
        ts = pd.Timestamp(d)
        if n > 0:
            future = sessions[sessions > ts]
            return future[n - 1].date() if len(future) >= n else future[-1].date()
        past = sessions[sessions < ts]
        return past[n].date() if len(past) >= -n else past[0].date()

    # -- intraday -------------------------------------------------------------

    def session_open(self, d: date) -> datetime:
        return self._cal.session_open(pd.Timestamp(d)).tz_convert(_ET).to_pydatetime()

    def session_close(self, d: date) -> datetime:
        return self._cal.session_close(pd.Timestamp(d)).tz_convert(_ET).to_pydatetime()

    def is_early_close(self, d: date) -> bool:
        if not self.is_session(d):
            return False
        return self.session_close(d).time() < dtime(16, 0)

    def is_open_now(self, now: Optional[datetime] = None) -> bool:
        now = now or pd.Timestamp.now(tz=_ET)
        return bool(self._cal.is_open_on_minute(pd.Timestamp(now)))


@functools.lru_cache(maxsize=1)
def get_calendar(name: str = "XNYS") -> TradingCalendar:
    """Shared calendar instance (calendar construction is expensive)."""
    return TradingCalendar(name)
