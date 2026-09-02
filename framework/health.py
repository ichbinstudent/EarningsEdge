"""Process readiness: lock + equity freshness + scan age + broker clock."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional


EQUITY_STALE_MIN = 30
SCAN_STALE_HOURS = 26


def _parse(ts) -> Optional[datetime]:
    if ts is None:
        return None
    if isinstance(ts, datetime):
        t = ts
    else:
        try:
            t = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        except ValueError:
            return None
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    return t


def health_ready(
    *,
    lock_held: bool,
    last_equity_ts: Optional[str] = None,
    last_scan_ts: Optional[str] = None,
    clock_ok: bool = False,
    now: Optional[datetime] = None,
    market_open: bool = False,
    weekday: Optional[bool] = None,
    equity_skipped_closed: bool = False,
) -> dict:
    """Return ``{"ready": bool, "reasons": list[str]}``.

    In RTH, last equity write must be < 30 min (or an explicit closed-market
    skip). On a weekday, last successful scan must be within ~26h.
    """
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    if weekday is None:
        weekday = now.weekday() < 5
    reasons: list[str] = []
    if not lock_held:
        reasons.append("no lock")
    if not clock_ok:
        reasons.append("clock unreachable")
    if market_open:
        eq = _parse(last_equity_ts)
        if eq is None or (now - eq) > timedelta(minutes=EQUITY_STALE_MIN):
            reasons.append("equity stale")
    elif not equity_skipped_closed and last_equity_ts is None:
        # Closed: an explicit skip or any prior snapshot is enough.
        pass
    if weekday:
        scan = _parse(last_scan_ts)
        if scan is None or (now - scan) > timedelta(hours=SCAN_STALE_HOURS):
            reasons.append("scan stale")
    return {"ready": not reasons, "reasons": reasons}
