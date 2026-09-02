"""Exactly one follow-up scan after a failed/empty run, 10–15 minutes later."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

RETRY_MINUTES = 12  # inside the 10–15 min window


def should_retry_scan(result: Optional[dict]) -> bool:
    if not result:
        return True
    if not result.get("success"):
        return True
    stats = result.get("stats") or {}
    if stats.get("candidate_count") == 0:
        return True
    return False


def should_chain_proposals(result: Optional[dict]) -> bool:
    if not result or not result.get("success"):
        return False
    stats = result.get("stats") or {}
    return int(stats.get("candidate_count") or 0) > 0


def next_retry(now: Optional[datetime] = None, minutes: int = RETRY_MINUTES) -> datetime:
    if not 10 <= minutes <= 15:
        raise ValueError("retry must be 10–15 minutes")
    now = now or datetime.now(timezone.utc)
    return now + timedelta(minutes=minutes)


def record_retry(minutes: int = RETRY_MINUTES) -> dict:
    """Pure record of the scheduled follow-up (tests inspect the offset)."""
    when = next_retry(minutes=minutes)
    return {"minutes": minutes, "next_run": when.isoformat(), "id": "scan_retry"}
