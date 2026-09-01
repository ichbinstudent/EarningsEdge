"""Kill switch: persisted halt of all order submission.

Tripped automatically by the risk manager (daily loss limit, repeated broker
rejections) or manually via Telegram ``/halt``. While halted, proposals are
still built and tagged but no order reaches the broker. State survives
restarts because it lives in the ``risk_state`` table.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from earnings_edge.db import risk_events_insert, risk_state_get, risk_state_set_halted

logger = logging.getLogger("framework.risk.killswitch")


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class KillSwitch:
    """Read/trip/resume the global halt flag."""

    def __init__(self):
        risk_state_get()

    def is_halted(self) -> bool:
        return bool(risk_state_get().get("halted"))

    def status(self) -> dict:
        row = risk_state_get()
        return {
            "halted": row.get("halted", 0),
            "reason": row.get("reason"),
            "tripped_at": row.get("tripped_at"),
            "tripped_by": row.get("tripped_by"),
        }

    def trip(self, reason: str, by: str = "system") -> None:
        logger.error("KILL SWITCH TRIPPED by %s: %s", by, reason)
        risk_state_set_halted(
            True, reason=reason, tripped_at=_utcnow(), tripped_by=by,
        )
        risk_events_insert("trip", f"{by}: {reason}")
        from framework.alerts import DEDUPER
        DEDUPER.emit(
            "kill_switch",
            f"🛑 KILL SWITCH TRIPPED by {by}: {reason} — no orders will submit until /resume.",
        )

    def resume(self, by: str = "operator") -> None:
        logger.warning("Kill switch resumed by %s", by)
        risk_state_set_halted(False)
        risk_events_insert("resume", by)


def record_event(event_type: str, detail: str,
                 strategy: Optional[str] = None) -> None:
    """Append an audit row to ``risk_events``."""
    risk_events_insert(event_type, detail, strategy=strategy)
