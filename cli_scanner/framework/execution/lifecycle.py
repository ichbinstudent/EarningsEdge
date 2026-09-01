"""Strategy lifecycle: paper → probation → live promotion path.

State lives in ``strategy_state`` (survives restarts). The risk manager
applies a size multiplier while on probation. On a live broker
(``ALPACA_LIVE=1``) ``RiskManager.check_trade`` vetoes ``lifecycle=paper``
— only ``probation`` and ``live`` may submit. Paper broker still accepts
the paper lifecycle.
"""

from __future__ import annotations

import logging
from typing import Optional

from earnings_edge.db import strategy_state_get, strategy_state_list, strategy_state_upsert

from ..risk.killswitch import record_event

logger = logging.getLogger("framework.execution.lifecycle")

LIFECYCLES = ("paper", "probation", "live")


class LifecycleManager:
    def __init__(self, probation_size_mult: float = 0.5):
        self.probation_size_mult = probation_size_mult

    def state(self, strategy: str) -> str:
        row = strategy_state_get(strategy)
        return row["lifecycle"] if row else "paper"

    def set_state(self, strategy: str, state: str, by: str = "operator") -> None:
        if state not in LIFECYCLES:
            raise ValueError(f"lifecycle must be one of {LIFECYCLES}")
        prev = self.state(strategy)
        strategy_state_upsert(strategy, lifecycle=state, updated_by=by)
        record_event(
            "promote" if LIFECYCLES.index(state) > LIFECYCLES.index(prev) else "demote",
            f"{prev} → {state} (by {by})", strategy=strategy,
        )
        logger.warning("lifecycle %s: %s → %s (by %s)", strategy, prev, state, by)

    def size_multiplier(self, strategy: str) -> float:
        return self.probation_size_mult if self.state(strategy) == "probation" else 1.0

    def all_states(self) -> dict[str, str]:
        return {r["name"]: r["lifecycle"] for r in strategy_state_list()}

    @staticmethod
    def eligible_for_promotion(stats: dict, min_closed_trades: int = 20,
                               min_win_rate: float = 0.50,
                               max_drawdown_pct: float = 0.10) -> bool:
        """Evaluate promotion criteria against a strategy's closed-trade stats."""
        if stats.get("closed_trades", 0) < min_closed_trades:
            return False
        if stats.get("win_rate", 0.0) < min_win_rate:
            return False
        if stats.get("max_drawdown_pct", 1.0) > max_drawdown_pct:
            return False
        return True
