"""Strategy registry: the single point where TOML configs meet runtime.

Wraps ``load_strategy_configs`` and resolves, per strategy *code* name (the
``trade.strategy`` string the engines actually emit), the effective risk
limits, lifecycle, execution mode and sizer. Also seeds ``strategy_state``
from configs at startup — operator changes via /promote //demote win because
seeding is INSERT OR IGNORE.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from ..risk.manager import RiskLimits
from .config import StrategyConfig, load_strategy_configs

logger = logging.getLogger("framework.core.registry")


class StrategyRegistry:
    def __init__(self, configs: Optional[dict[str, StrategyConfig]] = None,
                 config_dir: Optional[Path] = None,
                 base_limits: Optional[RiskLimits] = None):
        self.configs = configs if configs is not None else load_strategy_configs(config_dir)
        self.base_limits = base_limits or RiskLimits()

    # -- resolution -------------------------------------------------------------

    def get(self, name: str) -> Optional[StrategyConfig]:
        return self.configs.get(name)

    def limits_for(self, name: str) -> RiskLimits:
        """Global defaults with the strategy's [risk.limits] overrides applied."""
        cfg = self.configs.get(name)
        if cfg is None:
            return self.base_limits
        overrides = cfg.risk_limit_overrides()
        if not overrides:
            return self.base_limits
        return RiskLimits(**{**self.base_limits.__dict__, **overrides})

    def execution_mode(self, name: str) -> str:
        cfg = self.configs.get(name)
        return cfg.execution_mode if cfg else "approval"

    def is_enabled(self, name: str) -> bool:
        cfg = self.configs.get(name)
        return cfg.enabled if cfg else True  # unconfigured = enabled (legacy)

    def enabled_strategies(self, names: list[str]) -> list[str]:
        """Filter a code-level strategy list by config enabled flags."""
        return [n for n in names if self.is_enabled(n)]

    def sizer_spec(self, name: str) -> Optional[dict]:
        cfg = self.configs.get(name)
        return dict(cfg.sizer) if cfg and cfg.sizer else None

    # -- lifecycle seeding --------------------------------------------------------

    def sync_lifecycle(self) -> int:
        """Seed strategy_state from configs where no row exists yet.

        Never overwrites: an operator's /promote //demote persists across
        restarts. Returns the number of rows seeded.
        """
        from datetime import datetime, timezone
        from earnings_edge.db import strategy_state_insert_ignore
        now = datetime.now(timezone.utc).isoformat()
        seeded = 0
        for name, cfg in self.configs.items():
            seeded += strategy_state_insert_ignore(
                name, cfg.lifecycle, updated_at=now, updated_by="config",
            )
        if seeded:
            logger.info("lifecycle seeded from configs for %d strategies", seeded)
        return seeded


_REGISTRY: Optional[StrategyRegistry] = None


def get_registry() -> StrategyRegistry:
    """Process-wide registry (configs are small; reload via reset in tests)."""
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = StrategyRegistry()
    return _REGISTRY


def reset_registry() -> None:
    global _REGISTRY
    _REGISTRY = None
