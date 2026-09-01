"""Strategy configuration files (TOML, stdlib ``tomllib``).

One file per strategy under ``cli_scanner/strategies/``. Loaded and validated
at bot startup; a bad file disables that strategy and alerts, but never stops
the bot. Consumed by the risk manager (limits/sizer), the scheduler (schedule,
execution_mode) and the lifecycle manager (lifecycle).
"""

from __future__ import annotations

import logging
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger("framework.core.config")

DEFAULT_CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "strategies"

_EXECUTION_MODES = {"approval", "auto"}
_LIFECYCLES = {"paper", "probation", "live"}


class ConfigError(Exception):
    """Raised when a strategy config file fails validation."""


@dataclass(frozen=True)
class StrategyConfig:
    name: str
    enabled: bool = True
    schedule: str = ""                 # cron, interpreted in America/New_York
    execution_mode: str = "approval"
    engine: str = "immediate"          # 'immediate' or 'limit_ladder'
    lifecycle: str = "paper"
    universe: dict = field(default_factory=dict)
    sizer: dict = field(default_factory=dict)   # {"name": ..., params...}
    limits: dict = field(default_factory=dict)  # overrides for RiskLimits fields
    exits: list[dict] = field(default_factory=list)
    path: str = ""

    def risk_limit_overrides(self) -> dict:
        """Only the limits keys that map onto RiskLimits fields."""
        from ..risk.manager import RiskLimits
        valid = set(RiskLimits.__dataclass_fields__)
        return {k: v for k, v in self.limits.items() if k in valid}


def _validate(raw: dict, path: Path) -> StrategyConfig:
    sec = raw.get("strategy")
    if not isinstance(sec, dict):
        raise ConfigError(f"{path.name}: missing [strategy] section")
    name = sec.get("name")
    if not name or not isinstance(name, str):
        raise ConfigError(f"{path.name}: strategy.name is required")
    mode = sec.get("execution_mode", "approval")
    if mode not in _EXECUTION_MODES:
        raise ConfigError(f"{path.name}: execution_mode must be one of {sorted(_EXECUTION_MODES)}")
    engine = sec.get("engine", "immediate")
    lifecycle = sec.get("lifecycle", "paper")
    if lifecycle not in _LIFECYCLES:
        raise ConfigError(f"{path.name}: lifecycle must be one of {sorted(_LIFECYCLES)}")

    risk = raw.get("risk", {})
    if not isinstance(risk, dict):
        raise ConfigError(f"{path.name}: [risk] must be a table")
    sizer = risk.get("sizer", {})
    limits = risk.get("limits", {})
    if sizer and "name" not in sizer:
        raise ConfigError(f"{path.name}: [risk.sizer] requires a name")
    for k in limits:
        if not isinstance(limits[k], (int, float)):
            raise ConfigError(f"{path.name}: risk.limits.{k} must be numeric")

    exits = raw.get("exits", [])
    if not isinstance(exits, list) or any(not isinstance(e, dict) for e in exits):
        raise ConfigError(f"{path.name}: [[exits]] must be an array of tables")

    return StrategyConfig(
        name=name,
        enabled=bool(sec.get("enabled", True)),
        schedule=str(sec.get("schedule", "")),
        execution_mode=mode,
        engine=engine,
        lifecycle=lifecycle,
        universe=dict(raw.get("universe", {})),
        sizer=dict(sizer),
        limits=dict(limits),
        exits=list(exits),
        path=str(path),
    )


def load_strategy_configs(config_dir: Optional[Path] = None) -> dict[str, StrategyConfig]:
    """Load all valid strategy configs; invalid files are logged and skipped."""
    directory = Path(config_dir or DEFAULT_CONFIG_DIR)
    configs: dict[str, StrategyConfig] = {}
    if not directory.is_dir():
        logger.info("Strategy config dir %s does not exist — no configs loaded", directory)
        return configs
    for path in sorted(directory.glob("*.toml")):
        try:
            raw = tomllib.loads(path.read_text())
            cfg = _validate(raw, path)
        except (ConfigError, tomllib.TOMLDecodeError) as exc:
            logger.error("Strategy config %s INVALID: %s — strategy disabled", path.name, exc)
            continue
        if cfg.name in configs:
            logger.error("Duplicate strategy name %s in %s — skipping", cfg.name, path.name)
            continue
        configs[cfg.name] = cfg
        logger.info("Loaded strategy config %s (%s, mode=%s, lifecycle=%s)",
                    cfg.name, path.name, cfg.execution_mode, cfg.lifecycle)
    return configs
