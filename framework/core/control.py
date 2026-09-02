"""Runtime strategy enable/disable: operator overrides in ``strategy_state``.

The TOML ``[strategy] enabled`` flag is the static default; the operator can
pause/resume a strategy from Telegram without editing files or restarting.
The override lives in ``strategy_state.enabled`` (NULL = follow TOML) and is
consulted everywhere proposals are built. Exit management is NOT gated:
closing risk is always allowed, even for a paused strategy.
"""

from __future__ import annotations

import logging
from typing import Optional

from sqlalchemy.exc import SQLAlchemyError

from earnings_edge.db import (
    strategy_state_clear_enabled,
    strategy_state_clear_execution_mode,
    strategy_state_enabled_overrides,
    strategy_state_execution_mode_overrides,
    strategy_state_get,
    strategy_state_set_enabled,
    strategy_state_set_execution_mode,
)

from ..risk.killswitch import record_event

logger = logging.getLogger("framework.core.control")


def set_enabled(name: str, enabled: bool, by: str = "operator") -> None:
    """Persist an operator enable/disable override for a strategy."""
    strategy_state_set_enabled(name, enabled, updated_by=by)
    record_event("resume" if enabled else "pause",
                 f"strategy {'enabled' if enabled else 'disabled'} (by {by})",
                 strategy=name)
    logger.warning("strategy %s %s (by %s)", name,
                   "enabled" if enabled else "disabled", by)


def clear_override(name: str, by: str = "operator") -> None:
    """Remove the operator override → the TOML enabled flag decides again."""
    strategy_state_clear_enabled(name, updated_by=by)
    record_event("resume", f"enable override cleared (by {by})", strategy=name)


def enabled_overrides() -> dict[str, bool]:
    """All operator overrides currently in force."""
    return strategy_state_enabled_overrides()


def effective_enabled(name: str, toml_default: bool = True) -> bool:
    """Override if present, else the TOML default."""
    try:
        row = strategy_state_get(name)
    except SQLAlchemyError:
        return toml_default
    if row is None or row.get("enabled") is None:
        return toml_default
    return bool(row["enabled"])


def filter_enabled(names: list[str], toml_enabled) -> list[str]:
    """Apply DB overrides on top of the registry's TOML enabled filter.

    ``toml_enabled`` is a callable name -> bool (StrategyRegistry.is_enabled).
    A missing/unmigrated DB degrades to the TOML filter only.
    """
    base = [n for n in names if toml_enabled(n)]
    try:
        overrides = enabled_overrides()
    except SQLAlchemyError:
        return base
    return [n for n in base if overrides.get(n, True)]


# ---------------------------------------------------------------------------
# Execution mode (approval vs auto) — same override pattern as enabled.
# ---------------------------------------------------------------------------

_EXECUTION_MODES = {"approval", "auto"}


def set_execution_mode(name: str, mode: str, by: str = "operator") -> None:
    """Persist an operator execution-mode override ('approval' | 'auto').

    'approval' = proposals are pushed as cards, a human clicks Execute.
    'auto' = the bot executes proposals immediately (still risk-gated:
    RiskManager.check_trade + kill switch apply) and pushes a notification.
    """
    if mode not in _EXECUTION_MODES:
        raise ValueError(f"execution mode must be one of {sorted(_EXECUTION_MODES)}")
    strategy_state_set_execution_mode(name, mode, updated_by=by)
    record_event("execution_mode",
                 f"execution mode → {mode} (by {by})", strategy=name)
    logger.warning("strategy %s execution mode → %s (by %s)", name, mode, by)


def clear_execution_mode_override(name: str, by: str = "operator") -> None:
    """Remove the operator override → the TOML execution_mode decides again."""
    strategy_state_clear_execution_mode(name, updated_by=by)
    record_event("execution_mode",
                 f"execution-mode override cleared (by {by})", strategy=name)


def execution_mode_overrides() -> dict[str, str]:
    """All operator execution-mode overrides currently in force."""
    return strategy_state_execution_mode_overrides()


def effective_execution_mode(name: str, toml_default: str = "approval") -> str:
    """Override if present, else the TOML default. Missing DB/column → TOML."""
    try:
        row = strategy_state_get(name)
    except SQLAlchemyError:
        return toml_default
    if row is None or row.get("execution_mode") is None:
        return toml_default
    return row["execution_mode"]
