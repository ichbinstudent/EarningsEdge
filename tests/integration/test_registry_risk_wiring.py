"""Integration: TOML strategy configs -> StrategyRegistry -> RiskManager gate.

Uses the real strategies/*.toml files and a temp framework DB. Verifies the
wiring contract: per-strategy limits/sizer resolution, lifecycle seeding, and
that the risk chokepoint actually enforces what the TOML declares.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration

EXPECTED_STRATEGIES = {
    "calendar_call_ml",
    "earnings_quality",
    "short_straddle",
    "vol_risk_premium",
    "ff_ladder",
    "forward_factor_arb",
}


def test_registry_loads_real_toml_configs():
    from framework.core.registry import StrategyRegistry

    reg = StrategyRegistry()  # loads the real strategies/ directory

    assert EXPECTED_STRATEGIES <= set(reg.configs)

    # calendar_call_ml.toml: max_pct_per_trade = 0.10, pct_portfolio sizer 5%
    limits = reg.limits_for("calendar_call_ml")
    assert limits.max_pct_per_trade == 0.10
    assert limits.max_pct_per_underlying == 0.25
    assert limits.max_pct_per_strategy_day == 0.30

    sizer = reg.sizer_spec("calendar_call_ml")
    assert sizer is not None
    assert sizer["name"] == "pct_portfolio"
    assert sizer["pct"] == 0.05

    assert reg.is_enabled("calendar_call_ml") is True
    assert reg.execution_mode("calendar_call_ml") == "approval"
    cfg = reg.get("calendar_call_ml")
    assert cfg is not None and cfg.lifecycle == "paper"
    assert reg.is_enabled("short_straddle") is False
    assert reg.is_enabled("vol_risk_premium") is False
    assert reg.is_enabled("earnings_quality") is False

    # Unknown strategy: base limits, enabled by default (legacy contract)
    base = reg.limits_for("nonexistent_strategy")
    assert base.max_pct_per_trade == limits.max_pct_per_trade  # same default
    assert reg.is_enabled("nonexistent_strategy") is True


def test_registry_limits_drive_risk_gate(tmp_path):
    from framework.core.registry import StrategyRegistry
    from framework.risk.manager import RiskManager
    from sqlalchemy import text
    from earnings_edge.db import configure, engine as db_engine

    configure(tmp_path / "framework_test.db")
    reg = StrategyRegistry()

    # Lifecycle seeding: INSERT OR IGNORE, never overwrites operator state
    seeded = reg.sync_lifecycle()
    assert seeded == len(reg.configs)
    assert reg.sync_lifecycle() == 0  # idempotent
    with db_engine.get_session() as s:
        row = s.execute(
            text("SELECT lifecycle FROM strategy_state WHERE name = 'calendar_call_ml'")
        ).mappings().first()
    assert row["lifecycle"] == "paper"

    limits = reg.limits_for("calendar_call_ml")
    rm = RiskManager(limits=limits)

    # Small trade: approved
    d = rm.check_trade(
        "calendar_call_ml", "AAPL",
        est_cost=1_000.0, equity=100_000.0, buying_power=50_000.0,
    )
    assert d.approved, d.reason
    assert d.qty_multiplier == 1.0

    # Over the TOML per-trade cap (10% of buying power): vetoed
    d = rm.check_trade(
        "calendar_call_ml", "AAPL",
        est_cost=6_000.0, equity=100_000.0, buying_power=50_000.0,
    )
    assert not d.approved
    assert "buying power" in d.reason

    # Probation lifecycle halves size via the multiplier
    d = rm.check_trade(
        "calendar_call_ml", "AAPL",
        est_cost=1_000.0, equity=100_000.0, buying_power=50_000.0,
        lifecycle="probation",
    )
    assert d.approved
    assert d.qty_multiplier == limits.probation_size_mult == 0.5

    # Kill switch: every new entry vetoed while halted
    rm.killswitch.trip("integration test halt", by="pytest")
    d = rm.check_trade(
        "calendar_call_ml", "AAPL",
        est_cost=100.0, equity=100_000.0, buying_power=50_000.0,
    )
    assert not d.approved
    assert "kill switch" in d.reason
    rm.killswitch.resume(by="pytest")
    assert rm.killswitch.is_halted() is False
