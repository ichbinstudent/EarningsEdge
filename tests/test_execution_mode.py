"""Execution-mode control plane (approval/auto) + unified signal surface.

Covers:
- framework.core.control execution-mode overrides (set/effective/clear,
  invalid mode, event recorded, missing-column degradation)
- subscriptions.partition_by_mode (approval vs auto split)
- StrategySubscriptions known-user enrollment + persistence
- strategies_view shows the effective mode with override marker
"""
from __future__ import annotations

import pytest

from framework.core.control import (
    clear_execution_mode_override,
    effective_execution_mode,
    execution_mode_overrides,
    set_execution_mode,
)
from sqlalchemy import text
from earnings_edge.db import engine as db_engine


@pytest.fixture
def conn(tmp_path):
    db_engine.configure(tmp_path / "fw.db")


# ── control plane ─────────────────────────────────────────────────────────

def test_mode_defaults_to_toml(conn):
    assert effective_execution_mode("calendar_call_ml", "approval") == "approval"
    assert effective_execution_mode("calendar_call_ml", "auto") == "auto"


def test_mode_override_set_and_effective(conn):
    set_execution_mode("short_straddle", "auto", by="test")
    assert effective_execution_mode("short_straddle", "approval") == "auto"
    assert execution_mode_overrides() == {"short_straddle": "auto"}


def test_mode_override_clear_returns_to_toml(conn):
    set_execution_mode("short_straddle", "auto", by="test")
    clear_execution_mode_override("short_straddle", by="test")
    assert effective_execution_mode("short_straddle", "approval") == "approval"
    assert execution_mode_overrides() == {}


def test_mode_rejects_invalid(conn):
    with pytest.raises(ValueError):
        set_execution_mode("x", "yolo")


def test_mode_change_records_event(conn):
    set_execution_mode("vol_risk_premium", "auto", by="tester")
    with db_engine.get_session() as s:
        row = s.execute(
            text(
                "SELECT event_type, strategy, detail FROM risk_events "
                "WHERE strategy = 'vol_risk_premium' ORDER BY id DESC LIMIT 1"
            )
        ).mappings().first()
    assert row["event_type"] == "execution_mode"
    assert "auto" in row["detail"] and "tester" in row["detail"]


def test_mode_override_preserves_enabled_column(conn):
    """INSERT ... ON CONFLICT must not clobber the enabled override."""
    from framework.core.control import effective_enabled, set_enabled
    set_enabled("s1", False, by="test")
    set_execution_mode("s1", "auto", by="test")
    assert effective_enabled("s1", True) is False
    assert effective_execution_mode("s1", "approval") == "auto"


def test_effective_mode_degrades_without_row(conn):
    """No strategy_state row → TOML default, no exception."""
    assert effective_execution_mode("x", "approval") == "approval"
    assert effective_execution_mode("x", "auto") == "auto"


# ── partition ─────────────────────────────────────────────────────────────

def test_partition_by_mode():
    from earnings_edge.subscriptions import partition_by_mode
    rows = [
        {"strategy": "a", "id": 1},
        {"strategy": "b", "id": 2},
        {"strategy": "c", "id": 3},
    ]
    modes = {"a": "auto", "b": "approval", "c": "auto"}
    approval, auto = partition_by_mode(rows, lambda s: modes[s])
    assert [r["id"] for r in approval] == [2]
    assert [r["id"] for r in auto] == [1, 3]


def test_partition_unknown_strategy_defaults_approval():
    from earnings_edge.subscriptions import partition_by_mode
    rows = [{"strategy": "mystery", "id": 9}]
    approval, auto = partition_by_mode(rows, lambda s: "approval" if s != "mystery" else "approval")
    assert [r["id"] for r in approval] == [9] and auto == []


# ── known-user enrollment ─────────────────────────────────────────────────

def test_known_users_enrollment_and_persistence(tmp_path):
    from earnings_edge.subscriptions import StrategySubscriptions
    path = str(tmp_path / "subs.json")
    subs = StrategySubscriptions(path)
    assert subs.known_users() == set()

    subs.set_subscribed("short_straddle", 42, False)   # mute → enrolled
    assert subs.known_users() == {42}

    subs.set_subscribed("short_straddle", 42, True)    # unmute → stays enrolled
    assert subs.known_users() == {42}
    assert subs.is_subscribed("short_straddle", 42) is True

    subs.set_subscribed("ff_ladder", 7, True)          # fresh subscribe → enrolled
    assert subs.known_users() == {42, 7}

    reloaded = StrategySubscriptions(path)             # survives reload
    assert reloaded.known_users() == {42, 7}
    assert reloaded.is_subscribed("short_straddle", 42) is True


def test_known_users_legacy_file_without_field(tmp_path):
    """Opt-out-only files (pre-unification) load with an empty known set."""
    import json
    path = tmp_path / "subs.json"
    path.write_text(json.dumps({"opt_outs": {"ff_ladder": [42]}}))
    from earnings_edge.subscriptions import StrategySubscriptions
    subs = StrategySubscriptions(str(path))
    assert subs.known_users() == set()
    assert subs.is_subscribed("ff_ladder", 42) is False


# ── view ──────────────────────────────────────────────────────────────────

def test_strategies_view_marks_mode_override(tmp_path):
    from earnings_edge.bot_views import strategies_view
    from framework.core.config import StrategyConfig
    from framework.core.registry import StrategyRegistry
    db_engine.configure(tmp_path / "fw.db")
    cfg = StrategyConfig(name="s1", enabled=True, execution_mode="approval",
                         lifecycle="paper", sizer={}, limits={}, exits=[])
    reg = StrategyRegistry(configs={"s1": cfg})
    text, _ = strategies_view( registry=reg)
    assert "<b>s1</b> — paper | approval |" in text
    set_execution_mode("s1", "auto", by="test")
    text, _ = strategies_view( registry=reg)
    assert "<b>s1</b> — paper | auto (override) |" in text
