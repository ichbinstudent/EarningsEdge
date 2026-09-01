"""Tests for framework core: trading calendar, strategy config, universes."""

from __future__ import annotations

from datetime import date

import pytest

from framework.core.calendar import TradingCalendar
from framework.core.config import ConfigError, _validate, load_strategy_configs
from framework.core.universe import (
    EarningsCalendarUniverse, FileUniverse, StaticListUniverse, build_universe,
)


# ── Trading calendar (XNYS facts) ------------------------------------------

@pytest.fixture(scope="module")
def cal():
    return TradingCalendar()


def test_calendar_sessions_and_holidays(cal):
    assert cal.is_session(date(2026, 7, 6))          # Monday
    assert not cal.is_session(date(2026, 7, 4))      # Saturday
    assert not cal.is_session(date(2026, 7, 3))      # observed Independence Day
    assert not cal.is_session(date(2026, 11, 26))    # Thanksgiving


def test_calendar_next_prev_session(cal):
    assert cal.next_session(date(2026, 7, 3)) == date(2026, 7, 6)  # holiday → next Mon
    assert cal.next_session(date(2026, 7, 6)) == date(2026, 7, 6)  # session stays
    assert cal.next_session_after(date(2026, 7, 6)) == date(2026, 7, 7)
    assert cal.prev_session(date(2026, 7, 3)) == date(2026, 7, 2)


def test_calendar_add_sessions_skips_holidays(cal):
    # Jul 2 (Thu) + 2 sessions = Jul 7 (Tue): skips Jul-3 holiday and weekend
    assert cal.add_sessions(date(2026, 7, 2), 2) == date(2026, 7, 7)
    assert cal.add_sessions(date(2026, 7, 7), -2) == date(2026, 7, 2)


def test_calendar_early_close(cal):
    assert cal.is_early_close(date(2026, 11, 27))    # day after Thanksgiving
    assert not cal.is_early_close(date(2026, 7, 6))


# ── Strategy config ----------------------------------------------------------

def _raw(**over):
    base = {
        "strategy": {"name": "s1", "execution_mode": "approval", "lifecycle": "paper"},
        "risk": {"sizer": {"name": "pct_portfolio", "pct": 0.05},
                 "limits": {"max_pct_per_trade": 0.1}},
        "exits": [{"rule": "time", "days_after_entry": 3}],
    }
    base.update(over)
    return base


def test_validate_valid_config(tmp_path):
    cfg = _validate(_raw(), tmp_path / "s1.toml")
    assert cfg.name == "s1" and cfg.execution_mode == "approval"
    assert cfg.risk_limit_overrides() == {"max_pct_per_trade": 0.1}


def test_validate_rejects_bad_files(tmp_path):
    with pytest.raises(ConfigError):
        _validate({}, tmp_path / "x.toml")  # no [strategy]
    with pytest.raises(ConfigError):
        _validate(_raw(strategy={"execution_mode": "yolo"}), tmp_path / "x.toml")
    with pytest.raises(ConfigError):
        _validate(_raw(strategy={"name": "s", "lifecycle": "immortal"}), tmp_path / "x.toml")
    with pytest.raises(ConfigError):
        _validate(_raw(risk={"sizer": {"pct": 0.05}}), tmp_path / "x.toml")  # no sizer name


def test_load_configs_skips_invalid_and_duplicates(tmp_path):
    (tmp_path / "good.toml").write_text(
        '[strategy]\nname = "good"\n')
    (tmp_path / "bad.toml").write_text(
        '[strategy]\nname = "bad"\nexecution_mode = "yolo"\n')
    (tmp_path / "dup.toml").write_text(
        '[strategy]\nname = "good"\n')
    configs = load_strategy_configs(tmp_path)
    assert list(configs) == ["good"]


def test_shipped_example_configs_are_valid():
    configs = load_strategy_configs()
    # Every strategy the engines can emit has a config, named by its code name
    expected = {
        "calendar_call_ml", "short_straddle",
        "vol_risk_premium", "earnings_quality", "ff_ladder",
        "forward_factor_arb",
    }
    assert expected <= set(configs)
    for name in expected:
        cfg = configs[name]
        assert cfg.execution_mode in ("approval", "auto")
        assert cfg.sizer.get("name"), f"{name}: sizer missing"
        assert cfg.exits, f"{name}: exits missing"


# ── Universes ---------------------------------------------------------------

def test_static_universe():
    u = StaticListUniverse(["aapl", "MSFT"])
    assert u.symbols(date(2026, 7, 27)) == ["AAPL", "MSFT"]


def test_file_universe(tmp_path):
    f = tmp_path / "u.txt"
    f.write_text("AAPL\n# comment\n\nmsft\n")
    assert FileUniverse(f).symbols(date.today()) == ["AAPL", "MSFT"]
    assert FileUniverse(tmp_path / "nope.txt").symbols(date.today()) == []


def test_earnings_calendar_universe():
    class StubCandidate:
        def __init__(self, ticker):
            self.ticker = ticker

    class StubCollector:
        def fetch(self, on):
            return [StubCandidate("AAPL"), StubCandidate("MSFT"), StubCandidate("AAPL")]

    u = EarningsCalendarUniverse(collector=StubCollector())
    assert u.symbols(date(2026, 7, 27)) == ["AAPL", "MSFT"]


def test_build_universe(tmp_path):
    assert isinstance(build_universe({"type": "static", "symbols": ["AAPL"]}), StaticListUniverse)
    f = tmp_path / "u.txt"
    f.write_text("AAPL\n")
    assert isinstance(build_universe({"type": "file", "path": str(f)}), FileUniverse)
    assert isinstance(build_universe({"type": "earnings_calendar"}), EarningsCalendarUniverse)
    with pytest.raises(ValueError):
        build_universe({"type": "unknown"})
