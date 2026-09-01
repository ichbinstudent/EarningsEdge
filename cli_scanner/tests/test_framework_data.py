"""Tests for framework data layer: PIT catalog, model registry, job runs."""

from __future__ import annotations

import pytest

from framework.data.catalog import available_as_of, latest_availability, record
from framework.data.model_registry import get_active, promote_model, register_model, sha256_of
from framework.jobs import recent_runs, run_job
from sqlalchemy import text
from earnings_edge.db import engine as db_engine


@pytest.fixture
def conn(tmp_path):
    db_engine.configure(tmp_path / "fw.db")


# ── Data catalog -------------------------------------------------------------

def test_catalog_pit_query_respects_decision_time(conn):
    record("options_chain", "AAPL", "2026-07-20", source="alpaca",
           available_at="2026-07-20T20:00:00+00:00")
    record("options_chain", "AAPL", "2026-07-21", source="alpaca",
           available_at="2026-07-21T20:00:00+00:00")
    # Decision made midday Jul 21: only the Jul-20 dataset was available yet
    assert available_as_of("options_chain", "2026-07-21T13:00:00+00:00",
                           symbol="AAPL") == ["2026-07-20"]
    assert available_as_of("options_chain", "2026-07-22T13:00:00+00:00",
                           symbol="AAPL") == ["2026-07-20", "2026-07-21"]


def test_catalog_pit_only_excludes_unsafe_sources(conn):
    record("chain_snapshot", "AAPL", "2026-07-20", source="lse",
           available_at="2026-07-20T21:00:00+00:00", pit_safe=False)
    record("chain_snapshot", "AAPL", "2026-07-20", source="polygon",
           available_at="2026-07-20T21:00:00+00:00", pit_safe=True)
    assert available_as_of("chain_snapshot", "2026-07-21T00:00:00+00:00",
                           symbol="AAPL", pit_only=True) == ["2026-07-20"]
    rows = available_as_of("chain_snapshot", "2026-07-21T00:00:00+00:00",
                           symbol="AAPL", pit_only=False)
    assert rows == ["2026-07-20"]  # distinct as_of dates


def test_catalog_range_and_freshness(conn):
    for d in ("2026-07-20", "2026-07-21", "2026-07-22"):
        record("daily_bars", "MSFT", d, source="lse",
               available_at=d + "T21:00:00+00:00")
    assert available_as_of("daily_bars", "2026-07-23T00:00:00+00:00",
                           symbol="MSFT", as_of_start="2026-07-21",
                           as_of_end="2026-07-21") == ["2026-07-21"]
    latest = latest_availability("daily_bars", "MSFT")
    assert latest["as_of_date"] == "2026-07-22" and latest["source"] == "lse"


# ── Model registry -----------------------------------------------------------

def test_model_registry_register_promote_active(conn, tmp_path):
    m1 = tmp_path / "m1.joblib"
    m1.write_bytes(b"model-v1")
    m2 = tmp_path / "m2.joblib"
    m2.write_bytes(b"model-v2")

    register_model("calendar_filter", m1)
    register_model("calendar_filter", m2, promote=True)
    register_model("calendar_filter", m1)  # idempotent

    active = get_active("calendar_filter")
    assert active["sha256"] == sha256_of(m2)
    with db_engine.get_session() as s:
        rows = s.execute(text("SELECT * FROM model_registry")).mappings().all()
    assert len(rows) == 2

    promote_model("calendar_filter", sha256_of(m1))
    assert get_active("calendar_filter")["sha256"] == sha256_of(m1)


def test_model_registry_empty(conn):
    assert get_active("nothing") is None


# ── Job runs -----------------------------------------------------------------

def test_run_job_records_success(conn):
    result = run_job("test_job", lambda: 42, stats={"n": 1})
    assert result == 42
    runs = recent_runs("test_job")
    assert len(runs) == 1
    assert runs[0]["success"] == 1 and '"n": 1' in runs[0]["stats_json"]
    assert runs[0]["finished_at"]


def test_run_job_records_failure_and_reraises(conn):
    def boom():
        raise RuntimeError("explode")

    with pytest.raises(RuntimeError):
        run_job("bad_job", boom)
    runs = recent_runs("bad_job")
    assert runs[0]["success"] == 0 and "explode" in runs[0]["error"]
