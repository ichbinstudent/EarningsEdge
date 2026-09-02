"""Tests for bot_live text helpers and the monitor/status view enrichments."""
from __future__ import annotations

import pytest

from earnings_edge.bot_live import (
    SPINNER,
    fmt_duration,
    progress_text,
    sparkline,
    spinner_frame,
)


# ── text helpers ──────────────────────────────────────────────────────────

def test_spinner_cycles_through_frames():
    frames = [spinner_frame(i) for i in range(len(SPINNER))]
    assert len(set(frames)) == len(SPINNER)          # all frames distinct
    assert spinner_frame(len(SPINNER)) == spinner_frame(0)  # wraps


def test_fmt_duration():
    assert fmt_duration(7) == "7s"
    assert fmt_duration(75) == "1m 15s"
    assert fmt_duration(0) == "0s"
    assert fmt_duration(3723) == "1h 02m"


def test_sparkline_monotonic_series_ascends():
    s = sparkline([1, 2, 3, 4, 5, 6, 7, 8])
    assert s[0] == "▁" and s[-1] == "█"
    # non-decreasing for a monotonic input
    assert list(s) == sorted(s)


def test_sparkline_edge_cases():
    assert sparkline([]) == ""
    assert sparkline([5.0]) == ""
    flat = sparkline([3.0, 3.0, 3.0])
    assert len(set(flat)) == 1                       # flat series, no crash
    # downsamples long series to width
    assert len(sparkline(list(range(100)), width=16)) <= 16


def test_progress_text_contains_stage_and_elapsed():
    txt = progress_text("Scanning X", "fetching data", started=100.0,
                        tick=2, now=190.0)
    assert "Scanning X" in txt
    assert "fetching data" in txt
    assert "1m 30s" in txt
    assert spinner_frame(2) in txt


# ── views ─────────────────────────────────────────────────────────────────

@pytest.fixture
def conn(tmp_path):
    from earnings_edge.db import engine as db_engine
    db_engine.configure(tmp_path / "fw.db")


def _seed_equity(conn):
    from sqlalchemy import text
    from earnings_edge.db import engine as db_engine
    with db_engine.session_scope() as s:
        for i, eq in enumerate((100_000, 100_400, 100_250, 101_100)):
            s.execute(
                text(
                    "INSERT INTO equity_snapshots (ts, equity, buying_power, portfolio_value, source) "
                    "VALUES (:ts, :eq, :bp, :pv, 'test')"
                ),
                {"ts": f"2026-08-10T1{i}:00:00+00:00", "eq": eq, "bp": eq, "pv": eq},
            )


def test_status_view_with_extras(conn):
    from earnings_edge.bot_views import status_view
    _seed_equity(conn)
    text = status_view( market_open=True, pending_proposals=2, pending_exits=1,
                       next_events=["FF ladder step 14:15 ET", "Run Earnings Calendar 15:15 ET"],
                       funnel="funnel: 10 scanned → 4 decision → 2 proposals")
    assert "🟢 open" in text
    assert "Pending proposals: 2</b>" in text
    assert "<b>Next:</b> FF ladder step 14:15 ET" in text
    assert "funnel:" in text
    assert "▁" in text or "▄" in text               # sparkline rendered


def test_status_view_backward_compatible(conn):
    from earnings_edge.bot_views import status_view
    text = status_view()                        # no new kwargs
    assert "SYSTEM STATUS" in text
    assert "Next:" not in text


def test_monitor_view_renders_all_sections(conn):
    from earnings_edge.bot_views import monitor_view
    _seed_equity(conn)
    text = monitor_view( tick=3, pending_proposals=1, pending_exits=0,
                        next_events=["Exit rule evaluation 15:30 ET"])
    assert spinner_frame(3) in text
    assert "LIVE MONITOR" in text
    assert "<b>Kill switch:</b> 🟢 armed" in text
    assert "<b>Equity</b> $101,100" in text
    assert "FF ladders armed:</b> 0" in text
    assert "1 proposals" in text
    assert "<b>Next:</b> Exit rule evaluation" in text


def test_monitor_view_tolerates_empty_framework_db(tmp_path):
    """Fresh framework schema (no equity rows, no ff_ladders table) must not raise."""
    from earnings_edge.bot_views import monitor_view
    from earnings_edge.db import engine as db_engine
    db_engine.configure(tmp_path / "fresh.db")
    text = monitor_view(tick=0)
    assert "LIVE MONITOR" in text
    assert "FF ladders armed:</b> 0" in text
