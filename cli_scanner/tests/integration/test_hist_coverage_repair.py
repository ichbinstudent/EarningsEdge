"""Integration: hist-move repair drive -> SQLite write path.

ensure_hist_moves() runs for real against a temp DB; external providers
(Yahoo earnings dates, LSE/Polygon daily bars) are mocked. Covers:
1. backfill -> DB write path (rows land in snapshots tagged 'Backfill'),
2. coverage-gate behavior: the FF hist gate (hist_rms_move) flips from
   blocked to passing on repaired rows, and coverage.repair_candidates
   retires the ticker.
"""

from __future__ import annotations

import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

pytestmark = pytest.mark.integration

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

TODAY = date(2026, 8, 5)
EVENT_DATES = [date(2026, 4, 29), date(2026, 1, 28), date(2025, 10, 29)]


def _ms(d: date) -> int:
    return int(datetime(d.year, d.month, d.day).timestamp() * 1000)


def _fake_bars(ed: date, pre_close: float = 100.0, post_close: float = 112.0):
    """Three bars around the event: pre / earnings-day / post."""
    return [
        {"t": _ms(ed - timedelta(days=1)), "c": pre_close, "h": pre_close * 1.01,
         "l": pre_close * 0.99, "v": 1_000_000},
        {"t": _ms(ed), "c": post_close, "h": post_close * 1.05,
         "l": pre_close * 0.98, "v": 5_000_000},
        {"t": _ms(ed + timedelta(days=1)), "c": post_close * 1.01,
         "h": post_close * 1.02, "l": post_close * 0.99, "v": 2_000_000},
    ]


class _FakeLSE:
    def daily_bars(self, ticker, start, end):
        # infer the event date from the window (start = ed - 7d)
        ed = start + timedelta(days=7)
        return _fake_bars(ed)


class _FakeTicker:
    def __init__(self, ticker):
        self.ticker = ticker

    def get_earnings_dates(self, limit=12):
        idx = pd.DatetimeIndex([datetime(d.year, d.month, d.day) for d in EVENT_DATES])
        return pd.DataFrame({"EPS Estimate": [None] * len(idx)}, index=idx)


class _AncientTicker(_FakeTicker):
    """Yahoo sometimes serves ancient dates (ticker reuse); 2010 is beyond
    every provider's plan history and must be skipped without a bars call."""

    def get_earnings_dates(self, limit=12):
        dates = EVENT_DATES + [date(2010, 11, 5), date(2011, 3, 24)]
        idx = pd.DatetimeIndex([datetime(d.year, d.month, d.day) for d in dates])
        return pd.DataFrame({"EPS Estimate": [None] * len(idx)}, index=idx)


@pytest.fixture
def conn(tmp_db_path):
    import sqlite3

    c = sqlite3.connect(str(tmp_db_path), timeout=30)
    c.row_factory = sqlite3.Row
    yield c
    c.close()


@pytest.fixture
def mocked_providers(monkeypatch):
    import yfinance

    import earnings_edge.fwd_factor_ladder as ffl

    monkeypatch.setattr(yfinance, "Ticker", _FakeTicker)
    monkeypatch.setattr(ffl, "_lse_bars_client", lambda: _FakeLSE())
    monkeypatch.setattr(ffl, "_polygon_bars_client", lambda: None)
    ffl._hist_backfill_attempted.clear()
    return ffl


def _seed_outcome(conn, ticker, ed, move=8.0):
    conn.execute(
        """INSERT INTO snapshots
            (ticker, earnings_date, scan_date, timing, has_options,
             pre_earnings_close, post_earnings_close,
             actual_move_pct, actual_move_direction,
             max_intraday_range_pct, outcome_fetched_at)
           VALUES (?,?,?,?,0,100,108,?,?,1,'2026-08-01T00:00:00')""",
        (ticker, ed.isoformat(), ed.isoformat(), "Post Market", move, "UP"),
    )
    conn.commit()


def test_ensure_hist_moves_writes_backfill_rows(conn, mocked_providers):
    """Backfill path: 3 Yahoo event dates x LSE bars -> 3 new snapshot rows
    with computed outcome fields, tagged timing='Backfill'."""
    from earnings_edge.fwd_factor_ladder import ensure_hist_moves

    total = ensure_hist_moves("REPR", today=TODAY)
    assert total == 3

    rows = conn.execute(
        "SELECT earnings_date, timing, actual_move_pct, actual_move_direction,"
        " outcome_fetched_at FROM snapshots WHERE ticker='REPR' ORDER BY earnings_date"
    ).fetchall()
    assert len(rows) == 3
    for r in rows:
        assert r[1] == "Backfill"
        assert r[2] is not None and r[2] > 0  # actual_move_pct
        assert r[3] in ("UP", "DOWN")
        assert r[4] is not None  # outcome_fetched_at -> usable for the gate
    # fake bars move 100 -> 112 on the earnings day = +12%
    assert abs(rows[0][2] - 12.0) < 0.5


def test_hist_gate_flips_after_repair(conn, mocked_providers):
    """Coverage-gate behavior: ticker blocked (1 usable event < 3) passes
    hist_rms_move after repair, coverage goes 0% -> 100%, and the ticker
    leaves repair_candidates."""
    from earnings_edge.coverage import hist_move_coverage, repair_candidates
    from earnings_edge.fwd_factor_ladder import ensure_hist_moves, hist_rms_move

    _seed_outcome(conn, "GATECO", date(2026, 7, 29))

    rms, n = hist_rms_move(ticker="GATECO")
    assert rms is None and n == 1
    before = hist_move_coverage(universe=["GATECO"])
    assert before["covered"] == 0
    assert "GATECO" in repair_candidates()

    total = ensure_hist_moves("GATECO", today=TODAY)
    assert total >= 3

    rms, n = hist_rms_move(ticker="GATECO")
    assert rms is not None and rms > 0 and n >= 3
    after = hist_move_coverage(universe=["GATECO"])
    assert after["covered"] == 1 and after["pct"] == 100.0
    assert "GATECO" not in repair_candidates()


def test_ancient_event_dates_skipped_without_provider_call(conn, monkeypatch):
    """Event dates older than HIST_BACKFILL_MAX_AGE_DAYS are filtered before
    any bars request — beyond-plan dates 403 on Polygon and cost 45s+ of
    retries each."""
    import yfinance

    import earnings_edge.fwd_factor_ladder as ffl

    requested_windows: list = []

    class _RecordingLSE:
        def daily_bars(self, ticker, start, end):
            requested_windows.append((start, end))
            return _fake_bars(start + timedelta(days=7))

    monkeypatch.setattr(yfinance, "Ticker", _AncientTicker)
    monkeypatch.setattr(ffl, "_lse_bars_client", lambda: _RecordingLSE())
    monkeypatch.setattr(ffl, "_polygon_bars_client", lambda: None)
    ffl._hist_backfill_attempted.clear()

    total = ffl.ensure_hist_moves("OLDCO", today=TODAY)
    assert total == 3  # only the three in-plan events were backfilled
    cutoff = TODAY - timedelta(days=ffl.HIST_BACKFILL_MAX_AGE_DAYS)
    assert requested_windows, "expected bars requests for in-plan events"
    for start, _end in requested_windows:
        assert start >= cutoff - timedelta(days=7)
