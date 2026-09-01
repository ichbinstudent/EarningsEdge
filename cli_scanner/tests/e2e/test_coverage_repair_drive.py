"""E2E: coverage repair drive on a fixture universe into a temp DB.

Full pipeline for real: coverage.repair_candidates -> warm_hist_coverage
.repair_universe -> fwd_factor_ladder.ensure_hist_moves -> snapshots writes.
Only the outermost providers (Yahoo earnings dates, LSE bars) are mocked.
Asserts the hist-gate coverage delta over the fixture universe is > 0.
"""

from __future__ import annotations

import sqlite3
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

pytestmark = pytest.mark.e2e

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

TODAY = date(2026, 8, 5)
EVENT_DATES = [date(2026, 4, 29), date(2026, 1, 28), date(2025, 10, 29),
               date(2025, 7, 30)]


def _ms(d: date) -> int:
    return int(datetime(d.year, d.month, d.day).timestamp() * 1000)


class _FakeLSE:
    def daily_bars(self, ticker, start, end):
        ed = start + timedelta(days=7)
        return [
            {"t": _ms(ed - timedelta(days=1)), "c": 100.0, "h": 101.0, "l": 99.0},
            {"t": _ms(ed), "c": 110.0, "h": 115.0, "l": 98.0},
            {"t": _ms(ed + timedelta(days=1)), "c": 111.0, "h": 112.0, "l": 109.0},
        ]


class _FakeTicker:
    def __init__(self, ticker):
        self.ticker = ticker

    def get_earnings_dates(self, limit=12):
        idx = pd.DatetimeIndex([datetime(d.year, d.month, d.day) for d in EVENT_DATES])
        return pd.DataFrame({"EPS Estimate": [None] * len(idx)}, index=idx)


def _seed(conn, ticker, ed, has_options, usable_outcome):
    if isinstance(ed, str):
        ed = date.fromisoformat(ed)
    conn.execute(
        """INSERT INTO snapshots
            (ticker, earnings_date, scan_date, timing, has_options,
             actual_move_pct, actual_move_direction, outcome_fetched_at)
           VALUES (?,?,?,?,?,?,?,?)""",
        (ticker, ed.isoformat(), ed.isoformat(), "Post Market", has_options,
         6.0 if usable_outcome else None,
         "UP" if usable_outcome else None,
         "2026-08-01T00:00:00" if usable_outcome else None),
    )
    conn.commit()


def test_repair_drive_raises_hist_coverage(tmp_db_path, monkeypatch):
    import yfinance

    import earnings_edge.fwd_factor_ladder as ffl
    import warm_hist_coverage as whc
    from earnings_edge.coverage import hist_move_coverage, repair_candidates
    import sqlite3

    monkeypatch.setattr(yfinance, "Ticker", _FakeTicker)
    monkeypatch.setattr(ffl, "_lse_bars_client", lambda: _FakeLSE())
    monkeypatch.setattr(ffl, "_polygon_bars_client", lambda: None)
    ffl._hist_backfill_attempted.clear()

    conn = sqlite3.connect(str(tmp_db_path), timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        # Fixture universe:
        #  LIQ1: liquid, 2 usable outcomes (needs 1 more)
        #  LIQ2: liquid, 0 usable outcomes (needs 3)
        #  OTC1: non-liquid, 0 outcomes (skipped by liquid_only)
        _seed(conn, "LIQ1", "2026-07-29", has_options=1, usable_outcome=True)
        _seed(conn, "LIQ1", "2026-04-29", has_options=1, usable_outcome=True)
        _seed(conn, "LIQ2", "2026-07-30", has_options=1, usable_outcome=False)
        _seed(conn, "OTC1", "2026-07-30", has_options=0, usable_outcome=False)

        before = hist_move_coverage(conn)
        assert before["universe"] == 3 and before["covered"] == 0

        candidates = repair_candidates(conn, liquid_only=True)
        assert set(candidates) == {"LIQ1", "LIQ2"}  # OTC tail excluded
        assert candidates[0] == "LIQ1"  # most existing outcomes first

        result = whc.repair_universe(candidates, sleep=0)

        assert result["failed"] == 0
        delta = result["coverage_after"]["covered"] - result["coverage_before"]["covered"]
        assert delta > 0
        after = hist_move_coverage(conn)
        # both liquid names now pass the gate; OTC1 was never touched
        assert after["covered"] == 2 and after["universe"] == 3
        assert "LIQ1" not in repair_candidates(conn, liquid_only=True)
        assert "LIQ2" not in repair_candidates(conn, liquid_only=True)
        # repaired rows are usable for the FF ladder gate
        rms, n = ffl.hist_rms_move(ticker="LIQ2")
        assert rms is not None and n >= 3
    finally:
        conn.close()
