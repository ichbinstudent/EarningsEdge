"""Integration: approved-proposal -> realized-outcome join and aggregation.

Covers scripts/proposal_quality.py against a temp DB with fixture rows:
- ff_ladders (approved proposals, one filled + one expired-unfilled),
- managed_positions (recorded fill debit),
- snapshots (expected_move_pct + realized actual_move_pct outcomes).

No network: build_report() without --fetch-exit-prices is fully offline.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import proposal_quality as pq  # noqa: E402


@pytest.fixture
def pq_db(tmp_path):
    """Temp DB with earnings + framework + approval/ladder schema applied."""
    from earnings_edge.fwd_factor_ladder import DDL as FF_DDL
    from earnings_edge.trade_approval import _SCHEMA as PENDING_DDL
    from earnings_edge.db import configure

    path = tmp_path / "pq.db"
    configure(path)
    conn = sqlite3.connect(str(path), timeout=30)
    conn.executescript(FF_DDL)
    conn.executescript(PENDING_DDL)
    conn.commit()
    conn.close()
    return path


def _candidate(ticker: str, earnings_date: str, **over) -> str:
    cand = {
        "ticker": ticker,
        "earnings_date": earnings_date,
        "spot": 100.0,
        "strike": 100.0,
        "near_symbol": f"{ticker}260828C00100000",
        "far_symbol": f"{ticker}260918C00100000",
        "near_expiry": "2026-08-28",
        "far_expiry": "2026-09-18",
        "near_bid": 4.0,
        "near_ask": 4.4,
        "far_bid": 5.6,
        "far_ask": 6.0,
        "sigma_fwd": 0.30,
        "hist_rms_move": 0.02,
        "tau_days": 2,
        "d_start": 1.5,
        "d_cap": 1.6,
        "mid_debit": 1.4,
    }
    cand.update(over)
    return json.dumps(cand)


def _add_ladder(db_path, ticker, earnings_date, status, order_id=None, **cand_over):
    import sqlite3

    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "INSERT INTO ff_ladders (ticker, candidate_json, order_id, status, rung, "
        "armed_by, created_at, updated_at) VALUES (?,?,?,?,0,123,'2026-07-28 18:46:55',"
        "'2026-07-29 18:00:00')",
        (ticker, _candidate(ticker, earnings_date, **cand_over), order_id, status),
    )
    conn.commit()
    conn.close()


def _add_fill(db_path, group_id, near_symbol, far_symbol, entry_debit, qty=1.0):
    import sqlite3

    conn = sqlite3.connect(str(db_path))
    for sym in (near_symbol, far_symbol):
        conn.execute(
            "INSERT INTO managed_positions (symbol, strategy, group_id, qty, entry_price, "
            "status, order_id, opened_at) VALUES (?,?,?,?,?,?,?, '2026-07-29T18:00:00')",
            (sym, "ff_ladder", group_id, qty, entry_debit, "open", group_id),
        )
    conn.commit()
    conn.close()


def _add_outcome(db_path, ticker, earnings_date, expected=None, actual=None,
                 direction=None):
    import sqlite3

    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "INSERT INTO snapshots (ticker, earnings_date, scan_date, timing, "
        "expected_move_pct, actual_move_pct, actual_move_direction, data_source) "
        "VALUES (?,?,?,?,?,?,?, 'integration_fixture')",
        (ticker, earnings_date, earnings_date, "Post Market",
         expected, actual, direction),
    )
    conn.commit()
    conn.close()


def test_join_produces_correct_aggregates(pq_db):
    """Filled + unfilled approved proposals join to outcomes; hit rate and
    entry debit are computed exactly."""
    _add_ladder(pq_db, "META", "2026-07-29", "filled", order_id="ord-meta")
    _add_fill(pq_db, "ord-meta", "META260828C00100000", "META260918C00100000", 10.20)
    _add_outcome(pq_db, "META", "2026-07-29", expected=7.92, actual=-1.3144,
                 direction="DOWN")

    # approved (armed) but never filled -> still approved, not executed
    _add_ladder(pq_db, "BA", "2026-07-28", "expired")
    _add_outcome(pq_db, "BA", "2026-07-28", expected=5.36, actual=4.7565,
                 direction="UP")

    report = pq.build_report(pq_db)

    s = report["summary"]
    assert s["approved"] == 2
    assert s["executed"] == 1
    assert s["with_outcome"] == 2
    assert s["scorable"] == 2
    # both realized moves are smaller than the expected move -> both hits
    assert s["hits"] == 2
    assert s["hit_rate"] == 1.0
    assert s["significant"] is False  # n << 20
    assert s["total_pnl"] is None  # no --fetch-exit-prices

    trades = {t["ticker"]: t for t in report["trades"]}
    meta = trades["META"]
    assert meta["executed"] is True
    assert meta["entry_debit"] == 10.20
    assert meta["hit"] is True
    assert meta["expected_move_pct"] == 7.92
    assert meta["actual_move_pct"] == -1.3144
    assert meta["implied_event_move_pct"] is not None  # derived from quotes
    assert meta["hist_rms_move_pct"] == 2.0

    ba = trades["BA"]
    assert ba["executed"] is False
    assert ba["entry_debit"] is None
    assert ba["pnl"] is None
    assert ba["hit"] is True

    assert report["by_source"]["ff_ladder"]["approved"] == 2
    assert report["by_strategy"]["ff_ladder"]["executed"] == 1


def test_timing_aware_event_move_pure_math():
    """compute_event_move: AMC moves use the post-announcement session; BMO
    and unknown timing keep the stored on/after convention."""
    from datetime import date, datetime, timedelta

    ed = date(2026, 7, 29)
    bars = []
    for d, close in [(date(2026, 7, 28), 101.0), (ed, 102.0),
                     (date(2026, 7, 30), 110.0)]:
        ts = int(datetime(d.year, d.month, d.day).timestamp() * 1000)
        bars.append({"t": ts, "c": close})

    amc = pq.compute_event_move(bars, ed, "Post Market")
    assert amc == pytest.approx((110.0 - 102.0) / 102.0 * 100.0)
    bmo = pq.compute_event_move(bars, ed, "Pre Market")
    assert bmo == pytest.approx((102.0 - 101.0) / 101.0 * 100.0)
    unknown = pq.compute_event_move(bars, ed, None)
    assert unknown == pytest.approx(bmo)

    assert pq.exit_window_start(ed, "Post Market") == ed + timedelta(days=1)
    assert pq.exit_window_start(ed, "Pre Market") == ed
    assert pq.exit_window_start(ed, None) == ed

    # degenerate inputs never crash
    assert pq.compute_event_move([], ed, None) is None
    assert pq.compute_event_move(bars[:1], ed, None) is None


def test_empty_and_missing_outcomes_handled_gracefully(pq_db):
    """Empty DB -> zeroed report; executed trade without any outcome row ->
    unscorable, no fabricated hit or P&L."""
    report = pq.build_report(pq_db)
    s = report["summary"]
    assert s["approved"] == 0 and s["executed"] == 0
    assert s["hit_rate"] is None and s["total_pnl"] is None
    assert report["trades"] == []

    # executed trade whose ticker has no snapshots outcome at all
    _add_ladder(pq_db, "ZZZ", "2026-07-29", "filled", order_id="ord-zzz")
    _add_fill(pq_db, "ord-zzz", "ZZZ260828C00100000", "ZZZ260918C00100000", 1.50)
    # outcome row exists but expected_move_pct missing -> falls back to
    # implied event move derived from candidate quotes
    _add_ladder(pq_db, "QQQ", "2026-07-29", "filled", order_id="ord-qqq")
    _add_fill(pq_db, "ord-qqq", "QQQ260828C00100000", "QQQ260918C00100000", 1.50)
    _add_outcome(pq_db, "QQQ", "2026-07-29", expected=None, actual=0.5,
                 direction="UP")

    report = pq.build_report(pq_db)
    trades = {t["ticker"]: t for t in report["trades"]}
    assert trades["ZZZ"]["hit"] is None
    assert trades["ZZZ"]["actual_move_pct"] is None
    assert trades["ZZZ"]["pnl"] is None
    # QQQ: hit falls back to implied event move (realized 0.5% << implied)
    assert trades["QQQ"]["expected_move_pct"] is None
    assert trades["QQQ"]["hit"] is True
    s = report["summary"]
    assert s["executed"] == 2
    assert s["with_outcome"] == 1
    assert s["scorable"] == 1
