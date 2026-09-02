"""E2E: run scripts/proposal_quality.py end-to-end against a fixture DB.

Asserts the CLI entry point produces a report with the expected shape and
writes valid JSON — the same artifact the cron/docs pipeline consumes.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.e2e

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import proposal_quality as pq  # noqa: E402


@pytest.fixture
def fixture_db(tmp_path):
    from earnings_edge.fwd_factor_ladder import DDL as FF_DDL
    from earnings_edge.trade_approval import _SCHEMA as PENDING_DDL
    from earnings_edge.db import configure

    path = tmp_path / "e2e_pq.db"
    configure(path)
    conn = sqlite3.connect(str(path), timeout=30)
    conn.executescript(FF_DDL)
    conn.executescript(PENDING_DDL)
    conn.commit()
    conn.close()

    cand = {
        "ticker": "MSFT", "earnings_date": "2026-07-29", "spot": 397.32,
        "strike": 395.0, "near_symbol": "MSFT260828C00395000",
        "far_symbol": "MSFT260918C00395000", "near_expiry": "2026-08-28",
        "far_expiry": "2026-09-18", "near_bid": 20.97, "near_ask": 22.49,
        "far_bid": 24.81, "far_ask": 26.21, "sigma_fwd": 0.3136,
        "hist_rms_move": 0.00515, "tau_days": 2, "d_start": 9.54,
        "d_cap": 9.54, "mid_debit": 3.78,
    }
    conn = sqlite3.connect(str(path))
    conn.execute(
        "INSERT INTO ff_ladders (ticker, candidate_json, order_id, status, rung, "
        "armed_by, created_at, updated_at) VALUES (?,?,?,?,0,123,"
        "'2026-07-28 18:46:55','2026-07-29 18:00:00')",
        ("MSFT", json.dumps(cand), "ord-msft", "filled"),
    )
    for sym in ("MSFT260828C00395000", "MSFT260918C00395000"):
        conn.execute(
            "INSERT INTO managed_positions (symbol, strategy, group_id, qty, "
            "entry_price, status, order_id, opened_at) VALUES (?,?,?,?,?,?,?,"
            "'2026-07-29T18:00:00')",
            (sym, "ff_ladder", "ord-msft", 1.0, 6.55, "open", "ord-msft"),
        )
    conn.execute(
        "INSERT INTO snapshots (ticker, earnings_date, scan_date, timing, "
        "expected_move_pct, actual_move_pct, actual_move_direction, data_source) "
        "VALUES ('MSFT','2026-07-29','2026-07-28','Post Market',7.16,-0.7144,"
        "'DOWN','e2e_fixture')",
    )
    conn.commit()
    conn.close()
    return path


def test_script_end_to_end_report_shape(fixture_db, tmp_path, capsys):
    out = tmp_path / "report.json"
    report = pq.main(["--db", str(fixture_db), "--json", str(out)])

    # report dict shape
    for key in ("generated_at", "db_path", "summary", "by_source",
                "by_strategy", "trades", "notes"):
        assert key in report, key
    for key in ("approved", "executed", "with_outcome", "scorable", "hits",
                "hit_rate", "binomial_p", "total_pnl", "significant"):
        assert key in report["summary"], key

    assert report["summary"]["approved"] == 1
    assert report["summary"]["executed"] == 1
    assert report["summary"]["hits"] == 1  # 0.71% realized < 7.16% expected
    assert report["summary"]["hit_rate"] == 1.0
    assert report["summary"]["significant"] is False

    trade = report["trades"][0]
    assert trade["ticker"] == "MSFT"
    assert trade["entry_debit"] == 6.55
    assert trade["hit"] is True

    # JSON artifact on disk matches
    on_disk = json.loads(out.read_text())
    assert on_disk["summary"]["approved"] == 1
    assert len(on_disk["trades"]) == 1

    # stdout carries the human-readable summary
    stdout = capsys.readouterr().out
    assert "Proposal quality report" in stdout
    assert "stored outcome convention" in stdout
    assert "1/1" in stdout
