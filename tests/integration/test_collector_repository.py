"""Integration: earnings-calendar collector -> SQLite repository write path.

External sources (Investing.com, Finnhub) are mocked at the collector's
per-source fetch methods; the merge/fallback/persist logic runs for real
against a temp DB.
"""

from __future__ import annotations

from datetime import date

import pytest

pytestmark = pytest.mark.integration

TARGET = date(2026, 7, 29)


def _candidate(ticker, timing="Post Market", source="investing"):
    from earnings_edge.models import EarningsCandidate

    return EarningsCandidate(ticker=ticker, timing=timing, source=source)


@pytest.fixture
def collector(test_settings):
    from earnings_edge.collectors.earnings_calendar import EarningsCalendarCollector

    return EarningsCalendarCollector()


def _snapshot_row(ticker, timing="Post Market"):
    return {
        "ticker": ticker,
        "earnings_date": TARGET.isoformat(),
        "scan_date": date(2026, 7, 28).isoformat(),
        "timing": timing,
        "price": 150.0,
        "avg_volume_30d": 5_000_000,
        "has_options": 1,
        "data_source": "integration_fixture",
    }


def test_collector_merges_sources_and_persists(collector, monkeypatch, tmp_db_path):
    """investing + finnhub merge (dedup by ticker, non-Unknown timing wins),
    then every merged candidate lands in the snapshots table."""
    from sqlalchemy import text
    from earnings_edge.db import engine as db_engine, insert_snapshot

    monkeypatch.setattr(
        collector, "_investing_fetch",
        lambda d: [
            _candidate("AAPL", timing="Post Market", source="investing"),
            _candidate("MSFT", timing="Unknown", source="investing"),
        ],
    )
    monkeypatch.setattr(
        collector, "_finnhub_fetch",
        lambda d: [
            _candidate("MSFT", timing="Pre Market", source="finnhub"),
            _candidate("TSLA", timing="Post Market", source="finnhub"),
        ],
    )

    merged = collector.fetch(TARGET)

    assert {c.ticker for c in merged} == {"AAPL", "MSFT", "TSLA"}
    # MSFT existed in both; the finnhub row has real timing and must win.
    msft = next(c for c in merged if c.ticker == "MSFT")
    assert msft.timing == "Pre Market"
    assert msft.source == "finnhub"

    for c in merged:
        insert_snapshot(_snapshot_row(c.ticker, c.timing))
    with db_engine.get_session() as s:
        rows = s.execute(
            text("SELECT ticker, timing, data_source FROM snapshots ORDER BY ticker")
        ).mappings().all()

    assert [r["ticker"] for r in rows] == ["AAPL", "MSFT", "TSLA"]
    assert all(r["data_source"] == "integration_fixture" for r in rows)
    assert dict((r["ticker"], r["timing"]) for r in rows)["MSFT"] == "Pre Market"


def test_collector_fallback_and_duplicate_insert_ignored(
    collector, monkeypatch, tmp_db_path
):
    """investing.com failure falls back to finnhub; re-persisting the same
    scan is a silent no-op via the (ticker, earnings_date, scan_date, timing,
    data_source) unique index."""
    from sqlalchemy import text
    from earnings_edge.db import engine as db_engine, insert_snapshot

    def _boom(d):
        raise RuntimeError("investing.com 403")

    monkeypatch.setattr(collector, "_investing_fetch", _boom)
    monkeypatch.setattr(
        collector, "_finnhub_fetch",
        lambda d: [_candidate("NVDA", timing="Post Market", source="finnhub")],
    )

    merged = collector.fetch(TARGET)
    assert [c.ticker for c in merged] == ["NVDA"]
    assert merged[0].source == "finnhub"

    first = insert_snapshot(_snapshot_row("NVDA"))
    assert first > 0
    insert_snapshot(_snapshot_row("NVDA"))
    # INSERT OR IGNORE swallowed the duplicate: no new row written.
    with db_engine.get_session() as s:
        count = s.execute(text("SELECT COUNT(*) FROM snapshots")).scalar()

    assert count == 1
