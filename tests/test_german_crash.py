"""German-venue 5-minute crash detector: quotes, window, cooldown, bot wiring."""

from __future__ import annotations

import inspect
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from earnings_edge.german_crash import (
    Cooldown,
    CrashAlert,
    CrashAlertConfig,
    CrashDetector,
    CrashMonitor,
    format_alert,
    in_crash_poll_window,
    in_open_snapshot_window,
    parse_lseg_number,
    parse_tg_number,
    parse_trade_ts,
    quote_from_lseg,
    quote_from_tradegate,
    ric_parts,
    validate_quote,
)

NOW = datetime(2026, 9, 1, 16, 0, 0, tzinfo=timezone.utc)
CFG = CrashAlertConfig(
    threshold=0.20,
    window_secs=300,
    cooldown_secs=1800,
    stale_secs=180,
    min_price=1.0,
    tradegate_enabled=True,
    snapshot_open=False,
)


def _q(**over):
    base = dict(
        ticker="SAPG", venue="Gettex", last=100.0, bid=99.8, ask=100.2,
        ts=NOW, now=NOW, cfg=CFG, ric="SAPG.GTX", name="SAP SE", source="lseg",
        trade_ts=NOW,
    )
    base.update(over)
    return validate_quote(**base)


# ── parsers ──────────────────────────────────────────────────────────────


def test_parse_lseg_number_plus_prefix_and_sentinels():
    assert parse_lseg_number("+183.2") == 183.2
    assert parse_lseg_number("+0") == 0.0
    assert parse_lseg_number("4.5") == 4.5
    assert parse_lseg_number("-") is None
    assert parse_lseg_number("n.a.") is None
    assert parse_lseg_number(None) is None
    assert parse_lseg_number(float("nan")) is None


def test_parse_tg_number_german_and_plain():
    assert parse_tg_number(148.65) == 148.65
    assert parse_tg_number("1 076,00") == 1076.0
    assert parse_tg_number("1.076,00") == 1076.0
    assert parse_tg_number("148,65") == 148.65
    assert parse_tg_number("-") is None
    assert parse_tg_number("./.") is None


def test_ric_parts_and_trade_ts():
    assert ric_parts("SAPG.GTX") == ("SAPG", "Gettex")
    assert ric_parts("ADSGn.DE") == ("ADSGn", "Xetra")
    assert ric_parts("NVDA.F") == ("NVDA", "Frankfurt")
    ts = parse_trade_ts("01 SEP 2026", "17:47:33", NOW)
    assert ts is not None
    assert ts.tzinfo is not None
    # LSEG trade time is UTC: 17:47:33 == 17:47:33 UTC, not Berlin local.
    assert ts.utcoffset().total_seconds() == 0
    assert ts.hour == 17
    assert parse_trade_ts("01 SEP 2026", "17:40", NOW) is not None
    assert parse_trade_ts(None, "17:40", NOW) is None


# ── fail-closed validation ───────────────────────────────────────────────


def test_valid_two_sided_book_accepted():
    q = _q()
    assert q is not None
    assert q.price == pytest.approx(100.0)
    assert q.venue == "Gettex"


def test_reject_zero_last_without_book():
    assert _q(last=0.0, bid=0.0, ask=0.0) is None
    assert _q(last=None, bid=None, ask=None) is None


def test_reject_crossed_book():
    assert _q(last=10.0, bid=12.0, ask=11.0) is None


def test_reject_one_sided_or_zero_book():
    assert _q(bid=0.0, ask=10.0) is None
    assert _q(bid=10.0, ask=0.0) is None
    assert _q(bid=-1.0, ask=10.0) is None


def test_reject_stale_observation():
    old = NOW - timedelta(seconds=181)
    assert _q(ts=old) is None
    assert _q(ts=NOW - timedelta(seconds=179)) is not None


def test_reject_below_min_price():
    assert _q(last=0.5, bid=0.49, ask=0.51) is None


def test_stale_last_far_from_mid_uses_mid_if_trade_recent():
    trade = NOW - timedelta(seconds=120)
    q = _q(last=100.0, bid=70.0, ask=70.4, trade_ts=trade)
    assert q is not None
    assert q.price == pytest.approx(70.2)


def test_require_print_within_10_minutes():
    assert _q(trade_ts=None) is None
    assert _q(last=None, trade_ts=NOW) is None
    assert _q(trade_ts=NOW - timedelta(seconds=601)) is None
    assert _q(trade_ts=NOW - timedelta(seconds=600)) is not None


def test_fresh_last_near_mid_is_used():
    q = _q(last=100.0, bid=99.9, ask=100.1, trade_ts=NOW)
    assert q is not None
    assert q.price == pytest.approx(100.0)


def test_quote_from_lseg_and_tradegate():
    lseg = quote_from_lseg(
        {
            "q.RIC": "SAPG.GTX",
            "q._TRDPRC_1": "+100.0",
            "q._BID": "+99.8",
            "q._ASK": "+100.2",
            "q._DSPLY_NAME": "SAP SE",
            "q._TRADE_DATE": "01 SEP 2026",
            "q._TRDTIM_1": "18:00:00",
        },
        captured_at=NOW,
        cfg=CFG,
    )
    assert lseg is not None and lseg.ticker == "SAPG" and lseg.venue == "Gettex"

    # Tradegate index JSON has no last print → no alert input
    tg = quote_from_tradegate(
        {"isin": "DE0007164600", "name": "SAP SE", "bid": 99.8, "ask": 100.2},
        captured_at=NOW,
        cfg=CFG,
    )
    assert tg is None


def test_lseg_garbage_zero_book_no_quote():
    assert quote_from_lseg(
        {
            "q.RIC": "JUNK.GTX",
            "q._TRDPRC_1": "+12.0",
            "q._BID": "+0",
            "q._ASK": "+0",
        },
        captured_at=NOW,
        cfg=CFG,
    ) is None


def test_reject_xetra_stub_bid():
    # Live false alert 2026-09-02: LNC.DE bid=0.0001 ask=38.03
    assert _q(last=37.0, bid=0.0001, ask=38.03) is None


def test_reject_wide_spread_book():
    # Live false alert: CUFR.DE bid=71 ask=140
    assert _q(last=141.5, bid=71.0, ask=140.0) is None


def test_stub_book_does_not_create_false_crash():
    det = CrashDetector(CFG)
    t0 = NOW - timedelta(seconds=120)
    det.ingest([_px("LNC", "Xetra", 37.0, t0)], t0)
    stub = _q(ticker="LNC", venue="Xetra", last=37.0, bid=0.0001, ask=38.03, ric="LNC.DE")
    assert stub is None
    alerts = det.ingest([], NOW)  # no new valid quote
    assert alerts == []
    # even if someone ingested a raw mid of the stub, validate already blocked it
    alerts = det.ingest([_px("LNC", "Xetra", 37.0, NOW)], NOW)
    assert alerts == []


# ── drop definition: peak-to-trough, strictly > 20% ──────────────────────


def _px(ticker, venue, price, ts, bid=None, ask=None):
    bid = price - 0.1 if bid is None else bid
    ask = price + 0.1 if ask is None else ask
    # Validate as-of `ts` so historical samples are not rejected as stale.
    return _q(
        ticker=ticker, venue=venue, last=price, bid=bid, ask=ask,
        ts=ts, now=ts, ric=f"{ticker}.GTX",
    )


def test_drop_just_over_20_percent_alerts():
    det = CrashDetector(CFG)
    t0 = NOW - timedelta(seconds=120)
    q0 = _px("SAPG", "Gettex", 100.0, t0)
    q1 = _px("SAPG", "Gettex", 79.9, NOW)  # 20.1%
    alerts = det.ingest([q0], t0)
    assert alerts == []
    alerts = det.ingest([q1], NOW)
    assert len(alerts) == 1
    assert alerts[0].drop_pct == pytest.approx(0.201, abs=1e-6)
    assert alerts[0].high == pytest.approx(100.0)
    assert alerts[0].last == pytest.approx(79.9)


def test_drop_exactly_20_percent_does_not_alert():
    det = CrashDetector(CFG)
    t0 = NOW - timedelta(seconds=60)
    det.ingest([_px("SAPG", "Gettex", 100.0, t0)], t0)
    alerts = det.ingest([_px("SAPG", "Gettex", 80.0, NOW)], NOW)  # exactly 20%
    assert alerts == []


def test_drop_under_threshold_silent():
    det = CrashDetector(CFG)
    t0 = NOW - timedelta(seconds=60)
    det.ingest([_px("SAPG", "Gettex", 100.0, t0)], t0)
    alerts = det.ingest([_px("SAPG", "Gettex", 81.0, NOW)], NOW)
    assert alerts == []


def test_single_sample_never_alerts():
    det = CrashDetector(CFG)
    assert det.ingest([_px("SAPG", "Gettex", 50.0, NOW)], NOW) == []


def test_quotes_outside_window_ignored():
    det = CrashDetector(CFG)
    t_old = NOW - timedelta(seconds=301)  # just outside 300s
    det.ingest([_px("SAPG", "Gettex", 100.0, t_old)], t_old)
    alerts = det.ingest([_px("SAPG", "Gettex", 50.0, NOW)], NOW)
    # old high aged out; only one sample remains
    assert alerts == []


def test_quote_on_window_edge_counts():
    det = CrashDetector(CFG)
    t0 = NOW - timedelta(seconds=300)
    det.ingest([_px("SAPG", "Gettex", 100.0, t0)], t0)
    alerts = det.ingest([_px("SAPG", "Gettex", 70.0, NOW)], NOW)
    assert len(alerts) == 1


def test_recovery_does_not_alert():
    det = CrashDetector(CFG)
    t0 = NOW - timedelta(seconds=180)
    t1 = NOW - timedelta(seconds=90)
    det.ingest([_px("SAPG", "Gettex", 100.0, t0)], t0)
    det.ingest([_px("SAPG", "Gettex", 70.0, t1)], t1)  # would alert
    # recovered by the latest print
    alerts = det.ingest([_px("SAPG", "Gettex", 100.0, NOW)], NOW)
    assert alerts == []


def test_book_dump_without_print_uses_mid_and_alerts():
    det = CrashDetector(CFG)
    t0 = NOW - timedelta(seconds=60)
    det.ingest([_px("SAPG", "Gettex", 100.0, t0)], t0)
    q = _q(
        ticker="SAPG", venue="Gettex", last=100.0, bid=69.8, ask=70.2,
        ts=NOW, ric="SAPG.GTX", trade_ts=NOW - timedelta(seconds=60),
    )
    assert q is not None
    assert q.price == pytest.approx(70.0)
    alerts = det.ingest([q], NOW)
    assert len(alerts) == 1
    assert alerts[0].drop_pct == pytest.approx(0.30, abs=1e-6)


def test_venues_are_independent():
    det = CrashDetector(CFG)
    t0 = NOW - timedelta(seconds=60)
    det.ingest([
        _px("SAPG", "Gettex", 100.0, t0),
        _px("SAPG", "Xetra", 100.0, t0),
    ], t0)
    alerts = det.ingest([
        _px("SAPG", "Gettex", 70.0, NOW),
        _px("SAPG", "Xetra", 99.0, NOW),
    ], NOW)
    venues = {a.venue for a in alerts}
    assert venues == {"Gettex"}


# ── cooldown ─────────────────────────────────────────────────────────────


def test_cooldown_suppresses_same_ticker():
    cd = Cooldown(cooldown_secs=1800)
    a = CrashAlert(
        ticker="SAPG", venue="Gettex", drop_pct=0.25, high=100, last=75,
        window_secs=300, ts=NOW,
    )
    first = cd.filter([a], NOW)
    assert len(first) == 1
    second = cd.filter([a], NOW + timedelta(seconds=60))
    assert second == []
    later = cd.filter([a], NOW + timedelta(seconds=1801))
    assert len(later) == 1


def test_cooldown_is_per_venue_not_per_ticker():
    cd = Cooldown(cooldown_secs=1800)
    a1 = CrashAlert(ticker="SAPG", venue="Gettex", drop_pct=0.21, high=100, last=79,
                    window_secs=300, ts=NOW)
    a2 = CrashAlert(ticker="SAPG", venue="Xetra", drop_pct=0.40, high=100, last=60,
                    window_secs=300, ts=NOW)
    out = cd.filter([a1, a2], NOW)
    assert {a.venue for a in out} == {"Gettex", "Xetra"}
    again = cd.filter([a2], NOW + timedelta(seconds=60))
    assert again == []


def test_cooldown_persists_to_disk(tmp_path):
    path = str(tmp_path / "cd.json")
    a = CrashAlert(ticker="NVDA", venue="Gettex", drop_pct=0.3, high=10, last=7,
                   window_secs=300, ts=NOW)
    Cooldown(1800, path=path).filter([a], NOW)
    other = Cooldown(1800, path=path)
    assert other.filter([a], NOW + timedelta(seconds=10)) == []


# ── alert text ───────────────────────────────────────────────────────────


def test_format_alert_has_required_fields():
    a = CrashAlert(
        ticker="SAPG", venue="Gettex", drop_pct=0.214, high=126.4, last=99.3,
        window_secs=300, ts=NOW, bid=99.2, ask=99.4, name="SAP SE", ric="SAPG.GTX",
        high_ts=NOW - timedelta(seconds=90),
    )
    text = format_alert(a)
    assert "SAPG" in text
    assert "Gettex" in text
    assert "21.4%" in text
    assert "5m" in text
    assert "99.3" in text
    assert "126.4" in text
    assert "2026-09-01" in text
    assert "Bid/ask" in text
    assert "<b>" in text  # HTML for Telegram parse_mode


# ── monitor poll (mocked collectors) ─────────────────────────────────────


class _FakeGettex:
    def __init__(self, rows):
        self.rows = rows
        self.written = None
        self.fetched_rics = None

    def fetch_instruments(self, exchanges=None):
        return ["SAPG.GTX"]

    def fetch_quotes(self, rics):
        self.fetched_rics = list(rics)
        return self.rows

    def write_snapshot(self, quotes, now=None):
        self.written = (len(quotes), now)


class _FakeTg:
    def __init__(self, rows):
        self.rows = rows

    def fetch_index_quotes(self):
        return self.rows


def test_monitor_poll_emits_and_cools_down(tmp_path):
    lseg_t0 = [{
        "q.RIC": "SAPG.GTX", "q._TRDPRC_1": "+100", "q._BID": "+99.8",
        "q._ASK": "+100.2", "q._DSPLY_NAME": "SAP SE",
        "q._TRADE_DATE": "01 SEP 2026", "q._TRDTIM_1": "17:58:00",
    }]
    lseg_t1 = [{
        "q.RIC": "SAPG.GTX", "q._TRDPRC_1": "+70", "q._BID": "+69.8",
        "q._ASK": "+70.2", "q._DSPLY_NAME": "SAP SE",
        "q._TRADE_DATE": "01 SEP 2026", "q._TRDTIM_1": "18:00:00",
    }]
    fake = _FakeGettex(lseg_t0)
    mon = CrashMonitor(
        cfg=CFG, collector=fake, tradegate=_FakeTg([]),
        data_dir=str(tmp_path), cooldown_path=str(tmp_path / "cd.json"),
    )
    t0 = NOW - timedelta(seconds=120)
    r0 = mon.poll(t0)
    assert r0["n_alerts"] == 0
    fake.rows = lseg_t1
    r1 = mon.poll(NOW)
    assert r1["n_alerts"] == 1
    assert r1["tickers"] == ["SAPG"]
    r2 = mon.poll(NOW + timedelta(seconds=5))
    assert r2["n_alerts"] == 0  # cooldown


def test_monitor_skips_when_disabled(tmp_path):
    cfg = CrashAlertConfig(enabled=False)
    mon = CrashMonitor(cfg=cfg, collector=_FakeGettex([]), data_dir=str(tmp_path))
    assert mon.poll(NOW)["skipped"] == "disabled"


def test_monitor_does_not_alert_on_fetch_failure(tmp_path):
    class Boom:
        def fetch_instruments(self, exchanges=None):
            raise RuntimeError("network down")

        def fetch_quotes(self, rics):
            raise RuntimeError("network down")

    mon = CrashMonitor(
        cfg=CFG, collector=Boom(), tradegate=_FakeTg([]),
        data_dir=str(tmp_path), cooldown_path=str(tmp_path / "cd.json"),
    )
    out = mon.poll(NOW)
    assert out["n_alerts"] == 0
    assert out["n_fetched"] == 0


def test_in_open_snapshot_window():
    berlin_0730 = datetime(2026, 9, 1, 5, 30, tzinfo=timezone.utc)  # 07:30 CEST
    berlin_0800 = datetime(2026, 9, 1, 6, 0, tzinfo=timezone.utc)
    berlin_0900 = datetime(2026, 9, 1, 7, 0, tzinfo=timezone.utc)
    assert in_open_snapshot_window(berlin_0730)
    assert in_open_snapshot_window(berlin_0800)
    assert not in_open_snapshot_window(berlin_0900)


def test_in_crash_poll_window_0730_to_2300_berlin_weekdays():
    # Tue 2026-09-01 is CEST (UTC+2)
    assert in_crash_poll_window(datetime(2026, 9, 1, 5, 30, tzinfo=timezone.utc))  # 07:30
    assert in_crash_poll_window(datetime(2026, 9, 1, 21, 0, tzinfo=timezone.utc))   # 23:00
    assert not in_crash_poll_window(datetime(2026, 9, 1, 5, 29, tzinfo=timezone.utc))  # 07:29
    assert not in_crash_poll_window(datetime(2026, 9, 1, 21, 0, 1, tzinfo=timezone.utc))  # 23:00:01
    # Saturday
    assert not in_crash_poll_window(datetime(2026, 9, 5, 10, 0, tzinfo=timezone.utc))


def test_poll_skips_outside_berlin_window():
    collector = MagicMock()
    tradegate = MagicMock()
    mon = CrashMonitor(cfg=CFG, collector=collector, tradegate=tradegate)
    out = mon.poll(datetime(2026, 9, 1, 5, 0, tzinfo=timezone.utc))  # 07:00 CEST
    assert out["skipped"] == "outside_window"
    assert out["alerts"] == []
    collector.fetch_quotes.assert_not_called()
    tradegate.fetch_index_quotes.assert_not_called()


# ── bot wiring (real callers, not just the library) ──────────────────────


def test_bot_does_not_register_german_crash_jobs():
    import bot as bot_mod

    src = inspect.getsource(bot_mod.TradingBot._setup_scheduler)
    assert "german_crash" not in src
    assert not hasattr(bot_mod.TradingBot, "_german_crash_sync")


def test_crash_alert_process_owns_the_scheduler():
    import crash_alert as ca
    import inspect as ins

    src = ins.getsource(ca.main)
    assert 'id="german_crash"' in src
    assert 'id="german_crash_pre"' in src
    assert 'id="german_crash_close"' in src
    assert "*/2 8-22 * * mon-fri" in src
    assert "0 23 * * mon-fri" in src
    assert "poll_once" in ins.getsource(ca.tick)


def test_gettex_collector_write_snapshot_and_batching(tmp_path):
    from earnings_edge.collectors.gettex import GettexCollector

    calls = []

    class Sess:
        def get(self, url, headers=None, timeout=None):
            calls.append(url)
            resp = MagicMock()
            resp.status_code = 200
            if "quote/info" in url:
                rics = url.split("rics=")[1].split("&")[0].split(",")
                resp.json.return_value = {
                    "data": [{"q.RIC": r, "q._TRDPRC_1": "1"} for r in rics],
                }
            else:
                resp.json.return_value = {"data": []}
            return resp

    c = GettexCollector(str(tmp_path), batch_size=2, max_workers=1, session=Sess())
    rows = c.fetch_quotes(["A.GTX", "B.GTX", "C.GTX"])
    assert len(rows) == 3
    assert sum("quote/info" in u for u in calls) == 2  # 3 rics, batch 2
    path = c.write_snapshot(rows, NOW)
    assert path.endswith("gettex_quotes_2026-09-01.jsonl")
    text = open(path).read()
    assert text.count("\n") == 3
    assert "A.GTX" in text



