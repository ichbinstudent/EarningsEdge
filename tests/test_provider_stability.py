"""Stability regressions for the LSE → Yahoo provider chain.

A per-ticker catalog gap on LSE is routine (small-caps, OTC names); it must
not move the whole scan onto Yahoo (the backend this host gets IP-banned on).
"""

from __future__ import annotations

import math
import threading
from datetime import UTC, date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from earnings_edge.market_data_provider import (
    DataUnavailable,
    LSEProvider,
    OptionChainData,
    ResilientProvider,
    is_data_miss,
)


def _hist(close=100.0, days=5):
    idx = pd.date_range(end=datetime.today(), periods=days, freq="B").normalize()
    return pd.DataFrame(
        {"Open": close, "High": close, "Low": close, "Close": close, "Volume": 1e6},
        index=pd.Index(idx, name="Date"),
    )


def _chain(source):
    row = {
        "contractSymbol": "X",
        "strike": 100.0,
        "bid": 1.0,
        "ask": 1.2,
        "lastPrice": 1.1,
        "impliedVolatility": 0.4,
        "openInterest": 0,
        "volume": 1.0,
        "delta": 0.5,
        "inTheMoney": False,
    }
    return OptionChainData(calls=pd.DataFrame([row]), puts=pd.DataFrame([row]), source=source)


class Recording:
    """Provider stub that records calls; tickers in ``missing`` raise ``miss``."""

    max_expiries_hint = None

    def __init__(self, name, missing=(), miss=None, healthy=True):
        self.name = name
        self.missing = set(missing)
        self.miss = miss or (lambda t: DataUnavailable(f"{name} has no {t}"))
        self._healthy = healthy
        self.calls: list[tuple[str, str]] = []

    def healthy(self, timeout=6.0):
        return self._healthy

    def _hit(self, method, ticker):
        self.calls.append((method, ticker))
        if ticker in self.missing:
            raise self.miss(ticker)

    def history(self, ticker, period="1d"):
        self._hit("history", ticker)
        return _hist()

    def options_expiries(self, ticker):
        self._hit("options_expiries", ticker)
        return ["2099-01-15"]

    def option_chain(self, ticker, expiry):
        self._hit("option_chain", ticker)
        return _chain(self.name)


# ── ResilientProvider: miss vs outage ─────────────────────────────────────


def test_ticker_gap_on_lse_does_not_move_the_latch():
    lse = Recording("lse", missing={"SMALL"})
    yahoo = Recording("yahoo")
    r = ResilientProvider(lse=lse, yahoo=yahoo)
    assert r.options_expiries("SMALL") == ["2099-01-15"]
    assert r.active_name == "lse"
    r.options_expiries("AAPL")
    assert ("options_expiries", "AAPL") in lse.calls
    assert ("options_expiries", "AAPL") not in yahoo.calls


def test_ticker_is_pinned_to_the_provider_that_served_it():
    """Every expiry's chain for the gap ticker goes straight to Yahoo (one
    consistent source) instead of re-missing on LSE each time."""
    lse = Recording("lse", missing={"SMALL"})
    yahoo = Recording("yahoo")
    r = ResilientProvider(lse=lse, yahoo=yahoo)
    r.options_expiries("SMALL")
    lse.calls.clear()
    assert r.option_chain("SMALL", "2099-01-15").source == "yahoo"
    assert lse.calls == []


def test_pin_is_per_data_family():
    """An options gap must not push the ticker's candles off LSE."""
    lse = Recording("lse", missing=set())
    lse.options_expiries = lambda t: (_ for _ in ()).throw(DataUnavailable("no options"))
    yahoo = Recording("yahoo")
    r = ResilientProvider(lse=lse, yahoo=yahoo)
    r.options_expiries("SMALL")
    r.history("SMALL", "1d")
    assert ("history", "SMALL") in lse.calls
    assert ("history", "SMALL") not in yahoo.calls


def test_pin_expires():
    lse = Recording("lse", missing={"SMALL"})
    r = ResilientProvider(lse=lse, yahoo=Recording("yahoo"), affinity_ttl_secs=0.0)
    r.options_expiries("SMALL")
    lse.calls.clear()
    r.option_chain("SMALL", "2099-01-15")
    assert ("option_chain", "SMALL") in lse.calls  # pin already lapsed → LSE tried first


class _StatusError(Exception):
    def __init__(self, status):
        self.status = status
        super().__init__(f"[{status}]")


def test_http_404_counts_as_data_miss():
    lse = Recording("lse", missing={"OTC"}, miss=lambda t: _StatusError(404))
    r = ResilientProvider(lse=lse, yahoo=Recording("yahoo"))
    assert not r.history("OTC", "1d").empty
    assert r.active_name == "lse"


@pytest.mark.parametrize("status", [0, 429, 503])
def test_outage_statuses_still_latch(status):
    lse = Recording("lse", missing={"AAPL"}, miss=lambda t: _StatusError(status))
    r = ResilientProvider(lse=lse, yahoo=Recording("yahoo"))
    r.history("AAPL", "1d")
    assert r.active_name == "yahoo"


def test_is_data_miss_classification():
    assert is_data_miss(DataUnavailable("x"))
    assert is_data_miss(_StatusError(400))
    assert not is_data_miss(_StatusError(503))
    assert not is_data_miss(ConnectionError("reset"))


def test_miss_everywhere_raises_last_error():
    r = ResilientProvider(lse=Recording("lse", missing={"ZZZ"}), yahoo=Recording("yahoo", missing={"ZZZ"}))
    with pytest.raises(DataUnavailable):
        r.options_expiries("ZZZ")
    assert r.active_name == "lse"


def test_recheck_probe_runs_without_holding_the_lock():
    """The re-probe is a network call; holding the dispatch lock during it
    stalled every scan worker."""
    lse = Recording("lse", healthy=False)
    r = ResilientProvider(lse=lse, yahoo=Recording("yahoo"), recheck_calls=1)
    assert r.active_name == "yahoo"
    seen = {}

    def probe(timeout=6.0):
        seen["lock_free"] = r._lock.acquire(blocking=False)
        if seen["lock_free"]:
            r._lock.release()
        return True

    lse.healthy = probe
    r.history("AAPL", "1d")
    assert seen["lock_free"] is True
    assert r.active_name == "lse"


def test_probe_exception_counts_as_unhealthy():
    lse = Recording("lse")
    lse.healthy = lambda timeout=6.0: (_ for _ in ()).throw(RuntimeError("probe blew up"))
    r = ResilientProvider(lse=lse, yahoo=Recording("yahoo"))
    assert r.active_name == "yahoo"


# ── LSEProvider guards ────────────────────────────────────────────────────


def _fresh():
    return (datetime.now(UTC) - timedelta(hours=3)).isoformat().replace("+00:00", "Z")


def _stale():
    return (datetime.now(UTC) - timedelta(days=12)).isoformat().replace("+00:00", "Z")


class FakeVault:
    def __init__(self, rows=None, candles_rows=None):
        self.rows = rows or []
        self.candles_rows = candles_rows
        self.candle_calls = 0
        self.option_calls = 0

    def options(self, underlying, limit=5000, **kw):
        self.option_calls += 1
        return [dict(r) for r in self.rows]

    def candles(self, symbol, timeframe="1d", start=None, end=None, limit=5000, order="asc", dataset=None):
        self.candle_calls += 1
        if self.candles_rows is not None:
            return self.candles_rows
        return [
            {
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0 + i,
                "volume": 1000.0 * (i + 1),
                "timestamp": f"{date.today() - timedelta(days=d)}T00:00:00Z",
            }
            for i, d in enumerate((90, 40, 20, 3, 1))
        ]


EXP = (date.today() + timedelta(days=10)).isoformat()


def _opt(underlying, strike, ctype="call", updated=None, osi=None, **extra):
    row = {
        "ticker": osi or f"{underlying}{EXP}{ctype[0].upper()}{int(strike)}",
        "strike": strike,
        "expiry": EXP,
        "contract_type": ctype,
        "last_price": 2.5,
        "iv": 0.4,
        "delta": 0.5,
        "volume_today": 10,
        "underlying_price": 100.0,
        "updated_at": updated or _fresh(),
    }
    if underlying is not None:
        row["underlying"] = underlying
    row.update(extra)
    return row


def test_lse_ignores_another_companys_chain():
    """lse-data resolves unknown symbols by company-name match, so a ticker
    missing from the catalog can come back as a different company's chain."""
    vault = FakeVault([_opt("BBY", 100.0), _opt("BBY", 100.0, "put")])
    p = LSEProvider(api_key="x", client=vault)
    with pytest.raises(DataUnavailable):
        p.options_expiries("BE")


def test_lse_keeps_matching_rows_and_osi_rooted_rows():
    rows = [
        _opt("BRK.B", 100.0),
        _opt(None, 105.0, osi="BRKB261016C00105000"),  # no underlying field: OSI root
        _opt("OTHER", 110.0),
    ]
    p = LSEProvider(api_key="x", client=FakeVault(rows))
    ch = p.option_chain("BRK.B", EXP)
    assert list(ch.calls["strike"]) == [100.0, 105.0]


def test_lse_blanks_stale_quotes():
    rows = [_opt("T", 100.0), _opt("T", 105.0, updated=_stale()), _opt("T", 100.0, "put")]
    p = LSEProvider(api_key="x", client=FakeVault(rows))
    calls = p.option_chain("T", EXP).calls.set_index("strike")
    assert calls.loc[100.0, "lastPrice"] == 2.5
    assert math.isnan(calls.loc[105.0, "lastPrice"])
    assert math.isnan(calls.loc[105.0, "bid"]) and math.isnan(calls.loc[105.0, "impliedVolatility"])


def test_lse_all_stale_expiry_is_a_data_miss():
    rows = [_opt("T", 100.0, updated=_stale()), _opt("T", 100.0, "put", updated=_stale())]
    p = LSEProvider(api_key="x", client=FakeVault(rows))
    with pytest.raises(DataUnavailable):
        p.options_expiries("T")
    with pytest.raises(DataUnavailable):
        p.option_chain("T", EXP)


def test_lse_healthy_rejects_frozen_quotes():
    frozen = FakeVault([_opt("SPY", 500.0, updated=_stale())])
    assert LSEProvider(api_key="x", client=frozen).healthy() is False
    live = FakeVault([_opt("SPY", 500.0)])
    assert LSEProvider(api_key="x", client=live).healthy() is True


def test_lse_history_one_call_serves_all_short_periods():
    """Validator + analyzer ask 1d, 1mo, 3mo of the same ticker back to back."""
    vault = FakeVault()
    p = LSEProvider(api_key="x", client=vault)
    one = p.history("T", "1d")
    month = p.history("T", "1mo")
    quarter = p.history("T", "3mo")
    assert vault.candle_calls == 1
    assert len(one) == 1 and one["Close"].iloc[0] == 104.0
    assert len(month) == 3  # 20d, 3d, 1d bars inside 35 days
    assert len(quarter) == 5
    month.loc[month.index[0], "Close"] = -1.0  # callers get copies
    assert p.history("T", "1mo")["Close"].iloc[0] == 102.0


def test_lse_history_longer_period_refetches():
    vault = FakeVault()
    p = LSEProvider(api_key="x", client=vault)
    p.history("T", "3mo")
    p.history("T", "1y")
    assert vault.candle_calls == 2


def test_lse_history_cache_expires():
    vault = FakeVault()
    p = LSEProvider(api_key="x", client=vault)
    p._HISTORY_TTL_SECS = 0.0
    p.history("T", "1d")
    p.history("T", "1d")
    assert vault.candle_calls == 2


def test_lse_retries_transient_errors():
    vault = FakeVault()
    failures = iter([_StatusError(503), _StatusError(0)])

    def flaky(*a, **k):
        err = next(failures, None)
        if err is not None:
            raise err
        return FakeVault.candles(vault, *a, **k)

    vault.candles = flaky
    p = LSEProvider(api_key="x", client=vault)
    p._retry_delay = 0.0
    assert not p.history("T", "3mo").empty


@pytest.mark.parametrize("exc", [_StatusError(404), _StatusError(401), ValueError("bad row")])
def test_lse_does_not_retry_permanent_errors(exc):
    attempts = []

    def fail(*a, **k):
        attempts.append(1)
        raise exc

    vault = FakeVault()
    vault.candles = fail
    p = LSEProvider(api_key="x", client=vault)
    p._retry_delay = 0.0
    with pytest.raises(type(exc)):
        p.history("T", "3mo")
    assert len(attempts) == 1


def test_lse_retry_gives_up_after_max_attempts():
    attempts = []

    def fail(*a, **k):
        attempts.append(1)
        raise _StatusError(503)

    vault = FakeVault()
    vault.candles = fail
    p = LSEProvider(api_key="x", client=vault)
    p._retry_delay = 0.0
    with pytest.raises(_StatusError):
        p.history("T", "3mo")
    assert len(attempts) == LSEProvider._MAX_ATTEMPTS


def test_lse_client_gets_bounded_rest_timeout(monkeypatch):
    import lse

    seen = {}

    class Spy:
        def __init__(self, api_key=None, timeout=None, **kw):
            seen["timeout"] = timeout

    monkeypatch.setattr(lse, "LSE", Spy)
    LSEProvider(api_key="x").client  # noqa: B018 — property constructs the client
    assert seen["timeout"] == LSEProvider._REST_TIMEOUT_SECS < 60


def test_lse_caches_are_thread_safe_under_eviction():
    """8 scan workers share one provider; eviction used to iterate the cache
    dict while other workers inserted into it."""
    vault = FakeVault([_opt(None, 100.0), _opt(None, 100.0, "put")])
    p = LSEProvider(api_key="x", client=vault)
    p._CACHE_MAX_TICKERS = 4
    p._concurrency = threading.Semaphore(64)
    p._limiter.acquire = lambda: None
    errors = []

    def work(n):
        try:
            for i in range(60):
                t = f"T{n}_{i}"
                p.options_expiries(t)
                p.history(t, "1d")
        except Exception as exc:  # pragma: no cover — the failure mode under test
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(n,)) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert len(p._chain_cache) <= 4 and len(p._history_cache) <= 4


def test_lse_option_chain_nan_price_rows_keep_structure():
    rows = [_opt("T", k, updated=_stale() if k > 100 else None) for k in (95.0, 100.0, 105.0)]
    rows += [_opt("T", 100.0, "put")]
    ch = LSEProvider(api_key="x", client=FakeVault(rows)).option_chain("T", EXP)
    assert list(ch.calls["strike"]) == [95.0, 100.0, 105.0]
    assert np.isnan(ch.calls["lastPrice"].iloc[-1])
