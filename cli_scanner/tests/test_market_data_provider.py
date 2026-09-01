"""Tests for the resilient market-data provider layer (Yahoo primary, LSE/Polygon opt-in)."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from earnings_edge.analyzer import OptionsAnalyzer, _thin_expiries
from earnings_edge.market_data_provider import (
    LSEProvider,
    OptionChainData,
    PolygonProvider,
    ResilientProvider,
    YahooProvider,
)
from earnings_edge.models import AnalysisResult, EarningsCandidate
from earnings_edge.validator import StockValidator


# ── helpers ---------------------------------------------------------------

def _hist_df(close: float, days: int = 5, volume: float = 2_000_000) -> pd.DataFrame:
    idx = pd.date_range(end=datetime.today(), periods=days, freq="B").normalize()
    idx.name = "Date"
    return pd.DataFrame(
        {
            "Open": close * 0.99,
            "High": close * 1.01,
            "Low": close * 0.98,
            "Close": np.linspace(close * 0.95, close, days),
            "Volume": volume,
        },
        index=idx,
    )


def _chain_df(strike: float, iv: float, oi: int = 5000) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "contractSymbol": f"O:T{strike}",
                "strike": strike,
                "bid": 1.0,
                "ask": 1.2,
                "lastPrice": 1.1,
                "impliedVolatility": iv,
                "openInterest": oi,
                "volume": 100.0,
                "delta": 0.5,
                "inTheMoney": False,
            }
        ]
    )


class StubProvider:
    """Yahoo-shaped stub returning canned data."""

    name = "stub"
    max_expiries_hint = None

    def __init__(self, price: float = 100.0, oi_available: bool = True):
        self.price = price
        self.oi_available = oi_available
        today = date.today()
        self.expiries = [
            (today + timedelta(days=7)).isoformat(),
            (today + timedelta(days=30)).isoformat(),
            (today + timedelta(days=60)).isoformat(),
        ]

    def history(self, ticker, period="1d"):
        days = {"1d": 5, "1mo": 25, "3mo": 65}.get(period, 65)
        return _hist_df(self.price, days=days)

    def options_expiries(self, ticker):
        return list(self.expiries)

    def option_chain(self, ticker, expiry):
        iv = {self.expiries[0]: 0.80, self.expiries[1]: 0.50, self.expiries[2]: 0.40}[expiry]
        oi = 5000 if self.oi_available else 0
        return OptionChainData(
            calls=_chain_df(self.price, iv, oi),
            puts=_chain_df(self.price, iv, oi),
            oi_available=self.oi_available,
            source=self.name,
        )


# ── _thin_expiries ---------------------------------------------------------

def test_thin_expiries_keeps_anchors():
    dates = ["2026-08-01", "2026-08-08", "2026-08-15", "2026-08-22", "2026-09-20"]
    thinned = _thin_expiries(dates, 3)
    assert thinned[0] == dates[0]
    assert thinned[-1] == dates[-1]
    assert len(thinned) == 3


def test_thin_expiries_noop_when_small_or_none():
    dates = ["2026-08-01", "2026-08-08"]
    assert _thin_expiries(dates, 3) == dates
    assert _thin_expiries(dates, None) == dates


# ── PolygonProvider (HTTP stubbed) -----------------------------------------

class FakePolygon(PolygonProvider):
    """PolygonProvider with _get replaced by canned responses."""

    def __init__(self, spot: float = 100.0):
        super().__init__(api_key="test-key")
        self.spot = spot
        # no real sleeping in tests
        for limiter in self._limiters.values():
            limiter.acquire = lambda: None

    def _get(self, path, params=None, kind="stock"):
        today_ms = int(datetime.now().timestamp() * 1000)
        if "grouped" in path:
            return {"resultsCount": 1, "results": [
                {"T": "TEST", "o": self.spot, "h": self.spot * 1.01,
                 "l": self.spot * 0.99, "c": self.spot, "v": 3_000_000, "t": today_ms}
            ]}
        if "reference/options/contracts" in path:
            d1 = (date.today() + timedelta(days=7)).isoformat()
            d2 = (date.today() + timedelta(days=30)).isoformat()
            return {"results": [
                {"ticker": "O:TEST1", "contract_type": "call", "expiration_date": d1, "strike_price": self.spot},
                {"ticker": "O:TEST2", "contract_type": "put", "expiration_date": d1, "strike_price": self.spot},
                {"ticker": "O:TEST3", "contract_type": "call", "expiration_date": d2, "strike_price": self.spot * 1.2},
            ]}
        if "/range/1/day/" in path and "/O:" in path:
            # ATM call ~100 spot, 7 DTE, strike 100: fair value at IV 50% ≈ 2.2
            return {"resultsCount": 1, "results": [
                {"T": "O:TEST", "o": 2.2, "h": 2.3, "l": 2.1, "c": 2.2, "v": 500, "t": today_ms}
            ]}
        if "/range/1/day/" in path:
            bars = [
                {"T": "TEST", "o": self.spot, "h": self.spot * 1.01, "l": self.spot * 0.99,
                 "c": self.spot * (1 + 0.001 * i), "v": 3_000_000,
                 "t": today_ms - (30 - i) * 86_400_000}
                for i in range(30)
            ]
            return {"resultsCount": len(bars), "results": bars}
        raise AssertionError(f"unexpected path: {path}")


def test_polygon_history_1d_uses_grouped_daily():
    p = FakePolygon(spot=123.0)
    df = p.history("TEST", "1d")
    assert not df.empty
    assert float(df["Close"].iloc[-1]) == 123.0


def test_polygon_history_3mo_shape():
    p = FakePolygon()
    df = p.history("TEST", "3mo")
    assert len(df) == 30
    assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]


def test_polygon_expiries_sorted_unique():
    p = FakePolygon()
    exps = p.options_expiries("TEST")
    assert exps == sorted(set(exps))
    assert len(exps) == 2


def test_polygon_option_chain_sparse_atm():
    p = FakePolygon(spot=100.0)
    expiry = p.options_expiries("TEST")[0]
    chain = p.option_chain("TEST", expiry)
    assert chain.oi_available is False
    assert chain.source == "polygon"
    assert len(chain.calls) == 1 and len(chain.puts) == 1
    row = chain.calls.iloc[0]
    # bid/ask collapse to the EOD close (no quote entitlement)
    assert row["bid"] == row["ask"] == row["lastPrice"] == 2.2
    # IV solved from the close should be in a sane band (~0.5 expected)
    assert 0.2 < row["impliedVolatility"] < 1.5
    # ATM call delta should be near 0.5
    assert 0.3 < row["delta"] < 0.7


def test_polygon_missing_ticker_returns_empty_history():
    class EmptyPolygon(FakePolygon):
        def _get(self, path, params=None, kind="stock"):
            if "grouped" in path:
                return {"resultsCount": 1, "results": [
                    {"T": "OTHER", "o": 1, "h": 1, "l": 1, "c": 1, "v": 1,
                     "t": int(datetime.now().timestamp() * 1000)}
                ]}
            return super()._get(path, params, kind)

    p = EmptyPolygon()
    df = p.history("TEST", "1d")
    assert df.empty


# ── ResilientProvider failover ----------------------------------------------

class FailingYahoo:
    name = "yahoo"
    max_expiries_hint = None

    def __init__(self, healthy=False):
        self._healthy = healthy

    def healthy(self, timeout=6.0):
        return self._healthy

    def history(self, *a):
        raise ConnectionError("429 blocked")

    def options_expiries(self, *a):
        raise ConnectionError("429 blocked")

    def option_chain(self, *a):
        raise ConnectionError("429 blocked")


def test_resilient_starts_on_polygon_when_yahoo_unhealthy():
    r = ResilientProvider(yahoo=FailingYahoo(healthy=False), polygon=StubProvider())
    assert r.active_name == "stub"
    df = r.history("TEST", "1d")
    assert not df.empty


def test_resilient_default_chain_excludes_polygon():
    """Polygon is reserved for backtesting — the live default chain (no
    explicit polygon= passed, no LSE_API_KEY) must be Yahoo-only, so a dead
    Yahoo fails fast instead of grinding through a doomed Polygon fallback."""
    r = ResilientProvider()
    assert all(not isinstance(p, PolygonProvider) for p in r._order)
    assert r._order == [r._yahoo]


def test_resilient_explicit_polygon_kwarg_still_included():
    """Callers that want Polygon (tests, one-off scripts) can still opt in."""
    poly = StubProvider()
    r = ResilientProvider(yahoo=FailingYahoo(healthy=False), polygon=poly)
    assert poly in r._order


def test_resilient_failover_mid_run_and_latch():
    yahoo = FailingYahoo(healthy=True)  # healthy at init, then dies per-call
    poly = StubProvider()
    r = ResilientProvider(yahoo=yahoo, polygon=poly)
    assert r.active_name == "yahoo"
    df = r.history("TEST", "1d")  # yahoo raises → failover
    assert not df.empty
    assert r.active_name == "stub"
    # latched: subsequent calls go straight to polygon
    assert r.options_expiries("TEST") == poly.expiries


def test_resilient_recovers_when_yahoo_healthy_again():
    yahoo = FailingYahoo(healthy=False)
    r = ResilientProvider(yahoo=yahoo, polygon=StubProvider(), recheck_calls=2)
    assert r.active_name == "stub"
    yahoo._healthy = True
    yahoo.history = lambda *a, **k: _hist_df(100.0)  # noqa: E501 - now works
    r.history("TEST", "1d")  # call 1
    r.history("TEST", "1d")  # call 2 → recheck triggers
    assert r.active_name == "yahoo"


def test_yahoo_provider_healthy_uses_session():
    class FakeResp:
        status_code = 200

    class FakeSession:
        def get(self, url, timeout=None):
            assert "finance.yahoo.com" in url
            return FakeResp()

    y = YahooProvider(session=FakeSession())
    assert y.healthy() is True


# ── analyzer + validator integration with stub provider ---------------------

def test_analyzer_with_stub_provider():
    analyzer = OptionsAnalyzer()
    result = analyzer.compute_recommendation("TEST", provider=StubProvider())
    assert result.ok, result.error
    assert result.current_price == pytest.approx(100.0, rel=0.05)
    # declining IV term structure → negative slope
    assert result.term_slope < 0
    assert result.iv30_rv30 > 0


def test_analyzer_respects_expiries_hint():
    class HintedStub(StubProvider):
        max_expiries_hint = 2

        def __init__(self):
            super().__init__()
            self.chain_calls: list[str] = []

        def option_chain(self, ticker, expiry):
            self.chain_calls.append(expiry)
            return super().option_chain(ticker, expiry)

    stub = HintedStub()
    analyzer = OptionsAnalyzer()
    result = analyzer.compute_recommendation("TEST", provider=stub)
    assert result.ok, result.error
    # hint = 2 → near + far anchors only
    assert len(stub.chain_calls) == 2
    assert stub.chain_calls[0] == stub.expiries[0]
    assert stub.chain_calls[-1] == stub.expiries[-1]


class _FakeBrowser:
    def get_win_rate(self, ticker):
        return SimpleNamespace(win_rate=0.0, quarters=0)  # no data → gate skipped


def _passing_analysis(ticker: str, price: float) -> AnalysisResult:
    return AnalysisResult(
        ticker=ticker,
        current_price=price,
        recommendation="BUY",
        iv30_rv30=1.40,
        term_slope=-0.01,
        term_structure_valid=True,
        term_structure_tier2=False,
        expected_move="4.00%",
        avg_volume_pass=True,
        atm_call_delta=0.50,
        atm_put_delta=-0.50,
    )


def test_validator_skips_oi_gate_when_unavailable():
    stub = StubProvider(oi_available=False)
    analyzer = OptionsAnalyzer()
    analyzer.compute_recommendation = lambda t, ed=None, provider=None: _passing_analysis(t, 100.0)
    v = StockValidator(analyzer, _FakeBrowser(), provider=stub)
    cand = EarningsCandidate(ticker="TEST", timing="Post Market", earnings_date=date.today())
    result = v.validate(cand)
    assert "OI" not in result.reason, result.reason


def test_validator_oi_gate_applies_when_available():
    stub = StubProvider(oi_available=False)
    # OI present but tiny → must fail on the OI gate
    stub.oi_available = True
    analyzer = OptionsAnalyzer()
    analyzer.compute_recommendation = lambda t, ed=None, provider=None: _passing_analysis(t, 100.0)
    v = StockValidator(analyzer, _FakeBrowser(), provider=stub)
    # shrink OI below the 2000 gate
    orig_chain = stub.option_chain

    def low_oi_chain(ticker, expiry):
        ch = orig_chain(ticker, expiry)
        ch.calls["openInterest"] = 10
        ch.puts["openInterest"] = 10
        return ch

    stub.option_chain = low_oi_chain
    cand = EarningsCandidate(ticker="TEST", timing="Post Market", earnings_date=date.today())
    result = v.validate(cand)
    assert not result.passed
    assert "OI" in result.reason


# ── LSEProvider ------------------------------------------------------------

class FakeLSEClient:
    """Stub for the lse-data LSE client surface used by LSEProvider."""

    def __init__(self, spot: float = 320.0):
        self.spot = spot
        today = date.today()
        self.near = (today + timedelta(days=7)).isoformat()
        self.far = (today + timedelta(days=30)).isoformat()
        self.past = (today - timedelta(days=7)).isoformat()

    def candles(self, symbol, timeframe="1m", start=None, end=None, limit=5000,
                order="asc", dataset=None):
        return [
            {"symbol": symbol, "open": 100.0, "high": 102.0, "low": 99.0,
             "close": 101.0, "volume": 1_000_000,
             "timestamp": "2026-07-20T00:00:00.000000Z"},
            {"symbol": symbol, "open": 101.0, "high": 103.0, "low": 100.0,
             "close": 102.0, "volume": 2_000_000,
             "timestamp": "2026-07-21T00:00:00.000000Z"},
        ]

    def options(self, underlying, type=None, expiry=None, strike=None,
                min_dte=None, max_dte=None, limit=5000):
        rows = []
        for i, k in enumerate((315.0, 320.0, 325.0)):
            for ctype in ("call", "put"):
                rows.append({
                    "ticker": f"{underlying}{self.near}{ctype[0].upper()}{int(k)}",
                    "underlying": underlying, "strike": k, "expiry": self.near,
                    "contract_type": ctype, "last_price": 5.0 - i,
                    "volume_today": 100 + i, "iv": 0.40,
                    "delta": 0.55 - 0.1 * i if ctype == "call" else -0.45 - 0.1 * i,
                    "underlying_price": self.spot, "dte": 7,
                    "updated_at": "2026-07-24T20:00:00.000000Z",
                })
        rows.append({
            "ticker": "OLD", "underlying": underlying, "strike": 320.0,
            "expiry": self.past, "contract_type": "call", "last_price": 1.0,
            "volume_today": 5, "iv": 0.9, "delta": 0.1,
            "underlying_price": self.spot, "dte": -7,
            "updated_at": "2026-07-01T20:00:00.000000Z",
        })
        return rows


def test_lse_history_shape():
    p = LSEProvider(api_key="x", client=FakeLSEClient())
    df = p.history("TEST", "3mo")
    assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert df.index.name == "Date"
    assert len(df) == 2
    assert df["Close"].iloc[-1] == 102.0


def test_lse_history_1d_returns_last_session_only():
    p = LSEProvider(api_key="x", client=FakeLSEClient())
    df = p.history("TEST", "1d")
    assert len(df) == 1
    assert df["Volume"].iloc[0] == 2_000_000


def test_lse_history_failure_returns_empty():
    class BrokenClient(FakeLSEClient):
        def candles(self, *a, **k):
            raise ConnectionError("lse down")

    p = LSEProvider(api_key="x", client=BrokenClient())
    assert p.history("TEST", "3mo").empty


def test_lse_expiries_filter_past_and_sort():
    p = LSEProvider(api_key="x", client=FakeLSEClient())
    assert p.options_expiries("TEST") == [p.client.near]


def test_lse_option_chain_maps_columns():
    p = LSEProvider(api_key="x", client=FakeLSEClient())
    ch = p.option_chain("TEST", p.client.near)
    assert ch.source == "lse"
    assert ch.oi_available is False
    assert list(ch.calls.columns) == [
        "contractSymbol", "strike", "bid", "ask", "lastPrice",
        "impliedVolatility", "openInterest", "volume", "delta", "inTheMoney",
    ]
    assert len(ch.calls) == 3 and len(ch.puts) == 3
    assert list(ch.calls["strike"]) == [315.0, 320.0, 325.0]
    row = ch.calls.iloc[0]
    assert row["bid"] == row["lastPrice"] == 5.0
    assert row["impliedVolatility"] == 0.40
    assert row["volume"] == 100.0
    assert row["inTheMoney"]  # 320 spot > 315 strike


def test_lse_option_chain_unknown_expiry_raises():
    p = LSEProvider(api_key="x", client=FakeLSEClient())
    with pytest.raises(ValueError):
        p.option_chain("TEST", "2099-01-01")


def test_lse_healthy_requires_key():
    p = LSEProvider(api_key="", client=FakeLSEClient())
    assert p.healthy() is False
    p2 = LSEProvider(api_key="x", client=FakeLSEClient())
    assert p2.healthy() is True


def test_lse_concurrency_gate_matches_observed_vault_limit():
    """The scanner's worker pool runs up to 8 tickers in parallel, but the
    vault plan only accepts 2 concurrent connections — without this gate,
    the excess requests get rejected, which drives the shared rate limiter's
    backoff toward its 5s ceiling for every worker, not just the failing
    one. Locks the gate size to the documented plan limit."""
    p = LSEProvider(api_key="x", client=FakeLSEClient())
    assert p._concurrency.acquire(blocking=False) is True
    assert p._concurrency.acquire(blocking=False) is True
    assert p._concurrency.acquire(blocking=False) is False  # 3rd slot: none free


def test_lse_call_releases_concurrency_slot_after_completion():
    p = LSEProvider(api_key="x", client=FakeLSEClient())
    p._call(lambda: "ok")
    # both slots must be free again — a leaked slot would eventually starve
    # the whole scan down to 1 (or 0) concurrent LSE calls
    assert p._concurrency.acquire(blocking=False) is True
    assert p._concurrency.acquire(blocking=False) is True


def test_lse_call_releases_concurrency_slot_after_failure():
    p = LSEProvider(api_key="x", client=FakeLSEClient())

    def boom():
        raise ConnectionError("vault rejected")

    with pytest.raises(ConnectionError):
        p._call(boom)
    assert p._concurrency.acquire(blocking=False) is True
    assert p._concurrency.acquire(blocking=False) is True


class _FakeStatusError(Exception):
    """Stands in for lse.client.LSEError, which carries a real .status."""
    def __init__(self, status):
        self.status = status
        super().__init__(f"status={status}")


def test_lse_call_does_not_throttle_on_404_not_found():
    """A scan universe is mostly small-caps/OTC/delisted tickers the vault's
    catalog was never going to carry — that 404 is a fast, well-formed
    response, not an overload signal, and must not pin the shared limiter's
    backoff interval up (regression: this was making full scans crawl)."""
    p = LSEProvider(api_key="x", client=FakeLSEClient())
    with pytest.raises(_FakeStatusError):
        p._call(lambda: (_ for _ in ()).throw(_FakeStatusError(404)))
    assert p._limiter._interval == pytest.approx(0.4 * 0.9)


def test_lse_call_does_not_throttle_on_400_bad_request():
    p = LSEProvider(api_key="x", client=FakeLSEClient())
    with pytest.raises(_FakeStatusError):
        p._call(lambda: (_ for _ in ()).throw(_FakeStatusError(400)))
    assert p._limiter._interval == pytest.approx(0.4 * 0.9)


def test_lse_call_still_throttles_on_429_rate_limited():
    p = LSEProvider(api_key="x", client=FakeLSEClient())
    with pytest.raises(_FakeStatusError):
        p._call(lambda: (_ for _ in ()).throw(_FakeStatusError(429)))
    assert p._limiter._interval == pytest.approx(0.4 * 2)


def test_lse_call_throttles_on_unrecognized_exception():
    """No .status attribute at all (e.g. a raw connection error) — unknown
    shape, so the conservative default is still to back off."""
    p = LSEProvider(api_key="x", client=FakeLSEClient())
    with pytest.raises(ConnectionError):
        p._call(lambda: (_ for _ in ()).throw(ConnectionError("boom")))
    assert p._limiter._interval == pytest.approx(0.4 * 2)


# ── ResilientProvider with LSE in the chain ---------------------------------

class StubLSE:
    """Healthy LSE-shaped stub."""

    name = "lse"
    max_expiries_hint = None

    def __init__(self, healthy: bool = True):
        self._healthy = healthy

    def healthy(self, timeout=6.0):
        return self._healthy

    def history(self, ticker, period="1d"):
        return _hist_df(100.0)

    def options_expiries(self, ticker):
        return ["2026-08-01"]

    def option_chain(self, ticker, expiry):
        return OptionChainData(calls=_chain_df(100.0, 0.5),
                               puts=_chain_df(100.0, 0.5),
                               oi_available=False, source=self.name)


class FailingLSE(StubLSE):
    def history(self, *a):
        raise ConnectionError("lse down")

    def options_expiries(self, *a):
        raise ConnectionError("lse down")

    def option_chain(self, *a):
        raise ConnectionError("lse down")


def test_resilient_starts_on_lse_when_healthy():
    r = ResilientProvider(lse=StubLSE(), yahoo=FailingYahoo(healthy=False),
                          polygon=StubProvider())
    assert r.active_name == "lse"
    assert not r.history("TEST", "1d").empty


def test_resilient_skips_unhealthy_lse_for_yahoo():
    class HealthyYahoo(StubProvider):
        name = "yahoo"

        def healthy(self, timeout=6.0):
            return True

    r = ResilientProvider(lse=StubLSE(healthy=False), yahoo=HealthyYahoo(),
                          polygon=StubProvider())
    assert r.active_name == "yahoo"


def test_resilient_failover_lse_to_yahoo_to_polygon():
    yahoo = FailingYahoo(healthy=True)  # healthy at init, dies per-call
    r = ResilientProvider(lse=FailingLSE(healthy=True), yahoo=yahoo,
                          polygon=StubProvider())
    assert r.active_name == "lse"
    df = r.history("TEST", "1d")  # lse raises → yahoo raises → polygon
    assert not df.empty
    assert r.active_name == "stub"


def test_resilient_recovers_to_lse():
    lse = FailingLSE(healthy=False)
    r = ResilientProvider(lse=lse, yahoo=FailingYahoo(healthy=False),
                          polygon=StubProvider(), recheck_calls=2)
    assert r.active_name == "stub"
    lse._healthy = True
    lse.history = lambda *a, **k: _hist_df(100.0)
    r.history("TEST", "1d")
    r.history("TEST", "1d")  # recheck triggers
    assert r.active_name == "lse"
