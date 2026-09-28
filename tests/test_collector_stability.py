"""Stability regressions for the collectors and the Alpaca REST client."""

from __future__ import annotations

import ast
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from earnings_edge.alpaca_trading import AlpacaError, AlpacaTradingClient
from earnings_edge.collectors.base import BaseCollector, CircuitBreakerOpen
from earnings_edge.collectors.lse import LSECollector
from earnings_edge.collectors.polygon import PolygonClient


class _Status(Exception):
    def __init__(self, status):
        self.status = status
        super().__init__(f"[{status}]")


# ── BaseCollector classification hooks ───────────────────────────────────


class _Classifying(BaseCollector):
    def _is_data_miss(self, exc):
        return getattr(exc, "status", None) == 404

    def _is_retryable(self, exc):
        return getattr(exc, "status", None) != 401


def test_data_miss_is_not_retried_and_does_not_trip_breaker():
    c = _Classifying(name="t", max_retries=3, base_delay=0, circuit_threshold=2)
    calls = []

    def miss():
        calls.append(1)
        raise _Status(404)

    for _ in range(5):
        with pytest.raises(_Status):
            c.with_retry(miss)
    assert len(calls) == 5  # one attempt each
    assert c.is_healthy
    assert c.with_retry(lambda: "ok") == "ok"


def test_non_retryable_fails_fast_but_counts_toward_breaker():
    c = _Classifying(name="t", max_retries=3, base_delay=0, circuit_threshold=2)
    calls = []

    def auth():
        calls.append(1)
        raise _Status(401)

    for _ in range(2):
        with pytest.raises(_Status):
            c.with_retry(auth)
    assert len(calls) == 2
    with pytest.raises(CircuitBreakerOpen):
        c.with_retry(lambda: "never")


# ── PolygonClient ─────────────────────────────────────────────────────────


def _http_response(status, payload=None):
    resp = MagicMock(status_code=status)
    resp.json.return_value = payload or {}
    if status >= 400:
        resp.raise_for_status.side_effect = requests.HTTPError(f"{status}", response=resp)
    else:
        resp.raise_for_status.return_value = None
    return resp


@pytest.mark.parametrize("status", [403, 404])
def test_polygon_plan_or_ticker_miss_costs_one_request_and_no_sleep(status):
    """Each 403/404 used to cost 15s + 30s of retry sleeps and count toward
    the breaker (fwd_factor_ladder's backfill documents the 45s+ per miss)."""
    client = PolygonClient()  # conftest supplies a test key
    sleeps = []
    with (
        patch("earnings_edge.collectors.polygon.requests.get", return_value=_http_response(status)) as get,
        patch("earnings_edge.collectors.base.time.sleep", side_effect=sleeps.append),
    ):
        assert client.get("/v2/aggs/ticker/OLD/range/1/day/2019-01-01/2019-01-10") is None
    assert get.call_count == 1
    assert sleeps == []
    assert client.is_healthy


def test_polygon_server_error_is_still_retried():
    client = PolygonClient()
    responses = [_http_response(502), _http_response(200, {"resultsCount": 1, "results": [{"c": 1}]})]
    with (
        patch("earnings_edge.collectors.polygon.requests.get", side_effect=responses) as get,
        patch("earnings_edge.collectors.base.time.sleep"),
    ):
        assert client.get("/x") == {"resultsCount": 1, "results": [{"c": 1}]}
    assert get.call_count == 2


# ── LSECollector ──────────────────────────────────────────────────────────


def test_lse_collector_unknown_contract_is_not_retried():
    client = MagicMock()
    client.option_candles.side_effect = _Status(404)
    c = LSECollector(api_key="x", client=client, sleep=0)
    with (
        patch("earnings_edge.collectors.base.time.sleep") as sleep,
        pytest.raises(_Status),
    ):
        c.option_close("AAPL260801C00100000", date(2026, 7, 20))
    assert client.option_candles.call_count == 1
    sleep.assert_not_called()


def test_lse_collector_drops_other_underlyings_chain():
    client = MagicMock()
    client.options.return_value = [
        {
            "ticker": "BBY",
            "underlying": "BBY",
            "strike": 100.0,
            "expiry": "2026-08-01",
            "contract_type": "call",
        },
        {
            "ticker": "BE1",
            "underlying": "BE",
            "strike": 20.0,
            "expiry": "2026-08-01",
            "contract_type": "call",
        },
    ]
    c = LSECollector(api_key="x", client=client, sleep=0)
    assert [r["ticker"] for r in c.option_contracts("BE")] == ["BE1"]


# ── Alpaca REST client ────────────────────────────────────────────────────


@pytest.fixture
def alpaca():
    c = AlpacaTradingClient(api_key="k", api_secret="s")
    c.session = MagicMock()
    return c


def _resp(status, payload=None):
    r = MagicMock(status_code=status, text="")
    r.json.return_value = payload if payload is not None else {}
    return r


def test_alpaca_rate_limit_exhaustion_raises_instead_of_empty_success(alpaca):
    """Used to return {} — callers saw 'no chain' / an order without an id."""
    alpaca.session.request.return_value = _resp(429)
    with patch("earnings_edge.alpaca_trading.time.sleep"), pytest.raises(AlpacaError) as err:
        alpaca.get_account()
    assert err.value.status_code == 429
    assert alpaca.session.request.call_count == 3


def test_alpaca_get_retries_gateway_errors(alpaca):
    alpaca.session.request.side_effect = [_resp(503), _resp(200, {"buying_power": "1"})]
    with patch("earnings_edge.alpaca_trading.time.sleep"):
        assert alpaca.get_account() == {"buying_power": "1"}
    assert alpaca.session.request.call_count == 2


def test_alpaca_post_is_not_resent_after_gateway_error(alpaca):
    """A POST may have been applied before the gateway failed."""
    alpaca.session.request.return_value = _resp(502, {"message": "bad gateway"})
    with patch("earnings_edge.alpaca_trading.time.sleep"), pytest.raises(AlpacaError) as err:
        alpaca.submit_order("AAPL", 1, "buy", order_type="market")
    assert err.value.status_code == 502
    assert alpaca.session.request.call_count == 1


def test_alpaca_chain_pagination_is_capped(alpaca):
    page = {"snapshots": {"X": {"latestQuote": {"bp": 1.0, "ap": 1.1}}}, "next_page_token": "again"}
    alpaca.session.request.return_value = _resp(200, page)
    alpaca.MAX_CHAIN_PAGES = 3
    assert set(alpaca.get_options_chain_snapshots("AAPL")) == {"X"}
    assert alpaca.session.request.call_count == 3


# ── risk-event accounting ─────────────────────────────────────────────────


def test_each_ladder_failure_records_one_risk_event():
    """Every broad handler in fwd_factor_ladder wrote its silent_failure
    risk event (and traceback) twice — a duplicated exc-policy block."""
    tree = ast.parse(Path("earnings_edge/fwd_factor_ladder.py").read_text())
    for handler in (n for n in ast.walk(tree) if isinstance(n, ast.ExceptHandler)):
        n_events = sum(
            1
            for stmt in handler.body
            for node in ast.walk(stmt)
            if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "record_event"
        )
        assert n_events <= 1, f"line {handler.lineno}: {n_events} record_event calls"
