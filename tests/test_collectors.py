"""Tests for the collectors layer."""

import unittest
from unittest.mock import patch, MagicMock
from datetime import date

from earnings_edge.collectors.base import BaseCollector, CircuitBreakerOpen
from earnings_edge.models import EarningsCandidate


class TestBaseCollector(unittest.TestCase):
    def test_retry_succeeds_on_second_attempt(self):
        calls = []

        def flaky():
            calls.append(1)
            if len(calls) == 1:
                raise ConnectionError("transient")
            return "ok"

        c = BaseCollector(name="test", max_retries=3, base_delay=0.01)
        result = c.with_retry(flaky)
        self.assertEqual(result, "ok")
        self.assertEqual(len(calls), 2)

    def test_circuit_breaker_opens_after_failures(self):
        c = BaseCollector(name="test", max_retries=1, base_delay=0.01,
                          circuit_threshold=2)
        for _ in range(2):
            try:
                c.with_retry(lambda: (_ for _ in ()).throw(ConnectionError("fail")))
            except ConnectionError:
                pass
        with self.assertRaises(CircuitBreakerOpen):
            c.with_retry(lambda: "should not reach")

    def test_circuit_breaker_resets_on_success(self):
        c = BaseCollector(name="test", max_retries=3, base_delay=0.01,
                          circuit_threshold=2)
        c.with_retry(lambda: "ok")  # success
        self.assertFalse(c._circuit_open)

    def test_is_healthy_property(self):
        c = BaseCollector(name="test", circuit_threshold=1)
        self.assertTrue(c.is_healthy)
        c._on_failure()
        self.assertFalse(c.is_healthy)


class TestEarningsCalendarCollector(unittest.TestCase):
    def test_falls_back_to_finnhub_when_investing_fails(self):
        from earnings_edge.collectors.earnings_calendar import EarningsCalendarCollector

        collector = EarningsCalendarCollector()
        with patch.object(collector, '_investing_fetch', return_value=[]), \
             patch.object(collector, '_finnhub_fetch',
                          return_value=[EarningsCandidate(ticker="AAPL", timing="Post Market", source="finnhub")]):
            results = collector.fetch(date(2026, 6, 17))
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].ticker, "AAPL")
            self.assertEqual(results[0].source, "finnhub")

    def test_merges_sources_deduplicating(self):
        from earnings_edge.collectors.earnings_calendar import EarningsCalendarCollector

        collector = EarningsCalendarCollector()
        with patch.object(collector, '_investing_fetch',
                          return_value=[EarningsCandidate(ticker="AAPL", timing="Post Market", source="investing")]), \
             patch.object(collector, '_finnhub_fetch',
                          return_value=[EarningsCandidate(ticker="AAPL", timing="Post Market", source="finnhub"),
                                        EarningsCandidate(ticker="MSFT", timing="Pre Market", source="finnhub")]):
            results = collector.fetch(date(2026, 6, 17))
            tickers = {r.ticker for r in results}
            self.assertEqual(tickers, {"AAPL", "MSFT"})

    def test_returns_empty_when_all_sources_fail(self):
        from earnings_edge.collectors.earnings_calendar import EarningsCalendarCollector

        collector = EarningsCalendarCollector()
        with patch.object(collector, '_investing_fetch', return_value=[]), \
             patch.object(collector, '_finnhub_fetch', return_value=[]):
            results = collector.fetch(date(2026, 6, 17))
            self.assertEqual(results, [])


class TestPolygonClient(unittest.TestCase):
    def test_get_returns_none_without_api_key(self):
        from earnings_edge.collectors.polygon import PolygonClient
        with patch('earnings_edge.collectors.polygon.get_settings') as mock_s:
            mock_s.return_value = MagicMock(polygon_api_key="")
            c = PolygonClient()
            result = c.get("/v2/aggs/ticker/AAPL/range/1/day/2026-06-17/2026-06-20")
            self.assertIsNone(result)

    @patch('earnings_edge.collectors.polygon.requests')
    def test_get_daily_bars_returns_results(self, mock_requests):
        from earnings_edge.collectors.polygon import PolygonClient
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "resultsCount": 1,
            "results": [{"c": 150.0, "h": 155.0, "l": 148.0, "t": 1718600000000}],
        }
        mock_resp.raise_for_status = MagicMock()
        mock_requests.get.return_value = mock_resp

        with patch('earnings_edge.collectors.polygon.get_settings') as mock_s:
            mock_s.return_value = MagicMock(polygon_api_key="test_key")
            c = PolygonClient()
            bars = c.get_daily_bars("AAPL", "2026-06-17", "2026-06-20")
            self.assertEqual(len(bars), 1)
            self.assertEqual(bars[0]["c"], 150.0)


if __name__ == "__main__":
    unittest.main()


class TestLSECollector(unittest.TestCase):
    def _make(self, client):
        from earnings_edge.collectors.lse import LSECollector
        return LSECollector(api_key="x", client=client, sleep=0)

    def test_daily_bars_polygon_shape(self):
        from earnings_edge.collectors.lse import LSECollector  # noqa: F401
        client = MagicMock()
        client.candles.return_value = [
            {"symbol": "AAPL", "open": 1.0, "high": 2.0, "low": 0.5,
             "close": 1.5, "volume": 100,
             "timestamp": "2026-07-20T00:00:00.000000Z"},
            {"symbol": "AAPL", "open": 2.0, "high": 3.0, "low": 1.5,
             "close": 2.5, "volume": 200,
             "timestamp": "2026-07-21T00:00:00.000000Z"},
        ]
        c = self._make(client)
        bars = c.daily_bars("AAPL", date(2026, 7, 20), date(2026, 7, 21))
        self.assertEqual(len(bars), 2)
        self.assertEqual(bars[0]["c"], 1.5)
        self.assertEqual(bars[1]["v"], 200)
        self.assertIn("t", bars[0])
        client.candles.assert_called_once_with(
            "AAPL", "1d", start="2026-07-20", end="2026-07-21",
            limit=5000, order="asc")

    def test_option_contracts_filters_expiry_window(self):
        client = MagicMock()
        client.options.return_value = [
            {"ticker": "A", "strike": 100.0, "expiry": "2026-08-01",
             "contract_type": "call"},
            {"ticker": "B", "strike": 105.0, "expiry": "2026-08-15",
             "contract_type": "put"},
            {"ticker": "C", "strike": 110.0, "expiry": "2026-12-31",
             "contract_type": "call"},
        ]
        c = self._make(client)
        out = c.option_contracts(
            "AAPL", as_of=date(2026, 7, 20),
            expiry_gte=date(2026, 7, 25), expiry_lte=date(2026, 9, 1))
        self.assertEqual([r["ticker"] for r in out], ["A", "B"])
        self.assertEqual(out[0]["strike_price"], 100.0)
        self.assertEqual(out[0]["expiration_date"], "2026-08-01")

    def test_option_close_returns_last_close_in_window(self):
        client = MagicMock()
        client.option_candles.return_value = [
            {"close": 1.1, "timestamp": "2026-07-19T19:59:00Z"},
            {"close": 1.3, "timestamp": "2026-07-20T19:59:00Z"},
        ]
        c = self._make(client)
        px = c.option_close("AAPL260801C00100000", date(2026, 7, 20))
        self.assertEqual(px, 1.3)

    def test_option_close_ignores_bars_after_as_of(self):
        client = MagicMock()
        client.option_candles.return_value = [
            {"close": 1.1, "timestamp": "2026-07-20T19:59:00Z"},
            {"close": 9.9, "timestamp": "2026-07-21T13:30:00Z"},
        ]
        c = self._make(client)
        px = c.option_close("AAPL260801C00100000", date(2026, 7, 20))
        self.assertEqual(px, 1.1)

    def test_option_close_none_when_no_bars(self):
        client = MagicMock()
        client.option_candles.return_value = []
        c = self._make(client)
        self.assertIsNone(c.option_close("AAPL260801C00100000", date(2026, 7, 20)))

    def test_chain_cached_per_underlying(self):
        client = MagicMock()
        client.options.return_value = [
            {"ticker": "A", "strike": 100.0, "expiry": "2026-08-01",
             "contract_type": "call"},
        ]
        c = self._make(client)
        c.option_contracts("AAPL")
        c.option_contracts("AAPL")
        client.options.assert_called_once()
