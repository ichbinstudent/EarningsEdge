from unittest.mock import patch

from earnings_edge.browser import MarketChameleonBrowser
from earnings_edge.models import WinRateData


def test_browser_init_failure_breaker():
    browser = MarketChameleonBrowser()

    init_call_count = 0

    def fake_init_driver():
        nonlocal init_call_count
        init_call_count += 1
        raise RuntimeError("DevToolsActivePort file doesn't exist")

    with patch.object(browser, "_init_driver", side_effect=fake_init_driver):
        # 1. First failure
        res1 = browser.get_win_rate("AAPL")
        assert res1 == WinRateData()
        assert init_call_count == 1
        assert browser._consecutive_failures == 1

        # 2. Second failure
        res2 = browser.get_win_rate("TSLA")
        assert res2 == WinRateData()
        assert init_call_count == 2
        assert browser._consecutive_failures == 2

        # 3. Third failure - trips the breaker
        res3 = browser.get_win_rate("MSFT")
        assert res3 == WinRateData()
        assert init_call_count == 3
        assert browser._consecutive_failures == 3

        # 4. Subsequent call - should return default without calling init
        res4 = browser.get_win_rate("GOOG")
        assert res4 == WinRateData()
        assert init_call_count == 3  # init_driver not called again

        # Another one
        res5 = browser.get_win_rate("AMZN")
        assert res5 == WinRateData()
        assert init_call_count == 3
