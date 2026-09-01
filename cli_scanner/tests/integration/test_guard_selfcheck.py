"""Throwaway: proves the autouse network_guard actually fires."""

import pytest

pytestmark = pytest.mark.integration


def test_guard_blocks_requests():
    import requests

    with pytest.raises(AssertionError, match="real HTTP blocked"):
        requests.get("https://example.com", timeout=2)


def test_guard_blocks_socket():
    import socket

    with pytest.raises(AssertionError, match="must not open real network"):
        socket.create_connection(("example.com", 443), timeout=2)


def test_guard_blocks_live_alpaca():
    from earnings_edge.alpaca_trading import AlpacaTradingClient

    with pytest.raises(AssertionError, match="non-paper Alpaca"):
        AlpacaTradingClient(api_key="x", api_secret="y", paper=False)
