"""End-to-end test layer: full pipelines (collect -> scan -> propose) on
fixture data into a temp DB. No real network, no production data,
paper-only Alpaca. The network guard is autouse for every test here.
"""

import pytest


@pytest.fixture(autouse=True)
def _no_network(network_guard):
    """Apply the shared network/paper guard to every e2e test."""
    return network_guard
