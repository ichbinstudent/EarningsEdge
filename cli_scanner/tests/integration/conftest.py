"""Integration-test layer: component interactions against a temp DB with
mocked external providers. No real network, no production data, paper-only
Alpaca. The network guard is autouse for every test in this directory.
"""

import pytest


@pytest.fixture(autouse=True)
def _no_network(network_guard):
    """Apply the shared network/paper guard to every integration test."""
    return network_guard
