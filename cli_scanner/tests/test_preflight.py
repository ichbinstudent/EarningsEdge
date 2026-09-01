"""Tests for the pre-flight script (all externals mocked)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import scripts.preflight as preflight


def _run_main(env, account=None, lse_ok=True, halted=False):
    client = MagicMock()
    if account is None:
        client.get_account.return_value = {"equity": "100000", "buying_power": "50000"}
        client.get_clock.return_value = {"is_open": True}
    else:
        client.get_account.side_effect = account
    ks = MagicMock()
    ks.status.return_value = {"halted": halted, "reason": "test" if halted else None}
    with patch.dict("os.environ", env, clear=True), \
         patch("earnings_edge.alpaca_trading.create_client", return_value=client), \
         patch("earnings_edge.market_data_provider.LSEProvider") as lse, \
         patch("framework.risk.killswitch.KillSwitch", return_value=ks), \
         patch("requests.get") as tg:
        lse.return_value.healthy.return_value = lse_ok
        tg.return_value.status_code = 200
        return preflight.main()


GOOD_ENV = {
    "TELEGRAM_BOT_TOKEN": "t", "APCA_API_KEY_ID": "k", "APCA_API_SECRET_KEY": "s",
    "LSE_API_KEY": "l",
}


def test_preflight_passes_when_all_good():
    assert _run_main(GOOD_ENV) == 0


def test_preflight_fails_without_telegram_token():
    env = {k: v for k, v in GOOD_ENV.items() if k != "TELEGRAM_BOT_TOKEN"}
    assert _run_main(env) == 1


def test_preflight_fails_when_alpaca_down():
    assert _run_main(GOOD_ENV, account=ConnectionError("down")) == 1


def test_preflight_fails_when_halted():
    assert _run_main(GOOD_ENV, halted=True) == 1


def test_preflight_fails_when_lse_unhealthy():
    assert _run_main(GOOD_ENV, lse_ok=False) == 1
