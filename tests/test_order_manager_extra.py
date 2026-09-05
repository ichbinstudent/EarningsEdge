from unittest.mock import MagicMock

import pytest

from framework.execution.order_manager import (
    LimitWalkPolicy,
    MidPricePolicy,
    OrderManager,
    PricingPolicy,
)


def test_pricing_policy_base():
    with pytest.raises(NotImplementedError):
        PricingPolicy().walk(1.0, "buy")


def test_limit_walk_policy_stepping():
    with pytest.raises(ValueError):
        LimitWalkPolicy(steps=0)

    # Check stepping buy and sell
    pol = LimitWalkPolicy(steps=3, step_improve_bps=0.0, final_improve_bps=100.0)  # 0 to 1%
    mid = 100.0
    # Buy: walk UP. 100, 100.5, 101.0
    walk = pol.walk(mid, "buy")
    assert walk == [100.0, 100.5, 101.0]

    # Sell: walk DOWN. 100, 99.5, 99.0
    walk2 = pol.walk(mid, "sell")
    assert walk2 == [100.0, 99.5, 99.0]


def test_submit_refuses_no_limit_price():
    manager = OrderManager(client=MagicMock(), poll_secs=0, sleep=lambda x: None)
    with pytest.raises(ValueError, match="no limit price"):
        manager._submit([{"symbol": "A"}], 1, None, "cid", "day")


def test_submit_single_leg():
    client = MagicMock()
    client.submit_order.return_value = {"id": "oid1"}
    manager = OrderManager(client=client, poll_secs=0, sleep=lambda x: None)
    oid = manager._submit([{"symbol": "A", "side": "buy"}], 2, 10.0, "cid", "day")
    assert oid == "oid1"
    client.submit_order.assert_called_once()


def test_submit_api_retry_and_fail():
    client = MagicMock()
    client.submit_order.side_effect = Exception("API error")
    manager = OrderManager(client=client, poll_secs=0, sleep=lambda x: None)
    with pytest.raises(Exception, match="API error"):
        manager._submit([{"symbol": "A", "side": "buy"}], 2, 10.0, "cid", "day")
    assert client.submit_order.call_count == 3


def test_poll_fill_calculates_net_price():
    client = MagicMock()
    # No avg price on parent, but legs have it
    client.get_order.return_value = {
        "status": "filled",
        "filled_qty": 1,
        "filled_avg_price": None,
        "legs": [
            {"side": "buy", "filled_qty": 1, "filled_avg_price": 5.0, "ratio_qty": 1},
            {"side": "sell", "filled_qty": 1, "filled_avg_price": 3.0, "ratio_qty": 1},
        ],
    }
    manager = OrderManager(client=client, poll_secs=0, sleep=lambda x: None)
    filled, avg = manager._poll_fill("oid")
    assert filled == 1
    assert avg == 2.0  # buy 5, sell 3 -> net 2


def test_cancel_open_exception():
    client = MagicMock()
    client.get_order.return_value = {"status": "working"}
    client.cancel_order.side_effect = Exception("cancel error")
    manager = OrderManager(client=client, poll_secs=0, sleep=lambda x: None)
    manager._cancel_open("oid")  # Should swallow after 3 attempts
    assert client.cancel_order.call_count == 3


def test_managed_order_lifecycle():
    client = MagicMock()
    # 1. First poll fails get_order (exception swallowed, retries)
    # 2. Second poll returns partial, third returns filled
    client.get_order.side_effect = [
        Exception("temporary"),
        {"status": "filled", "filled_qty": 10, "filled_avg_price": 5.5},
    ]
    client.submit_multi_leg_order.return_value = {"id": "oid1"}

    manager = OrderManager(client=client, poll_secs=0, sleep=lambda x: None)
    mo = manager.execute(
        legs=[
            {"symbol": "A", "side": "buy", "ratio_qty": 1},
            {"symbol": "B", "side": "sell", "ratio_qty": 1},
        ],
        qty=10,
        policy=MidPricePolicy(),
        quote_fn=lambda: 5.5,
    )
    assert mo.state == "filled"
    assert mo.filled_qty == 10
    assert mo.filled_avg_price == 5.5
    assert mo.order_ids == ["oid1"]


def test_managed_order_no_quote():
    manager = OrderManager(client=MagicMock(), poll_secs=0, sleep=lambda x: None)
    mo = manager.execute([{"symbol": "A"}], 1, MidPricePolicy(), lambda: None)
    assert mo.state == "error"
    assert mo.detail == "no quote"
