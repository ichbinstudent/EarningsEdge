"""Daily-signal input regressions: hourly chain captures, delta tolerances,
chain-cache pagination and mid/greeks mapping."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime

import pandas as pd
import pytest

from earnings_edge.chain_cache import fetch_chain, row_for_contract
from earnings_edge.db import configure, insert_options_chain_rows, options_chain_df_latest
from earnings_edge.signals import compute_chain_signals, contract_market

CALL = "AAPL260918C00200000"
PUT = "AAPL260918P00200000"


def _snap(bid, ask, vol, **extra):
    return {
        "dailyBar": {"c": 5.0, "o": 5.0, "h": 5.0, "l": 5.0, "n": 1, "v": vol, "vw": 5.0},
        "latestQuote": {"bp": bid, "ap": ask, "bs": 1, "as": 1},
        **extra,
    }


def _capture(hour: int, contracts: dict) -> list[dict]:
    now = datetime(2026, 8, 21, hour, 0, tzinfo=UTC)
    return [row_for_contract(f"run{hour}", "AAPL", sym, snap, now=now) for sym, snap in contracts.items()]


# ── options_chain_df_latest ───────────────────────────────────────────────


def test_latest_chain_is_one_row_per_contract_at_its_latest_capture(tmp_path):
    """Two hourly captures of the same day used to come back together, so the
    cumulative dailyBar volume was counted once per capture."""
    configure(tmp_path / "ml.db")
    insert_options_chain_rows(_capture(14, {CALL: _snap(4.9, 5.1, 10), PUT: _snap(4.8, 5.0, 7)}))
    # 15:00 capture only re-saw the call; the put keeps its 14:00 row
    insert_options_chain_rows(_capture(15, {CALL: _snap(5.9, 6.1, 25)}))

    df = options_chain_df_latest("AAPL", "2026-08-21")
    assert len(df) == 2
    by_type = df.set_index("contract_type")
    assert by_type.loc["call", "volume"] == 25
    assert by_type.loc["call", "midpoint"] == pytest.approx(6.0)
    assert by_type.loc["put", "volume"] == 7

    sig = compute_chain_signals(df, as_of="2026-08-21")
    assert sig["option_volume"] == 32  # not 10 + 25 + 7


def test_contract_market_uses_latest_hourly_capture(tmp_path):
    db = tmp_path / "ml.db"
    configure(db)
    insert_options_chain_rows(_capture(15, {CALL: _snap(5.9, 6.1, 25)}))
    insert_options_chain_rows(_capture(14, {CALL: _snap(4.9, 5.1, 10)}))  # inserted later, older hour
    conn = sqlite3.connect(db)
    m = contract_market(conn, "AAPL", "call", 200.0, "2026-09-18", spot=200.0, as_of="2026-08-21")
    assert m is not None
    assert m["price"] == pytest.approx(6.0)


# ── compute_chain_signals tolerances ──────────────────────────────────────


def _chain(rows):
    return pd.DataFrame(
        [
            {
                "expiry": "2026-09-25",
                "strike": k,
                "contract_type": t,
                "volume": 1,
                "implied_volatility": iv,
                "delta": d,
            }
            for k, t, iv, d in rows
        ]
    )


def test_skew_refused_when_no_strike_near_25_delta():
    """Only near-ATM strikes quoted: the old code used them as the 25d wings
    and reported a ~0 skew (a fake 'flat skew' for momentum-skew picks)."""
    sig = compute_chain_signals(
        _chain([(100, "call", 0.30, 0.52), (100, "put", 0.31, -0.48)]), as_of="2026-08-22"
    )
    assert sig["atm_iv"] == pytest.approx(0.30)
    assert sig["skew_25d"] is None


def test_atm_refused_when_only_wings_are_quoted():
    sig = compute_chain_signals(
        _chain([(120, "call", 0.45, 0.20), (80, "put", 0.55, -0.22)]), as_of="2026-08-22"
    )
    assert sig["atm_iv"] is None


def test_placeholder_iv_rows_are_not_usable():
    sig = compute_chain_signals(
        _chain([(100, "call", 1e-5, 0.50), (101, "call", 0.33, 0.44)]), as_of="2026-08-22"
    )
    assert sig["atm_iv"] == pytest.approx(0.33)


# ── chain cache rows ──────────────────────────────────────────────────────


@pytest.mark.parametrize("bid,ask", [(0.0, 1.2), (1.3, 1.2), (None, 1.2)])
def test_no_midpoint_from_zero_bid_or_crossed_quote(bid, ask):
    row = row_for_contract("r", "AAPL", CALL, _snap(bid, ask, 1))
    assert row["midpoint"] is None


def test_snapshot_iv_and_greeks_are_kept_when_present():
    snap = _snap(1.0, 1.2, 1, impliedVolatility=0.42, greeks={"delta": 0.51, "gamma": 0.02, "vega": 0.3})
    row = row_for_contract("r", "AAPL", CALL, snap)
    assert row["midpoint"] == pytest.approx(1.1)
    assert (row["implied_volatility"], row["delta"], row["gamma"], row["vega"]) == (0.42, 0.51, 0.02, 0.3)
    assert row["theta"] is None


class _PagedClient:
    def __init__(self, pages):
        self.pages = pages
        self.tokens = []

    def chain_snapshot(self, underlying, page_token=None):
        self.tokens.append(page_token)
        idx = 0 if page_token is None else int(page_token)
        nxt = str(idx + 1) if idx + 1 < len(self.pages) else None
        return {"snapshots": self.pages[idx]}, nxt


def test_fetch_chain_follows_every_page():
    """Only the first page (front expiries) used to be stored."""
    client = _PagedClient([{"A": {}}, {"B": {}}, {"C": {}}])
    contracts, calls = fetch_chain(client, "AAPL")
    assert set(contracts) == {"A", "B", "C"}
    assert calls == 3
    assert client.tokens == [None, "1", "2"]


def test_fetch_chain_stops_at_page_cap():
    class Looping:
        def chain_snapshot(self, underlying, page_token=None):
            return {"snapshots": {"X": {}}}, "same-token"

    contracts, calls = fetch_chain(Looping(), "AAPL", max_pages=4)
    assert calls == 4 and contracts == {"X": {}}
