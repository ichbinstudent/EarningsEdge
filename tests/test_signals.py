"""Tests for the daily per-ticker signal computations (earnings_edge.signals)."""

from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

from earnings_edge.signals import (
    compute_chain_signals,
    compute_iv_percentile,
    compute_ts_momentum,
    compute_zscore,
    relative_momentum,
)


def _chain_row(expiry: str, strike: float, ctype: str, *, vol: float = 100,
               iv: float = 0.30, delta: float = 0.50) -> dict:
    return {
        "expiry": expiry, "strike": strike, "contract_type": ctype,
        "volume": vol, "implied_volatility": iv, "delta": delta,
    }


def _sample_chain() -> pd.DataFrame:
    rows = [
        # front expiry inside min_dte -> ignored
        _chain_row("2026-08-25", 100, "call", vol=1000, iv=0.90, delta=0.50),
        # usable expiry 2026-09-25 (34d out from AS_OF)
        _chain_row("2026-09-25", 100, "call", vol=500, iv=0.40, delta=0.52),  # ATM-ish
        _chain_row("2026-09-25", 105, "call", vol=200, iv=0.30, delta=0.25),  # 25d call
        _chain_row("2026-09-25", 95, "put", vol=300, iv=0.45, delta=-0.26),   # 25d put
        _chain_row("2026-09-25", 90, "put", vol=50, iv=0.60, delta=-0.10),
        # later expiry also usable but front one wins
        _chain_row("2026-12-18", 100, "call", vol=10, iv=0.35, delta=0.50),
    ]
    return pd.DataFrame(rows)


AS_OF = "2026-08-22"


def test_chain_signals_option_volume_sums_usable_expiries():
    sig = compute_chain_signals(_sample_chain(), as_of=AS_OF)
    # all rows count toward volume (liquidity is liquidity)
    assert sig["option_volume"] == 1000 + 500 + 200 + 300 + 50 + 10


def test_chain_signals_atm_iv_from_nearest_usable_expiry():
    sig = compute_chain_signals(_sample_chain(), as_of=AS_OF, min_dte=21)
    # the 0.90-IV front-expiry row (3d out) must be ignored; ATM = 0.52-delta row
    assert sig["atm_iv"] == pytest.approx(0.40)


def test_chain_signals_skew_25d_put_minus_call():
    sig = compute_chain_signals(_sample_chain(), as_of=AS_OF, min_dte=21)
    # skew = IV(25d put) - IV(25d call) = 0.45 - 0.30
    assert sig["skew_25d"] == pytest.approx(0.15)


def test_chain_signals_empty_chain():
    sig = compute_chain_signals(pd.DataFrame(), as_of=AS_OF)
    assert sig["option_volume"] is None
    assert sig["atm_iv"] is None
    assert sig["skew_25d"] is None


def test_iv_percentile_known_history():
    hist = [0.01 * i for i in range(1, 101)]  # 0.01 .. 1.00
    assert compute_iv_percentile(hist, 0.505) == pytest.approx(50.0)
    assert compute_iv_percentile(hist, 2.0) == pytest.approx(100.0)
    assert compute_iv_percentile(hist, 0.001) == pytest.approx(0.0)


def test_iv_percentile_insufficient_history():
    assert compute_iv_percentile([0.3] * 5, 0.3) is None


def test_zscore_known_history():
    hist = [1.0, 2.0, 3.0, 4.0, 5.0] * 5
    z, mean = compute_zscore(hist, 5.0)
    assert mean == pytest.approx(3.0)
    assert z == pytest.approx((5.0 - 3.0) / pd.Series(hist).std(ddof=1))


def test_zscore_insufficient_history():
    assert compute_zscore([1.0, 2.0], 2.0) == (None, None)


def test_ts_momentum_twelve_minus_one():
    # 252 bars rising 0.1% per day; momentum skips the 22 most recent bars
    bars = [{"c": 100.0 * (1.001 ** i)} for i in range(252)]
    mom = compute_ts_momentum(bars)
    expected = bars[-23]["c"] / bars[0]["c"] - 1.0
    assert mom == pytest.approx(expected, rel=1e-9)


def test_ts_momentum_too_few_bars():
    assert compute_ts_momentum([{"c": 1.0}] * 30) is None


def test_relative_momentum_ratio():
    # ticker +20%, benchmark +10% -> 1.2/1.1 > 1 (outperform)
    assert relative_momentum(0.20, 0.10) == pytest.approx(1.2 / 1.1)
    assert relative_momentum(None, 0.10) is None
    assert relative_momentum(0.10, None) is None


def test_chain_signals_tolerates_bad_expiry_strings():
    df = pd.DataFrame([
        _chain_row("not-a-date", 100, "call", vol=10, iv=0.9, delta=0.5),
        _chain_row(None, 100, "call", vol=10, iv=0.9, delta=0.5),
        _chain_row("2026-09-25", 100, "call", vol=10, iv=0.4, delta=0.5),
    ])
    sig = compute_chain_signals(df, as_of=AS_OF)
    assert sig["atm_iv"] == pytest.approx(0.4)  # bad rows skipped, not fatal


def test_enrich_chain_with_bs_fills_missing_iv_and_delta():
    from earnings_edge.signals import enrich_chain_with_bs
    df = pd.DataFrame([
        # missing IV/delta but has midpoint -> backfilled from BSM
        _chain_row("2026-09-25", 100, "call", vol=10, iv=None, delta=None) | {"midpoint": 5.0},
        # IV present -> untouched
        _chain_row("2026-09-25", 105, "call", vol=10, iv=0.33, delta=0.25),
    ])
    out = enrich_chain_with_bs(df, spot=100.0, r=0.045, as_of=AS_OF)
    iv0 = out.loc[0, "implied_volatility"]
    d0 = out.loc[0, "delta"]
    assert iv0 is not None and 0.05 < iv0 < 1.5  # solved from the $5 mid
    assert d0 is not None and 0.3 < d0 < 0.8     # near-ATM call
    # the $5 mid must round-trip through BSM at the solved IV
    from earnings_edge.option_math import black_scholes_price
    T = 34 / 365
    assert black_scholes_price(100.0, 100.0, T, 0.045, iv0, "call") == pytest.approx(5.0, abs=0.01)
    assert out.loc[1, "implied_volatility"] == pytest.approx(0.33)  # preserved


def test_enrich_chain_with_bs_no_midpoint_leaves_nulls():
    from earnings_edge.signals import enrich_chain_with_bs
    df = pd.DataFrame([_chain_row("2026-09-25", 100, "call", vol=10, iv=None, delta=None)])
    out = enrich_chain_with_bs(df, spot=100.0, r=0.045, as_of=AS_OF)
    assert out.loc[0, "implied_volatility"] is None


# ── contract market lookup (designer chain integration) ------------------------

def _chain_db(tmp_path):
    conn = sqlite3.connect(tmp_path / "c.db")
    conn.executescript("""
        CREATE TABLE options_chain (
            ticker TEXT, scan_date TEXT, contract_ticker TEXT, expiry TEXT,
            strike REAL, contract_type TEXT, volume REAL,
            implied_volatility REAL, delta REAL, midpoint REAL, close REAL
        );
    """)
    conn.execute(
        "INSERT INTO options_chain VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        ("XYZ", "2026-08-21", "XYZ260918C00100000", "2026-09-18", 100.0, "call",
         100, None, None, 5.10, 5.0),
    )
    conn.execute(
        "INSERT INTO options_chain VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        ("XYZ", "2026-08-21", "XYZ260918C00105000", "2026-09-18", 105.0, "call",
         50, 0.28, 0.25, 2.40, 2.35),
    )
    conn.commit()
    return conn


def test_contract_market_solves_iv_from_mid(tmp_path):
    from earnings_edge.signals import contract_market
    conn = _chain_db(tmp_path)
    m = contract_market(conn, "XYZ", "call", 100.0, "2026-09-18",
                        spot=100.0, r=0.045, as_of="2026-08-22")
    assert m is not None
    assert m["price"] == pytest.approx(5.10)
    assert 0.05 < m["iv"] < 1.5  # solved, not null
    from earnings_edge.option_math import black_scholes_price
    T = 27 / 365
    assert black_scholes_price(100.0, 100.0, T, 0.045, m["iv"], "call") == pytest.approx(5.10, abs=0.01)


def test_contract_market_prefers_stored_iv(tmp_path):
    from earnings_edge.signals import contract_market
    conn = _chain_db(tmp_path)
    m = contract_market(conn, "XYZ", "call", 105.0, "2026-09-18",
                        spot=100.0, r=0.045, as_of="2026-08-22")
    assert m["iv"] == pytest.approx(0.28)


def test_contract_market_missing_contract(tmp_path):
    from earnings_edge.signals import contract_market
    conn = _chain_db(tmp_path)
    assert contract_market(conn, "XYZ", "put", 100.0, "2026-09-18",
                           spot=100.0, r=0.045, as_of="2026-08-22") is None
