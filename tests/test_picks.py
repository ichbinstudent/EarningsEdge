"""Tests for the strategy picks engine (earnings_edge/picks.py).

Synthetic DataFrames exercise each filter boundary, sort orders, missing-column
tolerance, and the generate_picks orchestrator against a throwaway SQLite DB.
"""

from __future__ import annotations

import math
import sqlite3
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from earnings_edge.picks import (
    EARNINGS_COLUMNS,
    FORWARD_FACTOR_COLUMNS,
    MOMENTUM_SKEW_COLUMNS,
    VRP_COLUMNS,
    earnings_picks,
    forward_factor_picks,
    generate_picks,
    momentum_skew_picks,
    vrp_picks,
)

AS_OF = date(2026, 8, 20)  # a "today" for window tests
TOMORROW = AS_OF + timedelta(days=1)


# ── earnings_picks -----------------------------------------------------------

def _earnings_row(**over) -> dict:
    row = {
        "ticker": "AAA",
        "announcement_date": TOMORROW,
        "announcement_time": "BMO",
        "is_confirmed": True,
        "implied_move": 8.0,
        "option_volume": 50_000,
        "short_straddle_return": 0.12,
        "short_straddle_win_rate": 0.7,
        "avg_realized_move": 5.0,
        "avg_implied_move": 7.5,
        "implied_vs_avg_realized": 3.0,
        "term_structure_slope": -0.01,
        "historical_events_count": 12,
        "iv_rv": 1.4,
    }
    row.update(over)
    return row


def test_earnings_window_amc_today_and_bmo_tomorrow_pass():
    df = pd.DataFrame([
        _earnings_row(ticker="AMC_TODAY", announcement_date=AS_OF, announcement_time="AMC"),
        _earnings_row(ticker="BMO_TOMORROW", announcement_date=TOMORROW, announcement_time="BMO"),
    ])
    out = earnings_picks(df, as_of=AS_OF)
    assert sorted(out["ticker"]) == ["AMC_TODAY", "BMO_TOMORROW"]


def test_earnings_window_excludes_wrong_slots():
    df = pd.DataFrame([
        _earnings_row(ticker="BMO_TODAY", announcement_date=AS_OF, announcement_time="BMO"),
        _earnings_row(ticker="AMC_TOMORROW", announcement_date=TOMORROW, announcement_time="AMC"),
        _earnings_row(ticker="DAY_AFTER", announcement_date=TOMORROW + timedelta(days=1),
                      announcement_time="BMO"),
    ])
    out = earnings_picks(df, as_of=AS_OF)
    assert len(out) == 0
    # but all spec columns are still present on the empty frame
    assert list(out.columns) == EARNINGS_COLUMNS


def test_earnings_liquidity_boundary_and_missing_tolerance():
    df = pd.DataFrame([
        _earnings_row(ticker="AT_MIN", option_volume=10_000),
        _earnings_row(ticker="BELOW_MIN", option_volume=9_999),
        _earnings_row(ticker="MISSING", option_volume=np.nan),
    ])
    out = earnings_picks(df, as_of=AS_OF)
    assert sorted(out["ticker"]) == ["AT_MIN", "MISSING"]


def test_earnings_liquidity_threshold_configurable():
    df = pd.DataFrame([_earnings_row(option_volume=5_000)])
    assert len(earnings_picks(df, as_of=AS_OF, min_option_volume=10_000)) == 0
    assert len(earnings_picks(df, as_of=AS_OF, min_option_volume=5_000)) == 1


def test_earnings_requires_backwardation():
    df = pd.DataFrame([
        _earnings_row(ticker="BACK", term_structure_slope=-0.001),
        _earnings_row(ticker="FLAT", term_structure_slope=0.0),
        _earnings_row(ticker="CONTANGO", term_structure_slope=0.02),
        _earnings_row(ticker="NAN", term_structure_slope=np.nan),
    ])
    out = earnings_picks(df, as_of=AS_OF)
    assert list(out["ticker"]) == ["BACK"]


def test_earnings_iv_rv_strictly_above_one():
    df = pd.DataFrame([
        _earnings_row(ticker="AT_ONE", iv_rv=1.0),
        _earnings_row(ticker="ABOVE", iv_rv=1.01),
        _earnings_row(ticker="NAN", iv_rv=np.nan),
    ])
    out = earnings_picks(df, as_of=AS_OF)
    assert list(out["ticker"]) == ["ABOVE"]


def test_earnings_sort_short_straddle_return_desc():
    df = pd.DataFrame([
        _earnings_row(ticker="LOW", short_straddle_return=0.02),
        _earnings_row(ticker="HIGH", short_straddle_return=0.30),
        _earnings_row(ticker="MID", short_straddle_return=0.10),
        _earnings_row(ticker="NONE", short_straddle_return=np.nan),
    ])
    out = earnings_picks(df, as_of=AS_OF)
    assert list(out["ticker"]) == ["HIGH", "MID", "LOW", "NONE"]


def test_earnings_negative_straddle_history_filter():
    df = pd.DataFrame([
        _earnings_row(ticker="GOOD", short_straddle_return=0.05),
        _earnings_row(ticker="BAD", short_straddle_return=-0.05),
    ])
    # default: preferred but not required -> both pass
    assert len(earnings_picks(df, as_of=AS_OF)) == 2
    # opt-in hard filter
    out = earnings_picks(df, as_of=AS_OF, require_positive_straddle_return=True)
    assert list(out["ticker"]) == ["GOOD"]


def test_earnings_tolerates_missing_optional_columns():
    df = pd.DataFrame([
        {"ticker": "AAA", "announcement_date": TOMORROW, "announcement_time": "BMO",
         "term_structure_slope": -0.01, "iv_rv": 1.3},
    ])
    out = earnings_picks(df, as_of=AS_OF)
    assert list(out.columns) == EARNINGS_COLUMNS
    assert len(out) == 1
    assert math.isnan(out.iloc[0]["short_straddle_return"])
    assert math.isnan(out.iloc[0]["option_volume"])


def test_earnings_empty_input():
    out = earnings_picks(pd.DataFrame(), as_of=AS_OF)
    assert list(out.columns) == EARNINGS_COLUMNS
    assert len(out) == 0


# ── momentum_skew_picks -------------------------------------------------------

def _skew_row(**over) -> dict:
    row = {
        "ticker": "AAA",
        "next_earnings_date": date(2026, 10, 1),
        "direction": "call",
        "option_volume": 20_000,
        "ts_momentum": 0.15,
        "cs_momentum": 9,
        "relative_momentum": 1.1,
        "skew_value": -0.05,
        "skew_zscore": -2.0,
        "skew_mean": -0.01,
    }
    row.update(over)
    return row


def test_skew_zscore_boundary_inclusive():
    df = pd.DataFrame([
        _skew_row(ticker="AT", skew_zscore=-1.5),
        _skew_row(ticker="ABOVE", skew_zscore=-1.49),
        _skew_row(ticker="BELOW", skew_zscore=-3.0),
        _skew_row(ticker="NAN", skew_zscore=np.nan),
    ])
    out = momentum_skew_picks(df)
    assert sorted(out["ticker"]) == ["AT", "BELOW"]


def test_skew_call_side_requires_decile_8():
    df = pd.DataFrame([
        _skew_row(ticker="D8", direction="call", cs_momentum=8),
        _skew_row(ticker="D7", direction="call", cs_momentum=7),
        _skew_row(ticker="D10", direction="call", cs_momentum=10),
    ])
    out = momentum_skew_picks(df)
    assert sorted(out["ticker"]) == ["D10", "D8"]


def test_skew_put_side_requires_decile_le_3():
    df = pd.DataFrame([
        _skew_row(ticker="D3", direction="put", cs_momentum=3),
        _skew_row(ticker="D4", direction="put", cs_momentum=4),
        _skew_row(ticker="D1", direction="put", cs_momentum=1),
    ])
    out = momentum_skew_picks(df)
    assert sorted(out["ticker"]) == ["D1", "D3"]


def test_skew_volume_boundary_and_missing_tolerance():
    df = pd.DataFrame([
        _skew_row(ticker="AT", option_volume=5_000),
        _skew_row(ticker="BELOW", option_volume=4_999),
        _skew_row(ticker="MISSING", option_volume=np.nan),
    ])
    out = momentum_skew_picks(df)
    assert sorted(out["ticker"]) == ["AT", "MISSING"]


def test_skew_cs_momentum_decile_computed_across_universe():
    # 10 tickers with evenly spaced momentum -> deciles 1..10
    df = pd.DataFrame([
        _skew_row(ticker=f"T{i}", ts_momentum=float(i), cs_momentum=np.nan,
                  direction=None, skew_zscore=-2.0)
        for i in range(1, 11)
    ])
    out_all = momentum_skew_picks(df, min_option_volume=0)
    deciles = dict(zip(out_all["ticker"], out_all["cs_momentum"]))
    # bottom decile names trade put-side, top decile call-side
    assert deciles["T1"] == 1
    assert deciles["T10"] == 10
    assert out_all.loc[out_all["ticker"] == "T10", "direction"].iloc[0] == "call"
    assert out_all.loc[out_all["ticker"] == "T1", "direction"].iloc[0] == "put"
    # middle deciles have no tradeable direction -> excluded
    assert "T5" not in deciles


def test_skew_sort_zscore_asc():
    df = pd.DataFrame([
        _skew_row(ticker="MILD", skew_zscore=-1.6),
        _skew_row(ticker="STEEP", skew_zscore=-3.5),
        _skew_row(ticker="MID", skew_zscore=-2.0),
    ])
    out = momentum_skew_picks(df)
    assert list(out["ticker"]) == ["STEEP", "MID", "MILD"]


def test_skew_missing_relative_momentum_column_tolerated():
    df = pd.DataFrame([_skew_row()])
    df = df.drop(columns=["relative_momentum"])
    out = momentum_skew_picks(df)
    assert list(out.columns) == MOMENTUM_SKEW_COLUMNS
    assert math.isnan(out.iloc[0]["relative_momentum"])


def test_skew_empty_input():
    out = momentum_skew_picks(pd.DataFrame())
    assert list(out.columns) == MOMENTUM_SKEW_COLUMNS
    assert len(out) == 0


# ── forward_factor_picks ------------------------------------------------------

def test_ff_computed_from_variance_decomposition():
    # front 45% / back 40%, T1=30d, T2=60d
    # fwd = sqrt((0.16*60 - 0.2025*30)/30) = sqrt(0.1175) ~ 0.3428
    # FF  = (0.45 - 0.3428)/0.3428 ~ 0.3127
    df = pd.DataFrame([{
        "ticker": "AAA", "next_earnings_date": date(2026, 9, 1),
        "front_iv": 0.45, "back_iv": 0.40, "t1_dte": 30, "t2_dte": 60,
        "option_volume": 12_000,
    }])
    out = forward_factor_picks(df)
    assert len(out) == 1
    expected_fwd = math.sqrt((0.40 ** 2 * 60 - 0.45 ** 2 * 30) / 30)
    expected_ff = (0.45 - expected_fwd) / expected_fwd
    assert out.iloc[0]["forward_factor"] == pytest.approx(expected_ff, rel=1e-6)
    assert expected_ff >= 0.20  # sanity: the fixture is tradeable


def test_ff_boundary_exactly_0_20_passes():
    df = pd.DataFrame([
        {"ticker": "AT", "forward_factor": 0.20},
        {"ticker": "BELOW", "forward_factor": 0.1999},
    ])
    out = forward_factor_picks(df)
    assert list(out["ticker"]) == ["AT"]


def test_ff_threshold_configurable():
    df = pd.DataFrame([{"ticker": "AAA", "forward_factor": 0.15}])
    assert len(forward_factor_picks(df, min_ff=0.20)) == 0
    assert len(forward_factor_picks(df, min_ff=0.10)) == 1


def test_ff_uses_supplied_forward_iv_column():
    df = pd.DataFrame([{
        "ticker": "AAA", "front_iv": 0.48, "forward_iv": 0.40,
    }])
    out = forward_factor_picks(df)
    assert out.iloc[0]["forward_factor"] == pytest.approx(0.20)


def test_ff_degenerate_rows_excluded():
    # back variance smaller than front -> no real forward vol -> no FF
    df = pd.DataFrame([
        {"ticker": "NEG_VAR", "front_iv": 0.60, "back_iv": 0.30,
         "t1_dte": 30, "t2_dte": 60},
        {"ticker": "NAN_IVS", "front_iv": np.nan, "back_iv": np.nan,
         "t1_dte": 30, "t2_dte": 60},
        {"ticker": "GOOD", "forward_factor": 0.25},
    ])
    out = forward_factor_picks(df)
    assert list(out["ticker"]) == ["GOOD"]


def test_ff_sort_desc():
    df = pd.DataFrame([
        {"ticker": "LOW", "forward_factor": 0.21},
        {"ticker": "HIGH", "forward_factor": 0.60},
        {"ticker": "MID", "forward_factor": 0.35},
    ])
    out = forward_factor_picks(df)
    assert list(out["ticker"]) == ["HIGH", "MID", "LOW"]
    assert list(out.columns) == FORWARD_FACTOR_COLUMNS


def test_ff_empty_input():
    out = forward_factor_picks(pd.DataFrame())
    assert list(out.columns) == FORWARD_FACTOR_COLUMNS
    assert len(out) == 0


# ── vrp_picks -----------------------------------------------------------------

def _vrp_row(**over) -> dict:
    row = {
        "ticker": "AAA",
        "iv_pctl_1y": 50.0,
        "iv_rv": 1.2,
        "option_volume": 30_000,
        "next_earnings_date": date(2026, 10, 15),
        "term_structure_slope": 0.01,
        "iron_condor_mean_return": 0.04,
        "iron_condor_win_rate": 0.72,
    }
    row.update(over)
    return row


def test_vrp_iv_pctl_boundary_exclusive():
    df = pd.DataFrame([
        _vrp_row(ticker="BELOW", iv_pctl_1y=79.9),
        _vrp_row(ticker="AT", iv_pctl_1y=80.0),
        _vrp_row(ticker="ABOVE", iv_pctl_1y=95.0),
        _vrp_row(ticker="NAN", iv_pctl_1y=np.nan),  # tolerated: no data source
    ])
    out = vrp_picks(df)
    assert sorted(out["ticker"]) == ["BELOW", "NAN"]


def test_vrp_threshold_configurable():
    df = pd.DataFrame([_vrp_row(iv_pctl_1y=70.0)])
    assert len(vrp_picks(df, max_iv_pctl=80.0)) == 1
    assert len(vrp_picks(df, max_iv_pctl=60.0)) == 0


def test_vrp_requires_contango_or_flat():
    df = pd.DataFrame([
        _vrp_row(ticker="FLAT", term_structure_slope=0.0),
        _vrp_row(ticker="CONTANGO", term_structure_slope=0.02),
        _vrp_row(ticker="BACK", term_structure_slope=-0.001),
        _vrp_row(ticker="NAN", term_structure_slope=np.nan),
    ])
    out = vrp_picks(df)
    assert sorted(out["ticker"]) == ["CONTANGO", "FLAT"]


def test_vrp_sort_iron_condor_mean_return_desc():
    df = pd.DataFrame([
        _vrp_row(ticker="LOW", iron_condor_mean_return=0.01),
        _vrp_row(ticker="HIGH", iron_condor_mean_return=0.09),
        _vrp_row(ticker="NONE", iron_condor_mean_return=np.nan),
    ])
    out = vrp_picks(df)
    assert list(out["ticker"]) == ["HIGH", "LOW", "NONE"]


def test_vrp_backtest_columns_emitted_when_missing():
    df = pd.DataFrame([{"ticker": "AAA", "term_structure_slope": 0.01}])
    out = vrp_picks(df)
    assert list(out.columns) == VRP_COLUMNS
    for structure in ("iron_condor", "short_straddle", "short_strangle", "iron_butterfly"):
        assert math.isnan(out.iloc[0][f"{structure}_mean_return"])
        assert math.isnan(out.iloc[0][f"{structure}_win_rate"])


def test_vrp_empty_input():
    out = vrp_picks(pd.DataFrame())
    assert list(out.columns) == VRP_COLUMNS
    assert len(out) == 0


# ── generate_picks orchestrator ----------------------------------------------

def _build_test_db(path) -> None:
    from earnings_edge.db import engine as db_engine
    db_engine.configure(path)
    conn = sqlite3.connect(path)
    # current snapshot: AMC today, backwardated, IV/RV > 1
    conn.execute(
        "INSERT INTO snapshots (ticker, earnings_date, scan_date, timing, price, "
        "avg_volume_30d, total_open_interest, atm_iv_near, rv30, iv30_rv30, "
        "term_slope, expected_move_pct, actual_move_pct) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        ("AAA", str(AS_OF), str(AS_OF), "Post Market", 100.0, 1e6, 50_000,
         0.8, 0.5, 1.6, -0.01, 9.0, None),
    )
    # historical outcomes for AAA (avg |move| = 5%, 2 events, avg implied 7%)
    for i, (move, implied) in enumerate(((4.0, 6.0), (-6.0, 8.0))):
        conn.execute(
            "INSERT INTO snapshots (ticker, earnings_date, scan_date, timing, price, "
            "avg_volume_30d, total_open_interest, atm_iv_near, rv30, iv30_rv30, "
            "term_slope, expected_move_pct, actual_move_pct) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            ("AAA", f"2026-05-0{i + 1}", f"2026-04-3{i}", "Post Market", 90.0, 1e6, 40_000,
             0.7, 0.5, 1.4, -0.01, implied, move),
        )
    # a contango name for the VRP list (earnings far out, slope >= 0)
    conn.execute(
        "INSERT INTO snapshots (ticker, earnings_date, scan_date, timing, price, "
        "avg_volume_30d, total_open_interest, atm_iv_near, rv30, iv30_rv30, "
        "term_slope, expected_move_pct, actual_move_pct) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        ("BBB", "2026-11-01", str(AS_OF), "Pre Market", 50.0, 1e6, 20_000,
         0.4, 0.35, 1.15, 0.01, 4.0, None),
    )
    # ff row with precomputed sigma_fwd: FF = (0.48 - 0.40)/0.40 = 0.20
    conn.execute(
        "INSERT INTO ff_snapshots (ticker, scan_date, earnings_date, t1_iv, t2_iv, "
        "t1_dte, t2_dte, sigma_fwd, implied_event_move_pct, hist_rms_move_pct, "
        "premium_ratio) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        ("AAA", str(AS_OF), "2026-08-21", 0.48, 0.42, 30, 60, 0.40, 6.0, 5.0, 1.2),
    )
    # ff row without sigma_fwd -> computed via variance decomposition
    conn.execute(
        "INSERT INTO ff_snapshots (ticker, scan_date, earnings_date, t1_iv, t2_iv, "
        "t1_dte, t2_dte, sigma_fwd, implied_event_move_pct, hist_rms_move_pct, "
        "premium_ratio) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        ("CCC", str(AS_OF), "2026-09-10", 0.45, 0.40, 30, 60, None, 6.0, 5.0, None),
    )
    conn.commit()
    conn.close()


def test_generate_picks_against_read_only_db(tmp_path):
    db = tmp_path / "test.db"
    _build_test_db(db)
    conn = sqlite3.connect(db)
    n_before = conn.execute("SELECT COUNT(*) FROM snapshots").fetchone()[0]
    conn.close()
    picks = generate_picks(AS_OF)
    conn = sqlite3.connect(db)
    n_after = conn.execute("SELECT COUNT(*) FROM snapshots").fetchone()[0]
    conn.close()
    assert n_before == n_after

    assert set(picks) == {"earnings", "momentum_skew", "forward_factor", "vrp"}

    earnings = picks["earnings"]
    assert list(earnings["ticker"]) == ["AAA"]
    row = earnings.iloc[0]
    assert row["announcement_time"] == "AMC"  # Post Market -> AMC
    assert row["implied_move"] == pytest.approx(9.0)
    assert row["avg_realized_move"] == pytest.approx(5.0)
    assert row["avg_implied_move"] == pytest.approx(7.0)
    assert row["implied_vs_avg_realized"] == pytest.approx(4.0)  # 9.0 - 5.0
    assert row["historical_events_count"] == 2
    assert math.isnan(row["option_volume"])  # no option-volume source in the DB

    ff = picks["forward_factor"]
    assert list(ff["ticker"]) == ["CCC", "AAA"]  # CCC FF ~0.31 > AAA 0.20, sorted desc
    assert ff.loc[ff["ticker"] == "AAA", "forward_factor"].iloc[0] == pytest.approx(0.20)

    vrp = picks["vrp"]
    assert "BBB" in set(vrp["ticker"])
    assert math.isnan(vrp.iloc[0]["iv_pctl_1y"])  # no IV-percentile source

    # no momentum/skew data in the DB -> empty frame with the right columns
    assert len(picks["momentum_skew"]) == 0
    assert list(picks["momentum_skew"].columns) == MOMENTUM_SKEW_COLUMNS


def test_generate_picks_tolerates_missing_tables(tmp_path):
    from earnings_edge.db import engine as db_engine
    db = tmp_path / "empty.db"
    db_engine.configure(db)
    picks = generate_picks(AS_OF)
    assert set(picks) == {"earnings", "momentum_skew", "forward_factor", "vrp"}
    for df in picks.values():
        assert isinstance(df, pd.DataFrame)


def test_generate_picks_consumes_daily_signals(tmp_path):
    """daily_signals rows populate option_volume, iv_pctl_1y, momentum, skew."""
    db = tmp_path / "test.db"
    _build_test_db(db)
    conn = sqlite3.connect(db)
    ds_insert = (
        "INSERT INTO daily_signals (ticker, signal_date, option_volume, atm_iv, "
        "skew_25d, skew_zscore, skew_mean, iv_pctl_1y, ts_momentum, relative_momentum) "
        "VALUES (?,?,?,?,?,?,?,?,?,?)"
    )
    # 25 days of volume history for AAA -> 20d avg = 20000; latest-day signals
    for i in range(25):
        conn.execute(
            ds_insert,
            ("AAA", f"2026-07-{i+1:02d}", 20000.0, None, None, None, None, None, None, None),
        )
    conn.execute(
        ds_insert,
        ("AAA", str(AS_OF), 20000.0, 0.8, 0.05, -2.0, 0.04, 55.0, 0.35, 1.1),
    )
    conn.execute(
        ds_insert,
        ("BBB", str(AS_OF), 8000.0, 0.4, 0.03, -1.8, 0.03, 42.0, -0.20, 0.9),
    )
    conn.commit()
    conn.close()

    picks = generate_picks(AS_OF)

    earnings = picks["earnings"]
    assert earnings.iloc[0]["option_volume"] == pytest.approx(20000.0)

    vrp = picks["vrp"]
    bbb = vrp.loc[vrp["ticker"] == "BBB"].iloc[0]
    assert bbb["iv_pctl_1y"] == pytest.approx(42.0)
    assert bbb["option_volume"] == pytest.approx(8000.0)

    # AAA: skew_zscore -2.0 <= -1.5, ts_momentum 0.35 (top decile vs BBB -0.20),
    # option_volume 20000 >= 5000 -> a call-side momentum-skew pick
    ms = picks["momentum_skew"]
    assert list(ms["ticker"]) == ["AAA"]
    row = ms.iloc[0]
    assert row["direction"] == "call"
    assert row["skew_zscore"] == pytest.approx(-2.0)
    assert row["skew_value"] == pytest.approx(0.05)
    assert row["relative_momentum"] == pytest.approx(1.1)
    assert row["cs_momentum"] == pytest.approx(10.0)  # top decile of 2-ticker universe


# ── persistence -----------------------------------------------------------------

def test_persist_and_load_picks_roundtrip(tmp_path):
    from earnings_edge.db import engine as db_engine
    from earnings_edge.picks import load_picks, persist_picks
    db_engine.configure(tmp_path / "p.db")
    df = pd.DataFrame({
        "ticker": ["AAA", "BBB"],
        "forward_factor": [0.31, 0.20],
        "option_volume": [20000.0, float("nan")],
    })
    n = persist_picks({"forward_factor": df}, AS_OF)
    assert n == 2

    loaded = load_picks(str(AS_OF))
    assert len(loaded) == 2
    assert list(loaded["ticker"]) == ["AAA", "BBB"]  # rank order
    import json
    sig = json.loads(loaded.iloc[1]["signals_json"])
    assert sig["forward_factor"] == pytest.approx(0.20)
    assert sig["option_volume"] is None  # NaN serialized as null

    # re-persisting the same day replaces, not duplicates
    n2 = persist_picks({"forward_factor": df}, AS_OF)
    assert n2 == 2
    assert len(load_picks(str(AS_OF))) == 2
    assert len(load_picks(str(AS_OF), strategy="earnings")) == 0


def test_persist_picks_empty():
    from earnings_edge.picks import persist_picks
    assert persist_picks({"earnings": pd.DataFrame()}, AS_OF) == 0
