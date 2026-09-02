"""Strategy picks engine — replication of oquants.com's five daily pick lists.

A "pick" is a per-ticker row of signal values; structures/sizing are fixed
playbook rules, not data (see docs/oquants_feature_inventory.md §1). Four of
the five lists are implemented here as pure DataFrame functions:

  - earnings_picks        /plays/earnings           (short earnings vol)
  - momentum_skew_picks   /plays/momentum-skew
  - forward_factor_picks  /plays/forward-factors    (FF >= 0.20, ex-earnings)
  - vrp_picks             /plays/vrp-stock          (volatility risk premium)

(The fifth, /plays/pre-earnings-long-vol, is a model-classification output and
lives with the ML stack, not here.)

generate_picks(as_of) maps what the local earnings_ml.db actually holds
onto those signals. Documented data-source assumptions:

  - option_volume: 20-day average of daily_signals.option_volume (populated by
    scripts/collect_daily_signals.py from the options_chain table). NaN when
    the collector hasn't run — liquidity filters tolerate NaN (pass).
  - iv_pctl_1y / skew_zscore: from daily_signals; these need >= 20 days of
    accrued collector history, so they are NaN until then (filters exclude or
    tolerate per playbook, documented per function).
  - momentum/skew inputs: from daily_signals (ts_momentum via Polygon daily
    bars, skew_25d from the persisted options chain).
  - short_straddle_return / _win_rate (earnings) and all per-structure
    backtest stats (vrp): no per-structure backtest store exists yet.
  - is_confirmed: taken as "timing is known" (finnhub-sourced calendar).
  - avg_realized_move / avg_implied_move / historical_events_count: computed
    from the snapshots table's own outcome history (rows with actual_move_pct).
  - implied_vs_avg_realized = implied_move - avg_realized_move (percentage
    points), mirroring the dashboard's "Implied Vs. Actual" definition.

All pick functions are pure — no I/O — and tolerate missing optional columns
(emitted as NaN). generate_picks only ever runs SELECTs via the engine.
"""

from __future__ import annotations

import math
from datetime import date, datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from .fwd_factor import forward_iv

# ── Output schemas (column order mirrors the oquants tables) ────────────────

EARNINGS_COLUMNS = [
    "ticker", "announcement_date", "announcement_time", "is_confirmed",
    "implied_move", "option_volume", "short_straddle_return",
    "short_straddle_win_rate", "avg_realized_move", "avg_implied_move",
    "implied_vs_avg_realized", "term_structure_slope", "historical_events_count",
]

MOMENTUM_SKEW_COLUMNS = [
    "ticker", "next_earnings_date", "direction", "option_volume",
    "cs_momentum", "ts_momentum", "relative_momentum",
    "skew_value", "skew_zscore", "skew_mean",
]

FORWARD_FACTOR_COLUMNS = [
    "ticker", "next_earnings_date", "forward_factor", "option_volume",
]

VRP_STRUCTURES = ["iron_condor", "short_straddle", "short_strangle", "iron_butterfly"]

VRP_COLUMNS = [
    "ticker", "iv_pctl_1y", "iv_rv", "option_volume", "next_earnings_date",
] + [f"{s}_{stat}" for s in VRP_STRUCTURES for stat in ("mean_return", "win_rate")]

# DB timing strings -> oquants BMO/AMC enum
_TIMING_MAP = {"Pre Market": "BMO", "Post Market": "AMC"}


# ── helpers -------------------------------------------------------------------

def _with_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Return df with every column in *columns* present (missing -> NaN)."""
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            out[col] = np.nan
    return out


def _empty(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in columns})


def _as_date(series: pd.Series) -> pd.Series:
    """Coerce a str/date/datetime column to python dates (NaT stays NaT)."""
    return pd.to_datetime(series, errors="coerce").dt.date


# ── /plays/earnings — short earnings vol --------------------------------------

def earnings_picks(
    df: pd.DataFrame,
    as_of: date,
    min_option_volume: float = 10_000,
    min_iv_rv: float = 1.0,
    require_positive_straddle_return: bool = False,
) -> pd.DataFrame:
    """Short-earnings-vol picks (iron-fly playbook), sorted by straddle return.

    Input columns: the EARNINGS_COLUMNS schema plus ``iv_rv`` (IV/RV ratio,
    filter input only). Filters:
      - announcement between today's close and tomorrow's open:
        AMC on *as_of* or BMO on *as_of* + 1 calendar day;
      - option_volume >= min_option_volume (NaN tolerated — unknown liquidity);
      - term_structure_slope < 0 (front-month backwardation; NaN excluded);
      - iv_rv strictly > min_iv_rv (NaN excluded);
      - require_positive_straddle_return: the playbook only *prefers* names
        with losing long-straddle history, so this is opt-in.
    """
    cols = EARNINGS_COLUMNS
    if df.empty:
        return _empty(cols)
    df = _with_columns(df, cols + ["iv_rv"])

    ann_date = _as_date(df["announcement_date"])
    ann_time = df["announcement_time"].astype("string").str.upper()
    tomorrow = as_of + timedelta(days=1)
    in_window = ((ann_date == as_of) & (ann_time == "AMC")) | \
                ((ann_date == tomorrow) & (ann_time == "BMO"))

    vol = pd.to_numeric(df["option_volume"], errors="coerce")
    liquid = vol.isna() | (vol >= min_option_volume)

    slope = pd.to_numeric(df["term_structure_slope"], errors="coerce")
    backwardated = slope < 0

    iv_rv = pd.to_numeric(df["iv_rv"], errors="coerce")
    rich_iv = iv_rv > min_iv_rv

    mask = in_window & liquid & backwardated & rich_iv
    if require_positive_straddle_return:
        mask &= pd.to_numeric(df["short_straddle_return"], errors="coerce") > 0

    out = df.loc[mask, cols].copy()
    out["_sort"] = pd.to_numeric(out["short_straddle_return"], errors="coerce")
    out = out.sort_values("_sort", ascending=False, na_position="last")
    return out.drop(columns="_sort").reset_index(drop=True)


# ── /plays/momentum-skew -------------------------------------------------------

def _cs_momentum_decile(ts_momentum: pd.Series) -> pd.Series:
    """Cross-sectional momentum decile (1-10) across the input universe.

    Rank-based (robust to duplicate values); NaN momentum -> NaN decile.
    """
    mom = pd.to_numeric(ts_momentum, errors="coerce")
    decile = np.ceil(mom.rank(pct=True) * 10).clip(1, 10)
    return decile.where(mom.notna())


def momentum_skew_picks(
    df: pd.DataFrame,
    min_option_volume: float = 5_000,
    max_skew_zscore: float = -1.5,
    call_decile_min: int = 8,
    put_decile_max: int = 3,
) -> pd.DataFrame:
    """Momentum-skew picks (vertical debit spread playbook), sorted zscore asc.

    Input columns: the MOMENTUM_SKEW_COLUMNS schema. ``cs_momentum`` is used
    when supplied, otherwise computed as the cross-sectional decile of
    ``ts_momentum`` across the input universe. ``direction`` ("call"/"put") is
    used when supplied, otherwise derived from the decile (>= call_decile_min
    -> call, <= put_decile_max -> put). ``relative_momentum`` (vs SPY) is a
    pass-through and is NaN when absent.

    Criteria: skew_zscore <= max_skew_zscore; decile >= 8 for call-side,
    <= 3 for put-side; option_volume >= min_option_volume (NaN tolerated).
    """
    cols = MOMENTUM_SKEW_COLUMNS
    if df.empty:
        return _empty(cols)
    df = _with_columns(df, cols)

    cs = pd.to_numeric(df["cs_momentum"], errors="coerce")
    cs = cs.where(cs.notna(), _cs_momentum_decile(df["ts_momentum"]))
    df["cs_momentum"] = cs

    direction = df["direction"].astype("string").str.lower()
    derived = pd.Series(pd.NA, index=df.index, dtype="string")
    derived = derived.mask(cs >= call_decile_min, "call")
    derived = derived.mask(cs <= put_decile_max, "put")
    direction = direction.where(direction.notna(), derived)
    df["direction"] = direction

    z = pd.to_numeric(df["skew_zscore"], errors="coerce")
    steep = z <= max_skew_zscore

    vol = pd.to_numeric(df["option_volume"], errors="coerce")
    liquid = vol.isna() | (vol >= min_option_volume)

    side_ok = ((direction == "call") & (cs >= call_decile_min)) | \
              ((direction == "put") & (cs <= put_decile_max))

    out = df.loc[steep & liquid & side_ok, cols].copy()
    out["_sort"] = pd.to_numeric(out["skew_zscore"], errors="coerce")
    out = out.sort_values("_sort", ascending=True, na_position="last")
    return out.drop(columns="_sort").reset_index(drop=True)


# ── /plays/forward-factors -----------------------------------------------------

def _num(value) -> float:
    """Scalar -> float, NaN on None/NaN/non-numeric."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _row_forward_factor(row: pd.Series) -> float:
    """FF = (front_IV - forward_IV) / forward_IV for one row.

    Resolution order: explicit ``forward_factor`` column; explicit
    ``forward_iv`` column; variance decomposition of front/back IVs over
    t1_dte/t2_dte via fwd_factor.forward_iv. NaN when not computable (e.g.
    negative forward variance — earnings event inside the front expiry).
    """
    ff = _num(row.get("forward_factor"))
    if not math.isnan(ff):
        return ff
    front = _num(row.get("front_iv"))
    if math.isnan(front) or front <= 0:
        return np.nan
    fwd = _num(row.get("forward_iv"))
    if math.isnan(fwd):
        back, t1, t2 = (_num(row.get(c)) for c in ("back_iv", "t1_dte", "t2_dte"))
        if math.isnan(back) or math.isnan(t1) or math.isnan(t2):
            return np.nan
        fwd = forward_iv(front, t1 / 365.0, back, t2 / 365.0)
        if fwd is None:
            return np.nan
    if fwd <= 0:
        return np.nan
    return (front - fwd) / fwd


def forward_factor_picks(
    df: pd.DataFrame,
    min_ff: float = 0.20,
) -> pd.DataFrame:
    """Forward-factor calendar picks, sorted FF desc.

    Input columns: FORWARD_FACTOR_COLUMNS plus any of the FF inputs
    (``forward_factor``, or ``front_iv`` + ``forward_iv``, or ``front_iv`` /
    ``back_iv`` / ``t1_dte`` / ``t2_dte``). Keeps rows with FF >= min_ff
    (boundary inclusive — FF exactly 0.20 is tradeable).
    """
    cols = FORWARD_FACTOR_COLUMNS
    if df.empty:
        return _empty(cols)
    df = _with_columns(df, cols)
    df["forward_factor"] = df.apply(_row_forward_factor, axis=1)

    # epsilon absorbs float noise so a computed FF of exactly min_ff passes
    out = df.loc[df["forward_factor"] >= min_ff - 1e-12, cols].copy()
    out = out.sort_values("forward_factor", ascending=False, na_position="last")
    return out.reset_index(drop=True)


# ── /plays/vrp-stock -----------------------------------------------------------

def vrp_picks(
    df: pd.DataFrame,
    max_iv_pctl: float = 80.0,
    min_option_volume: Optional[float] = None,
) -> pd.DataFrame:
    """Volatility-risk-premium picks (iron-condor playbook), sorted by
    iron_condor_mean_return desc.

    Input columns: the VRP_COLUMNS schema plus ``term_structure_slope``
    (filter input only). Filters:
      - iv_pctl_1y < max_iv_pctl (strict; NaN tolerated — unknown percentile);
      - term_structure_slope >= 0 (contango/flat; NaN excluded);
      - optional option_volume >= min_option_volume (NaN tolerated).
    Per-structure backtest stats are pass-throughs — NaN when not supplied.
    """
    cols = VRP_COLUMNS
    if df.empty:
        return _empty(cols)
    df = _with_columns(df, cols + ["term_structure_slope"])

    pctl = pd.to_numeric(df["iv_pctl_1y"], errors="coerce")
    calm_enough = pctl.isna() | (pctl < max_iv_pctl)

    slope = pd.to_numeric(df["term_structure_slope"], errors="coerce")
    not_backwardated = slope >= 0

    mask = calm_enough & not_backwardated
    if min_option_volume is not None:
        vol = pd.to_numeric(df["option_volume"], errors="coerce")
        mask &= vol.isna() | (vol >= min_option_volume)

    out = df.loc[mask, cols].copy()
    out["_sort"] = pd.to_numeric(out["iron_condor_mean_return"], errors="coerce")
    out = out.sort_values("_sort", ascending=False, na_position="last")
    return out.drop(columns="_sort").reset_index(drop=True)


# ── orchestrator: earnings_ml.db -> all pick lists -----------------------------

def _latest_per_ticker(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "ticker" not in df.columns:
        return df
    sort_col = "scan_date" if "scan_date" in df.columns else "ticker"
    return df.sort_values(sort_col).drop_duplicates("ticker", keep="last")


def generate_picks(as_of: date) -> dict[str, pd.DataFrame]:
    """Build all pick lists from earnings_ml.db (READ-ONLY — SELECTs only).

    Returns {"earnings", "momentum_skew", "forward_factor", "vrp"}.
    Data-gap assumptions are documented in the module docstring.
    """
    from earnings_edge.db.repositories import (
        daily_signals_as_of_df,
        ff_snapshots_as_of_df,
        ff_universe_snapshots_as_of_df,
        snapshots_as_of_df,
    )

    cutoff = as_of.isoformat() if isinstance(as_of, (date, datetime)) else str(as_of)

    snaps = snapshots_as_of_df(cutoff)
    latest = _latest_per_ticker(snaps)

    # daily signals (option volume, IV percentile, momentum, skew) — populated
    # by scripts/collect_daily_signals.py; empty when the collector hasn't run.
    ds = daily_signals_as_of_df(cutoff)
    ds_latest = _latest_per_ticker(ds.rename(columns={"signal_date": "scan_date"}))
    vol20 = pd.Series(dtype=float)
    if not ds.empty and "option_volume" in ds.columns:
        vol20 = (ds.sort_values("signal_date")
                 .groupby("ticker")["option_volume"]
                 .apply(lambda s: s.dropna().tail(20).mean()))

    def ds_col(name: str) -> pd.Series:
        """latest daily_signals column as a ticker-indexed series (may be empty)."""
        if ds_latest.empty or name not in ds_latest.columns:
            return pd.Series(dtype=float)
        return ds_latest.set_index("ticker")[name]

    # ── earnings (short vol) ──
    earnings_in = pd.DataFrame()
    if not latest.empty:
        earnings_in = pd.DataFrame({
            "ticker": latest["ticker"],
            "announcement_date": latest.get("earnings_date"),
            "announcement_time": latest.get("timing", pd.Series(dtype="object")).map(_TIMING_MAP),
            "is_confirmed": latest.get("timing", pd.Series(dtype="object")).notna(),
            "implied_move": latest.get("expected_move_pct"),
            "option_volume": latest["ticker"].map(vol20),
            "term_structure_slope": latest.get("term_slope"),
            "iv_rv": latest.get("iv30_rv30"),
        })
        # historical move stats from the snapshots table's own outcome history
        if "actual_move_pct" in snaps.columns:
            hist = snaps[snaps["actual_move_pct"].notna()]
            if not hist.empty:
                stats = hist.groupby("ticker").agg(
                    avg_realized_move=("actual_move_pct", lambda s: s.abs().mean()),
                    avg_implied_move=("expected_move_pct", "mean"),
                    historical_events_count=("actual_move_pct", "size"),
                )
                earnings_in = earnings_in.merge(stats, on="ticker", how="left")
        earnings_in["implied_vs_avg_realized"] = (
            pd.to_numeric(earnings_in["implied_move"], errors="coerce")
            - pd.to_numeric(earnings_in.get("avg_realized_move"), errors="coerce")
        )

    # ── forward factors ──
    ff_legacy = ff_snapshots_as_of_df(cutoff)
    ff_univ = ff_universe_snapshots_as_of_df(cutoff)
    ff = pd.concat([ff_legacy, ff_univ], ignore_index=True) if not ff_univ.empty else ff_legacy
    ff_latest = _latest_per_ticker(ff)
    ff_in = pd.DataFrame()
    if not ff_latest.empty:
        ff_in = pd.DataFrame({
            "ticker": ff_latest["ticker"],
            "next_earnings_date": ff_latest.get("earnings_date"),
            "front_iv": ff_latest.get("t1_iv"),
            "back_iv": ff_latest.get("t2_iv"),
            "t1_dte": ff_latest.get("t1_dte"),
            "t2_dte": ff_latest.get("t2_dte"),
            "forward_iv": ff_latest.get("sigma_fwd"),
            "option_volume": ff_latest["ticker"].map(vol20),
        })

    # ── vrp ──
    vrp_in = pd.DataFrame()
    if not latest.empty:
        vrp_in = pd.DataFrame({
            "ticker": latest["ticker"],
            "iv_pctl_1y": latest["ticker"].map(ds_col("iv_pctl_1y")),
            "iv_rv": latest.get("iv30_rv30"),
            "option_volume": latest["ticker"].map(vol20),
            "next_earnings_date": latest.get("earnings_date"),
            "term_structure_slope": latest.get("term_slope"),
        })

    # ── momentum-skew ──
    # Fed by daily_signals (collect_daily_signals.py). skew_zscore/iv_pctl need
    # >= 20 days of accrued history, so they stay NaN (and the z-score filter
    # excludes the row) until the collector has run long enough.
    momentum_in = pd.DataFrame(columns=MOMENTUM_SKEW_COLUMNS)
    if not ds_latest.empty:
        earn_dates = (latest.set_index("ticker")["earnings_date"]
                      if not latest.empty and "earnings_date" in latest.columns
                      else pd.Series(dtype=object))
        momentum_in = pd.DataFrame({
            "ticker": ds_latest["ticker"],
            "next_earnings_date": ds_latest["ticker"].map(earn_dates),
            "option_volume": ds_latest["ticker"].map(vol20),
            "ts_momentum": ds_latest.get("ts_momentum"),
            "relative_momentum": ds_latest.get("relative_momentum"),
            "skew_value": ds_latest.get("skew_25d"),
            "skew_zscore": ds_latest.get("skew_zscore"),
            "skew_mean": ds_latest.get("skew_mean"),
        })

    return {
        "earnings": earnings_picks(earnings_in, as_of=as_of),
        "momentum_skew": momentum_skew_picks(momentum_in),
        "forward_factor": forward_factor_picks(ff_in),
        "vrp": vrp_picks(vrp_in),
    }


# ── persistence -----------------------------------------------------------------

PICKS_DDL = """
CREATE TABLE IF NOT EXISTS picks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pick_date TEXT NOT NULL,
    strategy TEXT NOT NULL,
    rank INTEGER NOT NULL,
    ticker TEXT NOT NULL,
    signals_json TEXT NOT NULL,
    created_at TEXT DEFAULT (datetime('now')),
    UNIQUE(pick_date, strategy, ticker)
);
CREATE INDEX IF NOT EXISTS idx_picks_date ON picks(pick_date, strategy);
"""


from earnings_edge.db.repositories import load_picks, persist_picks  # noqa: F401
