"""Daily per-ticker signals for the picks engine.

Computed once per trading day from the persisted options chain
(``options_chain`` table, populated by ``scripts/collect_options_snapshot.py``)
and from stock daily bars (Polygon), then persisted in the ``daily_signals``
table. Rolling statistics (IV percentile, skew z-score) accrue as the table
fills — they are None until enough history exists, never fabricated.

Signal definitions follow oquants.com (docs/oquants_feature_inventory.md):
  - option_volume: total contracts traded across the chain that day;
  - atm_iv: IV of the nearest-50-delta call at the front usable expiry;
  - skew_25d: IV(25-delta put) - IV(25-delta call) at the front usable expiry
    (positive = puts bid, the usual case for equities);
  - iv_pctl_1y: percentile of today's atm_iv within its own trailing history;
  - ts_momentum: 12-month-minus-1-month total return from daily bars;
  - relative_momentum: (1 + ts_momentum) / (1 + benchmark_momentum) — > 1
    means outperforming the benchmark (oquants uses SPY).

All computation functions are pure; ``upsert_daily_signals`` is the only I/O.
"""

from __future__ import annotations

import sqlite3
from datetime import date, datetime
from typing import Optional, Sequence

import numpy as np
import pandas as pd

# Minimum days-to-expiry for a chain expiry to count as "usable" — the very
# front expiry is dominated by the nearest event (earnings) and would poison
# ATM/skew readings, mirroring oquants' ex-earnings signal treatment.
DEFAULT_MIN_DTE = 21
# Minimum observations before rolling stats are reported.
MIN_HISTORY = 20
# Momentum window: 12 months of bars, skipping the most recent month.
MOM_LOOKBACK_BARS = 252
MOM_SKIP_BARS = 22
MOM_MIN_BARS = 60


def _to_date(value) -> Optional[date]:
    try:
        ts = pd.to_datetime(value, errors="coerce")
        if ts is None or pd.isna(ts):
            return None
        return ts.date()
    except Exception:
        return None


def enrich_chain_with_bs(
    chain: pd.DataFrame,
    *,
    spot: float,
    r: float,
    as_of: date | str,
) -> pd.DataFrame:
    """Backfill missing implied_volatility / delta from BSM.

    Alpaca's options snapshot carries quotes but no greeks on the current data
    tier, so IV and delta are solved from each row's midpoint (falling back to
    the daily-bar close) with the existing option_math functions. Rows that
    already carry IV/delta are kept verbatim; rows without a usable price keep
    their nulls (never fabricated).
    """
    from .option_math import black_scholes_delta, implied_volatility

    out = chain.copy()
    for col in ("implied_volatility", "delta", "midpoint", "close"):
        if col not in out.columns:
            out[col] = None

    ref = _to_date(as_of)
    for idx, row in out.iterrows():
        if row["implied_volatility"] is not None and pd.notna(row["implied_volatility"]):
            continue
        expiry = _to_date(row.get("expiry"))
        strike = row.get("strike")
        ctype = row.get("contract_type")
        price = row.get("midpoint")
        if price is None or pd.isna(price):
            price = row.get("close")
        if (ref is None or expiry is None or strike is None or pd.isna(strike)
                or price is None or pd.isna(price) or ctype not in ("call", "put")):
            continue
        T = max((expiry - ref).days, 0) / 365.0
        if T <= 0 or price <= 0:
            continue
        iv = implied_volatility(float(price), spot, float(strike), T, r, ctype)
        if iv is None or (isinstance(iv, float) and np.isnan(iv)):
            continue
        out.at[idx, "implied_volatility"] = float(iv)
        delta = black_scholes_delta(spot, float(strike), T, r, float(iv), ctype)
        if delta is not None and not (isinstance(delta, float) and np.isnan(delta)):
            out.at[idx, "delta"] = float(delta)
    return out


def compute_chain_signals(    chain: pd.DataFrame,
    *,
    as_of: date | str,
    min_dte: int = DEFAULT_MIN_DTE,
) -> dict:
    """Per-ticker signals from one day's options-chain rows.

    Expects columns: expiry, strike, contract_type, volume,
    implied_volatility, delta (missing/NaN tolerated). Returns a dict with
    ``option_volume``, ``atm_iv``, ``skew_25d`` (None when not computable).
    """
    out = {"option_volume": None, "atm_iv": None, "skew_25d": None}
    if chain is None or chain.empty:
        return out

    vol = pd.to_numeric(chain.get("volume"), errors="coerce")
    if vol.notna().any():
        out["option_volume"] = float(vol.sum())

    ref = _to_date(as_of)
    if ref is None:
        return out
    df = chain.copy()
    df["expiry_date"] = df["expiry"].map(_to_date)
    df["dte"] = df["expiry_date"].map(lambda d: (d - ref).days if d else None)
    df["iv"] = pd.to_numeric(df.get("implied_volatility"), errors="coerce")
    df["delta"] = pd.to_numeric(df.get("delta"), errors="coerce")
    usable = df[(df["dte"].fillna(-1) >= min_dte) & df["iv"].notna() & df["delta"].notna()]
    if usable.empty:
        return out

    front_expiry = min(usable["expiry_date"])
    front = usable[usable["expiry_date"] == front_expiry]

    calls = front[front["contract_type"] == "call"]
    if not calls.empty:
        atm = calls.iloc[(calls["delta"] - 0.50).abs().argmin()]
        out["atm_iv"] = float(atm["iv"])

    puts = front[front["contract_type"] == "put"]
    if not calls.empty and not puts.empty:
        c25 = calls.iloc[(calls["delta"] - 0.25).abs().argmin()]
        p25 = puts.iloc[(puts["delta"] + 0.25).abs().argmin()]
        out["skew_25d"] = float(p25["iv"] - c25["iv"])

    return out


def compute_iv_percentile(history: Sequence[float], current: float,
                          min_obs: int = MIN_HISTORY) -> Optional[float]:
    """Percentile (0-100) of *current* within *history*; None if thin."""
    hist = [float(h) for h in history if h is not None and np.isfinite(h)]
    if len(hist) < min_obs or current is None or not np.isfinite(current):
        return None
    below = sum(1 for h in hist if h < current)
    ties = sum(1 for h in hist if h == current)
    return float(100.0 * (below + 0.5 * ties) / len(hist))


def compute_zscore(history: Sequence[float], current: float,
                   min_obs: int = MIN_HISTORY) -> tuple[Optional[float], Optional[float]]:
    """(z-score, mean) of *current* vs *history*; (None, None) if thin."""
    hist = [float(h) for h in history if h is not None and np.isfinite(h)]
    if len(hist) < min_obs or current is None or not np.isfinite(current):
        return None, None
    arr = np.asarray(hist)
    std = float(arr.std(ddof=1))
    if std <= 0:
        return None, None
    mean = float(arr.mean())
    return float((current - mean) / std), mean


def compute_ts_momentum(bars: Sequence[dict]) -> Optional[float]:
    """12-month-minus-1-month total return from daily bars (oldest first).

    Bars are dicts with a close under key ``c`` (Polygon aggregate shape).
    Momentum = close[-1 - skip] / close[0] - 1 over the lookback window;
    None when there aren't at least MOM_MIN_BARS bars.
    """
    closes = [float(b["c"]) for b in bars
              if b.get("c") is not None and float(b.get("c")) > 0]
    if len(closes) < MOM_MIN_BARS:
        return None
    window = closes[-MOM_LOOKBACK_BARS:]
    end = window[-1 - MOM_SKIP_BARS] if len(window) > MOM_SKIP_BARS else window[0]
    return float(end / window[0] - 1.0)


def relative_momentum(ts_momentum: Optional[float],
                      benchmark_momentum: Optional[float]) -> Optional[float]:
    """(1 + ticker) / (1 + benchmark); > 1 means outperforming."""
    if ts_momentum is None or benchmark_momentum in (None, -1.0):
        return None
    return float((1.0 + ts_momentum) / (1.0 + benchmark_momentum))


# ── persistence -----------------------------------------------------------------

DAILY_SIGNALS_DDL = """
CREATE TABLE IF NOT EXISTS daily_signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    signal_date TEXT NOT NULL,
    option_volume REAL,
    atm_iv REAL,
    skew_25d REAL,
    skew_zscore REAL,
    skew_mean REAL,
    iv_pctl_1y REAL,
    ts_momentum REAL,
    relative_momentum REAL,
    created_at TEXT DEFAULT (datetime('now')),
    UNIQUE(ticker, signal_date)
);
CREATE INDEX IF NOT EXISTS idx_daily_signals_ticker_date
    ON daily_signals(ticker, signal_date);
"""

from earnings_edge.db.repositories import upsert_daily_signals  # noqa: F401


def contract_market(
    conn: sqlite3.Connection,
    ticker: str,
    kind: str,
    strike: float,
    expiry: date | str,
    *,
    spot: float,
    r: float = 0.045,
    as_of: date | str | None = None,
) -> Optional[dict]:
    """Market data for one contract from the latest persisted options_chain.

    Returns {"price", "iv", "delta", "scan_date"} or None when the contract
    isn't in the chain. Price = midpoint (falling back to daily-bar close).
    IV = the stored value when present, else solved from the price via BSM
    (Alpaca snapshots carry no greeks on the current data tier); None when
    neither is available — callers decide the fallback, nothing is fabricated.
    """
    expiry_iso = expiry.isoformat() if isinstance(expiry, date) else str(expiry)
    as_of_iso = (_to_date(as_of) or date.today()).isoformat()
    row = conn.execute(
        "SELECT midpoint, close, implied_volatility, delta, scan_date "
        "FROM options_chain "
        "WHERE ticker = ? AND contract_type = ? AND strike = ? AND expiry = ? "
        "AND scan_date <= ? ORDER BY scan_date DESC LIMIT 1",
        (ticker, kind, float(strike), expiry_iso, as_of_iso),
    ).fetchone()
    if row is None:
        return None

    mid, close, iv, delta, scan_date = row
    price = mid if mid is not None else close
    if iv is None and price and price > 0:
        ref = _to_date(as_of) or date.today()
        T = max((date.fromisoformat(expiry_iso) - ref).days, 0) / 365.0
        if T > 0:
            from .option_math import black_scholes_delta, implied_volatility
            solved = implied_volatility(float(price), spot, float(strike), T, r, kind)
            if solved is not None and not (isinstance(solved, float) and np.isnan(solved)):
                iv = float(solved)
                d = black_scholes_delta(spot, float(strike), T, r, iv, kind)
                if d is not None and not (isinstance(d, float) and np.isnan(d)):
                    delta = float(d)
    return {
        "price": float(price) if price is not None else None,
        "iv": float(iv) if iv is not None else None,
        "delta": float(delta) if delta is not None else None,
        "scan_date": scan_date,
    }
