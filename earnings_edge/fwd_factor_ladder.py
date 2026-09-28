from __future__ import annotations

import sqlite3

from framework.risk.killswitch import record_event

"""Forward-factor calendar ladder: candidate construction + patient limit execution.

Daily flow (US options session):
  13:45 ET  build candidates from the earnings calendar, push Telegram cards
            (✅ Arm / ❌ Skip). Arming = one human approval per trade.
  14:00 ET  armed ladders start at the combo mid, then concede every 15 min
            toward min(combo ask, 20%-premium debit D_cap) — never above
            either (LadderSpec.market_limit). Repricing = cancel + resubmit.
  each step refresh Alpaca chain quotes; if the mid runs away beyond the
            distance filter, cancel and disarm. Fills and disarms are
            reported back. Day orders — unfilled ladders die at the close.

State persists in the `ff_ladders` table so a bot restart doesn't strand
armed ladders (the runner re-reads open order ids on startup).
"""


import json
import logging
import math
import os
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Callable

from .alpaca_trading import AlpacaTradingClient
from .fwd_factor import (
    ET,
    LadderSpec,
    combo_debit,
    forward_iv,
    leg_quote_ok,
    occ_parse,
    target_debit,
    within_fill_range,
)
from .option_math import implied_volatility

logger = logging.getLogger("earnings_edge.fwd_factor_ladder")

MIN_PRICE = 3.0
MIN_HIST_EVENTS = 3  # gate: fewer realized events → no candidate
HIST_TARGET_EVENTS = 6  # backfill toward this many before trusting the RMS
DISTANCE_F = 0.15

# ── hardening knobs ────────────────────────────────────────────────────
MAX_QUOTE_AGE_SEC = 900  # hold step if either leg quote is older
SPOT_DRIFT_TOLERANCE = 0.03  # disarm if spot moved >3% from candidate build
BP_BUFFER = 1.1  # buying power must cover 110% of worst-case cost
TERMINAL_SUBMIT_STATUS = {401, 403, 404, 422}  # disarm on these broker errors


def _is_terminal_submit_error(exc: Exception) -> bool:
    """Broker said 'never going to work' (auth, permission, buying power,
    validation) vs transient network/5xx — only the former disarms."""
    from .alpaca_trading import AlpacaError

    return isinstance(exc, AlpacaError) and getattr(exc, "status_code", None) in TERMINAL_SUBMIT_STATUS


DDL = """
CREATE TABLE IF NOT EXISTS ff_ladders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    candidate_json TEXT NOT NULL,
    order_id TEXT,
    status TEXT NOT NULL DEFAULT 'armed',   -- armed | filled | disarmed | expired
    rung INTEGER DEFAULT 0,
    armed_by INTEGER,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
)
"""


@dataclass
class CalendarCandidate:
    ticker: str
    earnings_date: str
    spot: float
    strike: float
    near_symbol: str
    far_symbol: str
    near_expiry: str
    far_expiry: str
    near_bid: float
    near_ask: float
    far_bid: float
    far_ask: float
    sigma_fwd: float
    hist_rms_move: float
    tau_days: int
    d_start: float  # max debit at start_premium (25%)
    d_cap: float  # max debit at floor_premium (20%)
    mid_debit: float
    skip_reason: str | None = None
    strategy_override: str | None = None
    # "Pre Market" / "Post Market" / ... from the earnings calendar. Decides
    # whether an expiry ON the event date spans the move (BMO) or not (AMC),
    # and which session the post-event exit treats as the reaction day.
    timing: str | None = None


def near_leg_exit_by(near_expiry: str | None) -> date | None:
    """A calendar's structural exit deadline: the near leg's expiry date."""
    try:
        return date.fromisoformat(str(near_expiry)[:10]) if near_expiry else None
    except ValueError:
        return None


def is_after_close(timing: str | None) -> bool:
    """AMC-like timing: the price reaction lands on the NEXT session.

    Unknown timing is treated as after-close — the conservative reading for
    both expiry selection (expiry must be after the event date) and the
    post-event exit (wait for the next session).
    """
    t = (timing or "").lower()
    return not ("pre" in t or "bmo" in t or "before" in t)


# Liquidity gates for a tradeable calendar. The 2026-08 proposal-quality
# review found near-leg spreads of 50-80% of mid on filled names; a combo
# that wide cannot be entered or exited near its mid.
MAX_COMBO_SPREAD_OF_MID = 0.60
MIN_MID_DEBIT = 0.20


def combo_liquidity_reason(near_bid, near_ask, far_bid, far_ask) -> str | None:
    """Why a calendar's quotes are too thin to trade, or None when fine."""
    mid = combo_debit(near_bid, near_ask, far_bid, far_ask)
    if mid is None or mid < MIN_MID_DEBIT:
        return f"mid debit {mid if mid is not None else float('nan'):.2f} < {MIN_MID_DEBIT:.2f}"
    width = (near_ask - near_bid) + (far_ask - far_bid)
    if width / mid > MAX_COMBO_SPREAD_OF_MID:
        return f"combo spread {width:.2f} is {width / mid:.0%} of mid {mid:.2f}"
    return None


def hist_rms_move(ticker: str) -> tuple[float | None, int]:
    """RMS |actual_move_pct| over the ticker's realized events (as a fraction)."""
    from earnings_edge.db.repositories import snapshots_hist_abs_moves

    vals = [v / 100.0 for v in snapshots_hist_abs_moves(ticker)]
    if len(vals) < MIN_HIST_EVENTS:
        return None, len(vals)
    return math.sqrt(sum(v * v for v in vals) / len(vals)), len(vals)


# ── Historical-move backfill (self-healing coverage) ────────────────────
# The snapshots table only covers tickers the scanner happened to collect,
# so most of the earnings universe failed the hist gate ("hist events 0 < 3")
# and nothing could be armed. Finnhub's free tier only serves the current
# earnings window, so past announcement dates come from Yahoo (yfinance
# earnings calendar) and the move is computed from LSE daily bars via the
# shared outcome_from_bars transform. Rows land in `snapshots` tagged
# timing='Backfill' so hist_rms_move picks them up and later runs are free.
HIST_BACKFILL_MAX_EVENTS = 8  # ~2 years of quarters
HIST_BACKFILL_MAX_AGE_DAYS = 900  # neither LSE nor Polygon serves older bars
# (Polygon 403s beyond plan history, each 403
# costs 45s+ of retries — skip guaranteed fails)
_hist_backfill_attempted: set = set()  # per-process: never retry within a run


def _lse_bars_client():
    """Shared LSE collector for daily bars; None when unconfigured."""
    if not os.environ.get("LSE_API_KEY"):
        return None
    global _LSE_SINGLETON
    try:
        if _LSE_SINGLETON is None:
            from .collectors.lse import LSECollector

            _LSE_SINGLETON = LSECollector()
        return _LSE_SINGLETON
    except Exception as exc:
        # exc-policy: keep broad, ensure visibility
        record_event("silent_failure", f"fwd_factor_ladder: {exc}")
        import logging

        logging.getLogger(__name__).error("fwd_factor_ladder broad exception", exc_info=True)
        logger.info("hist backfill: LSE unavailable (%s)", exc)
        return None


_LSE_SINGLETON = None


def _polygon_bars_client():
    """Shared Polygon collector for daily bars; None when unconfigured."""
    if not os.environ.get("POLYGON_API_KEY"):
        return None
    global _POLYGON_SINGLETON
    try:
        if _POLYGON_SINGLETON is None:
            from .collectors.polygon import PolygonClient

            _POLYGON_SINGLETON = PolygonClient()
        return _POLYGON_SINGLETON
    except Exception as exc:
        # exc-policy: keep broad, ensure visibility
        record_event("silent_failure", f"fwd_factor_ladder: {exc}")
        import logging

        logging.getLogger(__name__).error("fwd_factor_ladder broad exception", exc_info=True)
        logger.info("hist backfill: Polygon unavailable (%s)", exc)
        return None


_POLYGON_SINGLETON = None

# Legacy backfill rows (timing='Backfill') were measured with the BMO window
# for every name, i.e. over the PRE-announcement session for after-close
# reporters (the bias docs/proposal_quality_2026-08.md documents for stored
# labels). That understated hist RMS and made the near leg look rich — a
# fake forward-factor edge. New rows carry the window they were measured
# with; legacy rows are recomputed by refresh_legacy_hist_moves().
LEGACY_BACKFILL_TAG = "Backfill"
_hist_refresh_attempted: set = set()


def event_timing_from_timestamp(ts) -> str | None:
    """'Pre Market' / 'Post Market' from a Yahoo earnings timestamp (ET).

    Yahoo reports the announcement time (e.g. 16:05 AMC, 06:30 BMO); a bare
    midnight means the time is unknown → None.
    """
    hour, minute = getattr(ts, "hour", None), getattr(ts, "minute", None)
    if hour is None or (hour == 0 and minute == 0):
        return None
    return "Pre Market" if hour < 12 else "Post Market"


def backfill_tag(timing: str | None) -> str:
    """snapshots.timing value recording which measurement window a row used."""
    return {"Pre Market": "Backfill:BMO", "Post Market": "Backfill:AMC"}.get(timing or "", "Backfill:2d")


def _yf_earnings_events(ticker: str, today: date) -> dict[date, str | None] | None:
    """Past announcement dates → timing from Yahoo, within provider history.

    None when Yahoo is unavailable or has nothing (callers keep what they have).
    """
    try:
        import yfinance as yf

        from .config import session

        try:
            ticker_obj = yf.Ticker(ticker, session=session)
        except TypeError:
            # test doubles (and older yfinance) without the session kwarg
            ticker_obj = yf.Ticker(ticker)
        df = ticker_obj.get_earnings_dates(limit=12)
    except Exception as exc:
        # exc-policy: keep broad, ensure visibility
        record_event("silent_failure", f"fwd_factor_ladder: {exc}")
        logger.error("fwd_factor_ladder broad exception", exc_info=True)
        logger.info("hist backfill %s: earnings dates unavailable (%s)", ticker, exc)
        return None
    if df is None or len(df) == 0:
        logger.info("hist backfill %s: no earnings dates", ticker)
        return None
    cutoff = today - timedelta(days=HIST_BACKFILL_MAX_AGE_DAYS)
    events: dict[date, str | None] = {}
    for ts in df.index:
        d = ts.date()
        if cutoff <= d < today:
            timing = event_timing_from_timestamp(ts)
            if d not in events or events[d] is None:
                events[d] = timing
    return events


def _event_outcome(ticker: str, ed: date, timing: str | None, lse, polygon) -> dict | None:
    """Timing-aware realized move for one event from LSE (then Polygon) bars."""
    from .services.outcome_service import OutcomeService

    bars = None
    if lse is not None:
        try:
            bars = lse.daily_bars(ticker, ed - timedelta(days=7), ed + timedelta(days=3))
        except Exception as exc:
            # exc-policy: keep broad, ensure visibility
            record_event("silent_failure", f"fwd_factor_ladder: {exc}")
            logger.error("fwd_factor_ladder broad exception", exc_info=True)
            logger.info("hist backfill %s %s: LSE bars failed (%s)", ticker, ed, exc)
    # Fall back to Polygon if LSE unavailable or returned empty
    if not bars and polygon is not None:
        try:
            start_iso = (ed - timedelta(days=7)).isoformat()
            end_iso = (ed + timedelta(days=3)).isoformat()
            bars = polygon.get_daily_bars(ticker, start_iso, end_iso)
        except Exception as exc:
            # exc-policy: keep broad, ensure visibility
            record_event("silent_failure", f"fwd_factor_ladder: {exc}")
            logger.error("fwd_factor_ladder broad exception", exc_info=True)
            logger.info("hist backfill %s %s: Polygon bars failed (%s)", ticker, ed, exc)
    if not bars:
        return None
    try:
        return OutcomeService.outcome_from_bars(bars, ed, timing=timing, span_both=timing is None)
    except Exception as exc:
        # exc-policy: keep broad, ensure visibility
        record_event("silent_failure", f"fwd_factor_ladder: {exc}")
        logger.error("fwd_factor_ladder broad exception", exc_info=True)
        logger.info("hist backfill %s %s: outcome computation failed (%s)", ticker, ed, exc)
        return None


def refresh_legacy_hist_moves(ticker: str, today: date | None = None) -> int:
    """Recompute a ticker's legacy ``timing='Backfill'`` rows timing-aware.

    Returns the number of rows rewritten. Once per ticker per process; never
    raises. Rows Yahoo no longer lists are left as they are.
    """
    from .db.repositories import snapshots_apply_hist_backfill_batch, snapshots_legacy_backfill_rows

    if ticker in _hist_refresh_attempted:
        return 0
    _hist_refresh_attempted.add(ticker)
    try:
        legacy = snapshots_legacy_backfill_rows(ticker)
    except Exception as exc:
        # exc-policy: keep broad, ensure visibility
        record_event("silent_failure", f"fwd_factor_ladder: {exc}")
        logger.error("fwd_factor_ladder broad exception", exc_info=True)
        return 0
    if not legacy:
        return 0
    today = today or datetime.now(UTC).date()
    lse, polygon = _lse_bars_client(), _polygon_bars_client()
    if lse is None and polygon is None:
        return 0
    events = _yf_earnings_events(ticker, today)
    if not events:
        return 0
    now_iso = datetime.now(UTC).isoformat()
    writes = []
    for row in legacy:
        try:
            ed = date.fromisoformat(str(row["earnings_date"])[:10])
        except ValueError:
            continue
        if ed not in events:
            continue
        outcome = _event_outcome(ticker, ed, events[ed], lse, polygon)
        if not outcome:
            continue
        writes.append(
            {
                "existing_id": row["id"],
                "ticker": ticker,
                "earnings_date": ed.isoformat(),
                "outcome": outcome,
                "fetched_at": now_iso,
                "timing_tag": backfill_tag(events[ed]),
                "set_timing": True,
            }
        )
    try:
        snapshots_apply_hist_backfill_batch(writes=writes)
    except Exception as exc:
        # exc-policy: keep broad, ensure visibility
        record_event("silent_failure", f"fwd_factor_ladder: {exc}")
        logger.error("fwd_factor_ladder broad exception", exc_info=True)
        return 0
    if writes:
        logger.info("hist refresh %s: %d legacy backfill rows re-measured timing-aware", ticker, len(writes))
    return len(writes)


def ensure_hist_moves(
    ticker: str,
    today: date | None = None,
    min_events: int = MIN_HIST_EVENTS,
) -> int:
    """Backfill realized earnings moves for under-covered tickers.

    Returns the number of usable events after the attempt. Never raises —
    any failure degrades to the pre-existing skip behaviour.
    """
    from .db.repositories import (
        snapshots_apply_hist_backfill_batch,
        snapshots_outcome_row,
        snapshots_usable_outcome_count,
    )

    today = today or datetime.now(UTC).date()
    have = snapshots_usable_outcome_count(ticker=ticker)
    if have >= min_events or ticker in _hist_backfill_attempted:
        return have
    _hist_backfill_attempted.add(ticker)

    # No bar source → nothing to measure; don't spend a Yahoo call first.
    lse = _lse_bars_client()
    polygon = _polygon_bars_client()
    if lse is None and polygon is None:
        return have

    events = _yf_earnings_events(ticker, today)
    if events is None:
        return have
    dates = sorted(events, reverse=True)[:HIST_BACKFILL_MAX_EVENTS]
    if not dates:
        return have

    now_iso = datetime.now(UTC).isoformat()
    written = 0
    writes: list[dict] = []
    for ed in dates:
        # Early exit optimization - stop once we have enough events
        if have + written >= min_events:
            logger.info(
                "hist backfill %s: early exit (have=%d, written=%d, min=%d)",
                ticker,
                have,
                written,
                min_events,
            )
            break

        outcome = _event_outcome(ticker, ed, events[ed], lse, polygon)
        if not outcome:
            continue
        existing = snapshots_outcome_row(
            ticker=ticker,
            earnings_date=ed.isoformat(),
        )
        try:
            if (
                existing
                and existing["actual_move_pct"] is not None
                and existing["outcome_fetched_at"] not in (None, "unavailable")
            ):
                continue  # already has a good outcome
            writes.append(
                {
                    "existing_id": existing["id"] if existing else None,
                    "ticker": ticker,
                    "earnings_date": ed.isoformat(),
                    "outcome": outcome,
                    "fetched_at": now_iso,
                    "timing_tag": backfill_tag(events[ed]),
                }
            )
            written += 1
        except Exception as exc:
            # exc-policy: keep broad, ensure visibility
            record_event("silent_failure", f"fwd_factor_ladder: {exc}")
            import logging

            logging.getLogger(__name__).error("fwd_factor_ladder broad exception", exc_info=True)
            logger.info("hist backfill %s %s: write failed (%s)", ticker, ed, exc)
    try:
        snapshots_apply_hist_backfill_batch(writes=writes)
    except Exception as exc:
        # exc-policy: keep broad, ensure visibility
        record_event("silent_failure", f"fwd_factor_ladder: {exc}")
        import logging

        logging.getLogger(__name__).error("fwd_factor_ladder broad exception", exc_info=True)
        pass
    total = have + written
    if written:
        logger.info("hist backfill %s: +%d events (total %d)", ticker, written, total)
    return total


def _pick_pair_tenor(
    chain: dict[str, dict],
    spot: float,
    today: date,
    *,
    t1_min_days: int = 30,
    t1_max_days: int = 60,
    t2_gap_days: int = 30,
) -> tuple[dict | None, dict | None]:
    """T1: expiry in [t1_min_days, t1_max_days] (closest to 45 if several).
    T2: expiry ~t2_gap_days days after T1 (closest wins). Strike: the one
    closest to spot that both expiries list. Calls only, as today."""
    by_expiry = _calls_by_expiry(chain, today)
    t1_cands = [e for e in by_expiry if t1_min_days <= (e - today).days <= t1_max_days]
    if not t1_cands:
        return None, None
    t1_exp = min(t1_cands, key=lambda e: abs((e - today).days - 45))
    t2_exp = _far_expiry(by_expiry, t1_exp, t2_gap_days)
    if t2_exp is None:
        return None, None
    return _same_strike_pair(by_expiry[t1_exp], by_expiry[t2_exp], spot)


def _pick_pair(
    chain: dict[str, dict],
    spot: float,
    today: date,
    event_date: date | None = None,
    timing: str | None = None,
) -> tuple[dict | None, dict | None]:
    """T1: the first expiration that spans the event; T2: ~30 days after T1.

    An expiry ON the event date only spans a before-open announcement; for
    after-close (or unknown) timing the near leg would settle before the
    move, so T1 must be strictly later. Chain = {occ_symbol: quote}; both
    legs share one strike (closest to spot among strikes both list).
    """
    by_expiry = _calls_by_expiry(chain, today)
    if not event_date:
        event_date = today
        timing = "Pre Market"  # no event: an expiry today is acceptable
    strict = is_after_close(timing)
    t1_cands = sorted(e for e in by_expiry if (e > event_date if strict else e >= event_date))
    if not t1_cands:
        return None, None
    t1_exp = t1_cands[0]
    t2_exp = _far_expiry(by_expiry, t1_exp, 30)
    if t2_exp is None:
        return None, None
    return _same_strike_pair(by_expiry[t1_exp], by_expiry[t2_exp], spot)


def _calls_by_expiry(chain: dict[str, dict], today: date) -> dict[date, list[tuple[str, dict]]]:
    by_expiry: dict[date, list[tuple[str, dict]]] = {}
    for sym in chain:
        p = occ_parse(sym)
        if p["option_type"] != "call" or p["expiry"] < today:
            continue
        by_expiry.setdefault(p["expiry"], []).append((sym, p))
    return by_expiry


def _far_expiry(by_expiry: dict, t1_exp: date, gap_days: int) -> date | None:
    cands = sorted((abs((e - t1_exp).days - gap_days), e) for e in by_expiry if (e - t1_exp).days > 0)
    return cands[0][1] if cands else None


def _same_strike_pair(near_entries: list, far_entries: list, spot: float) -> tuple[dict | None, dict | None]:
    """Near/far legs at ONE strike — the ATM-most strike both expiries list.

    Picking each expiry's own closest-to-spot strike built diagonals when the
    expiries' strike grids differ (2026-09 CCL: sold 22.5, bought 22.0), which
    the calendar math (target_debit, forward IV at one strike) does not price.
    """
    far_by_strike = {round(p["strike"], 4): (sym, p) for sym, p in far_entries}
    shared = [(sym, p) for sym, p in near_entries if round(p["strike"], 4) in far_by_strike]
    if not shared:
        return None, None
    near_sym, near_p = min(shared, key=lambda e: abs(e[1]["strike"] - spot))
    far_sym, far_p = far_by_strike[round(near_p["strike"], 4)]
    return (
        {"symbol": near_sym, "strike": near_p["strike"], "expiry": near_p["expiry"]},
        {"symbol": far_sym, "strike": far_p["strike"], "expiry": far_p["expiry"]},
    )


def _reject(ticker: str, earnings_date: date, spot: float, reason: str) -> CalendarCandidate:
    return CalendarCandidate(
        ticker=ticker,
        earnings_date=earnings_date.isoformat(),
        spot=spot or 0.0,
        strike=0.0,
        near_symbol="",
        far_symbol="",
        near_expiry="",
        far_expiry="",
        near_bid=0.0,
        near_ask=0.0,
        far_bid=0.0,
        far_ask=0.0,
        sigma_fwd=0.0,
        hist_rms_move=0.0,
        tau_days=0,
        d_start=0.0,
        d_cap=0.0,
        mid_debit=0.0,
        skip_reason=reason,
    )


def build_candidate(
    alpaca: AlpacaTradingClient,
    ticker: str,
    earnings_date: date,
    spec: LadderSpec = LadderSpec(),
    today: date | None = None,
    timing: str | None = None,
) -> CalendarCandidate:
    """Construct (or reject) one ladder candidate from live Alpaca quotes.

    ``timing`` is the calendar's announcement timing ("Pre Market" /
    "Post Market" / unknown); it decides whether an expiry on the event date
    spans the move and is carried on the candidate for the post-event exit.
    """
    today = today or datetime.now(UTC).date()
    spot = alpaca.get_stock_latest_trade(ticker)
    if not spot or spot < MIN_PRICE:
        return _reject(ticker, earnings_date, spot or 0.0, f"price {spot} < {MIN_PRICE}")

    # Cheap broker gates first: the hist backfill (Yahoo + LSE/Polygon bars)
    # used to run before these, spending calls on tickers that then died on
    # "no chain" (2026-09 logs: KMTS, CODA, HSCS).
    chain = alpaca.get_options_chain_snapshots(ticker)
    if not chain:
        return _reject(ticker, earnings_date, spot, "no chain")

    t1, t2 = _pick_pair(chain, spot, today, earnings_date, timing)
    if not t1 or not t2:
        return _reject(ticker, earnings_date, spot, "no same-strike T1/T2 pair spanning the event")

    q1, q2 = chain[t1["symbol"]], chain[t2["symbol"]]
    T1 = (t1["expiry"] - today).days / 365.0
    T2 = (t2["expiry"] - today).days / 365.0
    mid1 = (q1["bid"] + q1["ask"]) / 2.0
    mid2 = (q2["bid"] + q2["ask"]) / 2.0
    iv1 = implied_volatility(mid1, spot, t1["strike"], T1, 0.045, "call")
    iv2 = implied_volatility(mid2, spot, t2["strike"], T2, 0.045, "call")
    base = dict(
        ticker=ticker,
        earnings_date=earnings_date.isoformat(),
        spot=spot,
        strike=t1["strike"],
        near_symbol=t1["symbol"],
        far_symbol=t2["symbol"],
        near_expiry=t1["expiry"].isoformat(),
        far_expiry=t2["expiry"].isoformat(),
        near_bid=q1["bid"],
        near_ask=q1["ask"],
        far_bid=q2["bid"],
        far_ask=q2["ask"],
        sigma_fwd=0,
        hist_rms_move=0.0,
        tau_days=0,
        d_start=0,
        d_cap=0,
        mid_debit=0,
        timing=timing,
    )
    quotes_ok = leg_quote_ok(q1["bid"], q1["ask"]) and leg_quote_ok(q2["bid"], q2["ask"])
    thin = combo_liquidity_reason(q1["bid"], q1["ask"], q2["bid"], q2["ask"]) if quotes_ok else None
    if not quotes_ok or thin or not (math.isfinite(iv1) and math.isfinite(iv2)):
        # a zero-bid "mid" solves to a finite but meaningless IV; a combo
        # wider than MAX_COMBO_SPREAD_OF_MID can't be entered/exited near mid
        reason = (
            "invalid leg quotes (zero bid or crossed)"
            if not quotes_ok
            else f"illiquid: {thin}"
            if thin
            else "leg IV unsolvable"
        )
        return CalendarCandidate(**base, skip_reason=reason)

    # Realized event-move history. Legacy (BMO-window-for-everyone) backfill
    # rows are re-measured first — they understate after-close movers and
    # fake an edge. Under-covered tickers are topped up toward
    # HIST_TARGET_EVENTS: an RMS from exactly 3 quarters is mostly noise.
    refresh_legacy_hist_moves(ticker, today=today)
    rms, n_hist = hist_rms_move(ticker=ticker)
    if rms is None or n_hist < HIST_TARGET_EVENTS:
        ensure_hist_moves(ticker=ticker, today=today, min_events=HIST_TARGET_EVENTS)
        rms, n_hist = hist_rms_move(ticker=ticker)
    if rms is None:
        return _reject(ticker, earnings_date, spot, f"hist events {n_hist} < {MIN_HIST_EVENTS}")
    base["hist_rms_move"] = rms

    fwd = forward_iv(iv1, T1, iv2, T2)
    if fwd is None:
        return CalendarCandidate(**base, skip_reason="negative fwd variance")

    tau_days = max((earnings_date - today).days, 0) + 1
    tau = tau_days / 365.0
    d_start = target_debit(mid2, spot, t1["strike"], T1, fwd, tau, rms, spec.start_premium)
    d_cap = target_debit(mid2, spot, t1["strike"], T1, fwd, tau, rms, spec.floor_premium)
    mid = combo_debit(q1["bid"], q1["ask"], q2["bid"], q2["ask"]) or 0.0
    base.update(sigma_fwd=fwd, tau_days=tau_days, d_start=d_start or 0.0, d_cap=d_cap or 0.0, mid_debit=mid)

    if d_start is None or d_cap is None or d_cap <= 0:
        return CalendarCandidate(**base, skip_reason="target debit degenerate (<=0)")
    if not within_fill_range(mid, d_cap, DISTANCE_F):
        return CalendarCandidate(**base, skip_reason=f"mid {mid:.2f} too far from cap {d_cap:.2f}")
    return CalendarCandidate(**base)


@dataclass
class ArmedLadder:
    id: int
    candidate: CalendarCandidate
    order_id: str | None = None
    rung: int = 0
    status: str = "armed"
    created_at: str | None = None


class LadderRunner:
    """Manages armed ladders: place, reprice, watch fills, disarm on runaway.

    Hardening (2026-07-25):
    - DB via session_scope / ff_ladders_* repositories (thread-safe engine)
    - kill-switch consultation every step (framework risk_state, tolerant of
      the table missing so the runner stays usable standalone)
    - buying-power preflight at arm and step (110% of worst-case debit)
    - quote validity: positive bid/ask, ask >= bid, quote age <= 15 min
    - spot drift: disarm if spot moved >3% since candidate build (strike
      is no longer ATM)
    - terminal broker errors (401/403/404/422) disarm; transient errors hold
    - stale ladders (armed before today ET) are expired, never traded
    - arm dedupe per ticker
    """

    def __init__(
        self,
        alpaca: AlpacaTradingClient,
        db_path=None,
        spec: LadderSpec = LadderSpec(),
        now_fn: Callable[[], datetime] | None = None,
    ):
        self.alpaca = alpaca
        self.spec = spec
        # Injectable clock — production uses wall time; tests freeze it.
        # step(now) takes the tick time explicitly; this clock is for guards
        # inside arm() that need "today ET" (event-staleness refusal).
        self._now_fn = now_fn or (lambda: datetime.now(ET))
        if db_path is not None:
            from earnings_edge.db.engine import configure

            configure(db_path)
        self.events: list[str] = []  # drained by the bot for Telegram pushes
        self._bp_warned: set[int] = set()

    def _today_et(self) -> date:
        return self._now_fn().astimezone(ET).date()

    def _kill_switch_halted(self) -> bool:
        """Consult the framework kill switch; missing table = not halted."""
        try:
            from framework.risk.killswitch import KillSwitch
        except ImportError:
            return False
        try:
            return KillSwitch().is_halted()
        except sqlite3.Error:
            # exc-policy: narrowed to sqlite3.Error for missing table
            return False  # risk_state table not present in this DB

    def _buying_power(self) -> float | None:
        try:
            bp = self.alpaca.get_account().get("buying_power")
            return float(bp) if bp is not None else None
        except Exception as exc:
            # exc-policy: keep broad, ensure visibility
            record_event("silent_failure", f"fwd_factor_ladder: {exc}")
            import logging

            logging.getLogger(__name__).error("fwd_factor_ladder broad exception", exc_info=True)
            logger.warning("buying-power preflight failed: %s", exc)
            return None

    @staticmethod
    def _worst_case_cost(cand: CalendarCandidate) -> float:
        return max(cand.d_cap, 0.0) * 100.0 * BP_BUFFER

    def _risk_check_arm(self, cand: CalendarCandidate, est_cost: float):
        """Framework RiskManager gate for an arm. None = framework unavailable
        (standalone mode: the ad-hoc kill-switch/BP gates still apply)."""
        try:
            from framework.core.registry import get_registry
            from framework.execution.lifecycle import LifecycleManager
            from framework.risk.manager import RiskManager
        except ImportError:
            return None
        try:
            acct = self.alpaca.get_account()
            equity = float(acct.get("equity") or 0)
            bp = float(acct.get("buying_power") or 0)
            try:
                limits = get_registry().limits_for("ff_ladder")
            except Exception as exc:
                # exc-policy: keep broad, ensure visibility
                record_event("silent_failure", f"fwd_factor_ladder: {exc}")
                import logging

                logging.getLogger(__name__).error("fwd_factor_ladder broad exception", exc_info=True)
                limits = None
            return RiskManager().check_trade(
                strategy="ff_ladder",
                ticker=cand.ticker,
                est_cost=est_cost,
                equity=equity,
                buying_power=bp,
                lifecycle=LifecycleManager().state("ff_ladder"),
                limits=limits,
            )
        except Exception as exc:
            # exc-policy: keep broad, ensure visibility
            record_event("silent_failure", f"fwd_factor_ladder: {exc}")
            import logging

            logging.getLogger(__name__).error("fwd_factor_ladder broad exception", exc_info=True)
            logger.warning("ff arm risk check failed (%s) — ad-hoc gates only", exc)
            return None

    def _take_fill_if_any(self, ladder: ArmedLadder, cand: CalendarCandidate) -> bool:
        """If the resting order filled (or partial), book it and stop stepping.

        Call this *before* expire/disarm so a fill that lands after we
        decided to stop is not abandoned (ATLO/BA).
        """
        if not ladder.order_id:
            return False
        try:
            order = self.alpaca.get_order(ladder.order_id)
        except Exception as exc:
            record_event("silent_failure", f"fwd_factor_ladder: {exc}")
            logger.error("fwd_factor_ladder broad exception", exc_info=True)
            logger.info("ladder %s fill-check failed: %s", ladder.id, exc)
            return False
        status = (order.get("status") or "").lower()
        try:
            filled = float(order.get("filled_qty") or 0)
        except (TypeError, ValueError):
            filled = 0.0
        if status not in ("filled", "partially_filled") and filled <= 0:
            return False
        ladder.status = "filled"
        self._record_fill(cand, order)
        self.events.append(
            f"✅ FF ladder filled: {cand.ticker} calendar @ "
            f"{order.get('filled_avg_price')} ({cand.near_symbol} / {cand.far_symbol})"
        )
        return True

    def _record_fill(self, cand: CalendarCandidate, order: dict) -> None:
        """Framework bookkeeping for a filled ladder: risk spend + managed legs
        (without this, reconcile reports the spread as orphans and the
        assignment guard never sees the short leg)."""
        try:
            px = float(order.get("filled_avg_price") or 0)
        except (TypeError, ValueError):
            px = 0.0
        try:
            from framework.execution.managed import record_open_positions
            from framework.risk.manager import RiskManager
        except ImportError:
            return
        # Book under the strategy that proposed it (forward_factor_arb rides
        # the same runner) so its own TOML exit rules apply.
        strategy = cand.strategy_override or "ff_ladder"
        try:
            RiskManager().record_entry(strategy, cand.ticker, px * 100.0, detail="ladder fill")
        except Exception as exc:
            # A risk-ledger hiccup must not stop the legs being booked below —
            # unbooked legs are invisible to the exit engine.
            record_event("silent_failure", f"fwd_factor_ladder: risk ledger: {exc}")
            logger.error("fwd_factor_ladder broad exception", exc_info=True)
        try:
            legs = [
                {
                    "symbol": cand.near_symbol,
                    "side": "sell",
                    "ratio_qty": 1,
                    "option_type": "call",
                    "strike": cand.strike,
                    "expiry": cand.near_expiry,
                },
                {
                    "symbol": cand.far_symbol,
                    "side": "buy",
                    "ratio_qty": 1,
                    "option_type": "call",
                    "strike": cand.strike,
                    "expiry": cand.far_expiry,
                },
            ]
            record_open_positions(
                legs,
                strategy,
                group_id=str(order.get("id") or ""),
                order_id=order.get("id"),
                entry_price=px or None,
                metadata={
                    "earnings_date": cand.earnings_date,
                    "timing": cand.timing,
                    "side": "CALENDAR",
                    "credit": False,
                },
                # The near leg's expiry is the structural deadline. Without it
                # ScheduledExit never fires and the short leg rides into
                # expiry/assignment (2026-09-26: COST near leg assigned →
                # -100 shares).
                exit_by=near_leg_exit_by(cand.near_expiry),
            )
        except Exception as exc:
            record_event("silent_failure", f"fwd_factor_ladder: {exc}")
            logger.error("fwd_factor_ladder broad exception", exc_info=True)
            logger.warning("ff fill bookkeeping failed (non-fatal): %s", exc)

    # ── arm / state ──────────────────────────────────────────────────────

    def arm(self, cand: CalendarCandidate, armed_by: int | None = None) -> int | None:
        """Arm a ladder. Returns ladder id, or None if refused (with event)."""
        from earnings_edge.db.repositories import (
            ff_ladders_armed_id_for_ticker,
            ff_ladders_insert,
        )

        dupe = ff_ladders_armed_id_for_ticker(cand.ticker)
        if dupe:
            self.events.append(f"⚠️ FF ladder for {cand.ticker} already armed (#{dupe}) — arm refused")
            return None

        if self._kill_switch_halted():
            self.events.append(f"⛔ FF arm refused: {cand.ticker} — kill switch is halted")
            return None

        today_et = self._today_et()
        try:
            if date.fromisoformat(cand.earnings_date) < today_et:
                self.events.append(
                    f"⛔ FF arm refused: {cand.ticker} — earnings {cand.earnings_date} already passed"
                )
                return None
        except ValueError:
            pass  # unparseable earnings date — let the step guard handle it

        bp = self._buying_power()
        cost = self._worst_case_cost(cand)
        if bp is None:
            self.events.append(f"⛔ FF arm refused: {cand.ticker} — could not verify buying power")
            return None
        if bp < cost:
            self.events.append(
                f"⛔ FF arm refused: {cand.ticker} — buying power ${bp:,.0f} < worst-case cost ${cost:,.0f}"
            )
            return None

        # Framework risk gate: per-strategy caps (ff_ladder.toml), strategy-day
        # budget, lifecycle. Tolerant: framework absent → ad-hoc gates only.
        decision = self._risk_check_arm(cand, cost)
        if decision is not None and not decision.approved:
            self.events.append(f"⛔ FF arm refused: {cand.ticker} — risk veto: {decision.reason}")
            return None

        lid = ff_ladders_insert(cand.ticker, json.dumps(asdict(cand)), armed_by)
        logger.info(
            "ladder %d armed for %s (walk mid %.2f → min(ask, cap %.2f); D*25%% %.2f)",
            lid,
            cand.ticker,
            cand.mid_debit,
            cand.d_cap,
            cand.d_start,
        )
        return lid

    def _load_armed(self) -> list[ArmedLadder]:
        from earnings_edge.db.repositories import ff_ladders_load_armed

        rows = ff_ladders_load_armed()
        return [
            ArmedLadder(
                id=r["id"],
                candidate=CalendarCandidate(**json.loads(r["candidate_json"])),
                order_id=r["order_id"],
                rung=r["rung"] or 0,
                status=r["status"],
                created_at=r["created_at"],
            )
            for r in rows
        ]

    def _update(self, ladder: ArmedLadder) -> None:
        from earnings_edge.db.repositories import ff_ladders_update_state

        ff_ladders_update_state(ladder.id, ladder.order_id, ladder.rung, ladder.status)

    # ── broker actions ───────────────────────────────────────────────────

    def _place(self, cand: CalendarCandidate, limit: float) -> dict:
        legs = [
            {"symbol": cand.far_symbol, "ratio_qty": 1, "side": "buy"},
            {"symbol": cand.near_symbol, "ratio_qty": 1, "side": "sell"},
        ]
        return self.alpaca.submit_multi_leg_order(
            legs,
            order_type="limit",
            time_in_force="day",
            limit_price=limit,
            qty=1,
        )

    def _cancel_quietly(self, ladder: ArmedLadder) -> None:
        if not ladder.order_id:
            return
        try:
            self.alpaca.cancel_order(ladder.order_id)
        except Exception as exc:
            # exc-policy: keep broad, ensure visibility
            record_event("silent_failure", f"fwd_factor_ladder: {exc}")
            import logging

            logging.getLogger(__name__).error("fwd_factor_ladder broad exception", exc_info=True)
            logger.warning("cancel %s failed: %s", ladder.order_id, exc)
        ladder.order_id = None

    # ── step ─────────────────────────────────────────────────────────────

    def step(self, now: datetime) -> None:
        """One cadence tick: reprice every armed ladder to the current rung."""
        ladders = self._load_armed()
        if not ladders:
            return

        if self._kill_switch_halted():
            for ladder in ladders:
                self._cancel_quietly(ladder)
                ladder.status = "disarmed"
                self._update(ladder)
            self.events.append(f"⛔ Kill switch halted — {len(ladders)} FF ladder(s) cancelled + disarmed")
            return

        bp = self._buying_power()
        today_et = now.astimezone(ET).date()
        for ladder in ladders:
            try:
                self._step_one(ladder, ladder.candidate, now, today_et, bp)
            except Exception as exc:
                # exc-policy: keep broad, ensure visibility
                record_event("silent_failure", f"fwd_factor_ladder: {exc}")
                import logging

                logging.getLogger(__name__).error("fwd_factor_ladder broad exception", exc_info=True)
                if _is_terminal_submit_error(exc):
                    self._cancel_quietly(ladder)
                    ladder.status = "disarmed"
                    self.events.append(
                        f"🚫 FF ladder disarmed: {ladder.candidate.ticker} — broker rejected: {exc}"
                    )
                else:
                    logger.error("ladder %d (%s) step failed: %s", ladder.id, ladder.candidate.ticker, exc)
            finally:
                self._update(ladder)

    @staticmethod
    def _quote_ok(q: dict, now: datetime) -> bool:
        """Positive, ordered, fresh quote."""
        try:
            bp, ap = float(q.get("bp", 0)), float(q.get("ap", 0))
        except (TypeError, ValueError):
            return False
        if bp <= 0 or ap <= 0 or ap < bp:
            return False
        ts = q.get("t")
        if ts:
            try:
                age = (now - datetime.fromisoformat(str(ts).replace("Z", "+00:00"))).total_seconds()
                if age > MAX_QUOTE_AGE_SEC:
                    return False
            except ValueError:
                pass  # unparseable timestamp → can't verify, allow
        return True

    def _step_one(
        self, ladder: ArmedLadder, cand: CalendarCandidate, now: datetime, today_et, bp: float | None
    ) -> None:
        # 0. event already happened — the ladder is dead regardless of orders
        try:
            earnings_passed = date.fromisoformat(cand.earnings_date) < today_et
        except ValueError:
            earnings_passed = False
        if earnings_passed:
            if self._take_fill_if_any(ladder, cand):
                return
            self._cancel_quietly(ladder)
            ladder.status = "expired"
            self.events.append(
                f"⌛ FF ladder expired: {cand.ticker} — earnings {cand.earnings_date} passed unfilled"
            )
            return

        # 1. fill check on the resting order
        if ladder.order_id:
            order = self.alpaca.get_order(ladder.order_id)
            if order.get("status") == "filled":
                ladder.status = "filled"
                self._record_fill(cand, order)
                self.events.append(
                    f"✅ FF ladder filled: {cand.ticker} calendar @ "
                    f"{order.get('filled_avg_price')} ({cand.near_symbol} / {cand.far_symbol})"
                )
                return
            if order.get("status") in ("canceled", "expired", "rejected"):
                ladder.order_id = None  # fall through to re-place

        # 2. refresh quotes → validate → recompute mid + targets
        snaps = self.alpaca.get_option_snapshots_bulk(cand.near_symbol, cand.far_symbol)
        q1 = (snaps.get(cand.near_symbol) or {}).get("latestQuote") or {}
        q2 = (snaps.get(cand.far_symbol) or {}).get("latestQuote") or {}
        if not self._quote_ok(q1, now) or not self._quote_ok(q2, now):
            logger.info("ladder %d: bad/stale quotes for %s, holding", ladder.id, cand.ticker)
            return  # hold silently — transient feed issues shouldn't kill ladders

        # 3. spot drift — strike must still be ~ATM
        spot = self.alpaca.get_stock_latest_trade(cand.ticker)
        if spot is None:
            logger.info("ladder %d: no spot for %s, holding", ladder.id, cand.ticker)
            return
        if cand.spot > 0 and abs(spot - cand.spot) / cand.spot > SPOT_DRIFT_TOLERANCE:
            if self._take_fill_if_any(ladder, cand):
                return
            self._cancel_quietly(ladder)
            ladder.status = "disarmed"
            self.events.append(
                f"🚫 FF ladder disarmed: {cand.ticker} — spot {spot:.2f} drifted "
                f"{abs(spot - cand.spot) / cand.spot:.1%} from {cand.spot:.2f}, strike no longer ATM"
            )
            return

        near_bid, near_ask = float(q1["bp"]), float(q1["ap"])
        far_bid, far_ask = float(q2["bp"]), float(q2["ap"])
        mid = combo_debit(near_bid, near_ask, far_bid, far_ask)
        if mid is None:
            return
        far_mid = (far_bid + far_ask) / 2.0
        today = now.date()
        T1 = (date.fromisoformat(cand.near_expiry) - today).days / 365.0
        tau = max(cand.tau_days, 1) / 365.0
        d_start = target_debit(
            far_mid,
            cand.spot,
            cand.strike,
            T1,
            cand.sigma_fwd,
            tau,
            cand.hist_rms_move,
            self.spec.start_premium,
        )
        d_cap = target_debit(
            far_mid,
            cand.spot,
            cand.strike,
            T1,
            cand.sigma_fwd,
            tau,
            cand.hist_rms_move,
            self.spec.floor_premium,
        )

        # 4. runaway check → cancel + disarm
        if d_cap is None or d_cap <= 0 or not within_fill_range(mid, d_cap, DISTANCE_F):
            if self._take_fill_if_any(ladder, cand):
                return
            self._cancel_quietly(ladder)
            ladder.status = "disarmed"
            self.events.append(
                f"🚫 FF ladder disarmed: {cand.ticker} — mid {mid:.2f} ran beyond "
                f"cap {d_cap if d_cap else float('nan'):.2f} (+{DISTANCE_F:.0%})"
            )
            return

        # 5. buying-power gate before (re)placing
        cost = d_cap * 100.0 * BP_BUFFER
        if bp is None or bp < cost:
            if ladder.id not in self._bp_warned:
                self._bp_warned.add(ladder.id)
                self.events.append(
                    f"⚠️ FF ladder {cand.ticker}: insufficient buying power "
                    f"({'unknown' if bp is None else f'${bp:,.0f}'} < ${cost:,.0f}) — holding, not placing"
                )
            return

        # 6. reprice to the current rung: mid → min(combo ask, D_cap), never
        #    the model price itself (see LadderSpec.market_limit)
        rung = self.spec.rung_index(now)
        if rung is None:
            return  # outside the ladder window — resting order stays as-is
        ask_debit = combo_debit(near_bid, near_ask, far_bid, far_ask, executable=True)
        limit = self.spec.market_limit(rung, mid, ask_debit, d_cap)
        if ladder.order_id:
            order = self.alpaca.get_order(ladder.order_id)
            if abs(float(order.get("limit_price", 0)) - limit) < 1e-9:
                return  # already at the right price
            self._cancel_quietly(ladder)
        new_order = self._place(cand, limit)
        ladder.order_id = new_order.get("id")
        ladder.rung = self.spec.rung_index(now) or 0
        if new_order.get("status") == "filled":
            # marketable limit — filled on placement
            ladder.status = "filled"
            self._record_fill(cand, new_order)
            self.events.append(
                f"✅ FF ladder filled: {cand.ticker} calendar @ "
                f"{new_order.get('filled_avg_price')} ({cand.near_symbol} / {cand.far_symbol})"
            )
            return
        self.events.append(
            f"🪜 FF ladder {cand.ticker}: limit → {limit:.2f} "
            f"(rung {ladder.rung}, mid {mid:.2f}, cap {d_cap:.2f})"
        )

    def drain_events(self) -> list[str]:
        out, self.events = self.events, []
        return out
