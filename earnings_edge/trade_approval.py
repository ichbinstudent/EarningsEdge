"""Human-in-the-loop trade approval: propose -> Telegram confirm -> execute.

Flow:
1. `build_proposals()` runs the daily strategies, keeps only TAKE trades that
   pass the bridge's LOCAL gates (legs, DTE, position — zero API calls for
   filtered candidates), ranks by model score, and persists the top N as
   pending proposals with a pre-rendered card text.
2. The Telegram bot pushes each card with Execute/Skip buttons.
3. `execute_proposal()` re-validates at click time (pending? fresh? DTE and
   position still clean?) and only then submits to Alpaca.

Proposals are session-scoped: anything older than PROPOSAL_TTL_HOURS expires
and can never execute. Auto-execution only happens when the strategy's
effective mode is ``auto`` (TOML + operator override) AND the bot's
``partition_by_mode`` path runs it — live mode forces approval unless
``ALPACA_LIVE_ALLOW_AUTO=1``.
"""
from __future__ import annotations

import json
import logging
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from earnings_edge import cards, live_signals
from earnings_edge.alpaca_bridge import BridgeConfig, StrategyBridge, _resolve_strategy
from earnings_edge.alpaca_trading import create_client
from earnings_edge.db import (
    configure,
    get_engine,
    managed_positions_open,
    pending_trades_get,
    pending_trades_insert,
    pending_trades_list_pending,
    pending_trades_mark_decided,
    pending_trades_update_card,
    proposal_funnel_insert,
)
from earnings_edge.trading_types import DataBundle, Trade

logger = logging.getLogger(__name__)

PROPOSAL_TTL_HOURS = 8.0  # proposals are same-session only
FF_LADDER = "ff_ladder"
FF_WINDOW_END_ET = time(15, 45)  # armed ladders stop stepping here
DEFAULT_STRATEGIES = [
    "calendar_call_ml",
    "short_straddle",
    "vol_risk_premium",
    FF_LADDER,
    "forward_factor_arb",
    "earnings_quality"
]
# Per-strategy decisions that may become a proposal card. calendar_call_ml
# is model TAKE only.
TAKE_DECISIONS = {
    "calendar_call_ml": frozenset({"TAKE"}),
    "short_straddle": frozenset({"TAKE", "VOL_GATE"}),
    "vol_risk_premium": frozenset({"TAKE", "VOL_GATE"}),
}

# Funnel counters of the most recent build_proposals run (per strategy ->
# stage counts). The bot reads this to render the funnel summary line; it is
# also persisted to the proposal_funnel table for audit.
LAST_FUNNEL: dict = {}

# Kept for tests/scripts that apply the approval schema via executescript
# on a raw DB-API connection (create_all covers this for repository callers).
_SCHEMA = """
CREATE TABLE IF NOT EXISTS pending_trades (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at  TEXT NOT NULL,
    strategy    TEXT NOT NULL,
    ticker      TEXT NOT NULL,
    side        TEXT NOT NULL,
    trade_json  TEXT NOT NULL,
    card_text   TEXT NOT NULL,
    model_score REAL,
    status      TEXT NOT NULL DEFAULT 'pending',
    order_json  TEXT,
    note        TEXT,
    decided_by  INTEGER,
    decided_at  TEXT
);
CREATE TABLE IF NOT EXISTS proposal_funnel (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at      TEXT NOT NULL,
    strategies      TEXT NOT NULL,
    counts          TEXT NOT NULL,
    proposals_total INTEGER NOT NULL
)
"""


# ---------------------------------------------------------------------------
# Trade (de)serialization
# ---------------------------------------------------------------------------

def _json_default(o: Any) -> Any:
    if isinstance(o, (date, datetime)):
        return o.isoformat()
    try:  # numpy scalars leak out of pandas rows
        import numpy as np

        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
    except ImportError:
        pass
    return str(o)


def trade_to_json(t: Trade) -> str:
    return json.dumps(
        {
            "ticker": t.ticker,
            "earnings_date": t.earnings_date.isoformat(),
            "scan_date": t.scan_date.isoformat(),
            "strategy": t.strategy,
            "side": t.side,
            "entry_price": t.entry_price,
            "exit_price": t.exit_price,
            "pnl": t.pnl,
            "pnl_pct": t.pnl_pct,
            "features": t.features,
            "model_score": t.model_score,
            "ml_decision": t.ml_decision,
            "notes": t.notes,
        },
        default=_json_default,
    )


def trade_from_json(s: str) -> Trade:
    d = json.loads(s)
    return Trade(
        ticker=d["ticker"],
        earnings_date=date.fromisoformat(d["earnings_date"]),
        scan_date=date.fromisoformat(d["scan_date"]),
        strategy=d["strategy"],
        side=d["side"],
        entry_price=d["entry_price"],
        exit_price=d.get("exit_price", 0.0),
        pnl=d.get("pnl", 0.0),
        pnl_pct=d.get("pnl_pct", 0.0),
        features=d.get("features") or {},
        model_score=d.get("model_score"),
        ml_decision=d.get("ml_decision", "TAKE"),
        notes=d.get("notes", ""),
    )


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

class PendingTradeStore:
    """SQLite-backed pending-proposal store (lives in earnings_ml.db, WAL)."""

    def __init__(self, db_path: Optional[str] = None):
        self._db_path = Path(db_path) if db_path else None
        self._ensure_engine()

    def _ensure_engine(self) -> None:
        if self._db_path is not None:
            configure(self._db_path)
        else:
            get_engine()

    def add(self, trade: Trade, card_text: str) -> Optional[int]:
        """Insert a proposal; returns id, or None if an identical proposal is
        already pending (same strategy+ticker+side — no double-asking)."""
        self._ensure_engine()
        return pending_trades_insert(
            created_at=datetime.now(timezone.utc).isoformat(),
            strategy=trade.strategy,
            ticker=trade.ticker,
            side=trade.side,
            trade_json=trade_to_json(trade),
            card_text=card_text,
            model_score=trade.model_score,
        )

    def update_card(self, proposal_id: int, card_text: str) -> None:
        self._ensure_engine()
        pending_trades_update_card(proposal_id, card_text)

    def get(self, proposal_id: int) -> Optional[dict]:
        self._ensure_engine()
        return pending_trades_get(proposal_id)

    def list_pending(self) -> list[dict]:
        self._ensure_engine()
        return pending_trades_list_pending()

    def mark(
        self,
        proposal_id: int,
        status: str,
        *,
        order_json: Optional[dict] = None,
        note: Optional[str] = None,
        decided_by: Optional[int] = None,
    ) -> None:
        self._ensure_engine()
        pending_trades_mark_decided(
            proposal_id,
            status,
            order_json=json.dumps(order_json, default=_json_default) if order_json else None,
            note=note,
            decided_by=decided_by,
        )


# ---------------------------------------------------------------------------
# Proposal building (local filtering only — see alpaca_bridge cost discipline)
# ---------------------------------------------------------------------------

def _render_card(trade: Trade, legs: list[dict], proposal_id: str = "?") -> str:
    subtitle = (f"{cards.bold(trade.ticker)} {cards.esc(trade.side)} "
                f"(earnings {trade.earnings_date.isoformat()})")
    body = [
        f"  {cards.esc(leg['side'].upper())} {cards.esc(leg.get('ratio_qty', 1))} "
        f"{cards.code(leg['symbol'])}"
        for leg in legs
    ]
    meta = []
    if trade.entry_price:
        meta.append(f"est. debit ${trade.entry_price:.2f}")
    if trade.model_score is not None:
        meta.append(f"ML score {trade.model_score:.3f}")

    # Wire up position designer to show max loss / max profit
    try:
        import math
        import datetime
        from earnings_edge.designer import Leg, analyze

        designer_legs = []
        is_credit = trade.side in {"SHORT_STRADDLE", "SHORT_STRANGLE", "IRON_CONDOR"}
        target_premium = -float(trade.entry_price or 0.0) if is_credit else float(trade.entry_price or 0.0)
        
        assigned = False
        for l in legs:
            action = l["side"]
            ratio = int(l.get("ratio_qty", 1))
            
            price = 0.0
            if not assigned and ratio > 0:
                sq = ratio if action == "buy" else -ratio
                price = target_premium / sq
                assigned = True

            try:
                if isinstance(l.get("expiry"), str):
                    exp = datetime.date.fromisoformat(l["expiry"])
                else:
                    exp = l.get("expiry") or trade.earnings_date
            except Exception:
                exp = trade.earnings_date
                
            designer_legs.append(Leg(
                action=action,
                kind=l.get("option_type", "call"),
                strike=float(l.get("strike", 0.0)),
                expiry=exp,
                quantity=ratio,
                price=price,
                iv=0.30
            ))
            
        S = float((trade.features or {}).get("price") or designer_legs[0].strike)
        if S == 0.0:
            S = 100.0
            
        summary = analyze(designer_legs, S=S)
        mp = summary["max_profit"]
        ml = summary["max_loss"]
        
        profit_str = "Unlimited" if math.isinf(mp) else f"${mp:.0f}"
        loss_str = "Unlimited" if math.isinf(ml) or math.isinf(-ml) else f"${abs(ml):.0f}"
        meta.append(f"Max Profit: {profit_str} | Max Loss: {loss_str}")
        
    except Exception as e:
        logger.warning("designer analyze failed in proposal build: %s", e)

    if meta:
        body.append(cards.esc(" | ".join(meta)))
    if trade.notes:
        body.append(f"note: {cards.esc(trade.notes[:120])}")
        
    try:
        cmd_parts = [f"/designer {trade.ticker}"]
        for l in designer_legs:
            # action kind strike expiry qty
            cmd_parts.append(f"{l.action} {l.kind} {l.strike:g} {l.expiry} {l.quantity}")
        cmd_str = " ".join(cmd_parts)
        body.append("")
        body.append(f"<code>{cmd_str}</code>")
    except Exception:
        pass
        
    footer = cards.esc(f"Expires in {PROPOSAL_TTL_HOURS:.0f}h — confirm to execute.")
    return cards.card_frame(cards.ENTRY_EMOJI, f"Trade Proposal #{proposal_id} — {trade.strategy}",
                            subtitle, body, footer)


def _killswitch_note(db_path=None) -> str:
    """Card prefix when the kill switch is halted (proposals still build)."""
    try:
        from framework.risk.killswitch import KillSwitch
        if db_path is not None:
            from earnings_edge.db.engine import configure
            configure(db_path)
        ks = KillSwitch()
        if ks.is_halted():
            return cards.bold(
                f"🛑 KILL SWITCH HALTED ({ks.status().get('reason')}) — execution will be vetoed")
    except Exception:
        pass
    return ""


def _default_trade_source(db_path):
    """Live signal source: latest scan session -> per-strategy Trade lists."""
    df = live_signals.latest_scan_frame(db_path)
    return df, lambda name: live_signals.build_live_trades(df, name)


# ---------------------------------------------------------------------------
# FF ladder candidates as proposals (same store, same confirm path)
# ---------------------------------------------------------------------------

def ff_candidate_to_trade(cand) -> Trade:
    """Wrap an FF ladder candidate as a Trade so it persists in the
    pending_trades store exactly like every other strategy's proposal.
    The full candidate round-trips through features['candidate']."""
    from dataclasses import asdict

    return Trade(
        ticker=cand.ticker,
        earnings_date=date.fromisoformat(cand.earnings_date),
        scan_date=datetime.now(timezone.utc).date(),
        strategy=getattr(cand, "strategy_override", None) or FF_LADDER,
        side="CALENDAR",
        entry_price=cand.mid_debit,
        features={"candidate": asdict(cand)},
        model_score=None,
        ml_decision="TAKE",
        notes=f"ladder {cand.d_start:.2f} -> cap {cand.d_cap:.2f}",
    )


def ff_candidate_from_trade(trade: Trade):
    from earnings_edge.fwd_factor_ladder import CalendarCandidate

    return CalendarCandidate(**trade.features["candidate"])


def _render_ff_card(cand, proposal_id: str = "?") -> str:
    subtitle = f"{cards.bold(cand.ticker)} CALENDAR (earnings {cards.esc(cand.earnings_date)})"
    body = [
        f"  BUY  1 {cards.code(cand.far_symbol)}",
        f"  SELL 1 {cards.code(cand.near_symbol)}",
        cards.esc(f"spot {cand.spot:.2f} | strike {cand.strike:g}"),
        cards.esc(f"mid debit {cand.mid_debit:.2f} | ladder {cand.d_start:.2f} → cap {cand.d_cap:.2f}"),
        cards.esc(f"σ_fwd {cand.sigma_fwd:.1%} | hist RMS {cand.hist_rms_move:.1%} | τ {cand.tau_days}d"),
    ]
    
    try:
        # Extrapolate option kind (call/put) from OCC symbol
        near_kind = "put" if cand.near_symbol[-9] == "P" else "call"
        far_kind = "put" if cand.far_symbol[-9] == "P" else "call"
        cmd_str = f"/designer {cand.ticker} sell {near_kind} {cand.strike:g} {cand.near_expiry} 1 buy {far_kind} {cand.strike:g} {cand.far_expiry} 1"
        body.append("")
        body.append(f"<code>{cmd_str}</code>")
    except Exception:
        pass
        
    footer = ("Confirm = arm limit ladder 14:00→15:45 ET, tick up every 15 min.\n"
              "Expires 15:45 ET today.")
    return cards.card_frame(cards.FF_EMOJI, f"Trade Proposal #{proposal_id} — {FF_LADDER}",
                            subtitle, body, footer)


def build_ff_proposals(
    store: PendingTradeStore,
    candidates: list,
    *,
    max_proposals: int = 10,
) -> list[dict]:
    """Persist FF ladder candidates as pending proposals and return the rows.

    This is the same internal process as build_proposals(): store -> card ->
    Execute/Skip keyboard -> execute_proposal() on confirm (which arms the
    ladder instead of submitting a combo order)."""
    halted_note = _killswitch_note(store._db_path)
    proposals: list[dict] = []
    for cand in candidates:
        pid = store.add(ff_candidate_to_trade(cand), "")
        if pid is None:
            continue  # identical proposal already pending
        card = _render_ff_card(cand, str(pid))
        if halted_note:
            card = halted_note + "\n" + card
        store.update_card(pid, card)
        row = store.get(pid)
        if row is not None:
            proposals.append(row)
        if len(proposals) >= max_proposals:
            break
    logger.info("ff proposals built: %d (candidates: %d)", len(proposals), len(candidates))
    return proposals


def _persist_funnel(store: "PendingTradeStore", strategies: list[str], counts: dict, total: int) -> None:
    try:
        store._ensure_engine()
        proposal_funnel_insert(
            created_at=datetime.now(timezone.utc).isoformat(),
            strategies=json.dumps(strategies),
            counts=json.dumps(counts),
            proposals_total=total,
        )
    except Exception as exc:
        logger.warning("funnel persist failed (non-fatal): %s", exc)


def build_proposals(
    store: PendingTradeStore,
    *,
    strategies: Optional[list[str]] = None,
    max_proposals: int = 5,
    db_path: Optional[str] = None,
    bridge: Optional[StrategyBridge] = None,
    bundle: Optional[DataBundle] = None,
    trade_source=None,
) -> list[dict]:
    """Run live signal mappings, filter TAKE trades locally, persist top-N.

    Signals come from the LATEST SCAN SESSION via earnings_edge.live_signals
    (real strikes/expiries/quoted debits) — not from the backtest strategies,
    which replay historical rows and can never produce a live trade (their
    legs fall back to expiry = earnings_date -> DTE 0). `bundle` is accepted
    for backward compatibility only and is no longer used.

    Only genuinely actionable trades become proposals: valid legs, DTE inside
    [min, max], no existing position. Filtered candidates cost ZERO API calls
    (OCC construction + one cached position fetch). Per-stage funnel counts
    are persisted to proposal_funnel and exposed as LAST_FUNNEL.

    `trade_source`: optional callable strategy_name -> list[Trade] (tests).
    """
    bridge = bridge or StrategyBridge(client=create_client(), config=BridgeConfig())
    cfg = bridge.config

    # Strategy configs gate which strategies run ([strategy] enabled = false
    # disables without code changes); unconfigured strategies stay enabled.
    # Operator runtime overrides (strategy_state.enabled, set via the bot)
    # apply on top of the TOML flags.
    try:
        from framework.core.control import filter_enabled
        from framework.core.registry import get_registry
        from earnings_edge.db.engine import configure
        registry = get_registry()
        if store._db_path:
            configure(store._db_path)
        names = filter_enabled(strategies or DEFAULT_STRATEGIES,
                               registry.is_enabled)
        halted_note = _killswitch_note(store._db_path)
    except Exception as exc:
        logger.warning("registry unavailable (%s) — all strategies enabled", exc)
        names = strategies or DEFAULT_STRATEGIES
        halted_note = ""

    unmapped = [n for n in names if n not in live_signals.LIVE_STRATEGIES]
    if unmapped:
        logger.warning("strategies without a live signal mapping, skipped: %s", unmapped)
    names = [n for n in names if n in live_signals.LIVE_STRATEGIES]

    scan_df = None
    if trade_source is None:
        scan_df, trade_source = _default_trade_source(db_path or store._db_path)
        rows_scanned = len(scan_df)
    else:
        rows_scanned = -1  # injected source; input size not meaningful

    funnel: dict[str, dict] = {}
    candidates: list[tuple[Trade, list[dict]]] = []
    for name in names:
        stage = {
            "rows_scanned": rows_scanned,
            "decision_pass": 0,
            "legs_ok": 0,
            "dte_ok": 0,
            "position_ok": 0,
            "proposals_created": 0,
            "reject_model_skip": 0,
            "reject_no_quote": 0,
            "reject_dte": 0,
        }
        try:
            trades = trade_source(name)
        except Exception as exc:
            logger.error("live signal mapping for %s failed: %s", name, exc)
            funnel[name] = {**stage, "error": str(exc)}
            continue
        stage["decision_pass"] = len(trades)
        if name == "calendar_call_ml" and scan_df is not None:
            reasons = live_signals.calendar_funnel_reasons(scan_df)
            stage["reject_model_skip"] = reasons.get("model_skip", 0)
            stage["reject_no_quote"] = reasons.get("no_quote", 0)
        allowed = TAKE_DECISIONS.get(name, frozenset({"TAKE"}))
        for trade in trades:
            if trade.ml_decision not in allowed:
                continue
            legs = bridge._build_legs(trade)
            if not legs:
                continue
            stage["legs_ok"] += 1
            min_expiry = bridge._min_expiry(legs)
            if min_expiry is not None and trade.earnings_date:
                dte = (min_expiry - trade.earnings_date).days
                if not (cfg.max_dte_min <= dte <= cfg.max_dte_max):
                    stage["reject_dte"] += 1
                    continue
            stage["dte_ok"] += 1
            if cfg.skip_if_position_exists:
                positions = bridge._position_set()
                if any(leg["symbol"] in positions for leg in legs):
                    continue
            stage["position_ok"] += 1
            candidates.append((trade, legs))
        funnel[name] = stage

    # rank: model score first (None last); dedupe per (ticker, side)
    candidates.sort(
        key=lambda tl: (tl[0].model_score is not None, tl[0].model_score or 0.0),
        reverse=True,
    )
    seen: set[tuple[str, str]] = set()
    proposals: list[dict] = []
    for trade, legs in candidates:
        key = (trade.ticker, trade.side)
        if key in seen:
            continue
        seen.add(key)
        pid = store.add(trade, "")
        if pid is None:
            continue  # identical proposal already pending
        card = _render_card(trade, legs, str(pid))
        if halted_note:
            card = halted_note + "\n" + card
        store.update_card(pid, card)
        row = store.get(pid)
        if row is not None:
            proposals.append(row)
            funnel[trade.strategy]["proposals_created"] += 1
        if len(proposals) >= max_proposals:
            break
    global LAST_FUNNEL
    LAST_FUNNEL = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "candidates": len(candidates),
        "proposals": len(proposals),
        "strategies": funnel,
    }
    _persist_funnel(store, list(funnel), funnel, len(proposals))
    logger.info(
        "proposals built: %d (candidates: %d) | funnel: %s",
        len(proposals), len(candidates), json.dumps(funnel),
    )
    return proposals


# ---------------------------------------------------------------------------
# Execution (only ever triggered by explicit confirmation)
# ---------------------------------------------------------------------------

def _market_closed() -> Optional[str]:
    """None when the US market is open, else a human-readable refusal.

    Guards the click-time submission path: a stale card confirmed after the
    16:00 ET close must never submit an order."""
    try:
        clock = create_client().get_clock()
    except Exception as exc:
        return f"market clock check failed ({exc}) — refusing to submit blind"
    if not clock.get("is_open"):
        return "US market is closed — confirm during 09:30–16:00 ET"
    return None


def _execute_ff(
    store: PendingTradeStore,
    proposal_id: int,
    trade: Trade,
    row: dict,
    *,
    decided_by: Optional[int] = None,
    ff_runner=None,
    now: Optional[datetime] = None,
) -> dict:
    """Arm an FF ladder from a persisted proposal. Same confirm path as every
    other strategy; the ladder then walks 14:00–15:45 ET on the step cron.
    Arming itself is risk-gated inside LadderRunner.arm (kill switch, buying
    power, per-strategy caps)."""
    import pytz

    eastern = pytz.timezone("US/Eastern")
    now_utc = now or datetime.now(timezone.utc)
    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=timezone.utc)
    now_et = now_utc.astimezone(eastern)
    created = datetime.fromisoformat(row["created_at"])
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    created_et = created.astimezone(eastern)
    if created_et.date() != now_et.date():
        store.mark(proposal_id, "expired",
                   note=f"stale FF candidate (built {created_et.date()} ET)",
                   decided_by=decided_by)
        return {"ok": False,
                "error": f"proposal #{proposal_id} is stale (built {created_et.date()} ET) — wait for today's 13:45 ET batch"}
    if now_et.time() >= FF_WINDOW_END_ET:
        store.mark(proposal_id, "expired", note="ladder window closed",
                   decided_by=decided_by)
        return {"ok": False,
                "error": "ladder window (14:00–15:45 ET) has closed for today"}

    if ff_runner is None:
        from earnings_edge.fwd_factor_ladder import LadderRunner

        ff_runner = LadderRunner(create_client())
    lid = ff_runner.arm(ff_candidate_from_trade(trade), decided_by)
    if lid is None:
        events = ff_runner.drain_events()
        reason = events[-1] if events else "arm refused"
        store.mark(proposal_id, "error", note=reason, decided_by=decided_by)
        return {"ok": False, "error": reason}
    store.mark(proposal_id, "executed",
               order_json={"ladder_id": lid, "status": "armed"},
               decided_by=decided_by)
    logger.info("proposal %d armed FF ladder %d (%s)", proposal_id, lid, trade.ticker)
    return {"ok": True, "ladder_id": lid, "status": "armed",
            "order_id": f"ff-ladder-{lid}"}


def execute_proposal(
    store: PendingTradeStore,
    proposal_id: int,
    *,
    bridge: Optional[StrategyBridge] = None,
    decided_by: Optional[int] = None,
    ff_runner=None,
    now: Optional[datetime] = None,
) -> dict:
    """Execute one pending proposal after operator confirmation.

    Re-validates everything at click time — a stale card in a chat history
    must never submit an order.
    """
    row = store.get(proposal_id)
    if row is None:
        return {"ok": False, "error": f"proposal #{proposal_id} not found"}
    if row["status"] != "pending":
        return {"ok": False, "error": f"proposal #{proposal_id} already {row['status']}"}

    created = datetime.fromisoformat(row["created_at"])
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    now_utc = now or datetime.now(timezone.utc)
    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=timezone.utc)

    trade = trade_from_json(row["trade_json"])
    if trade.strategy in (FF_LADDER, "forward_factor_arb"):
        # FF cards carry their own freshness guards (same ET day + 14:00–15:45
        # arming window) inside _execute_ff; the generic session TTL does not
        # apply — a card built at the 13:45 batch is confirmed within minutes.
        return _execute_ff(store, proposal_id, trade, row,
                           decided_by=decided_by, ff_runner=ff_runner, now=now_utc)
    age = now_utc - created
    if age > timedelta(hours=PROPOSAL_TTL_HOURS):
        store.mark(proposal_id, "expired", note=f"age {age}", decided_by=decided_by)
        return {"ok": False, "error": f"proposal #{proposal_id} expired ({age.total_seconds() / 3600:.1f}h old)"}

    if bridge is None:
        # Production path (no injected bridge): never submit when the US
        # market is closed — a stale card confirmed after the 16:00 ET close
        # would otherwise queue an order overnight.
        closed = _market_closed()
        if closed:
            return {"ok": False, "error": closed}
    if bridge is None:
        # Risk-gated bridge: kill switch, portfolio caps, lifecycle state all
        # apply even though this path is human-approved.
        from framework.core.registry import get_registry
        from framework.execution.lifecycle import LifecycleManager
        from framework.risk.manager import RiskManager
        from earnings_edge.db.engine import configure
        if store._db_path:
            configure(store._db_path)
        try:
            registry = get_registry()
            resolver = registry.limits_for
            sizer_resolver = registry.sizer_spec
        except Exception:
            resolver = None
            sizer_resolver = None
        from earnings_edge.alpaca_bridge import (
            LIVE_FILL_POLL_ATTEMPTS, LIVE_FILL_POLL_SECS,
        )
        bridge = StrategyBridge(
            client=create_client(), config=BridgeConfig(),
            risk_manager=RiskManager(),
            lifecycle_manager=LifecycleManager(),
            limits_resolver=resolver,
            sizer_resolver=sizer_resolver,
        )
        bridge._fill_poll_attempts = LIVE_FILL_POLL_ATTEMPTS
        bridge._fill_poll_secs = LIVE_FILL_POLL_SECS
    # bridge.execute_trade re-runs the DTE + position guards live
    result = bridge.execute_trade(trade)
    if result is None:
        reasons = dict(bridge.skip_reasons)
        store.mark(proposal_id, "error", note=f"bridge skip: {reasons}", decided_by=decided_by)
        return {"ok": False, "error": f"skipped at execution: {reasons or 'bridge rejected'}"}

    order = {
        "order_id": result.order_id,
        "status": result.status,
        "filled_qty": result.filled_qty,
        "filled_avg_price": result.filled_avg_price,
    }
    store.mark(proposal_id, "executed", order_json=order, decided_by=decided_by)

    # Only book a position on an actual fill. An accepted-but-unfilled
    # limit is still working at the broker; reconcile adopts it on fill.
    if not result.filled_qty:
        return {"ok": True, **order, "booked": False}

    # Track legs as managed positions (reconcile + guards read these).
    try:
        from framework.positions.exits import CREDIT_SIDES
        store._ensure_engine()
        managed_positions_open(
            result.legs, trade.strategy,
            group_id=result.order_id, order_id=result.order_id,
            entry_price=trade.entry_price or None,
            exit_by=result.exit_by,
            metadata={
                "side": trade.side,
                "credit": trade.side in CREDIT_SIDES,
                "earnings_date": str(trade.earnings_date),
            },
        )
    except Exception as exc:
        logger.warning("managed-position record failed (non-fatal): %s", exc)
    return {"ok": True, **order}


def reject_proposal(store: PendingTradeStore, proposal_id: int, *, decided_by: Optional[int] = None) -> dict:
    row = store.get(proposal_id)
    if row is None:
        return {"ok": False, "error": f"proposal #{proposal_id} not found"}
    if row["status"] != "pending":
        return {"ok": False, "error": f"proposal #{proposal_id} already {row['status']}"}
    store.mark(proposal_id, "rejected", decided_by=decided_by)
    return {"ok": True}
