"""Per-strategy signal subscriptions for the Telegram bot.

Model: default ON with explicit opt-outs. The proposal flow historically
pushed every strategy's cards to every scanner subscriber; persisting only
opt-OUTS keeps that behavior for existing users (an empty/missing file =
everyone gets everything) while letting each user mute individual strategies
via /signals. The recipient universe is still the union of scanner
subscribers (or the TELEGRAM_APPROVAL_CHAT_ID override) — this store only
subtracts.

Persistence: data/strategy_subscribers.json, shape
    {"opt_outs": {"vol_risk_premium": [123, ...], ...}}

Pure Python, no telegram imports — unit-testable.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Iterable, Optional

logger = logging.getLogger(__name__)

# Strategies that produce signal cards users can mute via /signals.
# The four scan-chained strategies route via build_proposals (14:00 ET);
# ff_ladder builds its own 13:45 ET candidates but persists and confirms
# through the SAME proposal store (build_ff_proposals/execute_proposal) —
# /signals is the single toggle surface for all of them.
SIGNAL_STRATEGIES = [
    "calendar_call_ml",
    "vol_risk_premium",
    "short_straddle",
    "ff_ladder",
    "forward_factor_arb",
]


class StrategySubscriptions:
    """Opt-out store: which user muted which signal strategy.

    Also tracks KNOWN USERS: anyone who ever touched a signal toggle is
    enrolled in the proposal universe (the old pool was scanner subscribers
    only — with the unified /signals surface, interacting with /signals alone
    must be enough to receive cards).
    """

    def __init__(self, path: str):
        self._path = path
        self._opt_outs: dict[str, set[int]] = {}
        self._known: set[int] = set()
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self._path):
            return
        try:
            with open(self._path) as f:
                raw = json.load(f)
            self._opt_outs = {
                str(k): {int(u) for u in v}
                for k, v in (raw.get("opt_outs") or {}).items()
            }
            self._known = {int(u) for u in (raw.get("known_users") or [])}
        except Exception as exc:
            logger.error("Failed to load strategy subscriptions: %s", exc)
            self._opt_outs = {}
            self._known = set()

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self._path), exist_ok=True)
        tmp = self._path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(
                {"opt_outs": {k: sorted(v) for k, v in self._opt_outs.items()},
                 "known_users": sorted(self._known)},
                f,
                indent=2,
            )
        os.replace(tmp, self._path)

    def is_subscribed(self, strategy: str, uid: int) -> bool:
        return uid not in self._opt_outs.get(strategy, set())

    def set_subscribed(self, strategy: str, uid: int, on: bool) -> None:
        if strategy not in SIGNAL_STRATEGIES:
            raise KeyError(f"unknown signal strategy {strategy!r}")
        if on:
            outs = self._opt_outs.get(strategy)
            changed = False
            if outs and uid in outs:
                outs.discard(uid)
                if not outs:
                    del self._opt_outs[strategy]
                changed = True
            if uid not in self._known:
                self._known.add(uid)
                changed = True
            if changed:
                self._save()
        else:
            self._opt_outs.setdefault(strategy, set()).add(uid)
            self._known.add(uid)
            self._save()

    def known_users(self) -> set[int]:
        """Users who have ever interacted with /signals (enrolled in the pool)."""
        return set(self._known)

    def recipients(self, strategy: str, universe: Iterable[int]) -> set[int]:
        """Universe minus opt-outs for this strategy."""
        return set(universe) - self._opt_outs.get(strategy, set())


def partition_by_mode(rows: list[dict], mode_fn) -> tuple[list[dict], list[dict]]:
    """Split proposal rows into (approval, auto) by per-strategy mode.

    ``mode_fn`` is a callable strategy_name -> 'approval' | 'auto'
    (e.g. framework.core.control.effective_execution_mode bound to a conn
    and the registry's TOML defaults). Pure — the bot does IO around it.
    """
    approval = [r for r in rows if mode_fn(r["strategy"]) != "auto"]
    auto = [r for r in rows if mode_fn(r["strategy"]) == "auto"]
    return approval, auto


def route_proposals(
    proposals: list[dict],
    *,
    universe: set[int],
    subs: Optional[StrategySubscriptions],
    override_chat: Optional[int] = None,
) -> dict[int, list[dict]]:
    """Per-chat proposal routing.

    override_chat (TELEGRAM_APPROVAL_CHAT_ID) receives ALL strategies
    regardless of opt-outs. Otherwise each proposal goes to the universe
    minus that strategy's opt-outs. Proposals with zero recipients are
    dropped from the result (they remain persisted in pending_trades).
    """
    routed: dict[int, list[dict]] = {}
    for row in proposals:
        if override_chat is not None:
            targets = {override_chat}
        elif subs is not None:
            targets = subs.recipients(row["strategy"], universe)
        else:
            targets = set(universe)
        for uid in targets:
            routed.setdefault(uid, []).append(row)
    return routed


def funnel_line(funnel: Optional[dict]) -> str:
    """One-line plain-text funnel summary for the proposal batch message.

    Aggregated across strategies: max input rows, then summed stage counts.
    Empty/missing funnel -> empty string (caller omits the line).
    """
    if not funnel:
        return ""
    stages = funnel.get("strategies") or {}
    if not stages:
        return ""
    scanned = max(
        (s.get("rows_scanned", 0) for s in stages.values() if s.get("rows_scanned", 0) >= 0),
        default=0,
    )
    decision = sum(s.get("decision_pass", 0) for s in stages.values())
    legs = sum(s.get("legs_ok", 0) for s in stages.values())
    dte = sum(s.get("dte_ok", 0) for s in stages.values())
    pos = sum(s.get("position_ok", 0) for s in stages.values())
    proposals = funnel.get("proposals", 0)
    return (
        f"funnel: {scanned} scanned -> {decision} decision -> {legs} legs "
        f"-> {dte} dte -> {pos} no-pos -> {proposals} proposals"
    )
