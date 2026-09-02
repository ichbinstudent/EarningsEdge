"""Fail-closed operator authorization for the Telegram desk.

If TELEGRAM_APPROVAL_CHAT_ID (or TELEGRAM_APPROVAL_CHAT_IDS) is unset or
unparseable, *nobody* may halt, resume, promote, restart, execute, or close.
An empty allow-list is not “everyone” — that was the old fail-open hole.
"""
from __future__ import annotations

import os
from typing import Optional


AUTH_REFUSED = (
    "⛔ Operator lock is on and this chat is not authorized "
    "(set TELEGRAM_APPROVAL_CHAT_ID)."
)
AUTH_UNCONFIGURED = (
    "⛔ Operator lock is unconfigured — halt / execute / close / promote / "
    "restart are disabled until TELEGRAM_APPROVAL_CHAT_ID is set."
)


def operator_chat_ids(env: Optional[dict] = None) -> list[int]:
    """Parse the allow-list. Empty ⇒ lock is unconfigured (fail closed)."""
    src = env if env is not None else os.environ
    raw = (src.get("TELEGRAM_APPROVAL_CHAT_IDS") or src.get("TELEGRAM_APPROVAL_CHAT_ID") or "").strip()
    if not raw:
        return []
    out: list[int] = []
    for part in raw.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(part))
        except ValueError:
            continue
    return out


def operators_configured(env: Optional[dict] = None) -> bool:
    return bool(operator_chat_ids(env))


def is_operator(chat_id: Optional[int], env: Optional[dict] = None) -> bool:
    """True only when an allow-list exists *and* chat_id is on it."""
    if chat_id is None:
        return False
    allowed = operator_chat_ids(env)
    if not allowed:
        return False
    return int(chat_id) in allowed


def auth_message(env: Optional[dict] = None) -> str:
    return AUTH_UNCONFIGURED if not operators_configured(env) else AUTH_REFUSED
