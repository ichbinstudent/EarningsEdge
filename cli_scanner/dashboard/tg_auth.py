"""Telegram Mini App initData verification (HMAC) + operator gate."""
from __future__ import annotations

import hashlib
import hmac
import json
import os
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import parse_qsl

from earnings_edge.ops_auth import is_operator, operators_configured


class InitDataError(ValueError):
    """initData missing, forged, stale, or not an operator."""


def webapp_url(env: Optional[dict] = None) -> Optional[str]:
    """Public HTTPS origin of the Mini App, or None if unset/invalid."""
    src = env if env is not None else os.environ
    raw = (src.get("TELEGRAM_WEBAPP_URL") or "").strip()
    if raw.startswith("https://"):
        # Trailing slash avoids a Funnel/host redirect that drops #tgWebAppData.
        return raw if raw.endswith("/") else raw + "/"
    return None


def _bot_token(env: Optional[dict] = None) -> str:
    src = env if env is not None else os.environ
    return (src.get("TELEGRAM_BOT_TOKEN") or "").strip().strip('"').strip("'")


def verify_init_data(
    init_data: str,
    bot_token: Optional[str] = None,
    *,
    now: Optional[datetime] = None,
    max_age_s: int = 24 * 3600,
) -> dict:
    """Validate Telegram WebApp initData. Returns ``{user_id, auth_date, user}``."""
    token = (bot_token if bot_token is not None else _bot_token()).strip().strip('"').strip("'")
    if not token:
        raise InitDataError("bot token unset")
    raw = (init_data or "").strip()
    if not raw:
        raise InitDataError("initData missing")
    fields = dict(parse_qsl(raw, keep_blank_values=True))
    got = fields.pop("hash", "")
    if not got:
        raise InitDataError("hash missing")
    data_check = "\n".join(f"{k}={v}" for k, v in sorted(fields.items()))
    secret = hmac.new(b"WebAppData", token.encode(), hashlib.sha256).digest()
    expect = hmac.new(secret, data_check.encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expect, got):
        raise InitDataError("bad hash")
    try:
        auth_date = int(fields.get("auth_date") or "0")
    except ValueError as exc:
        raise InitDataError("bad auth_date") from exc
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    if max_age_s and (int(now.timestamp()) - auth_date) > max_age_s:
        raise InitDataError("initData expired")
    user = _json_field(fields.get("user"))
    chat = _json_field(fields.get("chat"))
    user_id = user.get("id")
    chat_id = chat.get("id")
    if user_id is None and chat_id is None:
        raise InitDataError("no user id")
    return {
        "user_id": int(user_id) if user_id is not None else None,
        "chat_id": int(chat_id) if chat_id is not None else None,
        "auth_date": auth_date,
        "user": user,
        "chat": chat,
    }


def _json_field(raw) -> dict:
    if not raw:
        return {}
    try:
        val = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return val if isinstance(val, dict) else {}


def require_operator(
    init_data: str,
    bot_token: Optional[str] = None,
    *,
    env: Optional[dict] = None,
    now: Optional[datetime] = None,
) -> int:
    """Verify initData and fail-closed operator lock. Returns Telegram user id."""
    parsed = verify_init_data(init_data, bot_token, now=now)
    uid = parsed.get("user_id")
    chat_id = parsed.get("chat_id")
    if not operators_configured(env):
        raise InitDataError("operator lock unconfigured")
    if is_operator(uid, env) or is_operator(chat_id, env):
        return int(uid if uid is not None else chat_id)
    raise InitDataError("not an operator")


def sign_init_data(fields: dict, bot_token: str) -> str:
    """Test helper: build a valid initData query string."""
    payload = dict(fields)
    payload.pop("hash", None)
    data_check = "\n".join(f"{k}={v}" for k, v in sorted(payload.items()))
    secret = hmac.new(b"WebAppData", bot_token.encode(), hashlib.sha256).digest()
    payload["hash"] = hmac.new(secret, data_check.encode(), hashlib.sha256).hexdigest()
    from urllib.parse import urlencode
    return urlencode(payload)
