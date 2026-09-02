"""Paper vs live Alpaca mode. Default paper. Live is fail-closed.

ALPACA_LIVE=1 switches create_client() onto the live trading URL and
(when set) APCA_LIVE_API_KEY_ID / APCA_LIVE_API_SECRET_KEY. Everything
else — tests, preflight without --i-mean-live, unset env — stays paper.
"""
from __future__ import annotations

import os
from typing import Optional


def _flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")


def alpaca_live_enabled() -> bool:
    return _flag("ALPACA_LIVE")


def allow_auto_on_live() -> bool:
    return _flag("ALPACA_LIVE_ALLOW_AUTO")


def force_approval_on_live() -> bool:
    """Live entries stay on Telegram cards unless ALPACA_LIVE_ALLOW_AUTO=1."""
    return alpaca_live_enabled() and not allow_auto_on_live()


def live_max_qty(paper_cap: int = 25) -> int:
    """Fat-finger cap. Live defaults to 1 contract; paper keeps paper_cap."""
    if not alpaca_live_enabled():
        return paper_cap
    raw = os.environ.get("ALPACA_LIVE_MAX_QTY", "1")
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return 1


def broker_label() -> str:
    return "live" if alpaca_live_enabled() else "paper"


def resolve_credentials(
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None,
    paper: Optional[bool] = None,
) -> tuple[str, str, bool]:
    """Return (key, secret, is_paper).

    ``paper is True`` (tests, explicit paper) always wins over ALPACA_LIVE.
    ``paper is False`` forces live credentials.
    ``paper is None`` follows ALPACA_LIVE.
    """
    if paper is True:
        is_paper = True
    elif paper is False:
        is_paper = False
    else:
        is_paper = not alpaca_live_enabled()

    if is_paper:
        key = api_key or os.environ.get("APCA_API_KEY_ID", "")
        secret = api_secret or os.environ.get("APCA_API_SECRET_KEY", "")
        return key, secret, True

    key = (
        api_key
        or os.environ.get("APCA_LIVE_API_KEY_ID")
        or os.environ.get("APCA_API_KEY_ID", "")
    )
    secret = (
        api_secret
        or os.environ.get("APCA_LIVE_API_SECRET_KEY")
        or os.environ.get("APCA_API_SECRET_KEY", "")
    )
    return key, secret, False
