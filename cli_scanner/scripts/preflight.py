#!/usr/bin/env python3
"""Pre-flight checks before starting the trading bot.

Verifies: env keys, Alpaca paper reachability + clock, LSE key, framework DB
writability, strategy configs, kill-switch status, trading calendar, Telegram
token. Exits 0 when every REQUIRED check passes, 1 otherwise (warnings only
for optional checks).

Usage:  python scripts/preflight.py
        python scripts/preflight.py --i-mean-live   # required when ALPACA_LIVE=1
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "", required: bool = True) -> bool:
    RESULTS.append((name, ok, detail))
    tag = "OK  " if ok else ("FAIL" if required else "WARN")
    print(f"[{tag}] {name}" + (f" — {detail}" if detail else ""))
    return ok


def _alpaca_account_checks(client, *, live: bool) -> bool:
    acct = client.get_account()
    paper_attr = getattr(client, "paper", True)
    paper = paper_attr if isinstance(paper_attr, bool) else True
    url = getattr(client, "base_url", "")
    url = url if isinstance(url, str) else ""
    label = "live" if live else "paper"
    if live and (paper or "paper-api" in url):
        return check("alpaca live endpoint", False, f"still on paper URL {url}")
    if not live and not paper:
        return check("alpaca paper endpoint", False, f"unexpected live URL {url}")
    status = str(acct.get("status") or "")
    blocked = str(acct.get("trading_blocked") or "").lower() in ("true", "1")
    acct_blocked = str(acct.get("account_blocked") or "").lower() in ("true", "1")
    ok = check(
        f"alpaca {label} account",
        status in ("", "ACTIVE") and not blocked and not acct_blocked,
        f"id={acct.get('account_number', '?')} status={status or 'n/a'} "
        f"equity=${float(acct.get('equity', 0) or 0):,.0f} "
        f"bp=${float(acct.get('buying_power', 0) or 0):,.0f} "
        f"pdt={acct.get('pattern_day_trader')} "
        f"options={acct.get('options_approved_level', acct.get('options_trading_level', '?'))}",
    )
    try:
        clock = client.get_clock()
        check("alpaca clock", True,
              f"is_open={clock.get('is_open')} next_open={str(clock.get('next_open'))[:16]}",
              required=False)
    except Exception as exc:
        check("alpaca clock", False, str(exc), required=False)
    return ok


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Pre-flight checks before starting the bot")
    parser.add_argument("--i-mean-live", action="store_true",
                        help="Confirm you intend to talk to the Alpaca LIVE API")
    args = parser.parse_args(argv if argv is not None else [])
    RESULTS.clear()
    all_ok = True

    from earnings_edge.alpaca_mode import alpaca_live_enabled

    live_wanted = alpaca_live_enabled()
    if live_wanted and not args.i_mean_live:
        all_ok &= check(
            "alpaca live confirmation", False,
            "ALPACA_LIVE=1 requires scripts/preflight.py --i-mean-live",
        )
    if args.i_mean_live and not live_wanted:
        all_ok &= check(
            "alpaca live confirmation", False,
            "--i-mean-live set but ALPACA_LIVE is not 1",
        )

    # 1. env keys
    required_keys = [
        ("TELEGRAM_BOT_TOKEN", True),
        ("APCA_API_KEY_ID", True),
        ("APCA_API_SECRET_KEY", True),
        ("LSE_API_KEY", True),
        ("POLYGON_API_KEY", False),
        ("FINNHUB_API_KEY", False),
    ]
    if live_wanted:
        required_keys.append(("APCA_LIVE_API_KEY_ID", True))
        required_keys.append(("APCA_LIVE_API_SECRET_KEY", True))
    for var, required in required_keys:
        ok = check(f"env {var}", bool(os.environ.get(var)), required=required)
        all_ok &= ok or not required

    # 2. Alpaca reachable + account
    try:
        from earnings_edge.alpaca_trading import create_client
        if args.i_mean_live and live_wanted:
            client = create_client()  # follows ALPACA_LIVE
            all_ok &= _alpaca_account_checks(client, live=True)
        else:
            client = create_client(paper=True)
            all_ok &= _alpaca_account_checks(client, live=False)
    except Exception as exc:
        all_ok &= check("alpaca account", False, str(exc))

    # 3. LSE provider reachable
    try:
        from earnings_edge.market_data_provider import LSEProvider
        ok = LSEProvider().healthy()
        all_ok &= check("lse provider health", ok)
    except Exception as exc:
        all_ok &= check("lse provider health", False, str(exc))

    # 4. framework DB writable
    try:
        from earnings_edge.db import risk_state_get
        risk_state_get()
        check("framework db writable", True)
    except Exception as exc:
        all_ok &= check("framework db writable", False, str(exc))

    # 5. strategy configs
    try:
        from framework.core.config import load_strategy_configs
        configs = load_strategy_configs()
        expected = {"calendar_call_ml", "short_straddle",
                    "vol_risk_premium", "earnings_quality", "ff_ladder",
                    "forward_factor_arb"}
        missing = expected - set(configs)
        all_ok &= check("strategy configs", not missing,
                        f"{len(configs)} loaded" + (f", MISSING: {sorted(missing)}" if missing else ""))
    except Exception as exc:
        all_ok &= check("strategy configs", False, str(exc))

    # 6. kill switch status (informational)
    try:
        from framework.risk.killswitch import KillSwitch
        status = KillSwitch().status()
        check("kill switch", not status.get("halted"),
              f"HALTED: {status.get('reason')}" if status.get("halted") else "armed")
        all_ok &= not status.get("halted")
    except Exception as exc:
        all_ok &= check("kill switch", False, str(exc))

    # 7. trading calendar loads
    try:
        from framework.core.calendar import get_calendar
        cal = get_calendar()
        from datetime import date
        check("trading calendar", True,
              f"today session: {cal.is_session(date.today())}")
    except Exception as exc:
        all_ok &= check("trading calendar", False, str(exc))

    # 8. Telegram reachable
    try:
        import requests
        token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
        resp = requests.get(f"https://api.telegram.org/bot{token}/getMe", timeout=10)
        all_ok &= check("telegram getMe", resp.status_code == 200)
    except Exception as exc:
        all_ok &= check("telegram getMe", False, str(exc))

    print()
    if all_ok:
        print("PRE-FLIGHT PASS — safe to start the bot")
        return 0
    print("PRE-FLIGHT FAIL — fix the FAIL items above first")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
