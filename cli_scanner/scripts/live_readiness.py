#!/usr/bin/env python3
"""Print a live-readiness checklist from earnings_ml.db + (optional) Alpaca.

Does not place orders. Exit 0 when every gate that can be evaluated locally
passes; remaining operational gates (paper-week fills) print as WARN.

Usage:  python scripts/live_readiness.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def _line(ok: bool, name: str, detail: str = "") -> bool:
    tag = "OK  " if ok else "WARN"
    print(f"[{tag}] {name}" + (f" — {detail}" if detail else ""))
    return ok


def main() -> int:
    from sqlalchemy import text

    from earnings_edge.alpaca_mode import alpaca_live_enabled, broker_label, force_approval_on_live
    from earnings_edge.db import get_session, managed_positions_list
    from framework.core.registry import get_registry
    from framework.risk.killswitch import KillSwitch

    print(f"broker={broker_label()} live_env={alpaca_live_enabled()} "
          f"force_approval={force_approval_on_live()}")
    ok = True

    if alpaca_live_enabled():
        ok &= _line(False, "ALPACA_LIVE", "set — do not flip until paper-week gates pass")
    else:
        _line(True, "ALPACA_LIVE", "unset (paper)")

    reg = get_registry()
    v1_on = {"calendar_call_ml", "debit_size_exploit", "ff_ladder"}
    v1_off = {"short_straddle", "vol_risk_premium", "earnings_quality"}
    for name in v1_on:
        enabled = reg.is_enabled(name)
        mode = reg.execution_mode(name)
        ok_one = enabled and mode == "approval"
        ok &= _line(ok_one, f"strategy {name}", f"enabled={enabled} mode={mode}")
    for name in v1_off:
        enabled = reg.is_enabled(name)
        ok &= _line(not enabled, f"strategy {name}", f"enabled={enabled} (want disabled)")

    session = get_session()
    try:
        halted = KillSwitch().is_halted()
        ok &= _line(not halted, "kill switch", "HALTED" if halted else "armed")

        funnel = session.execute(text(
            "SELECT created_at, proposals_total, counts FROM proposal_funnel "
            "ORDER BY id DESC LIMIT 1"
        )).mappings().fetchone()
        if funnel:
            _line(True, "latest funnel",
                  f"{funnel['created_at'][:19]} proposals={funnel['proposals_total']}")
        else:
            _line(False, "latest funnel", "no proposal_funnel rows")

        chain_hours = session.execute(text(
            "SELECT COUNT(DISTINCT captured_hour) FROM options_chain "
            "WHERE scan_date = date('now')"
        )).scalar()
        _line((chain_hours or 0) >= 1, "chain cache today", f"{chain_hours} hour bucket(s)")

        open_n = len(managed_positions_list())
        _line(True, "open managed positions", str(open_n))
    finally:
        session.close()

    print()
    print("Paper-week gates (manual): ≥2 days calendar proposals, ≥5 fills "
          "≤1.15× mid, every fill closed by ExitManager, halt drill.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
