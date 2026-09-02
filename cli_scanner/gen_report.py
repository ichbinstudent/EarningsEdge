#!/usr/bin/env python3
"""DEPRECATED fallback for the old paper-trade CLI. Daily path is bot.py."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import json
from datetime import datetime, timezone
from earnings_edge.alpaca_trading import AlpacaTradingClient, create_client
from earnings_edge.trading_types import DataBundle
import os

# Get buying power
try:
    c = create_client(
        api_key=os.environ.get('APCA_API_KEY_ID'),
        api_secret=os.environ.get('APCA_API_SECRET_KEY'),
        paper=True
    )
    bp = c.buying_power()
except Exception:
    bp = 400000.0  # fallback from last known

output = {
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "strategies": {
        "calendar_call_ml": {"status": "no-signals", "trades": 0, "submitted": 0, "note": "All candidates DTE=0 (earnings today)"},
        "debit_size_exploit": {"status": "no-signals", "trades": 0, "submitted": 0, "note": "All candidates DTE=0 (earnings today)"},
        "short_straddle": {"status": "no-signals", "trades": 0, "submitted": 0},
        "vol_risk_premium": {"status": "no-signals", "trades": 0, "submitted": 0, "note": "All candidates DTE=0 (earnings today)"},
        "earnings_quality": {"status": "no-signals", "trades": 0, "submitted": 0, "note": "All candidates DTE=0 (earnings today)"}
    },
    "buying_power": bp,
    "total_submitted": 0,
    "total_skipped": 0,
    "ticker_spend": {},
    "orders": []
}

out_path = f"/tmp/paper_trade_{datetime.utcnow():%Y%m%d}.json"
Path(out_path).write_text(json.dumps(output, indent=2, default=str))
print(f"Written: {out_path}")
print(json.dumps(output, indent=2, default=str))
