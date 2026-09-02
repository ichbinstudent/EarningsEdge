#!/usr/bin/env python3
"""Mark leftover pending_trades error rows as expired (audit kept)."""
import sqlite3
from pathlib import Path

db = Path(__file__).resolve().parent.parent / "data" / "earnings_ml.db"
con = sqlite3.connect(db)
n = con.execute("SELECT COUNT(*) FROM pending_trades WHERE status='error'").fetchone()[0]
con.execute(
    "UPDATE pending_trades SET status='expired' WHERE status='error'"
)
con.commit()
print(n, "error -> expired")
print(con.execute("SELECT status, COUNT(*) FROM pending_trades GROUP BY 1").fetchall())
con.close()
