#!/usr/bin/env python3
"""Mark leftover pending_trades error rows as expired (audit kept).

Writes directly to the production DB — the single exception to the
single-writer discipline, kept because it is a rare manual cleanup that
must run while the bot may be down. Asks for confirmation unless
--yes is passed, and refuses to run while the bot process is alive.
"""

import argparse
import sqlite3
import subprocess
from pathlib import Path

db = Path(__file__).resolve().parent.parent / "data" / "earnings_ml.db"

p = argparse.ArgumentParser()
p.add_argument("--yes", action="store_true", help="Skip confirmation prompt")
args = p.parse_args()

bot_alive = (
    subprocess.run(
        ["pgrep", "-f", "EarningsEdgeDetection/.venv/bin/python3.12 bot.py"],
        capture_output=True,
    ).returncode
    == 0
)
if bot_alive:
    raise SystemExit("Refusing: trading-bot process is running. Stop it first (single-writer discipline).")

if not args.yes:
    n = (
        sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        .execute("SELECT COUNT(*) FROM pending_trades WHERE status='error'")
        .fetchone()[0]
    )
    answer = input(f"{n} error rows -> expired. Proceed? [y/N] ")
    if answer.strip().lower() != "y":
        raise SystemExit("Aborted.")

con = sqlite3.connect(db)
n = con.execute("SELECT COUNT(*) FROM pending_trades WHERE status='error'").fetchone()[0]
con.execute("UPDATE pending_trades SET status='expired' WHERE status='error'")
con.commit()
print(n, "error -> expired")
print(con.execute("SELECT status, COUNT(*) FROM pending_trades GROUP BY 1").fetchall())
con.close()
