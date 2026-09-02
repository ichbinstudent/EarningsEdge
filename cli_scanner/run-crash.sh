#!/usr/bin/env bash
# German crash alerts — separate from trading-bot. Survives Hermes SIGTERM
# only if launched with setsid (see README). Do not long-poll Telegram.
set -euo pipefail
BOT_DIR="$HOME/EarningsEdgeDetection/cli_scanner"
PYTHON="$BOT_DIR/.venv/bin/python3.12"
LOG_DIR="$BOT_DIR/logs"
mkdir -p "$LOG_DIR"
exec >> "$LOG_DIR/crash-alert.log" 2>&1
echo "[$(date -Iseconds)] crash_alert supervisor started (PID $$)"
while true; do
    echo "[$(date -Iseconds)] launching crash_alert..."
    cd "$BOT_DIR"
    "$PYTHON" crash_alert.py
    rc=$?
    echo "[$(date -Iseconds)] crash_alert exited $rc, restarting in 10s..."
    sleep 10
done
