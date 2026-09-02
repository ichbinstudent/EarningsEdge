#!/usr/bin/env bash
# Supervisor loop for the Telegram trading bot. Prefer systemd user units
# (deploy/trading-bot.user.service). Running this *and* systemd double-polls
# Telegram (409 Conflict). This script refuses if the instance lock is held.
set -euo pipefail

LOCK="$HOME/EarningsEdgeDetection/data/trading-bot.lock"
if [[ -f "$LOCK" ]] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; then
    echo "trading-bot already running (lock $LOCK pid $(cat "$LOCK")). Use: systemctl --user restart trading-bot" >&2
    exit 1
fi

BOT_DIR="$HOME/EarningsEdgeDetection"
PYTHON="$BOT_DIR/.venv/bin/python3.12"
LOG_DIR="$BOT_DIR/logs"

mkdir -p "$LOG_DIR"

exec >> "$LOG_DIR/bot-supervisor.log" 2>&1
echo "[$(date -Iseconds)] supervisor started (PID $$)"

while true; do
    echo "[$(date -Iseconds)] launching bot..."
    cd "$BOT_DIR"
    "$PYTHON" bot.py
    rc=$?
    echo "[$(date -Iseconds)] bot exited with code $rc, restarting in 10s..."
    sleep 10
done
