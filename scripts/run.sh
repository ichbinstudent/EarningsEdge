#!/bin/bash
# EarningsEdge CLI Scanner Runner (lives in scripts/; invoke from repo root or
# via `bash scripts/run.sh`).

# Auto-detect worker count (half the cores, 2–6)
NUM_CORES=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
WORKERS=$(( NUM_CORES / 2 ))
(( WORKERS < 2 )) && WORKERS=2
(( WORKERS > 6 )) && WORKERS=6

cd "$(dirname "$0")/.."
exec python3 scanner.py --parallel $WORKERS "$@"
