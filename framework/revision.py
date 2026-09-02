"""Code SHA + process start time for /status."""
from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

STARTED_AT = datetime.now(timezone.utc)
_REPO = Path(__file__).resolve().parents[2]


def code_sha(cwd: Optional[Path] = None) -> str:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(cwd or _REPO),
            capture_output=True, text=True, timeout=2,
        )
        sha = (r.stdout or "").strip()
        return sha or "unknown"
    except Exception:
        return "unknown"


def started_at_iso() -> str:
    return STARTED_AT.isoformat()
