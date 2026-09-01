"""Job run observability: uniform run history for every scheduled job.

Wraps any callable, records start/finish/success/stats/error into
``job_runs``, and re-raises so the scheduler's own error handling still
applies. Replaces the scan-only ``scan_runs`` pattern for non-scan jobs
(equity snapshots, reconcile, exit evaluation, outcome labeling).
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Optional

from earnings_edge.db import job_runs_finish, job_runs_list, job_runs_start

logger = logging.getLogger("framework.jobs")


def run_job(
    name: str,
    fn: Callable[[], Any],
    stats: Optional[dict] = None,
) -> Any:
    """Run ``fn`` under a job_runs audit row. Returns fn's result."""
    stats = stats if stats is not None else {}
    run_id = job_runs_start(name)
    try:
        result = fn()
    except Exception as exc:
        job_runs_finish(
            run_id, success=0,
            stats_json=json.dumps(stats, default=str),
            error=str(exc)[:500],
        )
        logger.error("job %s failed: %s", name, exc)
        raise
    if isinstance(result, dict):
        stats.update(result)
    job_runs_finish(
        run_id, success=1, stats_json=json.dumps(stats, default=str),
    )
    return result


def recent_runs(name: Optional[str] = None, limit: int = 20) -> list[dict]:
    """Recent job runs (dashboard/health)."""
    return job_runs_list(name=name, limit=limit)
