"""Service layer for post-earnings outcome tracking.

Replaces the raw ``requests`` + ``time.sleep(13)`` approach in ``outcomes.py``
with :class:`earnings_edge.collectors.polygon.PolygonClient`, which enforces a
sliding-window rate limit internally (5 calls / 62s). The bar→outcome math is
kept in a pure helper (:meth:`OutcomeService.outcome_from_bars`) so it can be
unit-tested without any network dependency.

Also tracks stock-move outcomes for :class:`live_calendar_candidates` and
:class:`scanner_scan_outputs`.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from typing import Any, Optional

from earnings_edge.collectors.polygon import PolygonClient
from earnings_edge.db import (
    configure,
    fetch_pending_live_candidates,
    fetch_pending_outcomes,
    record_live_candidate_failure,
    record_snapshot_outcome_failure,
    update_live_candidate_move,
    update_outcome,
)

logger = logging.getLogger("earnings_edge.services.outcome")


class OutcomeService:
    """Fetches actual post-earnings moves from Polygon and labels snapshots.

    Parameters
    ----------
    polygon_client:
        Optional :class:`PolygonClient` (or any fake exposing
        ``get_daily_bars``). Injecting one makes the service fully unit-testable
        without hitting the Polygon API.
    db_path:
        Optional SQLite database path override.
    """

    def __init__(
        self,
        polygon_client: Optional[PolygonClient] = None,
        db_path: Optional[Any] = None,
    ) -> None:
        self._db_path = db_path
        self._polygon = polygon_client or PolygonClient()

    # -- public API -------------------------------------------------------

    def compute_outcome(
        self,
        ticker: str,
        earnings_date_str: str,
        timing: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        """Compute the actual earnings-move outcome for *ticker*.

        Looks at the close the day before earnings (pre) and the close on/after
        earnings (post), plus the max intraday range across the earnings day
        and the two following days. Returns ``None`` when there is not enough
        bar data. AMC / Post Market uses the first close *after* ``ed``.
        """
        ed = datetime.strptime(earnings_date_str, "%Y-%m-%d").date()
        # Wide window to guarantee a pre-earnings bar across weekends/holidays.
        from_d = (ed - timedelta(days=7)).isoformat()
        to_d = (ed + timedelta(days=3)).isoformat()
        bars = self._polygon.get_daily_bars(ticker, from_d, to_d)
        return self.outcome_from_bars(bars, ed, timing=timing)

    def run_outcomes(
        self,
        min_age_days: int = 2,
        limit: int = 0,
        max_retries: int = 2,
    ) -> dict[str, int]:
        """Process all pending outcomes; returns a stats dict.

        Unlike the legacy ``outcomes.run_outcomes`` there is no explicit
        ``time.sleep(13)`` between calls — :class:`PolygonClient` enforces the
        sliding-window rate limit. After ``max_retries`` consecutive "no data"
        results for a snapshot it is marked permanently unavailable.
        """
        updated = 0
        failed = 0
        if self._db_path:
            configure(self._db_path)
        pending = fetch_pending_outcomes(min_age_days=min_age_days)
        logger.info(
            "%d pending outcomes to fetch%s",
            len(pending),
            f" (processing up to {limit})" if limit else "",
        )
        if limit > 0:
            pending = pending[:limit]

        for row in pending:
            ticker = row["ticker"]
            ed = row["earnings_date"]
            timing = row["timing"] if "timing" in row.keys() else None
            logger.info("  %s (%s)", ticker, ed)

            outcome = self.compute_outcome(ticker, ed, timing=timing)
            if outcome:
                update_outcome(row["id"], outcome)
                updated += 1
                logger.info(
                    "    → %s %.2f%% (max range %.2f%%)",
                    outcome["actual_move_direction"],
                    outcome["actual_move_pct"],
                    outcome["max_intraday_range_pct"],
                )
            else:
                failed += 1
                record_snapshot_outcome_failure(row["id"], max_retries)

        logger.info(
            "Outcomes complete: %d updated, %d no data", updated, failed
        )
        return {
            "updated": updated,
            "failed": failed,
            "processed": updated + failed,
        }

    # -- internals --------------------------------------------------------

    @staticmethod
    def _is_amc(timing: Optional[str]) -> bool:
        return bool(timing) and "post" in str(timing).lower()

    @staticmethod
    def outcome_from_bars(
        bars: list[dict[str, Any]],
        ed: date,
        timing: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        """Pure bar→outcome transformation (no network, no DB).

        *bars* is a list of Polygon agg dicts with at least ``c`` (close),
        ``h`` (high), ``l`` (low) and ``t`` (ms epoch timestamp).

        Default / BMO: pre = last close *before* ``ed``, post = first close
        on/after ``ed``. AMC / Post Market: pre = last close on/before ``ed``
        (the pre-announcement session), post = first close *after* ``ed``.
        """
        if len(bars) < 2:
            return None

        pre_bar: Optional[dict[str, Any]] = None
        post_bar: Optional[dict[str, Any]] = None
        earnings_bar: Optional[dict[str, Any]] = None
        amc = OutcomeService._is_amc(timing)

        for i, bar in enumerate(bars):
            bar_date = datetime.fromtimestamp(bar["t"] / 1000).date()
            if amc:
                if bar_date <= ed:
                    pre_bar = bar
                if bar_date > ed and post_bar is None:
                    post_bar = bar
                    earnings_bar = bar
            else:
                if bar_date < ed and i < len(bars) - 1:
                    pre_bar = bar
                if bar_date >= ed and post_bar is None:
                    post_bar = bar
                    earnings_bar = bar

        # Fall back to the bar immediately before the post bar.
        if pre_bar is None and post_bar is not None:
            idx = bars.index(post_bar)
            if idx > 0:
                pre_bar = bars[idx - 1]

        if pre_bar is None or post_bar is None:
            return None

        pre_close = pre_bar["c"]
        post_close = post_bar["c"]
        if pre_close <= 0:
            return None

        actual_move_pct = ((post_close - pre_close) / pre_close) * 100
        direction = (
            "UP"
            if actual_move_pct > 0.5
            else "DOWN"
            if actual_move_pct < -0.5
            else "FLAT"
        )

        # Max intraday range across earnings day + following two days.
        max_range_pct = 0.0
        earnings_idx = bars.index(earnings_bar) if earnings_bar else -1
        for j in range(max(0, earnings_idx), min(len(bars), earnings_idx + 3)):
            bar = bars[j]
            range_pct = ((bar["h"] - bar["l"]) / pre_close) * 100
            max_range_pct = max(max_range_pct, range_pct)

        return {
            "pre_earnings_close": round(pre_close, 4),
            "post_earnings_close": round(post_close, 4),
            "actual_move_pct": round(actual_move_pct, 4),
            "actual_move_direction": direction,
            "max_intraday_range_pct": round(max_range_pct, 4),
            "outcome_fetched_at": datetime.now(timezone.utc).isoformat(),
        }

    # -- live_calendar_candidates outcomes ---------------------------------

    def run_live_candidate_outcomes(
        self,
        min_age_days: int = 2,
        limit: int = 0,
        max_retries: int = 2,
    ) -> dict[str, int]:
        """Process live_calendar_candidates missing a stock-move outcome.

        Returns ``updated``, ``failed``, ``processed`` counts.
        """
        updated = 0
        failed = 0
        if self._db_path:
            configure(self._db_path)
        pending = fetch_pending_live_candidates(min_age_days=min_age_days)
        logger.info(
            "%d pending live-candidate outcomes to fetch%s",
            len(pending),
            f" (processing up to {limit})" if limit else "",
        )
        if limit > 0:
            pending = pending[:limit]

        for row in pending:
            ticker = row["ticker"]
            ed = row["earnings_date"]
            cid = row["id"]
            logger.info("  live_can %s id=%d (%s)", ticker, cid, ed)

            outcome = self.compute_outcome(ticker, ed)
            if outcome:
                update_live_candidate_move(cid, outcome)
                updated += 1
                logger.info(
                    "    → %s %.2f%% (max range %.2f%%)",
                    outcome["actual_move_direction"],
                    outcome["actual_move_pct"],
                    outcome["max_intraday_range_pct"],
                )
            else:
                failed += 1
                record_live_candidate_failure(cid, max_retries)

        logger.info(
            "Live-candidate outcomes: %d updated, %d no data",
            updated,
            failed,
        )
        return {
            "updated": updated,
            "failed": failed,
            "processed": updated + failed,
        }
