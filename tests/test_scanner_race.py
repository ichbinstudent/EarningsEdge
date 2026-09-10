"""Regression: reports dict must not be mutated after scan_earnings returns.

Reproduces the Sep 10 production crash ("dictionary changed size during
iteration"): the collection loop's f.result(timeout=60) abandons a slow
worker but the thread keeps running, and shutdown(wait=False) never joins
it, so a straggler write lands in the reports dict AFTER scan_earnings has
returned it — blowing up downstream iteration in
scan_service.ScanService._process_result.
"""

import concurrent.futures
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

from earnings_edge.models import EarningsCandidate, ValidationMetrics, ValidationResult
from earnings_edge.scanner import EarningsScanner


def _result(tier: int = 1) -> ValidationResult:
    return ValidationResult(
        passed=True,
        tier=tier,
        near_miss=False,
        reason="OK",
        metrics=ValidationMetrics(),
    )


def test_straggler_write_never_lands_after_return():
    """A slow worker's write must be inside the returned dict at return time.

    Simulates the production race without waiting 60 wall-clock seconds:
    Future.result is patched so that a not-yet-done future raises
    TimeoutError immediately when called WITH a timeout — exactly what the
    old f.result(timeout=60) did to a worker parked behind the slow
    MarketChameleon browser. The straggler is then released strictly after
    the main thread has passed shutdown().

    Old code (abandon + shutdown(wait=False)): scan returns with 20
    reports while the straggler thread is still parked; its write mutates
    the already-returned dict ~0.3s later — the production crash window.

    Fixed code (as_completed + f.result() + shutdown(wait=True)): the
    collection loop blocks until the straggler finishes, so all 21
    reports are in the dict before it is ever returned, and nothing
    mutates it afterwards.
    """
    scanner = EarningsScanner()
    scanner.validator = MagicMock()

    straggler_started = threading.Event()
    release_straggler = threading.Event()

    def fake_validate(candidate):
        if candidate.ticker == "STRAGGLER":
            straggler_started.set()
            release_straggler.wait(timeout=10.0)
            return _result()
        return _result()

    scanner.validator.validate.side_effect = fake_validate

    candidates = [EarningsCandidate(ticker=f"OK{i}", timing="Post Market") for i in range(20)]
    candidates.append(EarningsCandidate(ticker="STRAGGLER", timing="Post Market"))

    original_result = concurrent.futures.Future.result

    def simulated_timeout_result(self, timeout=None):
        # Keyed on timeout==60: that is exactly the old collection loop's
        # f.result(timeout=60). The earnings-fetch pool uses timeout=30 and
        # must stay untouched so the candidate list is deterministic.
        if timeout == 60 and not self.done():
            # Mirrors the old code's f.result(timeout=60) giving up on a
            # still-running worker: the future raises, the worker thread
            # itself keeps running.
            raise TimeoutError("simulated 60s collection timeout")
        return original_result(self, timeout)

    def releaser():
        straggler_started.wait(timeout=10.0)
        # The main thread needs only microseconds to get from the abandoned
        # future through shutdown() and tier building to the return, so a
        # 0.3s delay guarantees the straggler write lands after the dict
        # has been handed back — the exact production failure shape.
        time.sleep(0.3)
        release_straggler.set()

    t = threading.Thread(target=releaser)
    t.start()

    with (
        patch("earnings_edge.scanner.fetch_earnings", return_value=candidates),
        patch("earnings_edge.scanner.scan_dates", return_value=("2026-09-10", "2026-09-10")),
        patch.object(concurrent.futures.Future, "result", simulated_timeout_result),
    ):
        res = scanner.scan_earnings(workers=2)
        # Measured immediately at return, while the straggler is still
        # parked (release happens no earlier than 0.3s in) — this is the
        # exact state downstream consumers receive the dict in.
        len_at_return = len(res.reports)

        # The returned dict must be complete at return time and stable
        # afterwards. Watch for a late straggler write the way any
        # downstream consumer (scan_service._process_result) would
        # experience it: as a size change during use.
        deadline = time.time() + 1.0
        grew_after_return = False
        while time.time() < deadline:
            if len(res.reports) != len_at_return:
                grew_after_return = True
                break
            time.sleep(0.01)

    release_straggler.set()
    t.join(timeout=5.0)
    assert len_at_return == 21, (
        f"scan_earnings returned with {len_at_return} of 21 reports — the "
        "straggler was abandoned and the returned dict is still mutable"
    )
    assert not grew_after_return, "reports dict was mutated after scan_earnings returned it"


def test_scan_earnings_joins_pool_before_building_tiers():
    """Direct property check: the worker pool is joined (wait=True) before
    tier separation, so no worker thread outlives the collection loop."""
    scanner = EarningsScanner()
    scanner.validator = MagicMock()
    scanner.validator.validate.side_effect = lambda candidate: _result(tier=1)

    shutdown_calls: list[tuple] = []
    original_shutdown = ThreadPoolExecutor.shutdown

    def patched_shutdown(self, wait=True, *, cancel_futures=False):
        shutdown_calls.append((wait, cancel_futures))
        return original_shutdown(self, wait=wait, cancel_futures=cancel_futures)

    candidates = [EarningsCandidate(ticker=f"T{i}", timing="Post Market") for i in range(5)]

    with (
        patch("earnings_edge.scanner.fetch_earnings", return_value=candidates),
        patch("earnings_edge.scanner.scan_dates", return_value=("2026-09-10", "2026-09-10")),
        patch.object(ThreadPoolExecutor, "shutdown", patched_shutdown),
    ):
        res = scanner.scan_earnings(workers=2)

    # scan_earnings also shuts down the short-lived earnings-fetch pool with
    # wait=False (fine — it only returns lists, no shared dict), so we
    # assert that at least one shutdown — the worker pool's — joined.
    assert any(wait for wait, _ in shutdown_calls), "worker pool must be joined (wait=True)"
    assert len(res.reports) == 5
    assert set(res.tier1) == {f"T{i}" for i in range(5)}
