import concurrent.futures
import threading
import time
from unittest.mock import MagicMock, patch

from earnings_edge.models import EarningsCandidate, ValidationResult
from earnings_edge.scanner import EarningsScanner


def test_scan_earnings_race_condition():
    scanner = EarningsScanner()

    slow_worker_event = threading.Event()
    worker_started_event = threading.Event()

    def fake_validate(candidate):
        if candidate.ticker == "SLOW":
            worker_started_event.set()
            slow_worker_event.wait(timeout=5.0)
            return ValidationResult(passed=True, tier=1, near_miss=False, reason="OK")
        else:
            return ValidationResult(passed=True, tier=1, near_miss=False, reason="OK")

    scanner.validator = MagicMock()
    scanner.validator.validate.side_effect = fake_validate

    def fake_fetch_earnings(date_str, *args, **kwargs):
        if True:
            # Lots of fast candidates to increase the chance of iteration during the race
            return [EarningsCandidate(ticker=f"FAST_{i}", timing="Post Market") for i in range(100)] + [
                EarningsCandidate(ticker="SLOW", timing="Post Market")
            ]
        return []

    original_result = concurrent.futures.Future.result

    def fake_result(self, timeout=None):
        if timeout == 60:  # old buggy code
            worker_started_event.wait(timeout=2.0)
            raise TimeoutError("Simulated timeout for old code")
        return original_result(self, timeout)

    def unblock_slow_worker():
        worker_started_event.wait(timeout=2.0)
        # Give main thread time to catch the mocked TimeoutError and start the dict iteration
        time.sleep(0.05)
        slow_worker_event.set()

    t = threading.Thread(target=unblock_slow_worker)
    t.start()

    with (
        patch("earnings_edge.scanner.fetch_earnings", side_effect=fake_fetch_earnings),
        patch("earnings_edge.scanner.scan_dates", return_value=("2026-09-10", "2026-09-10")),
        patch.object(concurrent.futures.Future, "result", new=fake_result),
    ):
        # If the code is buggy, a RuntimeError will be raised here.
        # Because SLOW will insert into `reports` while `scan_earnings` is iterating it.
        res = scanner.scan_earnings(workers=2)

        # The new code should safely wait for SLOW, so SLOW should be in the results.
        assert "SLOW" in res.reports

    t.join()
