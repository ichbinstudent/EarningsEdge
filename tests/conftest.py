"""Shared test fixtures for EarningsEdge test suite.

Layering:
- tests/test_*.py            unit tests (no fixtures required)
- tests/integration/         integration tests (temp DB, mocked providers, no network)
- tests/e2e/                 end-to-end pipeline tests (fixture data -> temp DB -> assertions)

The fixtures below are defined at the root so both layers can use them; the
integration/e2e conftests apply the network guard autouse.
"""

import tempfile
from datetime import date
from pathlib import Path

import pytest


def pytest_configure(config):
    """Point the engine at a throwaway DB before collection-time imports.

    Production callers (preflight, dashboard, KillSwitch) call get_engine()
    with no path and would otherwise open data/earnings_ml.db.
    """
    from earnings_edge.db import engine as db_engine

    path = Path(tempfile.mkdtemp()) / "collection_isolated.db"
    db_engine.configure(path)


def make_temp_db() -> Path:
    """Provide a temporary SQLite database path."""
    d = tempfile.mkdtemp()
    return Path(d) / "test.db"


def make_sample_candidate():
    """Return a sample EarningsCandidate for testing."""
    from earnings_edge.models import EarningsCandidate

    return EarningsCandidate(
        ticker="AAPL",
        timing="Post Market",
        earnings_date=date(2026, 6, 17),
        source="finnhub",
    )


def make_sample_metrics():
    """Return a sample ValidationMetrics for testing."""
    from earnings_edge.models import ValidationMetrics

    return ValidationMetrics(
        price=150.0,
        volume=5_000_000,
        days_to_expiry=7,
        open_interest=10_000,
        term_structure=-0.008,
        iv_rv_ratio=1.35,
        win_rate=65.0,
        win_quarters=8,
        expected_move_dollars=3.50,
        expected_move_pct=2.33,
    )


# ---------------------------------------------------------------------------
# Shared integration/e2e fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True, scope="function")
def _isolate_engine(tmp_path):
    """Bind the process-wide engine to a throwaway DB for every test.

    Tests that need a specific file call configure(their_path) in setup and
    override this. Teardown re-points at a second throwaway so other fixture
    teardowns cannot hit the production database.
    """
    from earnings_edge.db import engine as db_engine

    db_engine.configure(tmp_path / "isolated.db")
    yield
    db_engine.configure(tmp_path / "isolated_teardown.db")


@pytest.fixture
def tmp_db_path(tmp_path) -> Path:
    """Temp SQLite DB with the full earnings_edge schema — never production data.

    Schema is created by engine.configure (WAL mode, all migrations applied).
    The file lives under pytest's tmp_path and is destroyed with it.
    """
    from earnings_edge.db import engine as db_engine

    path = tmp_path / "test_earnings_ml.db"
    db_engine.configure(path)
    return path


@pytest.fixture(autouse=True)
def test_settings(monkeypatch):
    """Install a frozen test Settings as the process-wide singleton.

    Autouse: get_settings() is a process-wide singleton frozen on first
    call, and several production modules (bot.py, scripts/preflight.py, ...)
    call load_dotenv() at IMPORT time — pytest imports every test file's
    dependencies during collection, so this repo's real .env (real
    LSE_API_KEY etc.) silently leaks into the test process before any test
    runs. Whichever test happened to trigger the first get_settings() call
    would then bake that real value in for every other test in the session,
    regardless of file or run order. Applying this to every test via
    monkeypatch (auto-restored after each test) closes that off entirely.
    """
    import earnings_edge.settings as settings_mod
    from earnings_edge.settings import Settings

    s = Settings(
        polygon_api_key="test-polygon-key",
        finnhub_api_key="test-finnhub-key",
        telegram_bot_token="test-telegram-token",
        lse_api_key="",
    )
    monkeypatch.setattr(settings_mod, "_settings", s)
    return s


@pytest.fixture
def network_guard(monkeypatch):
    """Fail any test that attempts real network I/O or a non-paper Alpaca client.

    Guards at three layers so a missed mock fails loudly instead of leaking:
    1. socket layer (catches urllib3/raw sockets/DNS),
    2. requests + curl_cffi session layer (clear error naming the URL),
    3. AlpacaTradingClient constructor (paper=True enforced).
    """
    import socket

    def _raise_blocked(*args, **kwargs):
        raise AssertionError(
            "integration/e2e tests must not open real network connections — "
            "mock the provider/collector instead"
        )

    # Must stay subclassable: modules like PySocks do
    # ``class _BaseSocket(socket.socket):`` at import time.
    class _GuardedSocket(socket.socket):
        def connect(self, address):
            _raise_blocked()

        def connect_ex(self, address):
            _raise_blocked()

    monkeypatch.setattr(socket, "socket", _GuardedSocket)
    monkeypatch.setattr(socket, "create_connection", _raise_blocked)
    monkeypatch.setattr(socket, "getaddrinfo", _raise_blocked)

    import requests

    def _blocked_http(self, method, url, **kwargs):
        raise AssertionError(f"real HTTP blocked in tests: {method} {url}")

    monkeypatch.setattr(requests.Session, "request", _blocked_http)

    try:
        from curl_cffi import requests as cffi_requests

        monkeypatch.setattr(cffi_requests.Session, "request", _blocked_http)
    except ImportError:  # pragma: no cover - curl_cffi is a hard dep, defensive
        pass

    from earnings_edge import alpaca_trading

    real_init = alpaca_trading.AlpacaTradingClient.__init__

    def _guarded_init(self, *args, **kwargs):
        paper = kwargs.get("paper", args[2] if len(args) > 2 else True)
        if paper is not True:
            raise AssertionError("non-paper Alpaca endpoint blocked in tests — pass paper=True")
        real_init(self, *args, **kwargs)

    monkeypatch.setattr(alpaca_trading.AlpacaTradingClient, "__init__", _guarded_init)
    yield


@pytest.fixture
def seeded_db(tmp_db_path):
    """Seed the temporary database with representative rows for views to read."""
    from sqlalchemy import text

    from earnings_edge.db import engine as db_engine

    engine = db_engine.get_engine()
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO snapshots (ticker, earnings_date, scan_date, timing) "
                "VALUES ('AAPL', '2026-10-15', '2026-09-04', 'Post Market')"
            )
        )

        conn.execute(
            text(
                "INSERT INTO live_calendar_candidates (ticker, earnings_date, scan_timestamp, passed) "
                "VALUES ('AAPL', '2026-10-15', '2026-09-04T08:00:00.000', 1)"
            )
        )

        conn.execute(
            text(
                "INSERT INTO trade_events (symbol, strategy, event_type, price, detail, ts) "
                "VALUES ('AAPL260918C00250000', 'momentum', 'buy_to_open', 150.5, 'Bought 100 shares <foo>', '2026-09-04T08:00:00.123')"
            )
        )

        conn.execute(
            text(
                "INSERT INTO job_runs (job_name, started_at, success, stats_json, error) "
                "VALUES ('sync', '2026-09-04T08:00:00.000', 1, '{\"a\": 1, \"b\": 2}', '')"
            )
        )

        conn.execute(
            text(
                "INSERT INTO equity_snapshots (ts, equity, buying_power, portfolio_value) "
                "VALUES ('2026-09-03T16:00:00.000', 9950.0, 5000.0, 9950.0)"
            )
        )

        conn.execute(text("INSERT INTO strategy_state (name, enabled) VALUES ('momentum', 1)"))

        conn.execute(
            text(
                "INSERT INTO exit_proposals (group_id, strategy, ticker, rule, status, created_at) "
                "VALUES ('group1', 'momentum', 'AAPL', 'stop_loss', 'pending', '2026-09-04T08:00:00.000')"
            )
        )

        conn.execute(
            text("INSERT INTO ff_ladders (ticker, status, candidate_json) VALUES ('AAPL', 'armed', '{}')")
        )

        conn.execute(
            text(
                "INSERT INTO managed_positions (symbol, strategy, group_id, qty, status, opened_at) "
                "VALUES ('AAPL260918C00250000', 'momentum', 'group1', 1.0, 'open', '2026-09-04T08:00:00.000')"
            )
        )

        conn.execute(
            text(
                "INSERT INTO pending_trades (strategy, ticker, side, status, created_at, trade_json, card_text) "
                "VALUES ('momentum', 'AAPL', 'long', 'pending', '2026-09-04T08:00:00.000', '{}', 'some text')"
            )
        )

        conn.execute(
            text(
                "INSERT INTO scan_runs (scanner_name, scan_timestamp, trigger_type) "
                "VALUES ('daily_scan', '2026-09-04T08:00:00.000', 'cron')"
            )
        )
    return tmp_db_path
