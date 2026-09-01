# Test layers

| Layer | Location | What it covers | Constraints |
|---|---|---|---|
| unit | `tests/test_*.py` | Pure functions/classes in isolation | No DB, no providers |
| integration | `tests/integration/` | Component wiring: collector→DB, TOML→registry→risk gate | Temp DB, mocked providers, **no network**, paper-only Alpaca |
| e2e | `tests/e2e/` | Full pipelines: collect→persist→scan→propose | Same guards as integration |

Shared fixtures live in `tests/conftest.py`:

- `tmp_db_path` — temp SQLite DB with the full schema (WAL, migrations applied), zero production data.
- `test_settings` — frozen `Settings` with dummy keys installed as the singleton (Settings freezes at first `get_settings()`, so the swap must happen before code under test reads it).
- `network_guard` — fails the test on any real socket/DNS, any `requests`/`curl_cffi` HTTP call, or any `AlpacaTradingClient(paper=False)`. Autouse in `tests/integration/` and `tests/e2e/` via their local conftests.

Markers are registered in `pyproject.toml` (`integration`, `e2e`). Unit tests
are intentionally unmarked — they are the default.

## Running

```bash
cd cli_scanner

# everything (default; all layers are fast and hermetic)
.venv/bin/python -m pytest tests/ -q

# one layer only
.venv/bin/python -m pytest tests/ -q -m integration
.venv/bin/python -m pytest tests/ -q -m e2e

# unit only / everything except e2e
.venv/bin/python -m pytest tests/ -q -m "not integration and not e2e"
.venv/bin/python -m pytest tests/ -q -m "not e2e"
```

Use `.venv/bin/python -m pytest`, never `uv run pytest` — the package is not
pip-installed into the venv, so only the `python -m` form puts `cli_scanner/`
on `sys.path`.

## Rules for new code

Any code change must ship with integration and (when it touches a pipeline)
e2e coverage in these layers. New entry points must load `load_dotenv()`
before any `earnings_edge` import; tests bypass this by using the
`test_settings` fixture instead of real env.
