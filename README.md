# Earnings Edge

Paper-trading system for US earnings options, driven by a Telegram bot. Independent of the original EarningsEdgeDetection scanner; this tree is the live bot, strategy layer, and ML pipeline.

Python 3.12. Package metadata lives in `cli_scanner/pyproject.toml`.

## What it does

- **Earnings calendar scan** weekdays at 14:00 ET (20:00 Berlin). IV/RV, term structure, calendar quotes, ML TAKE/SKIP.
- **Proposal cards** on Telegram: approve or skip. Auto mode is per-strategy and still risk-gated.
- **Forward-factor ladders** (`ff_ladder`, `forward_factor_arb`): patient MLEG limit entry 14:00–15:45 ET.
- **German crash alerts**: separate process (`crash_alert.py` / `run-crash.sh`). Poll every 2 minutes 07:30–23:00 Europe/Berlin weekdays. Telegram if a name drops more than 20% in 5 minutes (tight book + print ≤10 min). Not a realtime stream.
- **Ops**: `/status` `/positions` `/orders` `/jobs` `/equity` `/signals` `/picks` `/halt` `/resume`. Kill switch and per-strategy pause.

Alpaca paper only. Resting limits at computed prices; no live cash.

## Strategies

TOML in `cli_scanner/strategies/`. Details: `cli_scanner/STRATEGIES.md`.

| Name | Idea | Execution |
|---|---|---|
| `calendar_call_ml` | ML-scored call calendar into earnings | approval card |
| `ff_ladder` | Implied event move vs RMS history, limit ladder | arm 13:45 ET |
| `forward_factor_arb` | Front IV vs forward vol, earnings-agnostic ladder | arm |
| `vol_risk_premium` | Short straddle when IV/RV ≥ 1.4 | approval / auto |
| `short_straddle` | Same family, IV/RV ≥ 1.2 | approval / auto |
| `earnings_quality` | Post-event drift (backtest; no live mapping yet) | — |

`debit_size_exploit` is disabled (`*.toml.disabled`) and not in the live signal path.

## Setup

```bash
git clone https://github.com/ichbinstudent/EarningsEdge.git
cd EarningsEdge/cli_scanner
uv sync --group dev
```

Python 3.12. `uv` is required (CI uses the same). Do not `pip install -e .` without the setuptools include list in `pyproject.toml`; the tree is a flat layout (`earnings_edge`, `framework`, …).

Copy secrets to `cli_scanner/.env` (gitignored):

```
TELEGRAM_BOT_TOKEN=
TELEGRAM_APPROVAL_CHAT_ID=
APCA_API_KEY_ID=
APCA_API_SECRET_KEY=
APCA_API_BASE_URL=https://paper-api.alpaca.markets/v2
LSE_API_KEY=
POLYGON_API_KEY=
FINNHUB_API_KEY=
```

`load_dotenv()` must run before any `earnings_edge` import. Entry points already do this; new scripts must too.

## Run

```bash
# Telegram bot (foreground). Prefer systemd trading-bot if sudo is available.
# If starting from Hermes, use setsid so SIGTERM to the agent does not kill it:
setsid ./run-bot.sh </dev/null >/dev/null 2>&1 &

# German crash alerts (own process — do not put this back in bot.py)
setsid ./run-crash.sh </dev/null >/dev/null 2>&1 &

# Health
curl -s http://127.0.0.1:8502/health

# CLI scanner
uv run python scanner.py --list

# Tests
uv run python -m pytest tests -q --tb=short
```

No sudo on the usual host: kill the **python** `bot.py` PID, not a bash wrapper, then relaunch. Wait 3–5s so Telegram polling can drop the old getUpdates lock.

Optional crash-alert knobs (defaults in parentheses): `GERMAN_CRASH_THRESHOLD` (0.20), `GERMAN_CRASH_WINDOW_SECS` (300), `GERMAN_CRASH_COOLDOWN_SECS` (1800), `GERMAN_CRASH_MAX_SPREAD` (0.10), `GERMAN_CRASH_TRADE_MAX_AGE_SECS` (600), `GERMAN_CRASH_ENABLED` (1).

## Layout

```
cli_scanner/
  bot.py                 Telegram entry
  scanner.py             CLI scan
  earnings_edge/         scanners, live signals, FF math, collectors
  framework/             risk, sizing, orders, exits
  strategies/*.toml      live strategy configs
  tests/
  data/                  SQLite + models (not in git)
```

Pricing: LSE primary when `LSE_API_KEY` is set, then Yahoo, then Polygon (`EARNINGS_PRICE_PROVIDER=auto`).

## License

MIT. See `LICENSE`.
