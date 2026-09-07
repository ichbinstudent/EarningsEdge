# Earnings Edge — Bot Architecture

Current as of 2026-09-04 (HEAD `2838e4e`). Grounded in the live tree — file names, schedules, and ports are as-deployed, not aspirational.

## Process topology

Three independent processes, each under its own systemd **user** unit (`~/.config/systemd/user/`, `Restart=always`, `RestartSec=10`):

| Unit | Executable | Role |
|---|---|---|
| `trading-bot.service` | `bot.py` | Telegram trading bot: scanning, proposals, execution, ops panels. Owns the scheduler. |
| `ee-crash-alert.service` | `crash_alert.py` | German-venue crash alerts (Gettex/Tradegate/Frankfurt). Telegram-only, never orders; shares the bot token but does not long-poll, so it can't collide with `bot.py`. |
| (deploy dir also ships) `ee-dashboard.service` | dashboard FastAPI | Optional; the dashboard is normally served *inside* `bot.py` instead (see below). |

`bot.py` additionally starts two localhost HTTP servers:
- **health** — `127.0.0.1:8502/health`
- **dashboard** — `127.0.0.1:8503` (FastAPI/uvicorn, panels + static mini-app; also exposed via Tailscale port-forward 8503 → 8449)

Both crash-alert polling (07:30–23:00 Berlin weekdays) and the bot's trading gates respect market-hours windows per venue.

## Repo layout (flat, by design)

```
bot.py                  TradingBot — wiring only: registration, scheduler, panel plumbing
crash_alert.py          German crash-alert process (standalone)
earnings_edge/          Domain layer: scanner, signals, picks, proposals, approval, views, rich messages
  db/                   SQLAlchemy engine/models/migrations + repositories (the only SQL-bearing layer)
  collectors/           Alpaca options, earnings calendar, LSE, Polygon, gettex/tradegate quotes
  services/             ScanService, OutcomeService (job-orchestrated persistence)
framework/              Trading framework: registry, risk, execution, positions, jobs, ops
strategies/             One TOML per strategy (see STRATEGIES.md for the math)
dashboard/              FastAPI server, desk actions, Telegram WebApp auth
scripts/                Pipeline + ops CLIs: scanner, collect, outcomes, train, backtest family,
                        calendar_call_backtest, polygon backfills, FF analysis, designer CLI
data/                   SQLite DBs, model artifacts, gettex quote cache
deploy/                 systemd unit files
```

History note: the repo was flattened from `cli_scanner/` to root on 2026-09-02; nothing lives in a sub-package anymore. On 2026-09-07 the root CLIs were consolidated into `scripts/`; the root keeps only `bot.py` + `crash_alert.py` (systemd `ExecStart` targets) and their supervisor scripts.

## Runtime: TradingBot (`bot.py`)

`TradingBot` is deliberately **wiring only** (~2,500 lines, 123 methods):

- Registers 24 `CommandHandler`s, each a one-line delegate to `earnings_edge/handlers.py` (23 `cmd_*` functions, moved verbatim from the class on 09-02; delegates kept so registration and test seams don't churn).
- Owns the APScheduler setup (see schedule table below).
- Panel plumbing: `_send_panel` (send + delete previous panel), `_edit_panel` (edit-in-place, fallback to new message), card/batch pushes, alert flushing.
- Long-polls Telegram (polling mode); auth is a **fail-closed allow-list** (`TELEGRAM_APPROVAL_CHAT_ID(S)` in `.env`, parsed by `earnings_edge/ops_auth.py`). Risk commands (`/halt`, `/resume`, `/promote`, `/demote`, restart) are dead until the list is configured.

### Scheduled jobs (APScheduler, America/New_York TZ)

| Job (id) | Crontab (ET) | What |
|---|---|---|
| `scanner_Earnings Calendar` | `0 14 * * mon-fri` | Earnings scan → **chained proposal build** (no separate proposal cron) |
| `ff_ladder_propose` | `45 13 * * mon-fri` | FF ladder arm/proposals |
| `ff_ladder_step` | `0,15,30,45 14-15 * * mon-fri` | Limit-ladder stepping |
| `equity_snapshot` | `*/15 9-16 * * mon-fri` | Equity snapshot + daily-loss check |
| `reconcile` | `*/30 9-16 * * mon-fri` | Broker reconciliation |
| `assignment_guard` | `45 15 * * mon-fri` | Ex-div ITM short-call guard |
| `exit_eval` | `*/15 9-16 * * mon-fri` | Exit rule evaluation |
| `chain_cache` | `5 9-16 * * mon-fri` | Hourly Alpaca options-chain cache |
| `daily_picks` | `0 7 * * mon-fri` | Refresh chains/signals, persist picks |
| `db_backup` | `15 0 * * *` | SQLite backup |
| `db_health_check` | `5 * * * *` | Integrity check (corruption caught ≤1h) |
| `scan_retry` (on demand) | one-shot, +12 min | Exactly one retry after a failed scan; never stacks |

Every scheduled job runs through `framework/jobs.py::run_job` — one `job_runs` audit row per run (success, stats JSON, error). Failed runs surface in `/jobs` and the pending inbox.

## Strategy layer

**TOML → registry → runtime.** `strategies/*.toml` is loaded by `framework/core/config.py` and resolved per strategy **code name** (the `trade.strategy` string engines emit) by `framework/core/registry.py::StrategyRegistry`: effective risk limits, lifecycle (`paper` → …), execution mode (`approval` | `auto`), and sizer. Runtime state lives in `strategy_state`; TOML seeds it with `INSERT OR IGNORE`, so operator `/promote` `/demote` wins over file changes until restart.

Live strategies: `calendar_call_ml`, `ff_ladder`, `forward_factor_arb`, `vol_risk_premium`, `short_straddle`, `earnings_quality` (backtest only). `debit_size_exploit.toml.disabled` — out of the signal path. Strategy math and entry/exit rules: `STRATEGIES.md`.

## Trading pipeline (synchronous strategies)

```
scan (14:00 ET, EarningsCalendarScanner + ScanService)
  → live_signals.py       latest scan frame → Trade candidates (calendar / straddle)
  → trade_approval.py    build_proposals(): risk gates → PendingTradeStore → Telegram cards
  → operator approves (✅) or auto mode
  → alpaca_bridge.py     preflight_combo(): every leg exists on Alpaca, two-sided book,
                         combo spread ≤ 40% of mid; cards priced at live Alpaca mid
  → execute_proposal()   → StrategyBridge.execute_trade → OrderManager (limit only)
  → trade_events + managed_positions rows
```

Design invariants:
- **Limits only, structurally.** `OrderManager._submit` refuses any submission without a limit price (the old silent market-order fallback was deleted). `PricingPolicy` subclasses: `MidPricePolicy`, `LimitWalkPolicy` (patient resting limits — used by ladders and exits).
- **Preflight before the card reaches your phone** — unlisted strikes (the PL 19.5 case) and wide spreads (HPE/AI case) are rejected as `reject_preflight` in the funnel, never shown as executable cards.
- **Veto reasons are threaded to the operator** — the Execute-click footer shows `last_look: spread 0.38 > 40% of mid 0.39`, not a bare counter dict (`skip_detail` since `d849b43`).
- The funnel (`proposal_funnel` table) counts every stage: candidates → gated → proposed → preflight-rejected → approved → executed.

**FF ladder async path** (`fwd_factor_ladder.py`, `fwd_factor.py`): proposals arm at 13:45 ET with a computed fair debit, then `LimitWalkPolicy` steps resting limits down toward the capped price every 15 min through 15:45 ET. Exits via `ScheduledExit` at front-leg expiry or rule-based exits.

## Risk

- `framework/risk/manager.py::RiskManager` — single chokepoint for `check_trade`: per-strategy spend caps, daily loss limits, repeated-rejection halts.
- `framework/risk/killswitch.py::KillSwitch` — DB-persisted global halt (survives restarts). `/halt` `/resume`; auto-trips on daily-loss breach from equity snapshots. Everything that submits orders checks it, including the FF ladder.
- `framework/risk/equity.py` — equity snapshots (equity, buying power) every 15 min in market hours; daily PnL vs day-start.

## Positions & exits

- `framework/execution/managed.py::open_groups` — the local book as position groups (legs, entry, event).
- `framework/positions/book.py::classify_book` — broker-truth reconciliation: **managed** (matched) / **orphan** (at broker, not local) / **missing** (local open, broker-closed). `/positions` renders this with per-item buttons (adopt / ignore / close / mark-closed — `book_actions.py`).
- `framework/positions/exits.py` — rule engine evaluated every 15 min: profit targets, stops, `ScheduledExit`; `guards.py` — assignment-risk (ex-div ITM short calls).
- Reconcile every 30 min; orphans alert once (dedup), then respect the ignore list.

## Data layer

- Single production DB: `data/earnings_ml.db` (SQLite, WAL). Legacy `earnings.db`/`scanner.db` are read-only history. Backups daily 06:15 Berlin into `data/backup/`.
- `earnings_edge/db/` — SQLAlchemy ORM (engine, `models.py` ~30 tables: `snapshots`, `live_calendar_candidates`, `ff_snapshots`, `ff_universe_snapshots`, `pending_trades`, `managed_positions`, `trade_events`, `job_runs`, `equity_snapshots`, `risk_events`, `proposal_funnel`, …).
- `repositories.py` — **the only SQL-bearing layer** (144 functions, one interface: every function opens its own session; the legacy leading-`sqlite3.Connection` compatibility form was removed 2026-09-02).
- `framework/data/` — catalog + model registry (deployed ML artifacts, versioned).
- Market data: `earnings_edge/market_data_provider.py` — resilient chain **LSE (primary) → Yahoo (fallback)**, optional Polygon (re-implements the yfinance surface); auto health-checks, re-probe every N calls. Yahoo access uses curl_cffi chrome impersonation (this host's IP gets blocked otherwise). LSE needs `LSE_API_KEY`.

## ML

- Screening model (`beat_expected_move`), deployed via `framework/data/model_registry.py`; calendar candidates scored TAKE/SKIP in `calendar_filter.py` (AUC ~0.76 on 397 dense rows, retrained 2026-08).
- Option models: magnitude regression family. Governance notes in `docs/model_governance_2026-08.md`.
- Backtest refresh report: `docs/backtest_refresh_2026-08.md` (19,346 snapshots / 624 calendar trades / 18 strategies).

## Telegram surfaces

- **Classic HTML panels** (`earnings_edge/bot_views.py`): `/status`, `/monitor`, `/positions`, `/strategies`, `/setups`, `/pending`, `/exits` — text + inline keyboards, edit-in-place on tap.
- **Rich-message tables** (`earnings_edge/rich_msg.py`, since `2838e4e`): `/picks`, `/orders`, `/jobs`, `/equity` render real `<table>` markup via raw `sendRichMessage` (Bot API 10.1 — python-telegram-bot 22.8 doesn't expose it, so these POST directly with httpx). Desk Refresh buttons edit rich panels in place via `editMessageText` with a `rich_message` payload. **Every rich attempt falls back to the classic rendering on any failure** — no page can break. Strategy names/OCC symbols contain underscores that silently break classic Markdown; that's why everything is HTML-escaped (`cards.esc` / `html.escape`).
- **Proposal/exit cards** (`earnings_edge/cards.py`, `trade_approval.py`): strategy-grouped batch pushes with per-item rows; entry cards show executable Alpaca-mid prices.
- **Mini App** (`dashboard/tg_auth.py` + `static/`): WebApp book/inbox/halt; auth is Telegram-initiated, `TELEGRAM_WEBAPP_URL` from `.env`.

## Ops & conventions

- **Secrets**: `.env` (gitignored). `load_dotenv()` MUST run before any `earnings_edge` import (Settings freeze at first import).
- **Paper only** — Alpaca paper API; `alpaca_mode.py::broker_label()` marks every surface.
- **Logging**: `framework/ops.py` secret-redaction filter; `InstanceLock` prevents double-starting (stale `data/trading-bot.lock` → remove before restart).
- **CI**: uv-synced venv (`.venv/`), Python 3.12; test suite `pytest tests/` — 787 passing as of `2838e4e`.
- **Conventions**: flat layout, no vestigial forks; changes land via agy/grok coding agents against scoped briefs, then independent verification (tests + live DB row shapes + traced callers) before commit; restart is manual and approval-gated.

## Known sharp edges

- Yahoo rate-limiting from this host can blind the scanner for days — LSE key is the fix; monitor via `/status` last-scan age.
- `day_start_equity()` / `daily_pnl()` return `None` outside market hours (views guard since `2838e4e`; older callers may not).
- The `.vexp/` dir is a local code-indexing daemon's workspace — its manifest diffs are benign tool artifacts, not repo changes.
