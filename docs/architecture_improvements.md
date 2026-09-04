# Architecture Improvements Plan

Written 2026-09-04 (HEAD `d8097c6`). Source: architect review of the live tree — every item cites a concrete incident or metric from this repo, not generic advice. Companion: `architecture.md` (current-state doc).

Execution model: waves of independent coding agents (agy/grok) in isolated git worktrees; each branch is verified independently (tests + live-tree reasoning + traced callers) before merging to main; deploy (bot restart) once per wave, not per item.

## Tier 1 — cheap, prevents real incidents

### 1. Contract tests against the real DB schema
**Problem (incident 2026-09-04):** three wrong dict keys (`strategy_name` vs `strategy`; `date`/`avg_equity` vs `d`/`e`) sailed through 11 green tests because fixtures encoded the same wrong guesses the code made. Tests validated code against itself, not against reality. The same fixture-guessing pattern exists across view tests generally.
**Fix:** a `tests/conftest.py` fixture that builds a fresh SQLite, runs the *real* migrations (`earnings_edge/db/migrations.py`), seeds representative rows, and view/repo tests run against that instead of `MagicMock` return values. Migrate the highest-risk tests first (bot_views, rich_msg, repositories, trade_approval).
**Verify:** mutation test — temporarily rename a column access in a view; the suite must go red.
**Effort:** ~half a day of agent work.

### 2. Typed row schemas + mypy in CI
**Problem:** repositories return `list[dict]` — untyped. The Tier-1 incident would have been a compile error instead of a runtime blank-column.
**Fix:** `TypedDict` per query shape in `earnings_edge/db/repositories.py` (start with the ~15 functions consumed by views/handlers: `trade_events_list`, `job_runs_list`, `equity_snapshots_*`, `strategy_state_list`, `exit_proposals_list_pending`, `scan_runs_*`, `job_runs_*`, `trade_events_list`, `adopted_positions_symbols`); add `mypy` to the dev group with `--strict` scoped to `earnings_edge/db`, `earnings_edge/rich_msg`, `earnings_edge/bot_views`, `framework/risk` first (project-wide strict is a later milestone). CI workflow gains a lint+type job next to pytest.
**Verify:** CI red on a deliberately wrong key; green on main.
**Effort:** ~half a day.

### 3. Exception policy by path
**Problem:** bot.py alone has 53 bare `except Exception`; trading paths (trade_approval: 11, alpaca_bridge: 14, fwd_factor_ladder: 16) can silently swallow failures. A swallowed failure in an execution path is worse than a crash — systemd restarts crashes, it can't see silent ones. (Related incident: the 09-01 SQLite corruption went unnoticed for hours because jobs failed silently into job_runs.)
**Fix:** policy — bare broad-excepts banned in `trade_approval.py`, `alpaca_bridge.py`, `fwd_factor_ladder.py`, `framework/execution/`, `framework/risk/`; each existing one must either re-raise, narrow to expected types, or write a `risk_event` row via `framework.risk.killswitch.record_event`. Cosmetic/panel code (bot.py view paths, bot_views, rich_msg) keeps broad catches — that's correct there. Enforce with a tiny CI lint (grep-based allowlist) so it can't regress.
**Verify:** the lint fails on a newly added bare except in a banned path; existing suite green.
**Effort:** ~2–3 hours.

## Tier 2 — structural

### 4. Finish the bot.py diet
**Problem:** still 2,489 lines / 123 methods despite "wiring only" being the stated design goal (was 2,783 on 09-02). Remaining mass: the `_handle_callback` elif chain (~30 branches), card/batch push logic, approval flow plumbing.
**Fix:** (a) callbacks → route map (`{"bk_": ..., "desk_": ..., "in_": ..., "st_": ...}` dispatch, mirroring CommandHandler registration); (b) card/batch push logic → `earnings_edge/push.py`; (c) approval flow → `earnings_edge/approval_flow.py`. One-line delegates + registration table stay. Target: bot.py < 800 lines. Behavior-identical: the 787-test suite must stay green without modifying any test.
**Verify:** line count; `git diff --stat` shows only moves; pytest green untouched.
**Effort:** ~1 agent-day. **Depends on:** nothing (but do after 1+2 land so contract tests guard the move).

### 5. Unified panel renderer
**Problem:** every rich page carries two code paths in its handler (try rich → fall back) — six copies of the same conditional; the remaining HTML pages (`/status` `/positions` `/strategies` `/pending` `/monitor`) would each need it again. Layout is hand-rolled per page (the /orders Detail-column problem).
**Fix:** view functions return a spec — `(title, columns, rows, paragraphs, keyboard)` — and one renderer emits both rich markup and classic HTML from it; one sender owns the try-rich-fallback-classic logic (currently duplicated in 4 handlers + 3 desk-refresh callbacks). Then converting remaining pages is data work, not plumbing.
**Verify:** `/picks` `/orders` `/jobs` `/equity` byte-identical output (golden tests on both formats); fallback still exercised by mocked failure.
**Effort:** ~1 agent-day. **Depends on:** item 4 (avoid moving code twice).

### 6. Telegram outbox
**Problem:** card pushes are fire-and-forget with `logger.error` on failure — a Telegram hiccup during a push loses the card until the next scan. Operator approval cards are the system's core UX.
**Fix:** `outbox` table (id, chat_id, payload, kind, status, attempts, next_retry_at) written in the same transaction as the proposal; a drain loop sends + retries with backoff; idempotent sends (dedupe by id). Reuse for crash alerts and risk-event pushes later.
**Verify:** integration test with a mocked flaky Telegram that eventually delivers; no duplicate cards after retry.
**Effort:** ~1 agent-day.

### 7. Scheduler hardening
**Problem A:** schedules are Berlin-crontab strings that merely *happen* to track ET for current slots — every new job is a DST landmine (documented convention, not enforced mechanism). **Problem B:** no `max_instances`/`misfire_grace_time`/`coalesce` — after a VM pause APScheduler defaults can silently pile up or drop runs.
**Fix:** (a) express the *intent* (ET clock times) directly — either `CronTrigger(hour=..., minute=..., timezone="America/New_York")` or a small `ET_CRON` table that generates the Berlin crontab (pick one, document it); (b) set explicit `max_instances=1`, `misfire_grace_time`, `coalesce=True` on every job in `_setup_scheduler`.
**Verify:** all 12 jobs still fire at the same absolute ET times (assert on trigger computation, not wall-clock); VM-pause simulation test for misfire handling.
**Effort:** ~3 hours.

### 8. Single-writer DB discipline
**Problem:** the 09-01 SQLite corruption came from concurrent writers (bot + recovery scripts hitting the DB simultaneously). Hourly integrity check + daily backup are symptom treatment.
**Fix:** structural — all writes through the bot process; scripts/crons get read-only access (SQLAlchemy `create_engine(..., isolation_level)` / a `--read-only` entry convention in scripts). Long-term option: WAL + busy_timeout everywhere as the floor. Postgres only if this ever chafes — at current scale SQLite + one writer is the *right* answer.
**Effort:** ~half a day. **Depends on:** item 6 (outbox) for any cross-process write needs.

## Tier 3 — bigger bets

### 9. Hang watchdog via systemd notify
**Problem:** `Restart=always` catches exits, not hangs. A hung provider call (no timeout on a data fetch) freezes scanning/exit/order management invisibly.
**Fix:** `Type=notify` + `sd_notify("WATCHDOG=1")` pings from the main loop / a watchdog thread in bot.py; unit gains `WatchdogSec=300` (5 min). Requires systemd ≥ 246 for user units (check host version).
**Effort:** ~3 hours.

### 10. Metrics endpoint
**Problem:** ops watch (EE-03 cron) greps logs to reconstruct job health; the data is already in `job_runs`/`equity_snapshots`.
**Fix:** tiny textfile/Prometheus-format export on the existing health port (8502) — job success rates, last-run age per job, equity, kill-switch state. The EE-03 cron then reads a URL instead of parsing journalctl.
**Effort:** ~3 hours.

## Explicit non-goals
Service split, message broker, event sourcing, Kubernetes. One operator, paper trading, one box — the monolith with enforced internal seams is correct. The value is boundaries being *enforced* (items 1–4), not more processes. Also: archive `data/` recovery debris (`recovered*.sql`, `dead_db_*`, sqlite-tools) now that the DB is stable.

## Sequencing

- **Wave 1 (parallel, no interdeps):** 1 + 2 + 3 + (7,9,10 as one small batch) — four isolated worktrees, four agents, one brief each.
- **Wave 2 (sequential after 1+2):** 4 → 5 → 6 → 8.
- **Deploy:** one bot restart per wave (after wave-2 items that affect behavior; wave-1 items 1–3 are test/lint-only, item 7/9/10 affect the running unit).
- Every merge: full suite green + independent verification by the supervisor (not the agent's self-report).
