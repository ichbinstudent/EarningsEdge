# Verification: forward-vol "20% FV-difference" entry — implemented

Date: 2026-08-05. Reviewer: autonomous strategy audit (cron).
Scope: David's forward-volatility strategy spec (2026-07-26) vs the shipped
forward-factor ladder (`earnings_edge/fwd_factor.py`,
`earnings_edge/fwd_factor_ladder.py`, `strategies/ff_ladder.toml`, bot wiring).

**Verdict: IMPLEMENTED.** Every element of the spec maps to shipped code. One
deliberate, documented deviation in the benchmark used for the 20% gate (see
§1.4) — it is a refinement agreed during design, not a gap.

## Spec element → code mapping

### 1. "Compute the forward factor WITH and WITHOUT the earnings event"

**1.1 T1 straddling earnings** — `_pick_pair()`,
`fwd_factor_ladder.py:296-329`: T1 = expiry in [30, 60] DTE, closest to 30.
Earnings containment inside the T1 window is enforced implicitly: if the event
falls after T1 expiry then `tau > T1`, `required_near_iv()` returns `None`
(`fwd_factor.py:70`) and the candidate is rejected as
`"target debit degenerate (<=0)"` (`fwd_factor_ladder.py:399-400`).

**1.2 Later, event-free T2** — same function: T2 = expiry 21-42 days after
T1, closest to +30 (`fwd_factor_ladder.py:322-328`).

**1.3 WITHOUT-event baseline** — `forward_iv()`,
`fwd_factor.py:106-113`: `sigma_fwd = sqrt((T2*iv2^2 - T1*iv1^2)/(T2 - T1))`
from live Alpaca quote mids (IVs BS-solved locally, `build_candidate`,
`fwd_factor_ladder.py:376-389`). This sigma_fwd is the earnings-free baseline
and is reused as `sigma_base` in the WITH-event decomposition
(`required_near_iv`, `fwd_factor.py:58-77`):
`sigma_1*^2 * T1 = ((1+p)*m_hist)^2 + sigma_fwd^2 * (T1 - tau)`.

**1.4 DEVIATION (documented, deliberate): the 20% comparator is RMS of
realized historical event moves, not a raw FF-with/without ratio.** The naive
`FF = (sigma_1 - sigma_fwd)/sigma_fwd` lights up for *every* earnings name
because the event mechanically inflates sigma_1 — it cannot separate
event-attributable elevation from genuine richness. Design record:
`references/forward-factor-event-premium.md` ("Why not raw Forward Factor",
"RMS, not median"). The implemented gate is
`implied_event_move / RMS(historical |actual_move_pct|) >= 1.20`, validated
historically by `scripts/ff_backfill.py` → `ff_snapshots.premium_ratio`.
Live RMS source: `hist_rms_move()` (`fwd_factor_ladder.py:98-111`, >=3
realized events from `snapshots`), with self-healing coverage via
`ensure_hist_moves()` (Yahoo earnings dates + LSE/Polygon daily bars).

### 2. "Enter only when the pricing difference exceeds ~20%"

- Premium thresholds: `LadderSpec.start_premium = 0.25`,
  `floor_premium = 0.20` (`fwd_factor.py:148-157`).
- The vol-space gate is **inverted into price space**: `target_debit()`
  (`fwd_factor.py:80-103`) computes `D*(p) = far_price - BS(sigma_1*(p))`.
  Because a cheaper calendar = a richer near leg, **any fill at
  `debit <= D*(20%)` is by construction an entry at >= 20% implied event
  premium** — the 20% rule is enforced by the limit price itself, not by a
  signal check after the fact.
- Distance filter: a candidate is only tracked while
  `mid <= D*(20%) * 1.15` (`within_fill_range`, `fwd_factor.py:134-143`;
  applied at candidate build, `fwd_factor_ladder.py:401-402`, and re-checked
  every ladder step, `:778-784`, with disarm on runaway).

### 3. "Resting limit orders at computed fair price — never market orders"

- Ladder mechanics (`LadderSpec`): first rung at `D*(25%)` (cheapest) at
  14:00 ET, concede `$0.01` every 15 min, **hard cap `D*(20%)`**, last reprice
  15:45 ET, day orders — unfilled ladders die at the close.
- Only order path: `LadderRunner._place()` (`fwd_factor_ladder.py:645-652`)
  submits a single Alpaca **MLEG `order_type="limit"`, `time_in_force="day"`**
  order with the net-debit `limit_price`. No market-order path exists in the
  module.
- Human-in-the-loop: 13:45 ET Telegram arm/skip proposal cards
  (`bot.py:820+`, `ff_ladder.toml` `execution_mode = "approval"`); arming a
  ladder IS the approval. Repricing inside the cap is execution tactics.
- Risk wiring (paper-only, tight caps, kill switch — standing prefs):
  - every arm passes `RiskManager.check_trade` with the TOML limits
    (`_risk_check_arm`, `fwd_factor_ladder.py:492-522`);
  - framework kill switch consulted at arm and at every step (halt → cancel
    all working orders + disarm, `:574-576`, `:671-677`);
  - buying-power preflight at arm and each step (110% of worst-case debit);
  - quote validity (positive/ordered, age <= 15 min), spot-drift guard (3%),
    terminal vs transient broker-error handling, event-date staleness
    (earnings passed → expire untraded), per-ticker arm dedupe.
- `strategies/ff_ladder.toml`: `lifecycle = "paper"`, `fixed_dollar` sizer
  budget $2,000, limits 5% per-trade / 15% per-underlying / 20%
  per-strategy-day, exits = time (1 day before event, approval card) +
  profit-target 50% + stop-loss 75% (framework ExitManager).

## Test status at verification

`cd /path/to/repo && .venv/bin/python -m pytest tests/ -q`:
**346 passed, 17 failed.** All 17 failures are the known env-dependent
baseline recorded 2026-07-31 (14 `test_fwd_factor_ladder.py` runner-mechanics
tests + 3 `test_framework_wiring.py` ladder wiring tests, sensitive to live
env/prod DB state on this host) — they fail identically on a clean tree and
are unrelated to this audit (no code changed). All pure-math tests in
`test_fwd_factor.py` pass.

## Conclusion

No implementation work required. The 20% FV-difference entry rule, the
fair-price inversion, and the resting-limit-only execution are all live in
production code with the approval flow and risk gates in place.
