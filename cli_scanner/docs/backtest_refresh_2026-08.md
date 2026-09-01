# Backtest refresh under the v2.0 framework — 2026-08-08

Re-run of the full strategy backtest on the current production DB
(`data/earnings_ml.db`: 19,346 snapshots, 624 calendar trades, 604 live
candidates, 9,754 scan outputs). The Jul 6 rankings predate the v2.0
framework (TOML strategies, risk gates, exit engine); this refresh is the
post-framework baseline.

Rerun commands:

```bash
cd cli_scanner
.venv/bin/python backtest.py --all --output /tmp/backtest.json   # report
# metrics below: same engine, full trade lists (no 100-trade truncation),
# per-strategy max drawdown from date-ordered cumulative P&L
```

**Units caveat.** Calendar family P&L is dollars per 1-lot; positional and
multi-strike families report their own P&L conventions (dollars);
`stock_drift_pead` and `earnings_quality` P&L is in percent. Compare within
families, not across. Drawdowns inherit the same mixed units.

## Ranked comparison

### Calendar-call family (P&L in $ per 1-lot)

| Rank | Strategy | Trades | Taken | Hit rate | Total P&L | Avg P&L/trade | Avg ret-on-debit | Max DD |
|---|---|---|---|---|---|---|---|---|
| 1 | debit_size_exploit † | 388 | 267 | 49.8% | **+7,017** | +26.28 | 24.9% | −1,013 |
| 2 | calendar_call_ml † | 555 | 346 | 44.2% | **+5,797** | +16.75 | 21.2% | −1,721 |
| 3 | iv_rv_mean_reversion | 407 | 269 | 47.2% | +4,692 | +17.44 | 22.0% | −1,723 |
| 4 | term_structure_steepener | 176 | 109 | 57.8% | +2,949 | +27.06 | 12.9% | −521 |
| 5 | calendar_call_high_conviction | 555 | 40 | 47.5% | +1,223 | +30.57 | 125.4% | −14 |
| 6 | calendar_call_no_ml | 555 | 555 | 37.1% | **−3,042** | −5.48 | 10.9% | −6,514 |
| — | dax_forward_vol | 0 | 0 | — | — | — | — | — | (no Eurex collection) |
| — | short_straddle (legacy) | 0 | 0 | — | — | — | — | — | (placeholder) |

† = TOML-registered strategy (live under the v2.0 framework).

The ML filter remains the difference between profit and loss on identical
candidates: `calendar_call_ml` +$5,797 vs `calendar_call_no_ml` −$3,042
(same 555 rows). `debit_size_exploit` overtakes it on total P&L with a
better hit rate and smaller drawdown — the ≤3%-debit envelope is doing real
work. `calendar_call_high_conviction` has the best risk shape (max DD −$14,
125% avg return on debit) but only 40 taken trades.

### Positional family (P&L in $)

| Rank | Strategy | Trades | Hit rate | Total P&L | Avg P&L/trade | Max DD |
|---|---|---|---|---|---|---|
| 1 | vol_risk_premium † | 233 | **94.8%** | **+4,410** | +18.93 | −13.2 |
| 2 | short_straddle † | 246 | 93.1% | +3,202 | +13.02 | −17.4 |
| 3 | directional_call | 4 | 0% | −1,827 | −456.85 | −1,827 |
| 4 | directional_put | 8 | 0% | −78,735 | −9,841.82 | −78,735 |
| 5 | long_straddle | 11 | 0% | −80,497 | −7,317.88 | −80,497 |

The two short-premium strategies keep their edge (93–95% hit rates, tiny
drawdowns). All three long-premium/directional strategies are 0%-hit-rate
money losers — consistent with the 2026-07-28 finding that the
magnitude/direction model heads carry no signal (`references/option-models.md`).

### Multi-strike family (P&L in $)

| Strategy | Trades | Hit rate | Total P&L | Max DD |
|---|---|---|---|---|
| iron_condor_real | 1 | 0% | −0.39 | −0.39 |
| butterfly_real | 295 | 29.5% | −664 | −666 |
| risk_reversal_real | 238 | 30.7% | −541 | −543 |

All negative; none are TOML-registered or traded live.

### Edge vs expected move

Only `stock_drift_pead` trades carry both `expected_move_pct` and
`actual_move_pct` features (501 of 1,293 rows): the expected move exceeds
the realized |move| by **+6.27pp on average** — implied earnings moves are
systematically richer than realized, which is exactly the premium the
short-vol strategies harvest. `stock_drift_pead` itself is a coin flip
(52.7% hit rate, +6.97% cumulative over 1,293 trades, max DD −571.8pp) and
`earnings_quality` is flat (50.5% hit rate, avg +1.11%/trade before costs —
directional surprise-chasing has no net edge).

## Divergence flags: backtest vs live behavior

Measured live behavior comes from `docs/proposal_quality_2026-08.md`
(EE-09, 2026-08-07: 8 approved proposals, 4 executed, all ff_ladder).

1. **ff_ladder — divergence, execution-driven.** No backtest exists for
   ff_ladder (live-only ladder execution), so the only measurement is live:
   −$1,225 realized over 3 priceable trades despite the vol thesis being
   right 6/8 (timing-aware). Cause is paper fill quality (~1.6x combo mid on
   entry), not signal. Until fills are benchmarked against combo mid, live
   ff_ladder results cannot be compared against any backtest convention.
2. **calendar_call_ml — divergence, flow-driven.** Backtest: +$5,797, the
   primary strategy. Live: the approval flow has produced **zero proposals**
   since 2026-07-25 (`proposals built: 0 (candidates: 0)` every weekday) —
   the local gate funnel silently rejects everything and has no per-stage
   rejection counters. The backtest ranking is currently not expressed in
   live trading at all.
3. **short_straddle / vol_risk_premium — divergence, sizer-driven.**
   Backtest: the two best risk-adjusted strategies (93–95% hit rate).
   Live: the `vol_target` sizer vetoes them (`risk_pct` 1% of ~$100k =
   $1,000 budget vs strike×100×0.20 notional-proxy max loss → qty 0 →
   `size_veto`). Zero live proposals expected until the sizer math is
   revisited (see `references/execution-wiring.md`).
4. **debit_size_exploit — no live data.** Top of this refresh's calendar
   ranking, TOML-registered, but has never produced a live proposal (same
   gated funnel as calendar_call_ml). Unverifiable until flag 2 is fixed.
5. **Exit engine never fires on ff_ladder fills** (ops finding from EE-09,
   restated): the 4 filled ladder positions are still `open` 9 days
   post-event — the TOML time exit (`days_before_event = 1`) cannot trigger
   for same-day fills. Backtest assumes on-time exits; live positions are
   abandoned. P&L attribution between entry and exit logic is currently
   impossible.

## Top-3 ranking (this refresh)

1. **debit_size_exploit** — +$7,017, 49.8% hit, max DD −$1,013 (calendar)
2. **calendar_call_ml** — +$5,797, 44.2% hit, avg 21.2% return on debit (calendar)
3. **vol_risk_premium** — +$4,410, 94.8% hit, max DD −$13 (positional)

All three are TOML-registered; none is currently expressing its backtest
edge live (flags 2–4).

## Tests

- `tests/integration/test_backtest_toml_refresh.py` (4): deterministic
  backtest metrics for TOML strategies on fixture data
  (`debit_size_exploit`, `earnings_quality`), exit engine driving fixture
  positions under the real `calendar_call_ml.toml` rules (profit-target
  auto-close; time-exit proposal + dedup).
- `tests/e2e/test_backtest_cli_report.py` (1): full `backtest.py` CLI
  subprocess run on a fixture DB produces the JSON report artifact with
  exact expected summaries.
