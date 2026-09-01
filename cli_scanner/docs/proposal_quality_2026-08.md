# Proposal quality measurement — 2026-08-07

Question: do the bot's operator-approved trade proposals beat the expected move
against realized outcomes? Measured, not assumed. Rerunnable:

```bash
cd cli_scanner
.venv/bin/python3.12 scripts/proposal_quality.py                      # offline, DB only
.venv/bin/python3.12 scripts/proposal_quality.py --fetch-exit-prices  # + Polygon event moves & exit marks
```

## Headline (honest, small sample)

**Sample is too small for significance** (8 approved proposals, 4 executed;
threshold for any retune decision was ~20). Descriptive stats only:

| Metric | Value |
|---|---|
| Approved proposals since 2026-07-25 | 8 (all ff_ladder; see findings) |
| Executed (filled) | 4: META, MSFT, QCOM, ICLR (earnings 2026-07-29) |
| Beat expected move, stored outcome convention | 8/8 (100%) — **flattering, see AMC caveat** |
| Beat expected move, timing-aware event move | 6/8 (75%), binomial p=0.29 vs 50% |
| Beat expected move, executed trades only (timing-aware) | 2/4 |
| Realized P&L (strategy exit convention) | **−$1,225** over 3 priceable trades |
| Statistical significance | none (n=8 scorable; n=4 executed) |

Per-trade detail (executed):

| Ticker | Entry debit | Exit credit (Jul 30) | P&L | Expected move | Realized event move | Hit? |
|---|---|---|---|---|---|---|
| META | $10.20 | $4.43 | −$577 | 7.92% | −7.95% | marginal miss |
| MSFT | $6.55 | $2.22 | −$433 | 7.16% | +15.51% | miss (2.2x expected) |
| QCOM | $4.35 | $2.20 | −$215 | n/a (scan-time fetch fail) | −2.62% | hit vs implied 13.6% |
| ICLR | $6.30 | unpriceable¹ | n/a | n/a | −7.10% | hit vs implied 11.3% |

¹ ICLR far leg (ICLR261016C00180000) has zero Polygon daily bars in the entire
window — the contract appears to have never printed. Near leg printed once
(6.70 on Jul 30). P&L excluded rather than estimated.

Unfilled approved proposals (ladder expired/disarmed): ATLO, HUBG, BA
(expired), V (disarmed, spot-drift guard). All four would have been "hits"
(realized < implied) — the ladder's price discipline kept us out of trades
that would have been fine, and let us into META/MSFT which were not.

## Why winning the vol thesis still lost money

The realized-vs-implied read was mostly right (6/8 under the honest event-move
convention), but the executed trades still lost ~30-50% of debit each. Cause:
**paper fill quality**. Entry fills came in at ~1.6x the combo mid
(META: filled 10.20 vs mid 6.39; MSFT 6.55 vs 3.78; QCOM 4.35 vs 2.57).
The ladder's D* cap (17.02 for META) was far above mid, so the marketable
limit crossed a very wide spread on Alpaca paper. The entry overpayment
exceeded the entire IV-crush edge even when the thesis was directionally
correct (QCOM: realized 2.62% vs implied 13.6% and still −$215).

Threshold-retune implication: **do not retune the 20% premium threshold on
this data**. The binding constraint is execution price, not signal quality.
Any retune needs (a) ≥20 executed proposals, (b) fills benchmarked against
combo mid at fill time, (c) the D* inversion re-checked against attainable
fills (candidate quotes had near-leg spreads of 50-80% of mid on the filled
names).

## Measurement caveat that matters: stored outcomes are one session early for AMC names

`snapshots.actual_move_pct` uses the outcome-service convention
(post_bar = first close ON/AFTER the earnings date). For "Post Market" (AMC)
reporters that close is the **pre-announcement** session. Measured on the four
filled trades (all AMC 2026-07-29):

| Ticker | stored actual_move_pct | true event move (Jul 29→30 close) |
|---|---|---|
| META | −1.31% | −7.95% |
| MSFT | −0.71% | +15.51% |
| QCOM | −4.42% | −2.62% |
| ICLR | −1.14% | −7.10% |

Under the stored convention the proposals "beat the expected move" 8/8; under
the timing-aware event window it is 6/8 (2/4 executed). Any hit-rate computed
from `snapshots.actual_move_pct` alone — including ML labels like
`beat_expected_move` — inherits this bias for AMC names (roughly half of US
earnings). `scripts/proposal_quality.py --fetch-exit-prices` recomputes the
event move timing-aware via `snapshots.timing`; the fix for the stored labels
themselves is a separate change to `OutcomeService.outcome_from_bars`.

## Ops findings (out of scope to fix here, flagged for David)

1. **The calendar-ML approval flow has never produced a proposal.**
   `pending_trades` is empty. Journal shows `proposals built: 0
   (candidates: 0)` every weekday since 2026-07-25 — the chained 15:15 ET
   build runs but zero TAKE trades pass the local gates (legs/DTE/position).
   Everything approved so far was FF ladder. If calendar-ML proposals are
   expected, the gate funnel needs a per-stage rejection count (currently
   invisible).
2. **The exit engine never fired on the 4 filled positions.** All
   `managed_positions` legs are still `status='open'` 9 days after the event;
   `exit_proposals` is empty despite the exit cron running every 15 min.
   The ff_ladder TOML time exit is `days_before_event = 1`, but the fills
   happened on the event day itself (2026-07-29), so that rule can never
   trigger for this strategy. Positions are effectively abandoned — the P&L
   above is what an on-time exit would have looked like.

## Artifacts

- Script: `cli_scanner/scripts/proposal_quality.py` (rerunnable; offline by
  default, `--fetch-exit-prices` adds Polygon event moves + exit marks at
  13s pacing)
- Full JSON: `cli_scanner/docs/proposal_quality_2026-08-07.json`
- Tests: `tests/integration/test_proposal_quality.py` (3), `tests/e2e/test_proposal_quality_e2e.py` (1)
