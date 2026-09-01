# EarningsEdge Strategy Architecture

This document outlines the core strategies available in the `cli_scanner` framework. 
Strategies are configured via `.toml` files in the `strategies/` directory and executed either asynchronously via limits (`ff_ladder`) or as immediate paper/live orders via the `trade_approval` pipeline.

## 1. `ff_ladder` (Forward-Factor Limit Ladder)
**Type:** Earnings Pre-Event Calendar Spread
**Risk Profile:** Defined Risk
**Execution:** Asynchronous Limit Ladder (`14:00 - 15:45 ET`)

**Architecture:**
- Focuses on capturing implied forward volatility underpricing ahead of earnings.
- Built via `fwd_factor.py` and `fwd_factor_ladder.py`.
- **Target Math:** Calculates the `required_near_iv` such that the implied earnings move matches a historical RMS move + 20-25% premium. Derives a theoretical limit debit price.
- **Entry:** Steps into the market starting from the cheapest price (25% premium) down to a capped price (20% premium).
- **Exit:** Uses `ScheduledExit` to hold the trade until the expiration date of the near (short) leg. 

## 2. `forward_factor_arb` (Pure Mathematical Forward Factor Arbitrage)
**Type:** Relative Value Volatility Arbitrage
**Risk Profile:** Defined Risk (Calendar Spread)
**Execution:** Asynchronous Limit Ladder

**Architecture:**
- Pure mathematical forward volatility arbitrage proposed to replace the earnings-jump-based logic.
- **Math (`forward_factor_arb.py`):**
  - Extracts the **Forward Volatility** between the front leg (30-60 DTE) and back leg (60-90 DTE).
  - Isolates the earnings-event variance from the front-leg IV to derive the **Ex-Earnings IV**.
  - Calculates the **Ex-Earnings Forward Factor**: `(Front Ex-Earn IV / Ex-Earn Fwd Vol) - 1`.
- **Entry:** Scans for `Factor > 1.1`. Builds a limit ladder starting at a debit where Factor = 1.5, stepping down to Factor = 1.25.
- **Exit:** `ScheduledExit` at the front leg's expiration.

## 3. `calendar_call_ml` (ML-Scored Calendar)
**Type:** Model-Driven Calendar Spread
**Risk Profile:** Defined Risk
**Execution:** Immediate Market/Mid Order upon Approval

**Architecture:**
- The primary proposal strategy for standard earnings trades.
- Uses a machine learning model artifact to score the `TAKE` or `SKIP` viability of a standard calendar call spread.
- **Exit:** Exited mechanically `90` minutes before the market close on the day of the near leg's expiration (`ScheduledExit`), or via a 50% profit target / 75% stop loss.

## 4. `vol_risk_premium` (VRP Harvest)
**Type:** Short Premium / Short Straddle
**Risk Profile:** Undefined Risk

**Architecture:**
- Sells short straddles purely to harvest systematic volatility risk premium (VRP).
- **Entry:** Scans the earnings universe for names where implied volatility vs. realized volatility (`iv_rv_ratio`) is `>= 1.4` and the implied Expected Move is `>= 6%`.
- **Sizing:** Uses `vol_target` sizing, aggressively budgeting for a stress scenario where the stock moves 2.0x the priced straddle move.
- **Exit:** Exited 2 days after entry.

## 5. `short_straddle` 
**Type:** Standard Short Straddle
**Risk Profile:** Undefined Risk

**Architecture:**
- Sister strategy to `vol_risk_premium`, but looking for slightly less extreme ratios (`iv_rv >= 1.2`).
- Designed specifically to capture the immediate overnight IV crush post-earnings.
- **Exit:** Mechanics explicitly trigger an exit exactly **1 day** after entry (the session immediately following the earnings announcement).

## 6. `earnings_quality` 
**Type:** Post-Earnings-Announcement Drift (PEAD)
**Risk Profile:** Defined Risk
**Execution:** Equity or Defined-Risk Options

**Architecture:**
- Trades the directional drift *after* an earnings announcement based on fundamental metrics (earnings quality).
- **Exit:** Held for `10` days (`days_after_entry = 10`) to capture the prolonged PEAD drift.

---
*(Note: `debit_size_exploit` has been fully removed from the active routing logic and disabled, as it was deemed non-executable in reality).*
