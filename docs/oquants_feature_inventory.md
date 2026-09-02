# Option Quants (oquants.com) — Feature Inventory

Recon date: 2026-08-20. Sources: sitemap, RSC payloads (`self.__next_f.push`), page JS chunks, embedded docs.
Auth note: the provided session token returned `401` on `/api/auth/token` and pages rendered with
`isAuthenticated:false, isSubscribed:false` — all findings below come from public/marketing content,
embedded docs, and client JS bundles (which expose full table schemas and API shapes). No data API
responses were obtained.

## Architecture / API surface

- Next.js (Vercel) app. Marketing pages server-rendered; app pages hydrate client-side.
- Data API base: `https://api2.oquants.com/api/v1/` (browser fetches with Bearer JWT).
- JWT obtained from `GET https://oquants.com/api/auth/token` (cookie session → JWT, cached in
  `localStorage["oq:auth:token"]`). Non-`research/` requests time out at 120 s; `research/` at 900 s.
- All data responses wrapped as `{ data: ... }`. POST endpoints take JSON bodies.
- Data vendor signals: field naming is ORATS-style (`orFcst20d`, `orIvFcst20d`, `avgOptVolu20d`,
  `pxAtmIv`, `ivPctile1y`, `stkPxChng1wk`, `tkOver`, `contango`, `ivHvXernRatio`).

### Endpoint map (from JS bundles)

- `scan/filter` (POST, infinite scroll: `{filters, limit, offset}` → `data.filtered_data[]`)
- `scan/universe`, `scan/get-custom-scans`, `scan/create-custom-scan` (POST),
  `scan/update-custom-scan` (POST), `scan/delete-custom-scan` (DELETE)
- Picks engines (all POST `{filters: []}`):
  `scan/get-earnings-strategy`, `scan/get-pre-earnings-long-vol-strategy`,
  `scan/get-momentum-skew-strategy`, `scan/get-forward-factor-strategy`,
  `scan/get-vrp-strategy`, `scan/get-vrp-stock-strategy`
- `dashboard/earnings/`: `card-data`, `card-data-batch` (`{tickers: "A, B", lookback: 7}`),
  `earnings-calendar`, `implied-move-timeseries`, `implied-moves-distribution`,
  `realized-moves-distribution`, `iv-crush`, `moves`, `quarterly-moves`, `returns`,
  `next-earnings-info`
- `dashboard/volatility/`: `backtest`, `forward-factors`, `forward-factors-timeseries`,
  `ivol-spot-corr`, `rv-vrp-scatter`, `skew`, `skew-timeseries`, `term-structure`,
  `ts-slope-timeseries`, `volcone`, `volume-oi`, `vrp`
- `dashboard/relative-value/`: `correlation`, `custom`, `iv-scatter`, `rolling-max`, `rv-scatter`,
  `skew`, `term-structure`, `vrp-scatter`
- `dashboard/directional/`: `implied-distribution`, `momentum`, `pead`, `skew`, `stock-chart`
- Position Designer: `positions/effective-fill-iv`, `positions/forecast-simulation`,
  `positions/interest-rates`, `positions/optimal-delta-hedge`, `live/option-chain`,
  `live/ticker-info`
- Portfolio: `portfolio/balance`, `portfolio/balance/transactions`, `portfolio/graphs`,
  `portfolio/metrics`, `portfolio/strategies`, `portfolio/structures`, `portfolio/trades`
- Research/backtest: `research/workspaces`, `research/workspace`, `research/create-workspace`,
  `research/update-workspace`, `research/delete-workspace`, `research/create-universe`,
  `research/research-variable`, `research/build-model`, `research/save-model`,
  `research/saved-models`, `research/update-model`, `research/delete-model`,
  `research/model-predictions`, `research/run-model-live`, `research/backtest`,
  `research/create-strategy`, `research/set-train-split`
- Auth/account: `auth:accounts`, `auth:user/preferences` (PATCH), `auth:user/watchlist`,
  `auth:billing/subscription`, `auth:billing/cancel-flow`

## 1. Strategy picks engine — `/plays/*` ("Top Plays")

Five daily pick lists, each a filterable/sortable table (tanstack table, server-side filters via
the `scan/*` endpoints). Header copy example: *"Top Earnings Plays for Aug 20, 2026 — Earnings
trades selected by our proprietary model based on the Volatility Vibes backtest criteria."*
A "pick" is a **row of signal values per ticker** — no legs/entry-exit rules are served by the API;
the playbooks (below) define the structure to trade.

### `/plays/earnings` (short earnings vol) — `scan/get-earnings-strategy`
Columns (default sort: `short_straddle_return` desc):
`ticker`, `announcement_date`, `announcement_time` (enum BMO/AMC), `is_confirmed`,
`implied_move` (%), `option_volume`, `short_straddle_return`, `short_straddle_win_rate`,
`avg_realized_move`, `avg_implied_move`, `implied_vs_avg_realized`, `term_structure_slope`,
`historical_events_count`.

Playbook filters (from embedded docs): announce between today's close and tomorrow's open;
≥10k avg daily option volume (or weeklys exist); stock price cap (~$100–150); front-month IV in
backwardation vs next expiry; IV/RV comfortably > 1; history of negative long-straddle returns for
that name. Recommended structure: **short ATM iron butterfly** (sell ATM straddle, buy 1–5Δ wings,
expiry right after earnings); max loss sized to 1–2% of strategy capital.

### `/plays/pre-earnings-long-vol` — `scan/get-pre-earnings-long-vol-strategy`
Model-driven long-vol picks. Columns (default sort `predicted_return_label` desc; default visible
set: ticker, announcement_date, announcement_time, is_confirmed, predicted_return_label,
option_volume): plus `implied_move`, `hist_avg_implied_move`, `hist_avg_realized_move`,
`historical_events_count`. `predicted_return_label` is a classification output of their model.

### `/plays/momentum-skew` — `scan/get-momentum-skew-strategy`
Columns (default sort `skew_zscore` asc): `ticker`, `next_earnings_date`, `direction`,
`option_volume`, `cs_momentum` (cross-sectional decile), `ts_momentum` (time-series),
`relative_momentum` (vs SPY, >1 = outperform), `skew_value`, `skew_zscore`, `skew_mean`.

Documented criteria: put/call skew z-score ≤ −1.5 (steep vs own history); CS momentum decile ≥ 8
for call-skew trades (≤ 3 for put-skew); ≥ 5k contracts/day liquidity (20k preferred). Trade:
asymmetric **vertical debit spread** (buy ~ATM/slightly OTM, sell further-OTM high-skew leg),
30–60 DTE, risk 0.5–2% of capital per trade, close before any earnings inside the trade window;
expect ~32% win rate with 5:1–10:1 reward:risk. Alternative: 1-3-2 ratio fly (buy 1 ATM 50Δ, sell 3
at 20–30Δ, buy 2 further OTM; wing gap = half body gap).

### `/plays/forward-factors` — `scan/get-forward-factor-strategy`
Columns (default sort `forward_factor` desc): `ticker`, `next_earnings_date`,
`forward_factor` (**Ex-Earn**), `option_volume` (20d avg).
Signal: `FF = (Front IV − Forward IV(1→2)) / Forward IV(1→2)` computed on ex-earnings IVs;
tradeable threshold FF ≥ 0.20; back leg typically < 100 DTE. Trade: long ATM calendar
(±35Δ double-calendar variant), max-debit discipline, size 1–4% of account, close as a spread
before front expiry; early exit if FF mean-reverts.

### `/plays/vrp-stock` — `scan/get-vrp-stock-strategy`
Columns (default sort `iron_condor_mean_return` desc): `ticker`, `iv_pctl_1y`, `iv_rv`,
`option_volume`, `next_earnings_date`, and backtest stats per structure:
`iron_condor_mean_return` / `_win_rate`, `short_straddle_mean_return` / `_win_rate`,
`short_strangle_mean_return` / `_win_rate`, `iron_butterfly_mean_return` / `_win_rate`.
(A sixth endpoint `scan/get-vrp-strategy` exists — presumably the ETF variant.)
Playbook: wide iron condor (sell 25–30Δ strangle, buy 1–5Δ wings, 30–60 DTE), IV percentile
< ~80%, contango/flat term structure, roll/exit ~7 DTE before expiry, exit if IV percentile
> 90–95% or term structure inverts; 1–5% of capital at risk per trade.

## 2. Backtester / Research — `/research` (workspaces), `/models`

Workspace wizard steps (from workspace JS chunk):

1. **Instruments** — universe builder with filters over the ~300 signals; explicit earnings
   handling toggle: "Excluding earnings" / "Including earnings" (`avoid_earnings` flag);
   "Preview Instruments".
2. **Strategy** — pick up to **3 pre-built strategies** from a library, each tagged
   Direction (Neutral/Bullish/Bearish) × Risk Profile (Defined/Undefined) × Volatility Exposure
   (Long/Short). Library (from asset names): long/short call, long/short put, long/short straddle,
   long/short strangle, long/short butterfly, long/short condor, long/short call vertical,
   long/short call directional fly, long/short put directional fly, call ratio, call ratio
   opposite, put ratio, put ratio opposite. DTE matching modes: `dte_below_equal` ("closest
   available DTE ≤ target") and `dte_above`. Positions initiated at or within 5% of advertised
   DTE; **held to expiration** (custom exits announced as future feature).
   Strategy stats table: Count, Mean, Standard Deviation, Minimum, Median (50th), …;
   charts "Returns Over Time" (time-series) and "Return Distribution" (histogram).
3. **Research (signal analysis)** — per signal: scatter plot vs return with line of best fit +
   p-value, decile returns chart, histogram. Transformations: `log1p`, `sigmoid`, `minmax`,
   `standardize`, `square_root`, `cube_root`, `yeo_johnson`, `rank`. Rule filters with operators
   (incl. `between`, Value/Value 2).
4. **Model** — three types: `rule` (manual thresholds), `regression_linear`, `regression_logistic`.
   Settings: Fit Intercept toggle; Linear Threshold (min predicted return %); Logistic Threshold
   (min probability %; training class threshold default return > 0). Metrics tabs Training/Test:
   linear → `train_rmse`, `test_rmse`, `train_r2`, `test_r2`; logistic → `train/test_accuracy`,
   `_auc`, `_precision`, `_recall`, `_f1`; Model Coefficients table (Feature/Coefficient).
   Return-metric table quadrants: `train_universe_returns`, `test_universe_returns`,
   `train_entry_returns`, `test_entry_returns` with Mean Return, Kelly Fraction, Min/Max Return,
   Median (p50). **Train/Test Split required** (`research/set-train-split`).
5. **Backtest** — inputs: Initial capital, Position size (0–100%), Limit concurrent positions /
   Max positions. Benchmark = buy-and-hold, chart portfolio vs benchmark. Metrics table:
   Final Portfolio Value, Total Return, CAGR, Sharpe Ratio, Max Drawdown, Total Trades; results
   split into Training Data / Testing Data. Models can be saved (`/models` page) and run live
   (`research/run-model-live`, `research/model-predictions` — table of ticker + predicted return).

Backtest methodology (docs/platform): cross-sectional across full ticker universe (no-code),
data from **2007**, near-EOD snapshots (15 min before close), **includes delisted tickers** (no
survivorship bias). Realistic fill model: deviates from mid based on stock/option volume, spread
width, historical trade-vs-mid deviations, OTM liquidity penalty. Capacity metric (max contracts
without price impact). Commissions = worst-case IBKR tiers: premium <$0.05 → $0.25/contract;
$0.05–$0.10 → $0.50; ≥$0.10 → $0.65. Margin via REG-T (IBKR standard) → return-on-margin metrics.

## 3. Position Designer — `/dashboard/designer/[symbol]`

Multi-leg builder over a live chain (`live/option-chain`, `live/ticker-info`,
`positions/interest-rates`). Leg fields: `action` (Buy/Sell), instrument (option Call/Put or
underlying Stock), `strike`, `expiration`, `quantity`, per-leg `price`/`iv`/`delta`/`gamma`/
`theta`/`vega`; positions serializable in URL (`positions` param, `frontDte`/`backDte`).

UI controls:
- Leg table with Fill IV (%) per leg → server computes `positions/effective-fill-iv`
  ("Effective Fill IV" vs "fair").
- **IV Adjustment sliders** — modes "Per Expiry" / "Per Position" (per-leg), "Link all sliders" /
  "Unlink sliders", "Reset to market IV" — i.e. IV scenario shocks per expiry or per leg.
- Date slider ("Date:", default Today) for P&L-before-expiry.
- Charts: P&L at Expiration + greek curves (delta/theta/gamma/vega toggles).
- Summary metrics: Max Profit, Max Loss, Breakeven, Win Rate, position Delta/Theta/Gamma/Vega.
- **RV Scenario Analysis**: POST `positions/forecast-simulation` with
  `{forecast_rv (decimal), legs, underlying_price, open_price, date}` → returns distribution
  stats: `mean_return`, `mean_pnl`, `return_std`, `win_rate`, `kelly_fraction`, `min_return`,
  `percentile_25`, `median_return`, `percentile_75`, `max_return` (+ `_pnl` variants). Displayed
  as "Mean Return ($/%)", "Return Std. Dev.", "Win %", "Kelly Fraction", Min/25th/Median/75th/Max.
- "Optimal Delta Hedge" (`positions/optimal-delta-hedge`).
- **Structures library**: save/load named multi-leg structures (`portfolio/structures`), filter
  by Direction (Bullish/Bearish/Neutral), Risk Profile (Defined/Undefined), Volatility Exposure
  (Long/Short); "Save to portfolio", "Add structure", "New empty structure", "Clear all".

## 4. Earnings dashboard — `/dashboard/earnings/[symbol]`

Header cards (`dashboard/earnings/card-data`): Next Earnings Date (+ Estimated flag, BMO/AMC
time), Implied earnings move, Average Move (historical avg realized), Average implied move,
Implied Vs. Actual (avg implied − avg actual), Average IV Crush, Average Straddle Return,
Average Put Return, Average Call Return.

Charts:
- **Implied vs Realized Moves** ("Expected vs Actual Earnings Moves") — per-event bars/line:
  `implied_move`, `realized_move`, `positive_move`, `negative_move` (`…/moves`).
- **Cumulative Returns** — long straddle/call/put entered day before, exited day after;
  individual + cumulative series (`straddle_returns`, `call_returns`, `put_returns`,
  `*_cumulative_returns`) (`…/returns`).
- **Pre vs Post IV / IV Crush** — `pre_iv`, `post_iv`, `iv_crush` per earnings date (`…/iv-crush`).
- **Implied Move Timeseries** — past quarter evolution (`…/implied-move-timeseries`).
- **Implied Moves Distribution** — probability density with Median, `probLessThan`,
  `probGreaterThan` markers (`…/implied-moves-distribution`).
- **Realized Moves Distribution** (`…/realized-moves-distribution`).
- **Seasonal Move Boxplot (Quarterly Moves)** — per calendar quarter: median, IQR box, whiskers,
  realized vs implied (`…/quarterly-moves`).
- Other endpoints: `…/earnings-calendar`, `…/next-earnings-info`, `card-data-batch`
  (batch `{tickers, lookback}` for screener cards).
- Standalone `/earnings-calendar` page: table of `announcement_date`, `ticker` (+ timing).

## 5. Other dashboards

### Volatility — `/dashboard/volatility/[symbol]`
Charts (each backed by an endpoint under `dashboard/volatility/`): VRP timeseries (`vrp`),
IV/RV-percentile vs VRP scatter (`rv-vrp-scatter`), **Strategy Backtest** chart (`backtest`) —
systematic per-asset backtests at selected DTE with delta-defined strikes: long/short 50Δ call,
long/short 50Δ put, long/short ATM straddle, short iron butterfly (50Δ body, ~5Δ/95Δ wings),
call/put directional flies (60/30/20Δ). Volatility Cone (`volcone`: min/max/median/25/75 pct
bands per horizon + current), Multi-Expiry Skew (`skew`, 10D–1Y, delta-normalized), Term
Structure IV & RV (`term-structure`), Skew Timeseries (`skew-timeseries`: call/put skew =
ATM−OTM, RR skew = OTM call − OTM put), TS Slope Timeseries (`ts-slope-timeseries`: 10–30D,
30–60D, 30–90D pairs), Option Volume & Put/Call OI ratio (`volume-oi`), Forward Factors bars
(`forward-factors`), FF/FwdVol timeseries (`forward-factors-timeseries`), Spot–IVol correlation
scatter (`ivol-spot-corr`).

### Relative Value — `/dashboard/relative-value/[symA]/[symB]`
Relative term structures (per-asset + difference), Relative skews (+ difference), IV/RV/VRP
scatter plots with quantile regression lines (median, 25th, 75th), custom comparison of any two
variables (ratio or difference, mean + std bands + histogram), return correlation scatter,
rolling return correlation (configurable return interval + window).

### Directional — `/dashboard/directional/[symbol]`
Stock chart + 200DMA; Cross-Sectional Momentum decile (stocks vs stocks, ETFs vs ETFs);
Time-Series Momentum; Relative Momentum vs SPY (>1 outperform); Sector Momentum; Turnover
(volume/shares outstanding); Proximity to 52-Week High; Risk-Neutral vs Historical return
distributions (from option skew vs simulated spot returns); **PEAD score** (−2..+2);
**Skew Score** (0–100, steepness + recent change; 100 = bullish skew).

## 6. Signal catalog — `/docs/signals` (~300 signals)

Grouped: Asset Specs (type, market cap, dividend fields, takeover flag, sector); Broad Market
(RFR 5w/LT, borrow rates 30d/2y from put-call parity); Contract Specs (DTE M1–M4, straddle
prices, smooth/forecast straddle prices, low/high strikes); Earnings (VDR, implied/forecast ER
effect, stdev of past 12 earnings moves, implied ER move, avg implied vs avg move, avg IV crush,
avg earnings straddle/put/call returns); Forward Volatility (Fwd IV / Flat Fwd IV / Flat Fwd
Ratio for pairs 20-30, 30-60, 60-90, 90-180, 30-90; all duplicated as Ex-Earn); IV (ATM/fit/
forecast IV M1–M4, constant-maturity IV 10d–1y, Ex-Earn variants, IV 200MA); IV Metrics (vol of
IV, IV percentile 1m/1y, IV z-score, IV stdev); IV/RV metrics (ratio, vs avg 1m/1y, stdev);
Liquidity (call/put volume & OI, IV spread 30d/LT, underlying volume, avg opt volume 20d);
Non-ATM IV at Δ5/25/75/95 × 7 tenors (+ Ex-Earn); RV (1d–1000d, close-to-close variants,
Ex-Earn variants); RV Metrics (vol of vol, forecast R²); Relative Value (best ETF, corr to
SPY/ETF 1m/1y, beta, IV percentile SPY/ETF, IV/SPY ratio stats, IV ETF ratio stats, IV/RV vs ETF,
skew slope vs ETF); Skew (slope 30d/LT, forecast, derivative, percentile, avg/stdev); Term
Structure (term slope); Underlying (price changes 1w/1m/6m/1y).

Definitions of note: "Ex-Earn" = IV/RV with earnings effect removed (term-structure model, root
time scaling); "slope" = 30-day skew slope; "contango" = term-structure slope (negative =
backwardation); Forward Factor = (front IV − fwd IV)/fwd IV.

## 7. Calculators (public)

- `/calculators/black-scholes` — BSM price + Greeks (Δ, Γ, Θ, Vega, Rho); real chain data or
  custom inputs; calls/puts, dividends, rates.
- `/calculators/forward-volatility` — suite: ForwardVolatilityFactorCalculator,
  CalendarSpreadCalculator (sizes calendar/double-calendar spreads, Max Debit), and a
  ForwardFactorScanner over expiration pairs.
- `/calculators/ex-earnings` — annualized vol with earnings impact removed per expiration
  (clean ex-earnings IV vs observed IV).
- `/calculators/correlation-matrix` — N-ticker return correlation matrix; daily/weekly/monthly,
  custom lookbacks, export/share.

## 8. Other pages

- `/screener` — custom scan builder over all signals ("Your screens" + "Common screens"
  presets: "Liquid ETF candidates for VRP trading", "Expensive Volatility", "Cheap Volatility",
  "Positive Momentum", "Negative Momentum"); create/copy/edit/delete screens
  (`scan/*-custom-scan`); filter operators incl. `between`.
- `/portfolio` — balance, transactions, trades, strategies, structures, metrics, graphs
  (position tracking; designer structures save here).
- `/models` — saved models list + live predictions table (ticker, predicted return; model types
  rule/linear/logistic).
- `/docs/platform` — full platform guide (chart-by-chart methodology; source of most details above).
- `/courses/options-fundamentals` — 4-section course + final exam (Options, Volatility, Market
  Efficiency & Edge, Real Trading).
- Marketing home embeds full strategy playbooks (VRP, Earnings, Momentum-Skew, Forward-Factor
  calendars) — the exact entry/exit/sizing rules summarized in §1.

## Replication takeaways for our bot

- Earnings pick = per-ticker signal row: implied move, hist avg implied/realized moves,
  implied_vs_avg_realized, term_structure_slope, short-straddle backtest return/win-rate,
  historical_events_count, liquidity. Structure (iron fly) and sizing are fixed rules, not data.
- Forward Factor on **ex-earnings** IV with threshold ≥ 0.20 is their marquee calendar signal.
- Backtester realism = fill-below-mid model + capacity caps + IBKR-tier commissions + REG-T
  margin; results reported as return on margin with train/test split discipline.
- Designer scenario engine: simulate terminal P&L distribution under a forecast RV to get
  mean/std/win-rate/Kelly/percentiles — a clean spec for our own position evaluator.
