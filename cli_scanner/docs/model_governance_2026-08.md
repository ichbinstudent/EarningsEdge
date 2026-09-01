# Model Governance Review - August 2026

## Executive Summary

The Aug-1 retrain of earnings screening model shows AUC degradation (0.802 → 0.762) that is **NOT statistically significant** given the tiny test set (11 positives). The scanner doesn't use this ML model for filtering anyway — it relies on hard-coded heuristic gates in `validator.py`. Option models show negative R² and below-baseline accuracy, confirming no predictive signal.

## Screening Model Analysis

### Metrics Comparison
- **Previous**: AUC 0.802, 449 rows, test precision 0.50
- **Current**: AUC 0.762, 397 rows, test precision 0.45
- **Test set**: Only 11 positive examples (same in both)

### Statistical Assessment

With n=11 positives, the 95% CI for AUC spans approximately ±0.074, making the confidence intervals:
- Previous: ~(0.728, 0.876)  
- Current: ~(0.688, 0.836)

These intervals substantially overlap. The 0.04 AUC drop is well within noise for this sample size. A proper significance test would require ~100 positive test examples minimum.

### Operational Impact: NONE

**Critical finding**: The `earnings_model` is trained but never used in production. The scanner filters via:
1. `StockValidator.validate()` with hard thresholds (price, IV/RV ratio, term structure, win rate)
2. Calendar filter ML model (`calendar_call_filter_ridge_allfeatures.joblib`) for calendar trades
3. No reference to `beat_expected_move` predictions in any scanner path

The screening model is dead code — a research artifact that doesn't affect live operations.

## Options Models Analysis

### Magnitude Model (gradient_boosting)
- **R² Score**: -0.145 (worse than predicting the mean)
- **MAE**: 7.31% on moves averaging 7.17%
- **Assessment**: No signal, pure noise

### Direction Model (gradient_boosting)  
- **Accuracy**: 39.6% (baseline ~45% for 3-class)
- **F1 Macro**: 0.338
- **Assessment**: Below majority-class baseline

These models correctly show no predictive power given current features. The negative results are trustworthy — don't use them for trading.

## Calendar Model (operational)

The `calendar_call_filter_ridge_allfeatures.joblib` remains the only ML model actually used in production via `bot_scanner.py`. This was not retrained in the Aug-1 batch and maintains its June-14 performance metrics.

## Recommendations

### 1. Keep Current State
- **Screening model**: Leave as-is. It's not used, retraining won't help with n=11 positives
- **Options models**: Correctly show no signal. Keep for monitoring but don't use
- **Calendar model**: Stable and operational, no action needed

### 2. Data Collection Priority
Focus collection on optionable names only (`has_options=1`). The 52 new outcomes since June were mostly OTC/foreign stocks without options chains, adding no value for model training.

### 3. Forward-Looking Features
The only path to predictive options models is adding forward-volatility features (σ_fwd from calendar spreads). Current spot features have no edge, as the models correctly demonstrate.

### 4. Threshold Tuning: NOT NEEDED
Since the screening model isn't used, there's no threshold to tune. The validator's hard gates (IV/RV > 1.10, term slope > -5%, etc.) are the actual filters.

## Testing Requirements

No code changes required based on this review, therefore no new tests needed. The models are performing as expected given the feature limitations.

## Sign-Off

Review conducted: 2026-08-04  
Next review: After 100+ new `has_options=1` outcomes collected