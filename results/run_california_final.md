# Regression Model Selection Report

## Dataset
- **Train rows:** 1600  |  **Test rows (held out):** 400
- **Features:** 8
- **Agent evaluations:** 5

## Agent Recommendation
**Winner:** `gradient_boosting`  
**Params:** `{'n_estimators': 500, 'max_depth': 4, 'learning_rate': 0.05, 'random_state': 0}`  
**CV RMSE (train):** 0.5789  
**R² (CV):** 0.7599  
**Training time:** 13960 ms

**Runner-up:** `random_forest` — CV RMSE 0.6183

### Justification
Gradient boosting is the recommended model with CV RMSE 0.5789 ± 0.0574 and R² 0.760 ± 0.039, beating random forest (0.6183) by ~0.04 RMSE — roughly two-thirds of one fold-std, a real but modest gap — and crushing the ridge baseline (0.847, R² 0.48). The data profile justifies this: only MedInc is linearly correlated with the target (r=0.68); the remaining signal lives in non-linear interactions, especially the spatial structure encoded by Latitude/Longitude (which have near-zero linear correlation but enormous explanatory power once you can split on them). Tree ensembles handle this natively, which is why ridge underfits so badly. Within GB, three configurations (depth 3 / n=200 / lr=0.1 → 0.5833; depth 4 / n=500 / lr=0.05 → 0.5789; depth 5 / n=300 / lr=0.05 / subsample=0.8 → 0.5806) all sit within 0.005 RMSE of each other — statistically tied given fold std 0.057 — so the chosen config is at the plateau rather than a uniquely optimal point. I skipped lasso (only 8 features, no need for L1 selection beyond what ridge offers), linear (essentially ridge with α→0), and decision_tree (strictly dominated by random_forest on a 2000-row noisy dataset).

### Caveats
1) **Capped target**: the maximum value is exactly 5.0 (the well-known California $500k cap), and the target is right-skewed (skew 0.93). Tree ensembles cannot extrapolate past this ceiling, so they will systematically underestimate truly high-value districts; this also makes RMSE near the cap look artificially good. A log or Box-Cox transform of the target — combined with explicit handling of capped rows (e.g. censored regression or dropping them) — is the most likely next-step improvement we did not test. 2) **Spatial generalisation risk**: Latitude/Longitude are key features and 5-fold CV folds will straddle different regions of California. Fold variance is visible (ridge r2_std=0.15; even GB r2_std=0.04 is non-trivial). The model may not generalise well to geographic areas under-represented in training; spatial feature engineering (distance to coast, distance to nearest metro, cluster IDs) is the most likely untapped performance gain. 3) **CV optimism**: hyperparameters were chosen by inspecting CV scores on the same folds used for reporting, so the headline RMSE of 0.5789 is mildly optimistic — the true generalisation RMSE is likely a touch higher. 4) **Limited search**: I did not test XGBoost/LightGBM, target transforms, feature engineering, or a wider hyperparameter grid (e.g. min_samples_leaf, regularisation). 5) **Small dataset**: 2000 rows amplifies fold-to-fold variance — the ~0.04 RMSE gap between GB and RF, while consistent, is not enormous, and on a different random split RF could plausibly be competitive.

## Agent-Evaluated Models
| Model | Params | CV RMSE ± std | R² ± std | Time (ms) |
|-------|--------|---------------|----------|-----------|
| gradient_boosting | n_estimators=500, max_depth=4, learning_rate=0.05 | 0.5789 ± 0.0574 | 0.7599 ± 0.0386 | 13960 |
| gradient_boosting | n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8 | 0.5806 ± 0.0587 | 0.7583 ± 0.0401 | 8575 |
| gradient_boosting | n_estimators=200, max_depth=3, learning_rate=0.1 | 0.5833 ± 0.0605 | 0.756 ± 0.0419 | 4492 |
| random_forest | n_estimators=200, max_features=sqrt | 0.6183 ± 0.0427 | 0.7269 ± 0.0279 | 5169 |
| ridge | alpha=1.0 | 0.8468 ± 0.1201 | 0.478 ± 0.1519 | 48 |


---

## Post-Hoc Baseline: GridSearchCV Exhaustive Search
*(Run after agent recommendation — agent had no visibility into these results)*

**Evaluations — Agent:** 5  |  **Baseline:** 143 combinations × 5 folds = 715 CV fits

**Baseline winner:** `gradient_boosting` — CV RMSE 0.5749, Test RMSE 0.5359, R² 0.763  
**Total search time:** 110461 ms

| Model | Best Params | CV RMSE | Test RMSE | R² |
|-------|-------------|---------|-----------|-----|
| gradient_boosting | learning_rate=0.03, max_depth=4, n_estimators=500, subsample=0.8 | 0.5749 | 0.5359 | 0.763 |
| random_forest | max_depth=20, min_samples_leaf=2, n_estimators=500 | 0.6322 | 0.6177 | 0.7142 |
| linear | — | 0.8486 | 0.6627 | 0.4757 |
| ridge | alpha=100.0 | 0.8024 | 0.675 | 0.5321 |
| lasso | alpha=0.1 | 0.8044 | 0.7516 | 0.5366 |
| decision_tree | max_depth=8, min_samples_leaf=10 | 0.7514 | 0.7584 | 0.5948 |

### Agent vs Baseline — Held-Out Test Set
| Metric | Agent (`gradient_boosting`) | Baseline (`gradient_boosting`) | Delta |
|--------|-------------------------------|------------------------------|-------|
| CV RMSE | 0.5789 | 0.5749 | +0.0040 |
| **Test RMSE** | **0.5415** | **0.5359** | **+0.0056** |
| R² (CV) | 0.7599 | 0.763 | -0.0031 |
| R² (test) | 0.7634 | 0.7683 | -0.0049 |

Baseline **beat** the agent on the held-out test set by 1.0% RMSE (0.0056 lower).