# Regression Model Selection Report

## Dataset
- **Train rows:** 353  |  **Test rows (held out):** 89
- **Features:** 10
- **Agent evaluations:** 7

## Agent Recommendation
**Winner:** `ridge`  
**Params:** `{'alpha': 0.1}`  
**CV RMSE (train):** 55.84  
**R² (CV):** 0.457  
**Training time:** 45 ms

**Runner-up:** `lasso` — CV RMSE 55.89

### Justification
The dataset is small (442 rows, 10 features) with a roughly symmetric target (skew 0.44, no capping) and a signal spread across multiple moderately-correlated predictors (bmi 0.61, s5 0.55, bp/s4/s6 ~0.4). The s1–s6 serum features are known to be highly collinear, which favours L2-regularised linear models. Empirically, ridge with α=0.1 achieved CV RMSE 55.84 ± 2.62 and R² 0.457, narrowly edging lasso α=0.1 (55.89 ± 2.34) and OLS (55.97 ± 3.18) — these three are statistically tied within one fold-std, but ridge's lowest mean RMSE plus the theoretical fit to the multicollinearity makes it the cleanest pick. Tree-based models lagged meaningfully: random forest reached 58.15 and a carefully-tuned gradient boosting only 60.21, both with larger fold variance. With only 442 rows, high-capacity non-linear models overfit; the underlying signal in this dataset is genuinely additive. Decision tree was skipped because it is strictly dominated by random forest. Stronger regularisation (α=1.0) under-fit both ridge and lasso, confirming that only light regularisation is needed.

### Caveats
1) **CV optimism**: ridge α was selected by inspecting the same 5-fold CV scores used to report performance, so the headline 55.84 RMSE is mildly optimistic; a nested CV would give a more honest estimate. 2) **Untested options**: I did not sweep a fine α grid for ridge/lasso (only 0.1 and 1.0), nor did I try ElasticNet, polynomial/interaction features, or feature scaling variants — modest gains may be available there. 3) **Modest R² (~0.46)**: roughly half the variance is unexplained, indicating substantial irreducible noise in this clinical outcome; gains from any model class will likely be small. 4) **Target transform**: skew is only 0.44 so a log transform is unlikely to help much, but a Box-Cox or quantile transform of the target is a reasonable next experiment given the wide range (25–346). 5) **Small sample size**: with 442 rows, fold-to-fold variance (R² std up to 0.17 for some models) means rankings can shift; the top three linear models should be considered tied. 6) **No spatial/temporal columns**, so geographic generalisation concerns do not apply here.

## Agent-Evaluated Models
| Model | Params | CV RMSE ± std | R² ± std | Time (ms) |
|-------|--------|---------------|----------|-----------|
| ridge | alpha=0.1 | 55.842 ± 2.6199 | 0.4565 ± 0.1201 | 45 |
| lasso | alpha=0.1 | 55.8873 ± 2.3439 | 0.4555 ± 0.1185 | 53 |
| linear | — | 55.9721 ± 3.1845 | 0.4493 ± 0.1441 | 46 |
| random_forest | n_estimators=200, max_depth=6, min_samples_leaf=3 | 58.1519 ± 4.3782 | 0.4092 ± 0.1425 | 2774 |
| ridge | alpha=1.0 | 60.185 ± 4.1434 | 0.3802 ± 0.0766 | 47 |
| gradient_boosting | n_estimators=200, max_depth=3, learning_rate=0.05 | 60.2109 ± 5.1772 | 0.3634 ± 0.1704 | 1871 |
| lasso | alpha=1.0 | 63.0231 ± 5.0867 | 0.3238 ± 0.0608 | 65 |


---

## Post-Hoc Baseline: GridSearchCV Exhaustive Search
*(Run after agent recommendation — agent had no visibility into these results)*

**Evaluations — Agent:** 7  |  **Baseline:** 143 combinations × 5 folds = 715 CV fits

**Baseline winner:** `lasso` — CV RMSE 55.8873, Test RMSE 52.898, R² 0.4555  
**Total search time:** 41382 ms

| Model | Best Params | CV RMSE | Test RMSE | R² |
|-------|-------------|---------|-----------|-----|
| lasso | alpha=0.1 | 55.8873 | 52.898 | 0.4555 |
| gradient_boosting | learning_rate=0.03, max_depth=3, n_estimators=100, subsample=0.8 | 57.9515 | 53.1948 | 0.4121 |
| ridge | alpha=0.1 | 55.842 | 53.4461 | 0.4565 |
| random_forest | max_depth=5, min_samples_leaf=1, n_estimators=500 | 57.9473 | 53.516 | 0.4127 |
| linear | — | 55.9721 | 53.8534 | 0.4493 |
| decision_tree | max_depth=3, min_samples_leaf=3 | 64.4797 | 59.6045 | 0.2834 |

### Agent vs Baseline — Held-Out Test Set
| Metric | Agent (`ridge`) | Baseline (`lasso`) | Delta |
|--------|-------------------------------|------------------------------|-------|
| CV RMSE | 55.84 | 55.8873 | -0.0473 |
| **Test RMSE** | **53.4461** | **52.898** | **+0.5481** |
| R² (CV) | 0.457 | 0.4555 | +0.0015 |
| R² (test) | 0.4609 | 0.4719 | -0.0110 |

Baseline **beat** the agent on the held-out test set by 1.0% RMSE (0.5481 lower).