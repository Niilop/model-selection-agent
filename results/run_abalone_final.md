# Regression Model Selection Report

## Dataset
- **Train rows:** 1600  |  **Test rows (held out):** 400
- **Features:** 7
- **Agent evaluations:** 6

## Agent Recommendation
**Winner:** `linear`  
**Params:** `{}`  
**CV RMSE (train):** 2.2689  
**R² (CV):** 0.5031  
**Training time:** 40 ms

**Runner-up:** `ridge` — CV RMSE 2.295

### Justification
On this 2000-row dataset with 7 morphometric features all moderately correlated with Rings (0.52–0.61) and no single dominant predictor, ordinary least squares achieved the lowest CV RMSE (2.269 ± 0.105, R² 0.503) and beat every non-linear alternative tested. Ridge (α=1) came within 1.2% at 2.295 — a statistical tie given the ~0.10 fold std. Random forest (2.301) and a well-regularised gradient boosting (2.301) both matched ridge but could not exceed linear, indicating the size→age relationship is dominantly linear at this sample size after the smooth morphometric features encode growth. Lasso (2.323) was worse, confirming no features should be zeroed out. The R² plateau near 0.50 across model families points to an irreducible noise floor (and likely a censored target at 29 rings) rather than under-modelling. I skipped decision_tree because it is strictly dominated by random_forest, and did not pursue further GB/RF tuning because the gap to linear is widening, not closing. Linear wins on accuracy, fit time (40 ms vs 5–11 s for ensembles), and interpretability — a clear pick.

### Caveats
Selection caveats: (1) Hyperparameters were chosen by inspecting the same 5-fold CV scores used for reporting, so the headline RMSE of 2.269 is mildly optimistic — a fully held-out evaluation would likely land slightly higher. (2) The target is right-skewed (skew 1.18) and its max of 29 is the well-known UCI cap, so the upper tail is plausibly censored; a log or Box-Cox transform of Rings (or modelling log(Rings+1.5) per the original paper) is the most likely accuracy lever I did not exercise. (3) I did not test feature engineering — particularly weight ratios (e.g. Shucked_weight/Whole_weight, which captures meat-to-shell composition) or polynomial size terms — that could plausibly push R² past 0.55. (4) The categorical Sex variable from the original Abalone dataset is absent here; if available, encoding M/F/I would likely add meaningful signal, especially for separating immature animals. (5) cv_rmse_std (0.08–0.14) is 4–6% of the mean RMSE, so fold heterogeneity is moderate but not alarming; the linear-vs-ridge-vs-RF differences are within one std and should be treated as ties. (6) At n=2000 the tree ensembles are likely under-served — with 10× more data, gradient boosting would probably overtake linear, so this recommendation is dataset-size-specific.

## Agent-Evaluated Models
| Model | Params | CV RMSE ± std | R² ± std | Time (ms) |
|-------|--------|---------------|----------|-----------|
| linear | — | 2.2689 ± 0.1047 | 0.5031 ± 0.0395 | 40 |
| ridge | alpha=1.0 | 2.295 ± 0.1351 | 0.4929 ± 0.0303 | 49 |
| random_forest | n_estimators=300, max_depth=None, min_samples_leaf=2 | 2.3005 ± 0.0808 | 0.4893 ± 0.0324 | 11217 |
| gradient_boosting | n_estimators=500, max_depth=2, learning_rate=0.03, subsample=0.8 | 2.3006 ± 0.1412 | 0.4889 ± 0.0505 | 5893 |
| lasso | alpha=0.01 | 2.3228 ± 0.141 | 0.4808 ± 0.0286 | 43 |
| gradient_boosting | n_estimators=300, max_depth=3, learning_rate=0.05 | 2.3413 ± 0.1429 | 0.4711 ± 0.0487 | 4883 |


---

## Post-Hoc Baseline: GridSearchCV Exhaustive Search
*(Run after agent recommendation — agent had no visibility into these results)*

**Evaluations — Agent:** 6  |  **Baseline:** 143 combinations × 5 folds = 715 CV fits

**Baseline winner:** `lasso` — CV RMSE 2.2688, Test RMSE 2.3442, R² 0.5032  
**Total search time:** 81160 ms

| Model | Best Params | CV RMSE | Test RMSE | R² |
|-------|-------------|---------|-----------|-----|
| lasso | alpha=0.0001 | 2.2688 | 2.3442 | 0.5032 |
| linear | — | 2.2689 | 2.3443 | 0.5031 |
| ridge | alpha=0.1 | 2.2678 | 2.3462 | 0.5039 |
| random_forest | max_depth=20, min_samples_leaf=5, n_estimators=500 | 2.2794 | 2.3485 | 0.4988 |
| gradient_boosting | learning_rate=0.05, max_depth=4, n_estimators=100, subsample=0.7 | 2.2968 | 2.384 | 0.4914 |
| decision_tree | max_depth=5, min_samples_leaf=10 | 2.4452 | 2.5216 | 0.4246 |

### Agent vs Baseline — Held-Out Test Set
| Metric | Agent (`linear`) | Baseline (`lasso`) | Delta |
|--------|-------------------------------|------------------------------|-------|
| CV RMSE | 2.2689 | 2.2688 | +0.0001 |
| **Test RMSE** | **2.3443** | **2.3442** | **+0.0001** |
| R² (CV) | 0.5031 | 0.5032 | -0.0001 |
| R² (test) | 0.5459 | 0.546 | -0.0001 |

Agent and baseline are **statistically tied** on the held-out test set.