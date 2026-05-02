# Regression Model Selection Report

## Dataset
- **Train rows:** 614  |  **Test rows (held out):** 154
- **Features:** 8
- **Agent evaluations:** 6

## Agent Recommendation
**Winner:** `gradient_boosting`  
**Params:** `{'n_estimators': 600, 'max_depth': 4, 'learning_rate': 0.03, 'random_state': 0}`  
**CV RMSE (train):** 0.3857  
**R² (CV):** 0.9985  
**Training time:** 3673 ms

**Runner-up:** `random_forest` — CV RMSE 0.6879

### Justification
Gradient boosting (n_estimators=600, max_depth=4, learning_rate=0.03) achieves CV RMSE 0.386 and R² 0.9985 on Y1, dramatically outperforming linear baselines (RMSE 2.93) and clearly beating random forest (RMSE 0.69). The 6× gap between linear and tree models confirms the heating-load target depends on strong non-linear interactions among the 8 building-geometry features (e.g., overall height × glazing area × orientation), which boosting captures most efficiently. Even though X5 and X4 alone correlate ~0.89 and ~0.86 with Y1, ridge offered no improvement over OLS, so multicollinearity is not the bottleneck — non-linearity is. Compared to the depth-3 GBM variants (RMSE 0.456 → 0.423), the deeper-and-slower configuration improved RMSE by ~16% and also reduced fold std from 0.060 to 0.032, indicating better stability rather than overfit. I skipped lasso (no evidence of irrelevant features — every X had non-trivial correlation) and a standalone decision tree (guaranteed worse than RF/GBM and prone to overfit at 768 rows). The target is well-behaved (skew 0.36, no ceiling), so no transform was warranted.

### Caveats
Selection caveats: (1) Hyperparameters were chosen by inspecting CV scores on the same 5 folds, so the reported RMSE 0.386 is mildly optimistic — a nested-CV or fresh hold-out would likely give a slightly higher number. (2) I did not test lasso, ElasticNet, polynomial-feature linear models, deeper GBM grids, learning-rate schedules, subsample/colsample stochastic boosting, or HistGradientBoosting; further small gains are plausible but unlikely to change the ranking given how flat the GBM tuning curve became. (3) The dataset is small (768 rows) and synthetic-looking — features X6 (orientation) and X8 (glazing distribution) are categorical-coded integers; one-hot encoding them might yield a marginal further improvement, particularly for the linear models. (4) No target transform was applied; the target's mild skew (0.36) and bounded but non-capped range (6.01–43.1) make this a low-priority lever, but a log1p transform could be tried to see if it tightens errors at the high end. (5) No spatial/temporal columns are present, so geographic generalisation is not a concern, but the dataset comes from simulated buildings under a specific parameter grid — performance on real-world buildings outside that grid is not assured. (6) Training time for the winning GBM is ~3.7 s; if inference latency or retraining cost matters, the depth-3 / 300-tree GBM (RMSE 0.456, ~1.7 s) is a reasonable lighter alternative.

## Agent-Evaluated Models
| Model | Params | CV RMSE ± std | R² ± std | Time (ms) |
|-------|--------|---------------|----------|-----------|
| gradient_boosting | n_estimators=600, max_depth=4, learning_rate=0.03 | 0.3857 ± 0.0324 | 0.9985 ± 0.0003 | 3673 |
| gradient_boosting | n_estimators=500, max_depth=3, learning_rate=0.05 | 0.4225 ± 0.0503 | 0.9982 ± 0.0004 | 2792 |
| gradient_boosting | n_estimators=300, max_depth=3, learning_rate=0.05 | 0.4559 ± 0.0599 | 0.9979 ± 0.0006 | 1660 |
| random_forest | n_estimators=300, max_features=sqrt | 0.6879 ± 0.0685 | 0.9952 ± 0.0011 | 3839 |
| linear | — | 2.9292 ± 0.0635 | 0.9145 ± 0.0042 | 49 |
| ridge | alpha=1.0 | 3.0013 ± 0.0927 | 0.9103 ± 0.0045 | 40 |


---

## Post-Hoc Baseline: GridSearchCV Exhaustive Search
*(Run after agent recommendation — agent had no visibility into these results)*

**Evaluations — Agent:** 6  |  **Baseline:** 143 combinations × 5 folds = 715 CV fits

**Baseline winner:** `gradient_boosting` — CV RMSE 0.3708, Test RMSE 0.4015, R² 0.9986  
**Total search time:** 39396 ms

| Model | Best Params | CV RMSE | Test RMSE | R² |
|-------|-------------|---------|-----------|-----|
| gradient_boosting | learning_rate=0.1, max_depth=4, n_estimators=500, subsample=1.0 | 0.3708 | 0.4015 | 0.9986 |
| random_forest | max_depth=20, min_samples_leaf=1, n_estimators=500 | 0.5066 | 0.4872 | 0.9974 |
| decision_tree | max_depth=8, min_samples_leaf=1 | 0.55 | 0.5891 | 0.997 |
| linear | — | 2.9292 | 3.0254 | 0.9145 |
| lasso | alpha=0.0001 | 2.9292 | 3.0259 | 0.9145 |
| ridge | alpha=0.01 | 2.9303 | 3.0309 | 0.9144 |

### Agent vs Baseline — Held-Out Test Set
| Metric | Agent (`gradient_boosting`) | Baseline (`gradient_boosting`) | Delta |
|--------|-------------------------------|------------------------------|-------|
| CV RMSE | 0.3857 | 0.3708 | +0.0149 |
| **Test RMSE** | **0.378** | **0.4015** | **-0.0235** |
| R² (CV) | 0.9985 | 0.9986 | -0.0001 |
| R² (test) | 0.9986 | 0.9985 | +0.0001 |

Agent **beat** the baseline on the held-out test set by 5.9% RMSE (0.0235 lower).