# Regression Model Selection Report

## Dataset
- **Train rows:** 1600  |  **Test rows (held out):** 400
- **Features:** 8
- **Agent evaluations:** 6

## Agent Recommendation
**Winner:** `gradient_boosting`  
**Params:** `{'n_estimators': 600, 'max_depth': 5, 'learning_rate': 0.05, 'subsample': 0.8}`  
**CV RMSE (train):** 0.152  
**R² (CV):** 0.669  
**Training time:** 18495 ms

**Runner-up:** `random_forest` — CV RMSE 0.1679

### Justification
The dataset (2000 rows, 8 numeric joint-angle features, target y with mean 0.71, std 0.26, near-zero skew) is the classic kin8nm robot-arm forward-kinematics problem. The strongest single linear correlation is theta3=0.519, all others ≤0.21, indicating that y is determined primarily by non-linear interactions among angles — exactly the regime where ridge bottoms out (RMSE 0.2019, R² 0.415) and where ensembled trees thrive. A single decision tree (RMSE 0.2293) actually performed worse than ridge, which confirms that the signal requires variance reduction via ensembling rather than a single high-capacity tree. Random forest immediately closed most of the gap (RMSE 0.1721 → 0.1679 with denser splits and leaf=2), but tuned gradient boosting (600 trees, depth=5, lr=0.05, subsample=0.8) reached RMSE 0.1520 and R² 0.669 — beating the best RF by 0.016 RMSE, which is well over one fold-std (0.009) and therefore statistically meaningful. Sequential residual fitting with moderate-depth trees captures the multiplicative angle interactions better than bagged averaging here. I skipped lasso (only 8 dense features — no sparsity story) and did not tune ridge further (linear ceiling already reached). Both top models are stable across folds (cv_rmse_std ≈ 6% of mean).

### Caveats
CV optimism: hyperparameters for both GB and RF were chosen by inspecting 5-fold CV scores on the same folds, so the reported RMSE of 0.1520 is mildly optimistic — expect a small degradation on truly unseen data. No held-out test set was evaluated. Untested options that could plausibly improve results: (a) richer GB grids (n_estimators 1000+, lower lr 0.01–0.03, depth 4–6, min_samples_leaf), (b) histogram-based GB or XGBoost/LightGBM, (c) feature engineering using sin/cos of angles or pairwise products — the natural basis for kinematics — which a tree must approximate piecewise and could unlock substantially more signal, (d) a small MLP, which often beats trees on smooth analytical targets like kin8nm. Target transform: y is already well-centred and not skewed (skew=0.07) and the max (1.45) is not a suspicious round number, so a log/Box-Cox transform is unlikely to help here. Spatial generalisation: not applicable — features are joint angles, not geographic, and there is no temporal column, so fold heterogeneity is benign (r2_std=0.027). Dataset size: at 2000 rows the high-capacity GB/RF ensembles are near their useful capacity for this number of features; doubling data would likely reduce RMSE further as kin8nm is known to scale well with sample size.

## Agent-Evaluated Models
| Model | Params | CV RMSE ± std | R² ± std | Time (ms) |
|-------|--------|---------------|----------|-----------|
| gradient_boosting | n_estimators=600, max_depth=5, learning_rate=0.05, subsample=0.8 | 0.152 ± 0.0087 | 0.6688 ± 0.0273 | 18495 |
| random_forest | n_estimators=500, max_features=0.5, min_samples_leaf=2 | 0.1679 ± 0.0078 | 0.5962 ± 0.0201 | 14413 |
| random_forest | n_estimators=300, max_features=sqrt | 0.1721 ± 0.0083 | 0.5757 ± 0.0203 | 7597 |
| gradient_boosting | n_estimators=300, max_depth=3, learning_rate=0.1 | 0.1777 ± 0.0099 | 0.5478 ± 0.0316 | 7525 |
| ridge | alpha=1.0 | 0.2019 ± 0.0098 | 0.4152 ± 0.0489 | 41 |
| decision_tree | max_depth=8 | 0.2293 ± 0.0057 | 0.2465 ± 0.0184 | 122 |


---

## Post-Hoc Baseline: GridSearchCV Exhaustive Search
*(Run after agent recommendation — agent had no visibility into these results)*

**Evaluations — Agent:** 6  |  **Baseline:** 143 combinations × 5 folds = 715 CV fits

**Baseline winner:** `gradient_boosting` — CV RMSE 0.1489, Test RMSE 0.1434, R² 0.6823  
**Total search time:** 122326 ms

| Model | Best Params | CV RMSE | Test RMSE | R² |
|-------|-------------|---------|-----------|-----|
| gradient_boosting | learning_rate=0.05, max_depth=6, n_estimators=500, subsample=0.7 | 0.1489 | 0.1434 | 0.6823 |
| random_forest | max_depth=None, min_samples_leaf=2, n_estimators=300 | 0.1662 | 0.159 | 0.6042 |
| linear | — | 0.2019 | 0.198 | 0.4152 |
| ridge | alpha=10.0 | 0.2019 | 0.198 | 0.4153 |
| lasso | alpha=0.0001 | 0.2019 | 0.198 | 0.4152 |
| decision_tree | max_depth=8, min_samples_leaf=10 | 0.2149 | 0.2055 | 0.3389 |

### Agent vs Baseline — Held-Out Test Set
| Metric | Agent (`gradient_boosting`) | Baseline (`gradient_boosting`) | Delta |
|--------|-------------------------------|------------------------------|-------|
| CV RMSE | 0.152 | 0.1489 | +0.0031 |
| **Test RMSE** | **0.1468** | **0.1434** | **+0.0034** |
| R² (CV) | 0.669 | 0.6823 | -0.0133 |
| R² (test) | 0.6598 | 0.6756 | -0.0158 |

Baseline **beat** the agent on the held-out test set by 2.4% RMSE (0.0034 lower).