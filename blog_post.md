# Does an LLM Reason Better Than Brute Force? A Regression Model Selection Experiment

Most machine learning workflows treat model selection as a search problem: define a grid of hyperparameters, let GridSearchCV exhaust it, pick the winner. It works, but it tells you nothing about *why* a model won. It also scales poorly — double the search space, double the training evaluations.

The question I wanted to answer: can an LLM reason about data characteristics well enough to approximate exhaustive search — using a fraction of the evaluations?

To find out, I built an agent that reads a dataset profile and reasons about which models to try, then compared it against two baselines: a wide GridSearchCV exhaustive search (715 CV fits) and Optuna, a Bayesian optimisation framework using Tree-structured Parzen Estimation (100 trials, 500 CV fits). All three were evaluated on a held-out 20% test set the agent never saw.

---

## The Setup

### The Agent

The agent follows a four-step workflow:

1. **Profile the data** — load the CSV, inspect target distribution, feature correlations, dataset size, and any structural signals (spatial columns, capping artefacts)
2. **Reason independently** — think out loud about which model families fit these characteristics, which to skip and why
3. **Evaluate chosen models** — run `train_and_evaluate` with thoughtful hyperparameter choices; 5-fold CV on the training set only
4. **Generate a report** — winner, runner-up, evidence-based justification, honest caveats

The agent has no access to the baseline results during reasoning. It works blind. The system prompt explicitly forbids mentioning a held-out test set — the agent selects purely on CV.

### The Baseline

GridSearchCV runs exhaustively over all six model families with wide grids:

- **Linear regression** — no hyperparameters (1 combination)
- **Ridge** — α ∈ {0.01, 0.1, 1.0, 10.0, 100.0} (5)
- **Lasso** — α ∈ {0.0001, 0.001, 0.01, 0.1, 1.0} (5)
- **Decision tree** — max_depth ∈ {3, 5, 8, 12, None} × min_samples_leaf ∈ {1, 3, 10} (15)
- **Random forest** — n_estimators ∈ {100, 300, 500} × max_depth ∈ {5, 10, 20, None} × min_samples_leaf ∈ {1, 2, 5} (36)
- **Gradient boosting** — n_estimators ∈ {100, 300, 500} × learning_rate ∈ {0.03, 0.05, 0.1} × max_depth ∈ {3, 4, 6} × subsample ∈ {0.7, 0.8, 1.0} (81)

**Total: 143 combinations × 5 folds = 715 CV fits.** This is a thorough search — not the narrow grids often used as straw-man baselines.

### Optuna (Bayesian Optimisation)

Optuna uses Tree-structured Parzen Estimation (TPE) to navigate the hyperparameter space sequentially, concentrating search in regions where previous trials performed well. Unlike GridSearchCV, it is not constrained to predefined discrete values — it samples continuously within ranges (e.g., learning rate log-uniform between 0.01 and 0.3, n_estimators between 50 and 800). This gives it access to configurations neither GridSearchCV nor the agent explicitly enumerate. Each run uses 100 trials with a fixed random seed, amounting to 500 CV fits — between the agent's ~30 and GridSearchCV's 715.

### The Evaluation

All three methods select their winner using 5-fold CV on the training set only, then are fit on the full training set and evaluated on a **held-out 20% test set** none of them touched during selection. Test RMSE is the final judge.

All experiments use an 80/20 train/test split (random_state=42). The agent is powered by Claude Opus 4.7 with adaptive thinking enabled.

---

## The Datasets

Five diverse regression benchmarks, chosen to cover different signal structures:

| Dataset | Rows | Features | Target | Signal type |
|---------|------|----------|--------|-------------|
| California Housing | 2,000 | 8 | Median house value ($100k) | Spatial, non-linear, capped |
| Diabetes Progression | 442 | 10 | Disease progression score | Linear-ish, small, noisy |
| Abalone Age | 2,000 | 7 | Ring count (age proxy) | Near-linear, collinear features |
| Energy Efficiency | 768 | 8 | Heating load (kWh/m²) | Strongly non-linear interactions |
| Kin8nm Kinematics | 2,000 | 8 | End-effector position | Non-linear, trigonometric |

---

## Results

| Dataset | Agent evals | Agent test RMSE | Optuna-100 test RMSE | GridSearch test RMSE |
|---------|-------------|-----------------|---------------------|---------------------|
| California | 5 | 0.5415 | 0.545 | **0.5359** |
| Diabetes | 7 | 53.446 | 53.497 | **52.898** |
| Abalone | 6 | 2.3443 | 2.3451 | **2.3442** |
| Energy | 6 | **0.378** | 0.4115 | 0.4015 |
| Kin8nm | 6 | 0.1468 | **0.1416** | 0.1434 |

**GridSearch: 4 wins. Agent: 1 win. Optuna: 1 win.** (Abalone is a three-way tie within 0.001 RMSE.)

**CV fits — Agent: ~30 per dataset · Optuna-100: 500 · GridSearch: 715**

The headline is not the win rate — it's the margin and the evaluation count. The agent's one clear win (energy, −5.9% vs GridSearch) holds against Optuna too (−8.1%). On the four datasets where it doesn't win, the losses are 1.0–2.4% — within fold standard deviation on these dataset sizes. The agent achieves this using roughly 1/24th the CV fits of GridSearch and 1/17th of Optuna.

---

## Dataset-by-Dataset Analysis

### California Housing — Baseline wins by 1.0%

**Data characteristics:** 2,000 California census blocks, predicting median house value. One dominant feature (MedInc, r=0.68), with Latitude/Longitude encoding geography. Target is capped at $500k — an artefact that creates an artificial performance ceiling for any model.

**Agent reasoning (5 evaluations):** The agent immediately diagnosed the two-tier signal:

> *"only one feature (MedInc, r=0.68) is linearly informative, while Latitude/Longitude carry strong non-linear/interaction signal that linear models cannot exploit. Ridge confirmed this with a weak CV RMSE of 0.847 (R²=0.48) and a very high R² std of 0.15 — the linear hyperplane simply cannot bend around California's geography."*

After confirming linear models were inappropriate, it ran two GB configurations and found them statistically tied on CV. It correctly flagged the capped target and recommended spatial feature engineering as the primary untapped gain.

**Result:** GridSearch found a marginally better configuration (0.5359 vs agent 0.5415, 1.0%). Optuna-100 scored 0.545 — slightly worse than the agent despite using 500 CV fits. All three methods converged on gradient boosting and were separated by less than 1.6%, with every difference well within fold standard deviation. All were constrained by the same ceiling: the spatial structure the flat features can't fully encode.

---

### Diabetes Progression — Baseline wins by 1.0%

**Data characteristics:** 442 patients, 10 blood serum measurements predicting disease progression. Small dataset, moderate signal spread across multiple features (bmi 0.60, s5 0.55, bp 0.44), no dominant predictor, mild right skew.

**Agent reasoning (7 evaluations):** The agent deployed a notably different diagnostic strategy here. Instead of jumping to its hypothesis, it ran linear regression first as a baseline probe, then tested non-linear models to confirm they offered no advantage:

> *"Random forest reached only 58.38 and gradient boosting only 60.29 — both >0.6 std worse than ridge while costing 30–100× more compute. These conditions strongly favour regularised linear models."*

It also noted the small dataset risk explicitly: with 89 test rows and fold std of ±2.6, model rankings carry substantial variance. Its winner was Ridge α=0.1.

**Result:** GridSearch picked Lasso (52.898), the agent picked Ridge (53.446), Optuna picked Ridge (53.497) — all within 0.6 RMSE of each other against a fold std of ±2.6. All three methods identified the linear family as correct; the small differences in test RMSE are noise on 89 test rows. The agent's structural reasoning was right. The test outcome simply wasn't large enough to distinguish methods on this dataset size.

---

### Abalone Age — Tied (0.004% difference)

**Data characteristics:** 2,000 abalones, 7 physical measurements (length, diameter, weights) predicting ring count. Features are highly collinear — all weight variants measure the same underlying "size." Target skew 1.18 with a long right tail.

**Agent reasoning (6 evaluations):** The agent identified the near-linear structure and tested L2 regularisation as the obvious choice for collinear features. After running Ridge, Lasso, and plain Linear, it found:

> *"All three linear models are statistically identical on CV (within 0.01 RMSE — far less than the ±0.08 fold std). This is the well-known abalone biological-noise floor; further gains require feature engineering, not a different model family."*

It recommended plain Linear OLS as the winner on parsimony grounds.

**Result:** Agent Linear 2.3443, Optuna Ridge 2.3451, GridSearch Lasso (α=0.0001) 2.3442 — three different methods, three slightly different model choices, all within 0.001 RMSE of each other. GridSearch's near-zero-alpha Lasso is functionally identical to OLS. All three converged on the same answer. The exhaustive search and Bayesian optimisation confirmed what the agent's reasoning predicted in 6 evaluations.

---

### Energy Efficiency — Agent wins by 5.9%

**Data characteristics:** 768 simulated buildings, 8 architectural parameters (compactness, surface area, wall/roof area, glazing) predicting heating load. Strong non-linear interactions between physical dimensions.

**Agent reasoning (6 evaluations):** The agent used a single decision tree as a non-linearity probe — a deliberate diagnostic move:

> *"A single depth-8 decision tree slashes RMSE to 0.55 — a 5.3× improvement over linear regression's 2.93. This proves substantial feature interactions exist (e.g., glazing × orientation × surface area)."*

It then went straight to gradient boosting and explicitly reasoned about stability, reducing cv_rmse_std by tuning toward more trees and lower learning rate:

> *"depth=4 with 500 trees and lr=0.05 gives the best stability — cv_rmse_std drops from 0.056 at default to 0.027. The shallower tree prevents memorising noise in the small dataset (768 rows)."*

**Result:** Agent 0.378, GridSearch 0.4015, Optuna-100 0.4115. The agent wins against both baselines — by 5.9% over GridSearch and 8.1% over Optuna. The mechanism matters: the agent's winning configuration (depth=4, lr=0.05, 500 estimators) was present in the baseline's grid. GridSearch evaluated it but CV selected a different configuration — one that did not generalise as well to the test set. Optuna, with its continuous search space and 100 trials, found a similar GB configuration but also could not match the agent's result. The agent did not win by accessing a better configuration; it won because its reasoning about stability guided it toward a configuration that CV noise pushed both search methods away from.

This is the clearest demonstration that reasoning about *why* matters, not just searching *what*.

---

### Kin8nm Kinematics — Baseline wins by 2.4%

**Data characteristics:** 2,000 samples from a forward kinematics simulation of an 8-joint robot arm, predicting end-effector position from joint angles. Strongly non-linear (trigonometric angular interactions), well-behaved target (symmetric, uncapped, low fold variance).

**Agent reasoning (6 evaluations):** The agent computed the linear ceiling analytically before running any model:

> *"The sum of squared single-feature correlations is ~0.38, predicting a linear R² ceiling around 0.4. Linear confirmed this exactly (RMSE 0.2019, R² 0.4152) — ~60% of variance lies in feature interactions."*

It then diagnosed why shallow GB underperformed: *"depth=3 stumps cannot capture multi-joint trigonometric interactions — these require deep trees to approximate."* It moved to depth=6 with 800 estimators and lr=0.03.

**Result:** Optuna-100 wins here at 0.1416, edging GridSearch (0.1434) and the agent (0.1468). This is the one dataset where Bayesian optimisation has a clear advantage — the continuous search space suits the trigonometric signal structure, and TPE's sequential learning found a finer-grained GB configuration than either the discrete grid or the agent's 6 evaluations reached. All three differences are within fold standard deviation, but kin8nm is the clearest case for Optuna's strengths.

---

## Patterns Across Datasets

**1. No single method dominates across all datasets.**

GridSearch wins on California, diabetes, and abalone — all narrow margins within fold variance. The agent wins on energy by a meaningful margin (5.9% over GridSearch, 8.1% over Optuna). Optuna wins on kin8nm, the one dataset where a continuous search space and sequential learning have the most traction. Abalone is a three-way tie.

**2. The agent beats both search methods when reasoning has leverage.**

On energy, the agent's explicit reasoning about stability and tree depth drove it to a configuration that CV noise pushed both GridSearch and Optuna away from. This is the strongest result: 30 CV fits outperforming 500 and 715.

**3. Model family selection was correct in all 5 cases.**

The agent correctly identified gradient boosting as optimal on California, energy, and kin8nm, and linear models as optimal on diabetes and abalone. On abalone, it chose plain OLS over Ridge — same family, slightly different regularisation level, but effectively tied in performance (RMSE difference 0.0001).

**4. The evaluation count tells a consistent story.**

Across all five datasets, the agent used 5–7 evaluations. That's 25–35 CV fits vs 715 — roughly 20–28× fewer. The agent didn't achieve this by getting lucky on early evaluations. It did it by skipping model families the data profile made obviously wrong, then exploring the promising region with thoughtful hyperparameter choices.

**5. The agent consistently flags what it can't know.**

Across all five reports, the agent raised: capped target concerns (California), spatial generalisation risk (California), CV optimism (all five), dataset-size-driven variance (diabetes), biological noise floors (abalone), and untested model families like XGBoost/LightGBM (kin8nm). None of these appear in any GridSearchCV output.

---

## Different Reasoning Approaches

One of the more interesting observations across runs: the agent doesn't apply a uniform strategy. Each dataset prompted different diagnostic behaviour.

On **California**, it ran a linear model first to establish the baseline, then moved directly to GB without wasting evaluations on intermediate families.

On **diabetes**, it ran *both* directions — tested linear first, then explicitly confirmed non-linear models were worse before committing to the linear recommendation. A more thorough but still efficient validation.

On **abalone**, it ran three linear variants side by side (Ridge, Lasso, OLS) to confirm they were statistically equivalent — a deliberate "is regularisation even doing anything here?" check.

On **energy**, it used a single shallow decision tree as a non-linearity probe before going to ensemble methods — treating tree depth as a diagnostic tool, not a competitive model.

On **kin8nm**, it computed the linear ceiling analytically from feature correlations *before* running any model, then used the first evaluation to confirm the prediction rather than explore.

Same underlying task, five different reasoning paths, each appropriate to the data profile — and in each case, the strategy converged on 5–7 evaluations. The efficiency appears robust to the reasoning approach, not dependent on any one diagnostic strategy working out.

---

## Limitations

**The agent and the baseline do not operate over the same search space.** GridSearchCV is constrained to the hyperparameter values defined in the grid — every combination evaluated is specified in advance by the practitioner. The agent has no such constraint: it selects hyperparameter values freely based on reasoning about the data, and can explore configurations the grid does not include. For kin8nm, the agent used 800 estimators — outside the baseline's maximum of 500. This asymmetry cuts both ways. It means the agent's losses cannot be taken as evidence that exhaustive grid search found the true optimum; it also means the agent's wins partly reflect access to a larger implicit search space rather than reasoning quality alone. In practical terms, however, the unconstrained nature of the agent is a usability advantage: it does not require the practitioner to predefine a grid, which is itself a non-trivial engineering decision.

**The agent is non-deterministic; the baseline is not.** GridSearchCV with a fixed random seed produces identical results on every run. The agent does not — reasoning paths vary across runs due to temperature and sampling in the underlying model. During development, California Housing was run twice and produced different outcomes: the agent won on one run and the baseline won on the other, both by narrow margins. This means individual run results carry more uncertainty than the tables suggest. A more robust evaluation would run each dataset multiple times and report mean and variance across agent runs. Single-run results should be interpreted as directional evidence, not precise measurements.

**CV on small datasets is noisy.** The diabetes result illustrates this. With 442 rows and 5 folds, each fold has ~88 samples. A 1.0% RMSE difference on 89 test rows is a single draw from a wide distribution. Running each dataset 10 times and averaging would give a cleaner signal.

**The datasets used here are relatively small and structurally legible.** All five benchmarks have fewer than 2,000 training rows and at most 10 features, with signal structures that map clearly onto known model families (spatial, linear, non-linear interactions). The agent's reasoning strategy — profile the data, identify the dominant signal type, select a model family, tune carefully — is well-suited to this regime. Whether the same approach scales to higher-dimensional datasets with less interpretable feature structures, noisier targets, or more ambiguous signal distributions is an open question. Problems where the data profile alone does not clearly favour one model family may require more evaluations or yield less reliable reasoning.

**Prompt engineering could further constrain agent behaviour.** Across five runs, the agent adopted meaningfully different diagnostic strategies — using decision trees as probes, testing non-linear models to rule them out, computing analytical bounds before running any evaluation. This variability is a source of non-determinism beyond temperature alone. More tightly specified prompting could standardise the reasoning approach across datasets, potentially improving consistency. Whether this would also improve predictive performance is unclear and likely dataset-dependent; there is a reasonable argument that adaptive strategy is itself desirable, and enforcing a uniform approach might eliminate useful diagnostic behaviour.

**The efficiency claim applies to model training evaluations, not raw compute.** The 1/24th ratio counts sklearn CV fits. It does not reflect total computational cost. A single LLM forward pass on Claude Opus 4.7 with adaptive thinking involves orders of magnitude more floating-point operations than fitting a Ridge regression on 400 rows, which completes in milliseconds on a modern CPU. What the agent saves is model training iterations and the engineering time required to define and run a grid. It does not save raw compute in any meaningful sense — it trades cheap training evaluations for expensive inference. This distinction matters when evaluating the approach for real-world use.

**Optuna's trial budget affects results in non-obvious ways.** Both 50-trial and 100-trial runs were tested. Results did not change substantially across the five datasets — the same model families won, and test RMSE differences between the two budgets were below 1% in all cases. The exception worth noting is California, where the 100-trial run scored marginally worse than the 50-trial run (0.545 vs 0.5438). This is a concrete illustration of CV overfitting under increasing search pressure: more trials means more exposure to CV noise, and TPE can converge on a configuration that scores better on CV but generalises slightly worse to the test set. The effect is small here, but it underlines a general principle — more search is not always better when selection is based on a noisy CV estimate.

**The benchmark datasets may not constitute a fair test of zero-shot reasoning.** California Housing, Diabetes, and Abalone are among the most widely used datasets in machine learning tutorials, textbooks, and public repositories. It is likely that extensive analyses of these exact datasets — including observations about dominant features, noise floors, and appropriate model families — were present in the model's training data. Some of the agent's correct diagnoses may reflect recalled priors about these specific problems rather than generalised reasoning from first principles. Testing the approach on novel, proprietary, or synthetic datasets with no public history would provide stronger evidence that the reasoning capability transfers to unseen problems.

**Five datasets is not a robust benchmark.** The results are directionally consistent but the sample size is too small for strong statistical claims. Energy is the most compelling single case; the others are narrow margins in both directions.

---

## Conclusion

Across three methods — semantic reasoning, Bayesian optimisation, and exhaustive enumeration — no single approach dominated. GridSearch won on three datasets, the agent on one, Optuna on one, with abalone a three-way tie. But the margins and the evaluation counts tell a more specific story.

The agent's energy win (−5.9% vs GridSearch, −8.1% vs Optuna-100) holds against both baselines and is the most meaningful result in the experiment. It used 30 CV fits against 500 and 715, and won because its reasoning about stability guided it toward a configuration that CV noise pushed both search methods away from. Reasoning about *why* produced a better *what*.

On the four datasets where the agent didn't win, the losses are 1.0–2.4% — within fold standard deviation on these dataset sizes. Abalone is the clearest illustration: three different methods, three slightly different model choices, all within 0.001 RMSE. Convergence on the same answer through different paths is its own form of validation.

The comparison with Optuna adds a useful data point. Bayesian optimisation with 500 CV fits did not systematically outperform exhaustive search with 715 fits across these datasets — GridSearch won on three, Optuna on one. This suggests that at these dataset sizes and model complexities, search budget matters more than search strategy. The agent's approach — skip obviously wrong families, reason carefully about the promising one — achieves a similar effect with a fraction of the evaluations.

There is also a practical dimension to the unconstrained search space. GridSearchCV requires a practitioner to define the grid before running — choosing which hyperparameters to vary and what values to include is itself a modelling decision that introduces its own bias. The agent eliminates this step: it selects parameters based on reasoning about the data, which may be a meaningful usability advantage in settings where API cost is not a binding constraint.

Where the agent has the clearest value, regardless of RMSE outcome, is the reasoning trail. On every dataset it produced a structured account of why it made the choices it made: which data characteristics drove the model family selection, why certain families were excluded, what the CV variance implies about generalisation risk, and what the next experiment should be. This output is not produced by GridSearchCV in any form. In contexts where model selection decisions must be explained — to stakeholders, auditors, or the team inheriting the code — the agent provides artefacts that exhaustive search does not.

The open question is whether the reasoning quality holds on harder problems: higher-dimensional data, noisier targets, more ambiguous signal structures, and datasets where domain knowledge encoded in training data does not clearly distinguish model families. These are the cases where both the promise and the risk of the approach are largest.

---

*All code and datasets are available at [github link]. The agent is built on the Anthropic Claude API with tool use and adaptive thinking. GridSearchCV and Optuna baselines use scikit-learn 1.5+ and optuna 4.x.*
