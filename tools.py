"""Tool implementations called by the agent."""

import time
from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split

from models import CANDIDATES

optuna.logging.set_verbosity(optuna.logging.WARNING)

CV_FOLDS = 5
TEST_SIZE = 0.2
_RMSE_SCORING = "neg_root_mean_squared_error"

# Module-level state shared across tool calls within one agent run
_X_train: pd.DataFrame | None = None
_X_test: pd.DataFrame | None = None
_y_train: pd.Series | None = None
_y_test: pd.Series | None = None
_baseline_result: dict | None = None
_optuna_result: dict | None = None
_evaluated_results: list[dict] = []


def load_data(csv_path: str, target_col: str) -> dict:
    global _X_train, _X_test, _y_train, _y_test, _evaluated_results, _baseline_result, _optuna_result

    _evaluated_results = []
    _baseline_result = None
    _optuna_result = None

    df = pd.read_csv(csv_path)
    if target_col not in df.columns:
        return {"error": f"Column '{target_col}' not found. Available: {list(df.columns)}"}

    df = df.select_dtypes(include=[np.number]).dropna()
    X = df.drop(columns=[target_col])
    y = df[target_col]

    _X_train, _X_test, _y_train, _y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=42
    )

    corr = _X_train.corrwith(_y_train).abs().sort_values(ascending=False)
    top_features = corr.head(5).round(3).to_dict()

    return {
        "total_rows": int(len(X)),
        "train_rows": int(len(_X_train)),
        "test_rows": int(len(_X_test)),
        "features": int(X.shape[1]),
        "feature_names": list(X.columns),
        "target_stats": {
            "mean": round(float(y.mean()), 4),
            "std": round(float(y.std()), 4),
            "min": round(float(y.min()), 4),
            "max": round(float(y.max()), 4),
            "skew": round(float(y.skew()), 4),
        },
        "top_feature_correlations": top_features,
        "candidate_models": list(CANDIDATES.keys()),
    }


def run_baseline() -> dict:
    global _baseline_result

    if _X_train is None or _y_train is None:
        return {"error": "Call load_data first."}

    all_results = []
    t0 = time.perf_counter()

    for name, cfg in CANDIDATES.items():
        param_grid = cfg["grid"]
        estimator = cfg["cls"](**cfg["fixed_kwargs"])

        search = GridSearchCV(
            estimator,
            param_grid if param_grid else {},
            scoring=_RMSE_SCORING,
            cv=CV_FOLDS,
            n_jobs=-1,
        )
        search.fit(_X_train, _y_train)
        best_cv_rmse = -search.best_score_
        best_params = search.best_params_

        # R² on train CV with best params
        best_est = cfg["cls"](**cfg["fixed_kwargs"], **best_params)
        r2_scores = cross_val_score(best_est, _X_train, _y_train, cv=CV_FOLDS, scoring="r2")

        # Test set evaluation
        best_est_test = cfg["cls"](**cfg["fixed_kwargs"], **best_params)
        best_est_test.fit(_X_train, _y_train)
        y_pred = best_est_test.predict(_X_test)
        test_rmse = float(np.sqrt(mean_squared_error(_y_test, y_pred)))
        test_r2 = float(r2_score(_y_test, y_pred))

        all_results.append({
            "model": name,
            "cv_rmse": round(float(best_cv_rmse), 4),
            "r2": round(float(r2_scores.mean()), 4),
            "test_rmse": round(test_rmse, 4),
            "test_r2": round(test_r2, 4),
            "best_params": best_params,
        })

    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    all_results.sort(key=lambda r: r["test_rmse"])
    winner = all_results[0]

    total_combinations = sum(
        max(1, len(list(__import__('itertools').product(*cfg["grid"].values()))))
        for cfg in CANDIDATES.values()
    )

    _baseline_result = {
        "winner": winner["model"],
        "winner_cv_rmse": winner["cv_rmse"],
        "winner_r2": winner["r2"],
        "winner_test_rmse": winner["test_rmse"],
        "winner_test_r2": winner["test_r2"],
        "winner_params": winner["best_params"],
        "all_results": all_results,
        "total_time_ms": elapsed_ms,
        "total_combinations": total_combinations,
    }
    return _baseline_result


def run_optuna_baseline(n_trials: int = 50) -> dict:
    global _optuna_result

    if _X_train is None or _y_train is None:
        return {"error": "Call load_data first."}

    # Param names are prefixed per model to avoid Optuna namespace conflicts
    _PREFIXES = {
        "ridge": "ridge_",
        "lasso": "lasso_",
        "decision_tree": "dt_",
        "random_forest": "rf_",
        "gradient_boosting": "gb_",
    }

    def objective(trial: optuna.Trial) -> float:
        model_name = trial.suggest_categorical("model", list(CANDIDATES.keys()))
        cfg = CANDIDATES[model_name]
        params: dict[str, Any] = {}

        if model_name == "ridge":
            params["alpha"] = trial.suggest_float("ridge_alpha", 1e-3, 1e3, log=True)
        elif model_name == "lasso":
            params["alpha"] = trial.suggest_float("lasso_alpha", 1e-4, 10.0, log=True)
        elif model_name == "decision_tree":
            params["max_depth"] = trial.suggest_int("dt_max_depth", 2, 20)
            params["min_samples_leaf"] = trial.suggest_int("dt_min_samples_leaf", 1, 20)
        elif model_name == "random_forest":
            params["n_estimators"] = trial.suggest_int("rf_n_estimators", 50, 800)
            params["max_depth"] = trial.suggest_int("rf_max_depth", 3, 30)
            params["min_samples_leaf"] = trial.suggest_int("rf_min_samples_leaf", 1, 10)
        elif model_name == "gradient_boosting":
            params["n_estimators"] = trial.suggest_int("gb_n_estimators", 50, 800)
            params["learning_rate"] = trial.suggest_float("gb_learning_rate", 0.01, 0.3, log=True)
            params["max_depth"] = trial.suggest_int("gb_max_depth", 2, 8)
            params["subsample"] = trial.suggest_float("gb_subsample", 0.5, 1.0)

        estimator = cfg["cls"](**cfg["fixed_kwargs"], **params)
        scores = cross_val_score(
            estimator, _X_train, _y_train, cv=CV_FOLDS, scoring=_RMSE_SCORING
        )
        return float(-scores.mean())

    t0 = time.perf_counter()
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials)
    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    best = study.best_trial
    model_name = best.params["model"]
    cfg = CANDIDATES[model_name]
    prefix = _PREFIXES.get(model_name, "")
    best_params = {
        k[len(prefix):]: v
        for k, v in best.params.items()
        if k != "model" and k.startswith(prefix)
    }

    estimator = cfg["cls"](**cfg["fixed_kwargs"], **best_params)
    estimator.fit(_X_train, _y_train)
    y_pred = estimator.predict(_X_test)

    _optuna_result = {
        "winner": model_name,
        "winner_params": best_params,
        "winner_cv_rmse": round(float(best.value), 4),
        "winner_test_rmse": round(float(np.sqrt(mean_squared_error(_y_test, y_pred))), 4),
        "winner_test_r2": round(float(r2_score(_y_test, y_pred)), 4),
        "n_trials": n_trials,
        "total_time_ms": elapsed_ms,
    }
    return _optuna_result


def train_and_evaluate(model_name: str, params: dict[str, Any] | None = None) -> dict:
    if _X_train is None or _y_train is None:
        return {"error": "Call load_data first."}
    if model_name not in CANDIDATES:
        return {"error": f"Unknown model '{model_name}'. Choose from: {list(CANDIDATES.keys())}"}

    cfg = CANDIDATES[model_name]
    merged = {**cfg["fixed_kwargs"], **(params or {})}
    estimator = cfg["cls"](**merged)

    t0 = time.perf_counter()
    rmse_scores = -cross_val_score(estimator, _X_train, _y_train, cv=CV_FOLDS, scoring=_RMSE_SCORING)
    r2_scores = cross_val_score(estimator, _X_train, _y_train, cv=CV_FOLDS, scoring="r2")
    fit_time_ms = int((time.perf_counter() - t0) * 1000)

    result = {
        "model": model_name,
        "params": merged,
        "cv_rmse": round(float(rmse_scores.mean()), 4),
        "cv_rmse_std": round(float(rmse_scores.std()), 4),
        "r2": round(float(r2_scores.mean()), 4),
        "r2_std": round(float(r2_scores.std()), 4),
        "fit_time_ms": fit_time_ms,
        "evaluation_number": len(_evaluated_results) + 1,
    }
    _evaluated_results.append(result)
    return result


def evaluate_on_test(model_name: str, params: dict[str, Any]) -> dict:
    """Fit on full training set, evaluate on held-out test set."""
    if _X_train is None or _X_test is None:
        return {"error": "Call load_data first."}
    if model_name not in CANDIDATES:
        return {"error": f"Unknown model '{model_name}'."}

    cfg = CANDIDATES[model_name]
    merged = {**cfg["fixed_kwargs"], **params}
    estimator = cfg["cls"](**merged)
    estimator.fit(_X_train, _y_train)
    y_pred = estimator.predict(_X_test)

    return {
        "model": model_name,
        "test_rmse": round(float(np.sqrt(mean_squared_error(_y_test, y_pred))), 4),
        "test_r2": round(float(r2_score(_y_test, y_pred)), 4),
        "test_rows": int(len(_y_test)),
    }


def generate_report(
    winner_model: str,
    winner_params: dict,
    winner_cv_rmse: float,
    winner_r2: float,
    winner_fit_time_ms: int,
    runner_up_model: str,
    runner_up_cv_rmse: float,
    justification: str,
    caveats: str = "",
) -> str:
    lines = ["# Regression Model Selection Report", ""]

    if _X_train is not None:
        lines += [
            "## Dataset",
            f"- **Train rows:** {len(_X_train)}  |  **Test rows (held out):** {len(_X_test)}",
            f"- **Features:** {_X_train.shape[1]}",
            f"- **Agent evaluations:** {len(_evaluated_results)}",
            "",
        ]

    lines += [
        "## Agent Recommendation",
        f"**Winner:** `{winner_model}`  ",
        f"**Params:** `{winner_params}`  ",
        f"**CV RMSE (train):** {winner_cv_rmse}  ",
        f"**R² (CV):** {winner_r2}  ",
        f"**Training time:** {winner_fit_time_ms} ms",
        "",
        f"**Runner-up:** `{runner_up_model}` — CV RMSE {runner_up_cv_rmse}",
        "",
        "### Justification",
        justification,
        "",
        "### Caveats",
        caveats,
        "",
    ]

    if _evaluated_results:
        lines += [
            "## Agent-Evaluated Models",
            "| Model | Params | CV RMSE ± std | R² ± std | Time (ms) |",
            "|-------|--------|---------------|----------|-----------|",
        ]
        for r in sorted(_evaluated_results, key=lambda x: x["cv_rmse"]):
            param_str = ", ".join(
                f"{k}={v}" for k, v in r["params"].items()
                if k not in CANDIDATES[r["model"]]["fixed_kwargs"]
            )
            lines.append(
                f"| {r['model']} | {param_str or '—'} | {r['cv_rmse']} ± {r['cv_rmse_std']} "
                f"| {r['r2']} ± {r['r2_std']} | {r['fit_time_ms']} |"
            )
        lines.append("")

    return "\n".join(lines)


def build_baseline_section(
    agent_winner_model: str,
    agent_winner_cv_rmse: float,
    agent_winner_r2: float,
    agent_test_rmse: float,
    agent_test_r2: float,
) -> str:
    if _baseline_result is None:
        return ""

    br = _baseline_result
    or_ = _optuna_result
    agent_evals = len(_evaluated_results)
    baseline_evals = br.get("total_combinations", "?")

    lines = [
        "---",
        "",
        "## Post-Hoc Baselines",
        "*(Run after agent recommendation — agent had no visibility into these results)*",
        "",
        f"**Agent evaluations:** {agent_evals} × {CV_FOLDS} folds = {agent_evals * CV_FOLDS} CV fits",
        f"**GridSearchCV:** {baseline_evals} combinations × {CV_FOLDS} folds = {baseline_evals * CV_FOLDS} CV fits  ({br['total_time_ms']} ms)",
    ]
    if or_:
        lines.append(
            f"**Optuna (TPE):** {or_['n_trials']} trials × {CV_FOLDS} folds = {or_['n_trials'] * CV_FOLDS} CV fits  ({or_['total_time_ms']} ms)"
        )

    lines += [
        "",
        "### GridSearchCV — All Models",
        "| Model | Best Params | CV RMSE | Test RMSE | R² |",
        "|-------|-------------|---------|-----------|-----|",
    ]
    for r in br["all_results"]:
        param_str = ", ".join(f"{k}={v}" for k, v in r["best_params"].items())
        lines.append(
            f"| {r['model']} | {param_str or '—'} | {r['cv_rmse']} | {r['test_rmse']} | {r['r2']} |"
        )

    # Three-way comparison table
    rmse_diff_gs = agent_test_rmse - br["winner_test_rmse"]
    pct_gs = rmse_diff_gs / br["winner_test_rmse"] * 100

    lines += [
        "",
        "### Three-Way Comparison — Held-Out Test Set",
        f"| Method | Evaluations (CV fits) | Model | Test RMSE | Test R² |",
        "|--------|----------------------|-------|-----------|---------|",
        f"| Agent | {agent_evals} × 5 = {agent_evals * CV_FOLDS} | `{agent_winner_model}` | **{agent_test_rmse}** | {agent_test_r2} |",
        f"| GridSearchCV | {baseline_evals} × 5 = {baseline_evals * CV_FOLDS} | `{br['winner']}` | **{br['winner_test_rmse']}** | {br['winner_test_r2']} |",
    ]
    if or_:
        lines.append(
            f"| Optuna TPE | {or_['n_trials']} × 5 = {or_['n_trials'] * CV_FOLDS} | `{or_['winner']}` | **{or_['winner_test_rmse']}** | {or_['winner_test_r2']} |"
        )

    # Verdict vs GridSearchCV
    if rmse_diff_gs < 0:
        verdict = f"Agent **beat** GridSearchCV by {abs(pct_gs):.1f}% RMSE on the held-out test set."
    elif abs(rmse_diff_gs) < 0.001:
        verdict = "Agent and GridSearchCV are **statistically tied** on the held-out test set."
    else:
        verdict = f"GridSearchCV **beat** the agent by {abs(pct_gs):.1f}% RMSE on the held-out test set."

    lines += ["", verdict]

    if or_:
        rmse_diff_opt = agent_test_rmse - or_["winner_test_rmse"]
        pct_opt = rmse_diff_opt / or_["winner_test_rmse"] * 100
        if rmse_diff_opt < 0:
            opt_verdict = f"Agent **beat** Optuna by {abs(pct_opt):.1f}% RMSE."
        elif abs(rmse_diff_opt) < 0.001:
            opt_verdict = "Agent and Optuna are **statistically tied**."
        else:
            opt_verdict = f"Optuna **beat** the agent by {abs(pct_opt):.1f}% RMSE."
        lines.append(opt_verdict)

    return "\n".join(lines)
