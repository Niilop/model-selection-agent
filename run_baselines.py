#!/usr/bin/env python3
"""Run GridSearchCV and Optuna baselines across all datasets without the agent."""

import argparse

import tools as tool_module
from datasets import DATASETS

_COL_W = 14


def _row(*cols):
    return "  ".join(str(c).ljust(_COL_W) for c in cols)


def run_dataset(name: str, n_trials: int, skip_gridsearch: bool = False, skip_optuna: bool = False) -> None:
    cfg = DATASETS[name]
    csv_path = f"data/{name}.csv"
    target = cfg["target"]

    print(f"\n{'='*60}")
    print(f"  {name.upper()}  —  target: {target}")
    print(f"{'='*60}")

    load_result = tool_module.load_data(csv_path, target)
    if "error" in load_result:
        print(f"  ERROR: {load_result['error']}")
        return
    print(f"  Rows: {load_result['train_rows']} train / {load_result['test_rows']} test  |  Features: {load_result['features']}")

    gs = None
    if not skip_gridsearch:
        print("  Running GridSearchCV...", flush=True)
        gs = tool_module.run_baseline()

    opt = None
    if not skip_optuna:
        print(f"  Running Optuna ({n_trials} trials)...", flush=True)
        opt = tool_module.run_optuna_baseline(n_trials=n_trials)

    print()
    print(_row("Method", "CV fits", "Winner model", "Test RMSE", "Test R²", "Time (ms)"))
    print("  " + "-" * (_COL_W * 6 + 10))

    if gs:
        gs_fits = gs["total_combinations"] * 5
        print(_row(
            "GridSearchCV",
            f"{gs['total_combinations']}x5={gs_fits}",
            gs["winner"],
            gs["winner_test_rmse"],
            gs["winner_test_r2"],
            gs["total_time_ms"],
        ))

    if opt:
        opt_fits = n_trials * 5
        print(_row(
            f"Optuna-{n_trials}",
            f"{n_trials}x5={opt_fits}",
            opt["winner"],
            opt["winner_test_rmse"],
            opt["winner_test_r2"],
            opt["total_time_ms"],
        ))

    if gs and opt:
        rmse_diff = opt["winner_test_rmse"] - gs["winner_test_rmse"]
        pct = rmse_diff / gs["winner_test_rmse"] * 100
        if abs(pct) < 0.5:
            verdict = "Tied"
        elif rmse_diff < 0:
            verdict = f"Optuna wins by {abs(pct):.1f}%"
        else:
            verdict = f"GridSearch wins by {abs(pct):.1f}%"
        print(f"\n  Verdict: {verdict}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GridSearchCV and Optuna on all datasets")
    parser.add_argument("--trials", type=int, default=50, help="Optuna trial count (default: 50)")
    parser.add_argument("--dataset", default=None, help="Run a single dataset by name (default: all)")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--optuna", action="store_true", help="Run Optuna only")
    group.add_argument("--gridsearch", action="store_true", help="Run GridSearchCV only")
    group.add_argument("--all", dest="run_all", action="store_true", help="Run both (default)")
    args = parser.parse_args()

    run_gs = args.gridsearch or args.run_all or not args.optuna
    run_opt = args.optuna or args.run_all or not args.gridsearch

    datasets = [args.dataset] if args.dataset else list(DATASETS.keys())

    for name in datasets:
        if name not in DATASETS:
            print(f"Unknown dataset '{name}'. Choose from: {list(DATASETS.keys())}")
            continue
        run_dataset(name, args.trials, skip_gridsearch=not run_gs, skip_optuna=not run_opt)

    print()


if __name__ == "__main__":
    main()
