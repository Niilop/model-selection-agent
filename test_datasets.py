"""Validate all dataset CSVs before running the agent."""

import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from datasets import DATASETS

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"


def check(label: str, condition: bool, detail: str = "") -> bool:
    status = PASS if condition else FAIL
    suffix = f"  ({detail})" if detail else ""
    print(f"  [{status}] {label}{suffix}")
    return condition


def test_dataset(name: str, cfg: dict) -> bool:
    print(f"\n-- {name} --")
    path = Path("data") / f"{name}.csv"
    ok = True

    ok &= check("file exists", path.exists(), str(path))
    if not ok:
        return False

    df = pd.read_csv(path)
    target = cfg["target"]

    ok &= check("target column present", target in df.columns,
                f"available: {list(df.columns)}")
    ok &= check("rows > 0", len(df) > 0, str(len(df)))
    ok &= check("more than 1 column", len(df.columns) > 1)

    if target in df.columns:
        ok &= check("target is numeric",
                    pd.to_numeric(df[target], errors="coerce").notna().all(),
                    str(df[target].dtype))
        ok &= check("target has variance", df[target].std() > 0)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c != target]
    ok &= check("at least 1 numeric feature", len(feature_cols) >= 1,
                f"{len(feature_cols)} numeric features")

    nan_pct = df[numeric_cols].isna().mean().max()
    ok &= check("no column >50% NaN", nan_pct < 0.5, f"max NaN%={nan_pct:.1%}")

    if cfg["n_rows"] is not None:
        ok &= check(f"row count <= {cfg['n_rows']}", len(df) <= cfg["n_rows"],
                    str(len(df)))

    print(f"  rows={len(df)}  features={len(feature_cols)}  "
          f"target_range=[{df[target].min():.3g}, {df[target].max():.3g}]")
    return ok


def main():
    results = []
    for name, cfg in DATASETS.items():
        try:
            passed = test_dataset(name, cfg)
        except Exception:
            print(f"  [{FAIL}] EXCEPTION")
            traceback.print_exc()
            passed = False
        results.append((name, passed))

    print("\n" + "=" * 40)
    total = len(results)
    passed = sum(1 for _, ok in results if ok)
    for name, ok in results:
        print(f"  {'OK' if ok else 'XX'} {name}")
    print(f"\n{passed}/{total} datasets ready")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
