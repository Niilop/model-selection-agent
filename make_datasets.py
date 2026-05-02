#!/usr/bin/env python3
"""Fetch and save all 5 benchmark datasets as CSVs."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing, load_diabetes
from sklearn.datasets import fetch_openml

from datasets import DATASETS


def _sample(df: pd.DataFrame, n_rows: int | None, seed: int) -> pd.DataFrame:
    if n_rows is None or n_rows >= len(df):
        return df.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(df), size=n_rows, replace=False)
    return df.iloc[sorted(idx)].reset_index(drop=True)


def fetch_california(seed: int) -> pd.DataFrame:
    data = fetch_california_housing(as_frame=True)
    return data.frame.rename(columns={"MedHouseVal": "median_house_value"})


def fetch_diabetes(seed: int) -> pd.DataFrame:
    data = load_diabetes(as_frame=True)
    return data.frame


def fetch_abalone(seed: int) -> pd.DataFrame:
    data = fetch_openml(name="abalone", version=1, as_frame=True, parser="auto")
    df = data.frame
    # Find the rings/target column regardless of what OpenML named it
    target_name = data.target.name
    target_col = next(
        (c for c in df.columns if "ring" in c.lower()),
        target_name,
    )
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df = df.rename(columns={target_col: "Rings"})
    df = df.select_dtypes(include=[np.number])  # drops Sex (M/F/I)
    return df


def fetch_energy(seed: int) -> pd.DataFrame:
    data = fetch_openml(name="energy_efficiency", version=1, as_frame=True, parser="auto")
    df = data.frame.select_dtypes(include=[np.number])
    # Dataset has two targets Y1 (heating) and Y2 (cooling) — keep both, agent uses Y1
    return df


def fetch_kin8nm(seed: int) -> pd.DataFrame:
    data = fetch_openml(name="kin8nm", version=1, as_frame=True, parser="auto")
    return data.frame.select_dtypes(include=[np.number])


FETCHERS = {
    "california": fetch_california,
    "diabetes": fetch_diabetes,
    "abalone": fetch_abalone,
    "energy": fetch_energy,
    "kin8nm": fetch_kin8nm,
}


def make_all(output_dir: str = "data", seed: int = 42) -> None:
    Path(output_dir).mkdir(exist_ok=True)

    for name, cfg in DATASETS.items():
        print(f"Fetching {name}...", end=" ", flush=True)
        df = FETCHERS[name](seed)
        df = _sample(df, cfg["n_rows"], seed)
        path = Path(output_dir) / f"{name}.csv"
        df.to_csv(path, index=False)
        target = cfg["target"]
        features = [c for c in df.columns if c != target]
        print(f"{len(df)} rows × {len(df.columns)} cols → {path}")
        print(f"  target: {target}  |  features ({len(features)}): {features[:5]}{'...' if len(features) > 5 else ''}")
        print(f"  {cfg['notes']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch and save all benchmark datasets")
    parser.add_argument("--output", default="data", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    make_all(args.output, args.seed)
