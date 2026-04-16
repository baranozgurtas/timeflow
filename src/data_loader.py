"""Load the Store Item Demand Forecasting dataset and split temporally.

The Kaggle dataset has columns: date, store, item, sales (Jan 2013 - Dec 2017).
If it's missing we generate a deterministic synthetic replica so the pipeline
runs end-to-end for reviewers who don't have Kaggle credentials.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


def _synthetic_dataset(
    start: str = "2013-01-01",
    end: str = "2017-12-31",
    n_stores: int = 10,
    n_items: int = 50,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a Kaggle-lookalike dataset for offline runs."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="D")
    rows = []
    # store and item base demand levels
    store_mult = rng.uniform(0.7, 1.3, n_stores)
    item_mult = rng.uniform(0.4, 2.5, n_items)
    trend = np.linspace(0, 8, len(dates))
    doy = np.arange(len(dates))
    yearly = 10 * np.sin(2 * np.pi * doy / 365.25)
    weekly = 6 * np.sin(2 * np.pi * doy / 7)
    for s in range(1, n_stores + 1):
        for i in range(1, n_items + 1):
            base = 25 * store_mult[s - 1] * item_mult[i - 1]
            noise = rng.normal(0, 3, len(dates))
            sales = base + trend + yearly + weekly + noise
            sales = np.clip(np.round(sales), 0, None).astype(int)
            rows.append(pd.DataFrame({"date": dates, "store": s, "item": i, "sales": sales}))
    df = pd.concat(rows, ignore_index=True)
    return df


def load_raw(path: str | Path) -> pd.DataFrame:
    """Load raw CSV or synthesize if missing."""
    path = Path(path)
    if path.exists():
        df = pd.read_csv(path, parse_dates=["date"])
    else:
        print(f"[data_loader] {path} not found — generating synthetic dataset.")
        df = _synthetic_dataset()
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
    df = df.sort_values(["store", "item", "date"]).reset_index(drop=True)
    return df


def temporal_split(
    df: pd.DataFrame,
    train_end: str,
    val_end: str,
    test_end: str,
    date_col: str = "date",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split into train / val / test respecting time ordering."""
    train_end_ts = pd.Timestamp(train_end)
    val_end_ts = pd.Timestamp(val_end)
    test_end_ts = pd.Timestamp(test_end)

    train = df[df[date_col] <= train_end_ts].copy()
    val = df[(df[date_col] > train_end_ts) & (df[date_col] <= val_end_ts)].copy()
    test = df[(df[date_col] > val_end_ts) & (df[date_col] <= test_end_ts)].copy()
    return train, val, test
