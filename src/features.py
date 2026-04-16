"""Feature engineering — lag, rolling, calendar, and target-encoding features.

All rolling features are shifted by 1 day to avoid leakage: we never use today's
sales to predict today's sales.
"""
from __future__ import annotations

from typing import Iterable, List

import numpy as np
import pandas as pd


def add_calendar_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    d = df[date_col]
    df["dayofweek"] = d.dt.dayofweek
    df["day"] = d.dt.day
    df["dayofyear"] = d.dt.dayofyear
    df["month"] = d.dt.month
    df["quarter"] = d.dt.quarter
    df["year"] = d.dt.year
    df["weekofyear"] = d.dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["is_month_start"] = d.dt.is_month_start.astype(int)
    df["is_month_end"] = d.dt.is_month_end.astype(int)
    df["sin_doy"] = np.sin(2 * np.pi * df["dayofyear"] / 365.25)
    df["cos_doy"] = np.cos(2 * np.pi * df["dayofyear"] / 365.25)
    df["sin_dow"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["cos_dow"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
    return df


def add_lag_features(
    df: pd.DataFrame,
    target: str,
    group_cols: List[str],
    lags: Iterable[int],
) -> pd.DataFrame:
    g = df.groupby(group_cols, sort=False)[target]
    for lag in lags:
        df[f"lag_{lag}"] = g.shift(lag)
    return df


def add_rolling_features(
    df: pd.DataFrame,
    target: str,
    group_cols: List[str],
    windows: Iterable[int],
) -> pd.DataFrame:
    shifted = df.groupby(group_cols, sort=False)[target].shift(1)
    for w in windows:
        df[f"roll_mean_{w}"] = shifted.groupby([df[c] for c in group_cols]).transform(
            lambda s: s.rolling(w, min_periods=1).mean()
        )
        df[f"roll_std_{w}"] = shifted.groupby([df[c] for c in group_cols]).transform(
            lambda s: s.rolling(w, min_periods=1).std()
        )
        df[f"roll_min_{w}"] = shifted.groupby([df[c] for c in group_cols]).transform(
            lambda s: s.rolling(w, min_periods=1).min()
        )
        df[f"roll_max_{w}"] = shifted.groupby([df[c] for c in group_cols]).transform(
            lambda s: s.rolling(w, min_periods=1).max()
        )
    return df


def add_expanding_mean(df: pd.DataFrame, target: str, group_cols: List[str]) -> pd.DataFrame:
    shifted = df.groupby(group_cols, sort=False)[target].shift(1)
    df["expanding_mean"] = shifted.groupby([df[c] for c in group_cols]).transform(
        lambda s: s.expanding().mean()
    )
    return df


def add_target_encodings(
    df: pd.DataFrame,
    train_mask: pd.Series,
    target: str,
    group_cols: List[str],
) -> pd.DataFrame:
    """Add out-of-time target-encoded means using only training rows."""
    train_df = df.loc[train_mask]
    for col in group_cols:
        means = train_df.groupby(col)[target].mean()
        df[f"te_{col}"] = df[col].map(means).fillna(train_df[target].mean())
    # cross encoding
    if len(group_cols) >= 2:
        cross = train_df.groupby(group_cols)[target].mean()
        df["te_cross"] = df.set_index(group_cols).index.map(cross)
        df["te_cross"] = df["te_cross"].fillna(train_df[target].mean())
    return df


def build_features(
    df: pd.DataFrame,
    target: str,
    group_cols: List[str],
    lags: Iterable[int],
    windows: Iterable[int],
    train_end: str,
    date_col: str = "date",
    use_target_encoding: bool = True,
) -> pd.DataFrame:
    """Full feature pipeline."""
    df = df.sort_values(group_cols + [date_col]).reset_index(drop=True)
    df = add_calendar_features(df, date_col=date_col)
    df = add_lag_features(df, target, group_cols, lags)
    df = add_rolling_features(df, target, group_cols, windows)
    df = add_expanding_mean(df, target, group_cols)
    if use_target_encoding:
        train_mask = df[date_col] <= pd.Timestamp(train_end)
        df = add_target_encodings(df, train_mask, target, group_cols)
    return df


def feature_columns(df: pd.DataFrame, target: str, date_col: str = "date") -> List[str]:
    """Return the list of model-input columns (everything except date + target)."""
    return [c for c in df.columns if c not in (target, date_col)]
