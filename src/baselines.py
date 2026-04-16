"""Baseline forecasters — the numbers LightGBM has to beat."""
from __future__ import annotations

import numpy as np
import pandas as pd


def seasonal_naive(
    full_df: pd.DataFrame,
    eval_start: pd.Timestamp,
    eval_end: pd.Timestamp,
    season_length: int = 7,
    target: str = "sales",
    group_cols: tuple = ("store", "item"),
    date_col: str = "date",
) -> pd.DataFrame:
    """y_hat(t) = y(t - season_length) within each (store, item) series."""
    df = full_df.sort_values(list(group_cols) + [date_col]).copy()
    df["y_pred"] = df.groupby(list(group_cols))[target].shift(season_length)
    mask = (df[date_col] >= eval_start) & (df[date_col] <= eval_end)
    out = df.loc[mask, list(group_cols) + [date_col, "y_pred"]].copy()
    fill_mean = full_df.loc[full_df[date_col] < eval_start, target].mean()
    out["y_pred"] = out["y_pred"].fillna(fill_mean)
    return out


def ets_forecast_long(
    history_long: pd.DataFrame,
    horizon: int,
    season_length: int = 7,
) -> pd.DataFrame:
    """ETS via statsforecast. Expects columns: unique_id, ds, y."""
    try:
        from statsforecast import StatsForecast
        from statsforecast.models import AutoETS
    except ImportError as e:
        raise ImportError(
            "statsforecast not installed. Run: pip install statsforecast"
        ) from e

    sf = StatsForecast(
        models=[AutoETS(season_length=season_length)],
        freq="D",
        n_jobs=-1,
    )
    sf.fit(history_long)
    fc = sf.predict(h=horizon)
    if "unique_id" not in fc.columns:
        fc = fc.reset_index()
    fc = fc.rename(columns={"AutoETS": "y_pred"})
    return fc[["unique_id", "ds", "y_pred"]]
