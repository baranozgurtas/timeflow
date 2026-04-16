"""Evaluation metrics for point and probabilistic forecasts."""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    denom = np.where(denom == 0, 1.0, denom)
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100.0)


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
    diff = y_true - y_pred
    loss = np.where(diff >= 0, quantile * diff, (quantile - 1) * diff)
    return float(np.mean(loss))


def coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    return float(np.mean((y_true >= lower) & (y_true <= upper)))


def interval_width(lower: np.ndarray, upper: np.ndarray) -> float:
    return float(np.mean(upper - lower))


def point_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "RMSE": rmse(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "SMAPE": smape(y_true, y_pred),
    }


def quantile_metrics(
    y_true: np.ndarray,
    q_preds: Dict[float, np.ndarray],
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for q, yhat in q_preds.items():
        out[f"Pinball@{q}"] = pinball_loss(y_true, yhat, q)
    if 0.1 in q_preds and 0.9 in q_preds:
        out["Coverage@80"] = coverage(y_true, q_preds[0.1], q_preds[0.9])
        out["Width@80"] = interval_width(q_preds[0.1], q_preds[0.9])
    return out


def slice_metrics(
    df: pd.DataFrame,
    y_col: str,
    pred_col: str,
    group_col: str,
) -> pd.DataFrame:
    """Metrics broken down by a group column (e.g. store or item)."""
    rows = []
    for name, g in df.groupby(group_col):
        m = point_metrics(g[y_col].values, g[pred_col].values)
        m[group_col] = name
        rows.append(m)
    return pd.DataFrame(rows)[[group_col, "RMSE", "MAE", "SMAPE"]].sort_values(group_col)
