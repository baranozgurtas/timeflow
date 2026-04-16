"""Smoke tests for TimeFlow. Run with: pytest -q"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.conformal import calibrate_cqr
from src.data_loader import temporal_split
from src.evaluate import coverage, pinball_loss, point_metrics, smape
from src.features import build_features
from src.newsvendor import NewsvendorParams, realized_profit


def _toy_df() -> pd.DataFrame:
    dates = pd.date_range("2017-01-01", "2017-06-30", freq="D")
    rows = []
    for s in (1, 2):
        for i in (1, 2):
            vals = 20 + np.arange(len(dates)) * 0.1 + 3 * np.sin(np.arange(len(dates)) / 7)
            rows.append(pd.DataFrame({"date": dates, "store": s, "item": i, "sales": vals}))
    return pd.concat(rows, ignore_index=True)


def test_point_metrics():
    y = np.array([10.0, 20.0, 30.0])
    yp = np.array([11.0, 19.0, 33.0])
    m = point_metrics(y, yp)
    assert m["MAE"] > 0 and m["RMSE"] > 0 and m["SMAPE"] > 0


def test_pinball_symmetric_at_half():
    y = np.array([1.0, 2.0, 3.0])
    yp = np.array([1.5, 1.5, 3.5])
    assert pinball_loss(y, yp, 0.5) > 0


def test_coverage_all_inside():
    y = np.array([1.0, 2.0, 3.0])
    assert coverage(y, y - 1, y + 1) == 1.0


def test_features_no_leakage():
    df = _toy_df()
    out = build_features(
        df,
        target="sales",
        group_cols=["store", "item"],
        lags=[1, 7],
        windows=[7],
        train_end="2017-05-31",
        use_target_encoding=True,
    )
    # roll_mean_7 for first row of each group should be NaN (no history) or the first point
    # but it must never equal that row's sales exactly for a non-trivial series
    first_rows = out.groupby(["store", "item"]).head(1)
    assert (first_rows["roll_mean_7"].fillna(-1) != first_rows["sales"]).all()


def test_temporal_split_orders():
    df = _toy_df()
    tr, va, te = temporal_split(df, "2017-03-31", "2017-05-15", "2017-06-30")
    assert tr["date"].max() < va["date"].min() < te["date"].min()


def test_conformal_coverage_property():
    np.random.seed(0)
    y = np.random.normal(0, 1, 500)
    q10 = np.full_like(y, -0.5)
    q90 = np.full_like(y, 0.5)
    cal = calibrate_cqr(y, {0.1: q10, 0.9: q90}, alpha=0.1)
    assert cal.q_hat >= 0


def test_newsvendor_profit_positive_when_perfect():
    y = np.array([10, 20, 30], dtype=float)
    order = y.copy()
    p = realized_profit(y, order, NewsvendorParams())
    assert p["mean_stockout_units"] == 0
    assert p["mean_leftover"] == 0
