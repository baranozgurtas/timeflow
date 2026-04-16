"""Train all models end-to-end and write a results table.

Usage: python -m src.train_all [--config configs/config.yaml]

Produces:
  models/lgbm_point.txt
  models/lgbm_q10.txt  lgbm_q50.txt  lgbm_q90.txt
  models/conformal.json
  models/mlp.pt                 (if torch is available)
  reports/metrics.csv           (per-model metrics on val + test)
  reports/feature_importance.csv
  reports/predictions_test.parquet
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .baselines import seasonal_naive
from .conformal import calibrate_cqr
from .data_loader import load_raw, temporal_split
from .evaluate import point_metrics, quantile_metrics, coverage, interval_width
from .features import build_features, feature_columns
from .newsvendor import NewsvendorParams, compare_policies
from .quantile_lgbm import (
    feature_importance,
    predict_point,
    predict_quantiles,
    save_models,
    train_point,
    train_quantile_models,
)
from .utils import ensure_dir, load_config, set_seed


def _split_xy(df: pd.DataFrame, target: str, date_col: str):
    feats = feature_columns(df, target=target, date_col=date_col)
    X = df[feats]
    y = df[target]
    return X, y, feats


def main(config_path: str = "configs/config.yaml") -> None:
    cfg = load_config(config_path)
    set_seed(cfg.get("random_seed", 42))

    paths = cfg["paths"]
    dcfg = cfg["data"]
    fcfg = cfg["features"]

    ensure_dir(paths["models_dir"])
    ensure_dir(paths["reports_dir"])

    print("[1/6] Loading data…")
    raw = load_raw(paths["raw_data"])
    print(f"      {len(raw):,} rows, date range {raw[dcfg['date_col']].min().date()} → {raw[dcfg['date_col']].max().date()}")

    print("[2/6] Building features…")
    full = build_features(
        raw,
        target=dcfg["target_col"],
        group_cols=dcfg["group_cols"],
        lags=fcfg["lags"],
        windows=fcfg["rolling_windows"],
        train_end=dcfg["train_end"],
        date_col=dcfg["date_col"],
        use_target_encoding=fcfg["use_target_encoding"],
    )

    # drop rows that don't have lag history (earliest dates)
    max_lag = max(fcfg["lags"]) + max(fcfg["rolling_windows"])
    full = full.dropna(subset=[f"lag_{max(fcfg['lags'])}"]).reset_index(drop=True)

    train, val, test = temporal_split(
        full, dcfg["train_end"], dcfg["val_end"], dcfg["test_end"], dcfg["date_col"]
    )
    print(f"      train={len(train):,}  val={len(val):,}  test={len(test):,}")

    X_tr, y_tr, feats = _split_xy(train, dcfg["target_col"], dcfg["date_col"])
    X_va, y_va, _ = _split_xy(val, dcfg["target_col"], dcfg["date_col"])
    X_te, y_te, _ = _split_xy(test, dcfg["target_col"], dcfg["date_col"])

    results_rows = []

    # -------- baseline: seasonal naive --------
    print("[3/6] Baseline: seasonal naive…")
    sn_val = seasonal_naive(
        raw,
        eval_start=val[dcfg["date_col"]].min(),
        eval_end=val[dcfg["date_col"]].max(),
        season_length=cfg["baselines"]["seasonal_length"],
        target=dcfg["target_col"],
        group_cols=tuple(dcfg["group_cols"]),
        date_col=dcfg["date_col"],
    )
    sn_test = seasonal_naive(
        raw,
        eval_start=test[dcfg["date_col"]].min(),
        eval_end=test[dcfg["date_col"]].max(),
        season_length=cfg["baselines"]["seasonal_length"],
        target=dcfg["target_col"],
        group_cols=tuple(dcfg["group_cols"]),
        date_col=dcfg["date_col"],
    )

    def _align(base_df, eval_df):
        keys = dcfg["group_cols"] + [dcfg["date_col"]]
        m = eval_df[keys + [dcfg["target_col"]]].merge(base_df, on=keys, how="left")
        return m["y_pred"].values, m[dcfg["target_col"]].values

    sn_val_pred, sn_val_true = _align(sn_val, val)
    sn_te_pred, sn_te_true = _align(sn_test, test)
    m = point_metrics(sn_val_true, sn_val_pred)
    m.update({"model": "SeasonalNaive", "split": "val"})
    results_rows.append(m)
    m = point_metrics(sn_te_true, sn_te_pred)
    m.update({"model": "SeasonalNaive", "split": "test"})
    results_rows.append(m)

    # -------- LightGBM point --------
    print("[4/6] LightGBM point model…")
    point_params = dict(cfg["model"]["point"])
    point_model = train_point(X_tr, y_tr, X_va, y_va, point_params)
    p_val = predict_point(point_model, X_va)
    p_te = predict_point(point_model, X_te)

    m = point_metrics(y_va.values, p_val)
    m.update({"model": "LGBM_point", "split": "val"})
    results_rows.append(m)
    m = point_metrics(y_te.values, p_te)
    m.update({"model": "LGBM_point", "split": "test"})
    results_rows.append(m)

    # feature importance
    fi = feature_importance(point_model, feats, top_k=30)
    fi.to_csv(Path(paths["reports_dir"]) / "feature_importance.csv", index=False)

    # -------- LightGBM quantile --------
    print("[5/6] LightGBM quantile models (τ=0.1, 0.5, 0.9)…")
    q_params = dict(cfg["model"]["quantile"])
    quantiles = q_params.pop("quantiles")
    q_models = train_quantile_models(X_tr, y_tr, X_va, y_va, q_params, quantiles)

    q_val = predict_quantiles(q_models, X_va)
    q_te = predict_quantiles(q_models, X_te)

    m = quantile_metrics(y_va.values, q_val)
    m.update({"model": "LGBM_quantile", "split": "val", "RMSE": np.nan, "MAE": np.nan, "SMAPE": np.nan})
    results_rows.append(m)
    m = quantile_metrics(y_te.values, q_te)
    m.update({"model": "LGBM_quantile", "split": "test", "RMSE": np.nan, "MAE": np.nan, "SMAPE": np.nan})
    results_rows.append(m)

    # -------- Conformal calibration on val --------
    alpha = cfg["model"]["conformal"]["alpha"]
    cal = calibrate_cqr(y_va.values, q_val, alpha=alpha, lower_q=0.1, upper_q=0.9)
    conf_te = cal.apply(q_te)
    cov_te = coverage(y_te.values, conf_te["lower"], conf_te["upper"])
    wid_te = interval_width(conf_te["lower"], conf_te["upper"])

    results_rows.append(
        {
            "model": "Conformal@90",
            "split": "test",
            "Coverage": cov_te,
            "Width": wid_te,
            "q_hat": cal.q_hat,
        }
    )

    # save conformal
    with open(Path(paths["models_dir"]) / "conformal.json", "w") as f:
        json.dump(
            {"q_hat": cal.q_hat, "alpha": cal.alpha, "lower_q": cal.lower_q, "upper_q": cal.upper_q},
            f,
            indent=2,
        )

    # -------- Newsvendor policy comparison --------
    nv_params = NewsvendorParams(**cfg["newsvendor"])
    nv_compare = compare_policies(y_te.values, p_te, q_te, nv_params)
    results_rows.append(
        {
            "model": "Policy_point",
            "split": "test",
            "mean_profit": nv_compare["point_policy"]["mean_profit"],
            "service_level": nv_compare["point_policy"]["service_level"],
        }
    )
    results_rows.append(
        {
            "model": "Policy_newsvendor",
            "split": "test",
            "mean_profit": nv_compare["newsvendor_policy"]["mean_profit"],
            "service_level": nv_compare["newsvendor_policy"]["service_level"],
        }
    )

    # -------- Save models, predictions, metrics --------
    print("[6/6] Saving models and reports…")
    save_models(point_model, q_models, paths["models_dir"])

    preds_out = test[dcfg["group_cols"] + [dcfg["date_col"], dcfg["target_col"]]].copy()
    preds_out["y_pred"] = p_te
    preds_out["q10"] = q_te[0.1]
    preds_out["q50"] = q_te[0.5]
    preds_out["q90"] = q_te[0.9]
    preds_out["conformal_lower"] = conf_te["lower"]
    preds_out["conformal_upper"] = conf_te["upper"]
    preds_out.to_parquet(Path(paths["reports_dir"]) / "predictions_test.parquet", index=False)

    metrics = pd.DataFrame(results_rows)
    metrics.to_csv(Path(paths["reports_dir"]) / "metrics.csv", index=False)
    print("\nMetrics:")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    main(args.config)
