"""LightGBM trainers — point forecaster + quantile forecasters.

Point:     L1 regression (robust to outliers, demand data is spiky).
Quantile:  three separate boosters for tau in {0.1, 0.5, 0.9} => 80% interval.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd


# ---------- Point model ----------

def train_point(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    params: Dict[str, Any],
    categorical: List[str] | None = None,
    seed: int = 42,
) -> lgb.Booster:
    p = dict(params)
    num_boost_round = p.pop("num_boost_round", 1500)
    early_stop = p.pop("early_stopping_rounds", 100)
    p.setdefault("verbosity", -1)
    p.setdefault("seed", seed)

    train_set = lgb.Dataset(
        X_train, label=y_train, categorical_feature=categorical or "auto"
    )
    val_set = lgb.Dataset(
        X_val, label=y_val, categorical_feature=categorical or "auto", reference=train_set
    )

    model = lgb.train(
        p,
        train_set,
        num_boost_round=num_boost_round,
        valid_sets=[train_set, val_set],
        valid_names=["train", "val"],
        callbacks=[
            lgb.early_stopping(early_stop, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    return model


# ---------- Quantile models ----------

def train_quantile_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    params: Dict[str, Any],
    quantiles: List[float],
    categorical: List[str] | None = None,
    seed: int = 42,
) -> Dict[float, lgb.Booster]:
    models: Dict[float, lgb.Booster] = {}
    num_boost_round = params.get("num_boost_round", 1500)
    early_stop = params.get("early_stopping_rounds", 100)

    train_set = lgb.Dataset(
        X_train, label=y_train, categorical_feature=categorical or "auto"
    )
    val_set = lgb.Dataset(
        X_val, label=y_val, categorical_feature=categorical or "auto", reference=train_set
    )

    for q in quantiles:
        p = {k: v for k, v in params.items() if k not in ("num_boost_round", "early_stopping_rounds", "quantiles")}
        p.update(
            {
                "objective": "quantile",
                "alpha": q,
                "metric": "quantile",
                "verbosity": -1,
                "seed": seed,
            }
        )
        m = lgb.train(
            p,
            train_set,
            num_boost_round=num_boost_round,
            valid_sets=[val_set],
            valid_names=["val"],
            callbacks=[
                lgb.early_stopping(early_stop, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
        models[q] = m
    return models


# ---------- Inference helpers ----------

def predict_point(model: lgb.Booster, X: pd.DataFrame) -> np.ndarray:
    return model.predict(X, num_iteration=model.best_iteration)


def predict_quantiles(
    models: Dict[float, lgb.Booster], X: pd.DataFrame
) -> Dict[float, np.ndarray]:
    out: Dict[float, np.ndarray] = {}
    for q, m in models.items():
        out[q] = m.predict(X, num_iteration=m.best_iteration)
    # enforce non-crossing: lower <= median <= upper
    qs_sorted = sorted(out.keys())
    stacked = np.vstack([out[q] for q in qs_sorted])
    stacked = np.sort(stacked, axis=0)
    for i, q in enumerate(qs_sorted):
        out[q] = stacked[i]
    return out


def save_models(
    point_model: lgb.Booster,
    quantile_models: Dict[float, lgb.Booster],
    out_dir: str | Path,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    point_model.save_model(str(out_dir / "lgbm_point.txt"))
    for q, m in quantile_models.items():
        m.save_model(str(out_dir / f"lgbm_q{int(q * 100):02d}.txt"))


def load_models(
    out_dir: str | Path, quantiles: List[float]
) -> Tuple[lgb.Booster, Dict[float, lgb.Booster]]:
    out_dir = Path(out_dir)
    point_model = lgb.Booster(model_file=str(out_dir / "lgbm_point.txt"))
    q_models = {
        q: lgb.Booster(model_file=str(out_dir / f"lgbm_q{int(q * 100):02d}.txt"))
        for q in quantiles
    }
    return point_model, q_models


def feature_importance(model: lgb.Booster, feature_names: List[str], top_k: int = 25) -> pd.DataFrame:
    gains = model.feature_importance(importance_type="gain")
    splits = model.feature_importance(importance_type="split")
    df = pd.DataFrame({"feature": feature_names, "gain": gains, "split": splits})
    df = df.sort_values("gain", ascending=False).head(top_k).reset_index(drop=True)
    return df
