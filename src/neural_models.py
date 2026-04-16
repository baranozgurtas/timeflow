"""A small PyTorch MLP baseline on the same engineered features.

The goal isn't to beat LightGBM on tabular data (it usually can't) — it's to
demonstrate that the engineered feature set is strong enough that a minimal
neural net already forecasts sensibly, and to have a neural comparison row in
the results table. Trains in a couple of minutes on CPU.

If PyTorch is unavailable, train_mlp() raises ImportError with a clear message.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd


@dataclass
class MLPArtifacts:
    model: object          # torch.nn.Module
    feature_mean: np.ndarray
    feature_std: np.ndarray
    target_mean: float
    target_std: float


def _require_torch():
    try:
        import torch
        import torch.nn as nn
    except ImportError as e:
        raise ImportError(
            "PyTorch not installed. Install with: pip install torch"
        ) from e
    return torch, nn


def train_mlp(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    hidden: Tuple[int, int] = (256, 128),
    epochs: int = 30,
    batch_size: int = 4096,
    lr: float = 1e-3,
    seed: int = 42,
) -> MLPArtifacts:
    torch, nn = _require_torch()
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    X_tr = X_train.values.astype(np.float32)
    X_va = X_val.values.astype(np.float32)
    y_tr = y_train.values.astype(np.float32)
    y_va = y_val.values.astype(np.float32)

    # z-score from training set (robust to feature scale)
    feat_mean = np.nanmean(X_tr, axis=0)
    feat_std = np.nanstd(X_tr, axis=0)
    feat_std = np.where(feat_std < 1e-6, 1.0, feat_std)
    X_tr = np.nan_to_num((X_tr - feat_mean) / feat_std)
    X_va = np.nan_to_num((X_va - feat_mean) / feat_std)

    target_mean = float(y_tr.mean())
    target_std = float(y_tr.std() + 1e-6)
    y_tr_n = (y_tr - target_mean) / target_std
    y_va_n = (y_va - target_mean) / target_std

    class MLP(nn.Module):
        def __init__(self, d_in: int):
            super().__init__()
            h1, h2 = hidden
            self.net = nn.Sequential(
                nn.Linear(d_in, h1),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(h1, h2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(h2, 1),
            )

        def forward(self, x):
            return self.net(x).squeeze(-1)

    model = MLP(X_tr.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.SmoothL1Loss()

    x_tr_t = torch.from_numpy(X_tr).to(device)
    y_tr_t = torch.from_numpy(y_tr_n).to(device)
    x_va_t = torch.from_numpy(X_va).to(device)
    y_va_t = torch.from_numpy(y_va_n).to(device)

    n = x_tr_t.shape[0]
    best_val = float("inf")
    best_state = None
    patience, bad = 5, 0

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            opt.zero_grad()
            pred = model(x_tr_t[idx])
            loss = loss_fn(pred, y_tr_t[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_pred = model(x_va_t)
            val_loss = loss_fn(val_pred, y_va_t).item()
        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return MLPArtifacts(
        model=model,
        feature_mean=feat_mean,
        feature_std=feat_std,
        target_mean=target_mean,
        target_std=target_std,
    )


def predict_mlp(artifacts: MLPArtifacts, X: pd.DataFrame) -> np.ndarray:
    torch, _ = _require_torch()
    X_np = X.values.astype(np.float32)
    X_np = np.nan_to_num((X_np - artifacts.feature_mean) / artifacts.feature_std)
    device = next(artifacts.model.parameters()).device
    artifacts.model.eval()
    with torch.no_grad():
        x = torch.from_numpy(X_np).to(device)
        pred_n = artifacts.model(x).cpu().numpy()
    return pred_n * artifacts.target_std + artifacts.target_mean
