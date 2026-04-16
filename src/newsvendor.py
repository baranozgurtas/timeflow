"""Newsvendor stocking from quantile forecasts.

Classical newsvendor: optimal order quantity for a single period is the critical-
fractile quantile of the demand distribution:

    q*  =  (p - c) / (p - c + h)     (lost-sale case, no penalty)

When a stockout penalty s is added, the critical fractile becomes:

    q*  =  (p - c + s) / (p - c + s + h)

We pick from the set of available quantile predictions the one closest to q*
and use that as the order quantity. Then we report expected profit/cost at that
choice evaluated against realized demand.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class NewsvendorParams:
    unit_cost: float = 1.0
    unit_price: float = 3.0
    holding_cost: float = 0.2
    stockout_penalty: float = 1.5

    @property
    def critical_fractile(self) -> float:
        margin = self.unit_price - self.unit_cost
        return (margin + self.stockout_penalty) / (
            margin + self.stockout_penalty + self.holding_cost
        )


def pick_quantile(
    q_preds: Dict[float, np.ndarray], target_fractile: float
) -> np.ndarray:
    """Return the prediction vector for the quantile closest to target fractile."""
    qs = np.array(sorted(q_preds.keys()))
    chosen = qs[np.argmin(np.abs(qs - target_fractile))]
    return q_preds[float(chosen)]


def realized_profit(
    y_true: np.ndarray,
    order_qty: np.ndarray,
    params: NewsvendorParams,
) -> Dict[str, float]:
    sold = np.minimum(y_true, order_qty)
    leftover = np.maximum(order_qty - y_true, 0)
    stockout_units = np.maximum(y_true - order_qty, 0)

    revenue = params.unit_price * sold
    procurement = params.unit_cost * order_qty
    holding = params.holding_cost * leftover
    penalty = params.stockout_penalty * stockout_units

    profit = revenue - procurement - holding - penalty
    return {
        "mean_profit": float(np.mean(profit)),
        "total_profit": float(np.sum(profit)),
        "mean_leftover": float(np.mean(leftover)),
        "mean_stockout_units": float(np.mean(stockout_units)),
        "service_level": float(np.mean(y_true <= order_qty)),
    }


def compare_policies(
    y_true: np.ndarray,
    point_pred: np.ndarray,
    q_preds: Dict[float, np.ndarray],
    params: NewsvendorParams,
) -> Dict[str, Dict[str, float]]:
    """Compare ordering = point prediction vs ordering = newsvendor quantile."""
    order_point = np.round(np.clip(point_pred, 0, None))
    order_nv = np.round(
        np.clip(pick_quantile(q_preds, params.critical_fractile), 0, None)
    )
    return {
        "point_policy": realized_profit(y_true, order_point, params),
        "newsvendor_policy": realized_profit(y_true, order_nv, params),
    }
