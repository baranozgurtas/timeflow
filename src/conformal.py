"""Split conformal prediction on top of a quantile forecaster.

Given validation residuals with respect to the quantile predictions, we compute
an additive correction that guarantees (in expectation) coverage of `1 - alpha`
on exchangeable test data. Implementation: Conformalized Quantile Regression
(Romano, Patterson, Candès 2019), which inflates the lower/upper quantiles by
the empirical quantile of the signed non-conformity scores.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class ConformalCalibrator:
    """Stores the conformal correction term learned on validation data."""

    q_hat: float
    alpha: float
    lower_q: float
    upper_q: float

    def apply(
        self, q_preds: Dict[float, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        lower = q_preds[self.lower_q] - self.q_hat
        upper = q_preds[self.upper_q] + self.q_hat
        # keep median if present
        out: Dict[str, np.ndarray] = {"lower": lower, "upper": upper}
        if 0.5 in q_preds:
            out["median"] = q_preds[0.5]
        return out


def calibrate_cqr(
    y_val: np.ndarray,
    q_preds_val: Dict[float, np.ndarray],
    alpha: float = 0.1,
    lower_q: float = 0.1,
    upper_q: float = 0.9,
) -> ConformalCalibrator:
    """Learn the conformal correction on a held-out calibration set."""
    lo = q_preds_val[lower_q]
    hi = q_preds_val[upper_q]
    scores = np.maximum(lo - y_val, y_val - hi)
    n = len(y_val)
    # conservative (1-alpha)(1+1/n) quantile
    k = int(np.ceil((n + 1) * (1 - alpha))) - 1
    k = min(max(k, 0), n - 1)
    q_hat = float(np.sort(scores)[k])
    return ConformalCalibrator(q_hat=q_hat, alpha=alpha, lower_q=lower_q, upper_q=upper_q)
