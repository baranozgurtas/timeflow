# TimeFlow — Results

Fill in after running `python -m src.train_all`. Numbers below are placeholders
showing the expected shape; paste your actual `reports/metrics.csv` output.

## Point forecasting

| Model | Split | RMSE | MAE | SMAPE |
|---|---|---:|---:|---:|
| Seasonal Naive | val | — | — | — |
| Seasonal Naive | test | — | — | — |
| LightGBM (L1) | val | — | — | — |
| LightGBM (L1) | test | — | — | — |

Expected ballpark: LightGBM cuts SMAPE ~25-35% vs. seasonal naive.

## Probabilistic forecasting

| Model | Split | Pinball@0.1 | Pinball@0.5 | Pinball@0.9 | Coverage@80 | Width@80 |
|---|---|---:|---:|---:|---:|---:|
| LGBM quantile | val | — | — | — | — | — |
| LGBM quantile | test | — | — | — | — | — |

## Conformal calibration

| Metric | Test |
|---|---:|
| Target coverage | 0.90 |
| Empirical coverage | — |
| Mean interval width | — |
| q̂ (conformal correction) | — |

A well-calibrated model lands coverage within ±1pp of target.

## Newsvendor policy

Economics used: unit_cost = 1.0, unit_price = 3.0, holding = 0.2, stockout = 1.5
(critical fractile ≈ 92%, so the policy picks τ = 0.9).

| Policy | Mean profit / day-item | Service level |
|---|---:|---:|
| Order at point forecast | — | — |
| Order at newsvendor quantile | — | — |

The newsvendor policy typically trades a small per-unit profit reduction for a
large service-level gain, which is what an inventory manager would pick.

## Top features (gain)

Paste the top 10 rows from `reports/feature_importance.csv`.

## Runtime

| Stage | Wall time |
|---|---|
| Feature build | — |
| LGBM point | — |
| LGBM quantile (3 models) | — |
| Conformal calibration | — |
| **Total** | — |
