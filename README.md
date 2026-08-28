# TimeFlow — Demand Forecasting with Calibrated Uncertainty and Cost-Aware Stocking Decisions

End-to-end demand forecasting on the [Kaggle Store Item Demand Forecasting](https://www.kaggle.com/c/demand-forecasting-kernels-only) dataset — 500 time series (10 stores × 50 items), 5 years, daily. Unlike a typical LightGBM demo that stops at point accuracy, this project connects **point forecasting, calibrated uncertainty, and quantile-based stocking decisions**.

<img width="1440" height="807" alt="Screenshot 2026-04-16 at 9 57 47 PM" src="https://github.com/user-attachments/assets/a360f933-656d-4b0f-85b5-5ea15e89f2f9" />

*Forecast viewer dashboard — 2017 H2 test window for Store 1, Item 1, showing the point forecast, actual sales, and calibrated 90% conformal prediction interval.*

<img width="1440" height="809" alt="Screenshot 2026-04-16 at 9 58 36 PM" src="https://github.com/user-attachments/assets/e2b69709-99b6-4300-9d1c-6c8c4fce036a" />

*Same viewer, different configuration — Store 5, Item 25, with the raw 10–90% quantile band. The higher-volume series exhibits wider intrinsic variability.*

## Why this project is different

Most demand-forecasting notebooks stop at a point estimate and an RMSE number. Inventory decisions additionally require uncertainty estimates and an explicit mapping from forecasts to order quantities.

TimeFlow implements three components:

1. **Quantile forecasts** — separate LightGBM models at τ = 0.1, 0.5, and 0.9 with non-crossing enforcement, providing distributional estimates rather than a single prediction.
2. **Conformal calibration** — split Conformalized Quantile Regression on validation residuals, reaching **89.1% empirical coverage at a 90% target** on the chronological test set.
3. **Newsvendor decision layer** — maps the theoretical critical fractile to the nearest available quantile forecast and compares realized profit against point-forecast ordering.

## Headline results

### Point forecasting — 31.5% SMAPE reduction over baseline

| Model             | Split    |     RMSE |      MAE |      SMAPE |
| ----------------- | -------- | -------: | -------: | ---------: |
| Seasonal Naive    | test     |    12.11 |     9.21 |     17.04% |
| **LightGBM (L1)** | **test** | **8.08** | **6.21** | **11.67%** |

### Calibrated intervals — empirical coverage within 1pp of target

| Metric                        |      Value |
| ----------------------------- | ---------: |
| Target coverage               |     90.00% |
| **Empirical coverage (test)** | **89.13%** |
| Mean interval width           |      24.81 |
| Conformal correction q̂       |      2.942 |

### Newsvendor policy — 1.98% higher mean profit under asymmetric costs

The configured unit economics are `unit_cost = 1.0`, `unit_price = 3.0`, `holding_cost = 0.2`, and `stockout_penalty = 1.5`. These imply a theoretical critical fractile of approximately 94.6%.

Because the trained quantile models are limited to τ = 0.1, 0.5, and 0.9, the implementation selects τ = 0.9 as the nearest available approximation rather than estimating the exact optimal quantile.

| Policy                                      | Mean profit / day-item | Profit lift | Service level |
| ------------------------------------------- | ---------------------: | ----------: | ------------: |
| Point-forecast ordering                     |                 107.33 |           — |         50.0% |
| **Approximate newsvendor policy (τ = 0.9)** |             **109.45** |  **+1.98%** |     **89.1%** |

The newsvendor approximation increases mean realized profit from 107.33 to 109.45 per day-item. The accompanying service-level increase is an expected consequence of ordering at the 0.9 quantile rather than near the conditional median, not an independent measure of forecasting improvement.

## Dashboard

Four interactive tabs: a forecast viewer with togglable intervals, a consolidated metrics table, global feature importance, and a newsvendor simulator that recomputes profit and service outcomes under configurable unit economics.

<img width="1438" height="767" alt="Screenshot 2026-04-16 at 10 00 29 PM" src="https://github.com/user-attachments/assets/294a6a63-8db6-405e-af97-c6f38dba46ca" />

*Metrics tab — model results across validation and test splits in one consolidated table.*

<img width="1439" height="771" alt="Screenshot 2026-04-16 at 10 00 44 PM" src="https://github.com/user-attachments/assets/a89d10a2-cd9a-4bc5-9532-ed910f3388ec" />

*Feature importance — `roll_mean_14`, `roll_mean_7`, and `roll_mean_30` dominate, followed by `dayofweek` and `lag_7`, reflecting short-to-medium demand trends and weekly seasonality.*

<img width="1440" height="660" alt="Screenshot 2026-04-16 at 10 00 55 PM" src="https://github.com/user-attachments/assets/146ea7ba-3682-4416-95f4-a497569f0fc6" />

*Newsvendor simulator — change the unit economics and inspect the resulting critical fractile, nearest available quantile, realized profit, and service level.*

## Repo layout

```text
timeflow/
├── configs/
│   └── config.yaml              # all hyperparameters in one place
├── src/
│   ├── data_loader.py           # Kaggle loader + synthetic fallback + temporal split
│   ├── features.py              # lag / rolling / calendar / target encoding
│   ├── baselines.py             # seasonal naive + ETS wrapper
│   ├── quantile_lgbm.py         # point + quantile LightGBM
│   ├── conformal.py             # split CQR calibrator
│   ├── newsvendor.py            # critical fractile + profit comparison
│   ├── neural_models.py         # PyTorch MLP baseline (optional)
│   ├── evaluate.py              # RMSE / MAE / SMAPE / pinball / coverage / width
│   ├── utils.py                 # config loader and seeding
│   └── train_all.py             # end-to-end orchestrator
├── app/
│   └── streamlit_app.py         # four-tab dashboard
├── tests/
│   └── test_pipeline.py         # pipeline and metric tests
├── requirements.txt
├── README.md                
```

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download the data with the Kaggle CLI
kaggle competitions download -c demand-forecasting-kernels-only
unzip demand-forecasting-kernels-only.zip -d data/

# If Kaggle data is unavailable, the pipeline generates a synthetic replica
# automatically on the first run.

# 3. Train the pipeline end to end
python -m src.train_all --config configs/config.yaml

# 4. Launch the dashboard
streamlit run app/streamlit_app.py
```

Training runs in approximately 5–10 minutes on a laptop without a GPU. The orchestrator writes:

| Output                             | Purpose                            |
| ---------------------------------- | ---------------------------------- |
| `models/lgbm_point.txt`            | Point forecaster                   |
| `models/lgbm_q{10,50,90}.txt`      | Quantile forecasters               |
| `models/conformal.json`            | Calibrated correction term         |
| `reports/metrics.csv`              | Metrics across models and splits   |
| `reports/feature_importance.csv`   | Top 30 features by gain            |
| `reports/predictions_test.parquet` | Test-set predictions and intervals |

## Evaluation protocol

Strict chronological train, validation, and test splits with leakage controls.

| Split      | Date range              |    Rows |
| ---------- | ----------------------- | ------: |
| Train      | 2013-01-01 → 2016-12-31 | 716,500 |
| Validation | 2017-01-01 → 2017-06-30 |  90,500 |
| Test       | 2017-07-01 → 2017-12-31 |  92,000 |

* **Point metrics:** RMSE, MAE, and SMAPE.
* **Probabilistic metrics:** pinball loss, empirical coverage, and mean interval width.
* **Conformal evaluation:** calibrated on the chronological validation set and evaluated empirically on the test set. Formal distribution-free coverage guarantees require exchangeability, which is not assumed for this time-series application.
* **Decision evaluation:** realized profit and service level for point-forecast and approximate newsvendor ordering under configurable unit economics.

## Feature engineering (40+ features)

| Category | Features                                                                                                                  |
| -------- | ------------------------------------------------------------------------------------------------------------------------- |
| Lag      | 1, 7, 14, and 28-day sales lags                                                                                           |
| Rolling  | 7/14/30-day mean, standard deviation, minimum, and maximum, shifted by one day to exclude contemporaneous targets         |
| Calendar | `dow`, `day`, `doy`, `month`, `quarter`, `year`, `weekofyear`, `is_weekend`, `is_month_start/end`, and cyclical encodings |
| Encoding | Store, item, and store-item means computed from training rows only                                                        |
| Trend    | Expanding mean per series                                                                                                 |

## Tests

```bash
pytest -q
```

The tests cover metric correctness, temporal split ordering, leakage controls in rolling features, conformal calibration behavior, and newsvendor profit accounting.

## Technical decisions

* **Why L1 for the point model?** Demand observations are spiky, and the L1 objective is robust to large residuals while aligning naturally with median-oriented forecasting.
* **Why separate quantile models?** Independent boosters estimate three distributional points, followed by sorting-based non-crossing enforcement to ensure monotonic predictions.
* **Why split conformal?** It provides lightweight, one-shot interval calibration without retraining. Coverage is reported empirically because the chronological observations are not assumed to be exchangeable.
* **Why newsvendor?** The single-period critical fractile maps forecast uncertainty to an order quantity. Because only three quantiles are trained, the implementation uses the available quantile nearest to the theoretical fractile.
* **Why LightGBM?** Gradient-boosted trees are well suited to tabular lag, rolling, calendar, and encoding features while remaining efficient across 500 series. The PyTorch MLP is retained as an optional secondary baseline.

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.
