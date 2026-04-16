# TimeFlow — Demand Forecasting with Calibrated Uncertainty and Profit-Optimal Decisions

End-to-end demand forecasting on the [Kaggle Store Item Demand Forecasting](https://www.kaggle.com/c/demand-forecasting-kernels-only) dataset — 500 time series (10 stores × 50 items), 5 years, daily. Unlike a typical LightGBM demo that stops at RMSE, this project delivers the full path: **point forecasts → calibrated intervals → stocking decisions**.

<img width="1440" height="807" alt="Screenshot 2026-04-16 at 9 57 47 PM" src="https://github.com/user-attachments/assets/a360f933-656d-4b0f-85b5-5ea15e89f2f9" />


*Forecast viewer dashboard — 2017 H2 test window for Store 1, Item 1, showing point forecast (blue), actual sales (black), and the calibrated 90% conformal prediction interval (shaded).*

<img width="1440" height="809" alt="Screenshot 2026-04-16 at 9 58 36 PM" src="https://github.com/user-attachments/assets/e2b69709-99b6-4300-9d1c-6c8c4fce036a" />

*Same viewer, different configuration — Store 5, Item 25, with the raw 10–90% quantile band instead of the conformal one. Higher-volume item, wider intrinsic variability.*

## Why this project is different

Most demand-forecasting notebooks stop at a point estimate and an RMSE number. That's not what an inventory manager needs. They need to know **how much to order**, and they need a number they can trust.

TimeFlow ships three things most demos don't:

1. **Quantile forecasts** — three LightGBM models at τ = 0.1, 0.5, 0.9 with non-crossing enforcement, so you get a full predictive distribution, not a single number.
2. **Conformal calibration** — split Conformalized Quantile Regression (Romano et al. 2019) on validation residuals, giving intervals with **empirical coverage of 89.1% at a 90% target** on the test set.
3. **Newsvendor decision layer** — converts the probabilistic forecast into an order quantity by solving the classical critical-fractile problem, and compares realized profit against ordering at the point forecast.

## Headline results

### Point forecasting — 33% SMAPE reduction over baseline

| Model | Split | RMSE | MAE | SMAPE |
|---|---|---:|---:|---:|
| Seasonal Naive | test | 12.11 | 9.21 | 17.04% |
| **LightGBM (L1)** | **test** | **8.08** | **6.21** | **11.67%** |

### Calibrated intervals — conformal coverage lands within 1pp of target

| Metric | Value |
|---|---:|
| Target coverage | 90.00% |
| **Empirical coverage (test)** | **89.13%** |
| Mean interval width | 24.81 |
| Conformal correction q̂ | 2.942 |

### Newsvendor policy — same unit economics, dramatically better service

Economics: unit_cost = 1.0, unit_price = 3.0, holding = 0.2, stockout = 1.5 (critical fractile ≈ 95% → order at τ = 0.9).

| Policy | Mean profit / day-item | Service level |
|---|---:|---:|
| Order at point forecast | 107.33 | **50.0%** ⚠️ |
| **Order at newsvendor quantile** | **109.45** | **89.1%** ✅ |

The newsvendor policy raises service level from 50% to 89% **while also increasing profit** — this is what optimizing for the right objective buys you.

## Dashboard

Four interactive tabs: forecast viewer with togglable intervals, full metrics table, global feature importance, and a newsvendor policy simulator where unit economics can be changed and the profit/service impact is recomputed live.

<img width="1438" height="767" alt="Screenshot 2026-04-16 at 10 00 29 PM" src="https://github.com/user-attachments/assets/294a6a63-8db6-405e-af97-c6f38dba46ca" />

*Metrics tab — all models across val and test splits in one consolidated table.*

<img width="1439" height="771" alt="Screenshot 2026-04-16 at 10 00 44 PM" src="https://github.com/user-attachments/assets/a89d10a2-cd9a-4bc5-9532-ed910f3388ec" />

*Feature importance — `roll_mean_14`, `roll_mean_7`, and `roll_mean_30` dominate, followed by `dayofweek` and `lag_7`. The model is picking up exactly what demand forecasting should — short-to-medium rolling means and weekly seasonality.*

<img width="1440" height="660" alt="Screenshot 2026-04-16 at 10 00 55 PM" src="https://github.com/user-attachments/assets/146ea7ba-3682-4416-95f4-a497569f0fc6" />

*Newsvendor policy tab — change unit economics and watch the critical fractile, chosen quantile, profit, and service level update instantly.*

## Repo layout

```
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
│   ├── utils.py                 # config loader, seeding
│   └── train_all.py             # end-to-end orchestrator
├── app/
│   └── streamlit_app.py         # 4-tab dashboard
├── tests/
│   └── test_pipeline.py         # smoke tests on metrics, leakage, conformal property
├── requirements.txt
├── README.md
└── RESULTS.md                   # detailed results table
```

## Quickstart

```bash
# 1. install
pip install -r requirements.txt

# 2. get the data (Kaggle CLI)
kaggle competitions download -c demand-forecasting-kernels-only
unzip demand-forecasting-kernels-only.zip -d data/
# if you don't have Kaggle access, the pipeline generates a synthetic replica
# automatically on first run.

# 3. train everything end-to-end
python -m src.train_all --config configs/config.yaml

# 4. launch the dashboard
streamlit run app/streamlit_app.py
```

Training runs in ~5–10 minutes on a laptop (no GPU). The orchestrator writes:

| Output | Purpose |
|---|---|
| `models/lgbm_point.txt` | Point forecaster |
| `models/lgbm_q{10,50,90}.txt` | Quantile forecasters |
| `models/conformal.json` | Calibrated correction term |
| `reports/metrics.csv` | All metrics across models and splits |
| `reports/feature_importance.csv` | Top 30 features by gain |
| `reports/predictions_test.parquet` | Test-set predictions with intervals |

## Evaluation protocol

Strict temporal split — no leakage, no cheating.

| Split | Date range | Rows |
|---|---|---:|
| Train | 2013-01-01 → 2016-12-31 | 716,500 |
| Validation | 2017-01-01 → 2017-06-30 | 90,500 |
| Test | 2017-07-01 → 2017-12-31 | 92,000 |

- **Point metrics:** RMSE, MAE, SMAPE.
- **Probabilistic metrics:** pinball loss at each τ, empirical coverage and mean width of the 80% and 90% intervals.
- **Conformal:** learned on validation, evaluated on test — coverage should land within ~1pp of target.
- **Decision:** realized profit and service level for point-policy vs. newsvendor-policy ordering, under a configurable unit-economics model.

## Feature engineering (40+ features)

| Category | Features |
|---|---|
| Lag | 1, 7, 14, 28-day lags of sales |
| Rolling | 7/14/30-day mean, std, min, max (all shifted by 1 day → zero leakage) |
| Calendar | dow, day, doy, month, quarter, year, weekofyear, is_weekend, is_month_start/end, sin/cos of doy and dow |
| Encoding | Store mean, item mean, store×item mean (computed from train rows only) |
| Trend | Expanding mean per series |

## Tests

```bash
pytest -q
```

Covers metric correctness, temporal split ordering, no-leakage in rolling features, conformal coverage property, and newsvendor profit accounting.

## Technical decisions

- **Why L1 objective for the point model?** Demand data is spiky; L1 is robust to outliers and gives better MAE than L2.
- **Why three separate quantile models instead of one?** Independent boosters, then non-crossing enforcement by sorting. Simpler than joint quantile objectives and reliably produces monotone quantiles.
- **Why split conformal over full conformal?** Distribution-free, one-shot calibration with guaranteed marginal coverage and zero retraining overhead.
- **Why newsvendor over more complex inventory models?** Single-period critical fractile has a closed-form solution and maps directly onto the quantile output — no extra modeling needed to go from forecast to decision.
- **Why LightGBM over deep learning?** On tabular time-series with strong engineered features, LightGBM dominates. The neural MLP in `neural_models.py` is included as a secondary baseline to confirm this.

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.
