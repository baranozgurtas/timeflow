# TimeFlow — Demand Forecasting with Probabilistic Outputs and Decisions

Store–item demand forecasting on the [Kaggle Store Item Demand Forecasting](https://www.kaggle.com/c/demand-forecasting-kernels-only) dataset (500 series, 5 years, daily). Unlike a typical LightGBM demo, this project ships **point forecasts, calibrated intervals, and a downstream ordering policy** — the full path from raw data to a decision.

## Highlights

- **Baselines** — seasonal naive (and optional AutoETS) to quantify the lift from ML over classical methods.
- **Point model** — LightGBM with L1 objective on 40+ engineered features (lags, rolling stats, calendar, target encoding, expanding mean).
- **Probabilistic forecasts** — three LightGBM quantile regressors (τ = 0.1, 0.5, 0.9) with non-crossing enforcement.
- **Calibrated intervals** — split Conformalized Quantile Regression (Romano et al. 2019) using validation residuals, with guaranteed marginal coverage.
- **Decision layer** — newsvendor critical-fractile ordering: take the quantile forecast, solve the stocking problem, compare realized profit against ordering at the point forecast.
- **Dashboard** — Streamlit UI with forecast viewer, metrics table, feature importance, and an interactive newsvendor policy tab where you can change unit economics and see the profit impact.

## Repo layout

```
timeflow/
├── configs/
│   └── config.yaml
├── src/
│   ├── data_loader.py       # Kaggle loader + synthetic fallback + temporal split
│   ├── features.py          # lag / rolling / calendar / target encoding
│   ├── baselines.py         # seasonal naive + ETS wrapper
│   ├── quantile_lgbm.py     # point + quantile LightGBM
│   ├── conformal.py         # split CQR calibrator
│   ├── newsvendor.py        # critical fractile + profit comparison
│   ├── neural_models.py     # PyTorch MLP baseline (optional)
│   ├── evaluate.py          # RMSE / MAE / SMAPE / pinball / coverage / width
│   ├── utils.py             # config loader, seeding
│   └── train_all.py         # end-to-end orchestrator
├── app/
│   └── streamlit_app.py     # 4-tab dashboard
├── tests/
│   └── test_pipeline.py
├── requirements.txt
└── README.md
```

## Quickstart

```bash
# 1. install
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. put the Kaggle train.csv at data/train.csv
#    (or skip — a synthetic dataset is generated automatically the first time
#     data/train.csv is missing)

# 3. train everything
python -m src.train_all --config configs/config.yaml

# 4. launch dashboard
streamlit run app/streamlit_app.py
```

## What the training script produces

| File | Purpose |
|---|---|
| `models/lgbm_point.txt` | Point forecaster |
| `models/lgbm_q10.txt` / `q50.txt` / `q90.txt` | Quantile forecasters |
| `models/conformal.json` | Conformal correction term |
| `reports/metrics.csv` | All metrics (val + test, every model) |
| `reports/feature_importance.csv` | Top 30 features by gain |
| `reports/predictions_test.parquet` | Test-set predictions with intervals |

## Evaluation protocol

| Split | Date range |
|---|---|
| Train | 2013-01-01 → 2016-12-31 |
| Validation | 2017-01-01 → 2017-06-30 |
| Test | 2017-07-01 → 2017-12-31 |

- **Point metrics:** RMSE, MAE, SMAPE.
- **Probabilistic metrics:** pinball loss at each τ, empirical coverage and mean width for the 80% quantile interval.
- **Conformal:** coverage and width of the calibrated 90% interval on the test set — coverage should sit very close to 0.90.
- **Decision:** realized profit and service level under point-policy vs. newsvendor-policy ordering (see `RESULTS.md` after running).

## Feature engineering

| Category | Features |
|---|---|
| Lag | 1, 7, 14, 28-day lags of sales |
| Rolling | 7/14/30-day mean, std, min, max (all shifted by 1 day → no leakage) |
| Calendar | dow, day, doy, month, quarter, year, weekofyear, is_weekend, is_month_start/end, sin/cos of doy and dow |
| Encoding | Store mean, item mean, store×item mean (computed from train rows only) |
| Trend | Expanding mean per series |

## Tests

```bash
pytest -q
```

Covers: metric correctness, conformal coverage property, temporal split ordering, no-leakage in rolling features, newsvendor profit accounting.

## License

MIT.
