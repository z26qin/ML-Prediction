# PJM Power & Demand Forecasting Dashboard

A local, interview-ready Streamlit app for PJM demand forecasting using historical load + optional weather features.

## Features

- 4 dashboard tabs:
  - Data Overview
  - Demand Forecast
  - Forecast Performance
  - Scenario / Weather Sensitivity
- CSV upload first (primary input path)
- Optional CSV URL loader
- Robust preprocessing for dirty data
- Baselines (lag-24 / lag-168) + RandomForest model
- Metrics: MAE, RMSE, MAPE, peak load error, directional accuracy
- Temperature sensitivity scenario (+/- °F)

## Expected CSV format

Preferred columns:

- `datetime`
- `load_mw`
- `zone`
- `temperature` (optional)
- `humidity` (optional)
- `wind_speed` (optional)

Supported aliases include:
- `timestamp`/`time` -> `datetime`
- `load`/`demand` -> `load_mw`
- `temp` -> `temperature`
- `wind` -> `wind_speed`

The app handles missing weather columns and missing values gracefully.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Then upload your CSV in the sidebar.

## Project structure

- `app.py` - Streamlit dashboard UI + workflow
- `src/data_loader.py` - loading and cleaning data
- `src/features.py` - feature engineering
- `src/model.py` - baseline/model training + scenario inference
- `src/metrics.py` - forecast metrics
- `sample_data.csv` - synthetic sample data for quick demo

## Notes

- No DB or cloud requirements.
- Built for MVP simplicity and readability.
