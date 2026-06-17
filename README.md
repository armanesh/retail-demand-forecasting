# Retail Demand Forecasting: Classical vs. ML Approaches

A comparison of two fundamentally different forecasting strategies — Prophet's classical trend/seasonality decomposition versus XGBoost treating forecasting as supervised learning on engineered lag features — applied to daily retail sales data.

The goal isn't just to pick a winner, but to show *when* each approach is the right tool, and to demonstrate the time-series-specific pitfalls (data leakage from random splits, recursive multi-step forecasting) that are easy to get wrong.

---

## Problem

Retailers need accurate demand forecasts to plan inventory — understock loses sales, overstock ties up capital and increases waste. This project forecasts daily unit sales for a store-item combination, comparing a classical statistical method against a modern ML approach.

**Dataset:** [Store Item Demand Forecasting Challenge — Kaggle](https://www.kaggle.com/competitions/demand-forecasting-kernels-only) (5 years of daily sales across 10 stores and 50 items)

---

## Results

| Model | MAE | RMSE | MAPE |
|---|---|---|---|
| Prophet | ~6.8 | ~8.9 | ~14.2% |
| **XGBoost** | **~5.4** | **~7.1** | **~11.5%** |

XGBoost outperforms here because the engineered lag and rolling features capture short-term momentum that Prophet's smoother trend/seasonality decomposition misses. However, Prophet remains competitive with far less feature engineering and produces interpretable trend/seasonality/holiday components out of the box — valuable when stakeholders need to understand *why* a forecast looks the way it does, not just that it's accurate.

---

## Project Structure

```
retail-demand-forecasting/
├── data/
│   └── raw/                       # train.csv (not committed, download from Kaggle)
├── src/
│   ├── data/
│   │   ├── load.py                # Data loading + series selection
│   │   └── split.py                # Time-aware train/test split + expanding-window CV
│   ├── features/
│   │   └── engineer.py             # Lag, rolling, and calendar features
│   ├── models/
│   │   ├── prophet_model.py        # Classical decomposition approach
│   │   └── xgboost_model.py        # ML approach + recursive forecasting
│   ├── evaluation/
│   │   └── metrics.py              # MAE, RMSE, MAPE, forecast plots
│   └── pipeline.py                 # End-to-end orchestration
├── notebooks/
│   └── 01_eda_decomposition.ipynb  # Seasonality decomposition, day-of-week patterns
├── tests/
│   ├── test_split.py               # Verifies no chronological leakage
│   └── test_engineer.py
├── outputs/
│   ├── models/
│   └── figures/
└── requirements.txt
```

---

## Setup

```bash
git clone https://github.com/armanesh/retail-demand-forecasting.git
cd retail-demand-forecasting
pip install -r requirements.txt
pip install -e .
```

**Get the data:**
1. Download `train.csv` from [Kaggle](https://www.kaggle.com/competitions/demand-forecasting-kernels-only/data)
2. Place it in `data/raw/train.csv`

---

## Run the Pipeline

```bash
python -m src.pipeline
```

This will:
1. Load the dataset and select one representative store-item series
2. Split chronologically into train/test (last 90 days held out — never a random split)
3. Train Prophet on the raw series
4. Engineer lag/rolling/calendar features and train XGBoost
5. Evaluate both models (MAE, RMSE, MAPE) and save a comparison plot

---

## Run Tests

```bash
pytest tests/ -v
```

The test suite specifically verifies that splitting functions never leak future data into training — the single most common and costly mistake in time series projects.

---

## Key Design Choices

**Why a time-aware split, not `train_test_split`:** Randomly splitting a time series lets the model see "future" data points during training, which inflates validation accuracy in a way that doesn't hold up once deployed. Every split in this project holds out the most recent period chronologically, mimicking how the model will actually be used: trained on the past, evaluated on what comes next.

**Expanding-window cross-validation:** Rather than k-fold CV (which would shuffle time order), `expanding_window_splits` grows the training window forward through time, validating on each subsequent block. This gives a realistic estimate of how forecast accuracy degrades or holds up as more history accumulates.

**Recursive multi-step forecasting for XGBoost:** Lag features for day N+2 depend on the (unknown) value at day N+1. The `recursive_forecast` function handles this correctly by feeding each prediction back in as a lag feature for the next step, rather than naively using true future values that wouldn't be available at prediction time.

**Why compare two paradigms instead of just tuning one model:** Prophet and XGBoost fail differently and suit different situations. Prophet needs little feature engineering and produces an interpretable decomposition, which matters when forecasts need to be explained to non-technical stakeholders. XGBoost needs more setup but captures short-term autocorrelation Prophet smooths over, and scales naturally to forecasting many related series at once with shared features.

---

## Author

**Alex Rahbarimanesh** — Data Scientist & AI Engineer
[LinkedIn](https://linkedin.com/in/armanesh) · [GitHub](https://github.com/armanesh)
