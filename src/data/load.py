"""
Data loading for the Store Item Demand Forecasting dataset.
Source: https://www.kaggle.com/competitions/demand-forecasting-kernels-only
Expected columns: date, store, item, sales
"""
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def load_raw(filename: str = "train.csv") -> pd.DataFrame:
    """Load raw sales data from data/raw/."""
    path = DATA_DIR / "raw" / filename
    df = pd.read_csv(path, parse_dates=["date"])
    print(f"Loaded {len(df):,} rows | {df['date'].min().date()} to {df['date'].max().date()}")
    return df


def filter_series(df: pd.DataFrame, store: int = 1, item: int = 1) -> pd.DataFrame:
    """
    Filter to a single store-item time series for focused modeling.
    The full dataset has many store/item combinations; this project
    demonstrates the methodology on one representative series.
    """
    series = df[(df["store"] == store) & (df["item"] == item)].copy()
    series = series.sort_values("date").reset_index(drop=True)
    print(f"Store {store}, Item {item}: {len(series)} daily observations")
    return series[["date", "sales"]]
