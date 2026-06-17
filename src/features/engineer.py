"""
Feature engineering for the XGBoost forecasting approach.
Converts a univariate time series into a supervised learning problem
using lag features, rolling statistics, and calendar features.
"""
import pandas as pd
import numpy as np

LAG_DAYS = [1, 7, 14, 28]
ROLLING_WINDOWS = [7, 14, 28]


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract calendar-based features from the date column."""
    df = df.copy()
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_month"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    return df


def add_lag_features(df: pd.DataFrame, target_col: str = "sales", lags: list = LAG_DAYS) -> pd.DataFrame:
    """
    Add lagged sales values as features.
    e.g. lag_7 = sales value exactly 7 days before the current row.
    """
    df = df.copy()
    for lag in lags:
        df[f"lag_{lag}"] = df[target_col].shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame, target_col: str = "sales", windows: list = ROLLING_WINDOWS) -> pd.DataFrame:
    """
    Add rolling mean/std features computed on lagged data (shifted by 1
    to avoid leaking the current day's value into its own features).
    """
    df = df.copy()
    shifted = df[target_col].shift(1)
    for window in windows:
        df[f"rolling_mean_{window}"] = shifted.rolling(window=window).mean()
        df[f"rolling_std_{window}"] = shifted.rolling(window=window).std()
    return df


def build_features(df: pd.DataFrame, target_col: str = "sales") -> pd.DataFrame:
    """Full feature engineering pipeline for the ML forecasting approach."""
    df = add_calendar_features(df)
    df = add_lag_features(df, target_col)
    df = add_rolling_features(df, target_col)

    # Drop rows with NaN from lag/rolling windows (start of series)
    df = df.dropna().reset_index(drop=True)
    return df


def get_feature_columns(df: pd.DataFrame, target_col: str = "sales", exclude: list = None) -> list:
    """Return the list of feature column names for model training."""
    exclude = exclude or []
    exclude_cols = ["date", target_col] + exclude
    return [c for c in df.columns if c not in exclude_cols]
