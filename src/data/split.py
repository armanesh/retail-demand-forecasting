"""
Time-aware train/test splitting and cross-validation.

CRITICAL: forecasting data must never be split randomly. Random splits leak
future information into training (the model would learn from "future" rows
when predicting "past" ones), producing artificially inflated accuracy that
collapses in production. All splits here respect chronological order.
"""
import pandas as pd
from typing import List, Tuple


def train_test_split_by_date(df: pd.DataFrame, test_days: int = 90) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a time series into train/test by holding out the last `test_days`
    days as the test set. This mimics real deployment: train on the past,
    evaluate on what was actually the future at training time.
    """
    df = df.sort_values("date").reset_index(drop=True)
    split_date = df["date"].max() - pd.Timedelta(days=test_days)

    train = df[df["date"] <= split_date].reset_index(drop=True)
    test = df[df["date"] > split_date].reset_index(drop=True)

    print(f"Train: {len(train)} days ({train['date'].min().date()} to {train['date'].max().date()})")
    print(f"Test:  {len(test)} days ({test['date'].min().date()} to {test['date'].max().date()})")
    return train, test


def expanding_window_splits(df: pd.DataFrame, n_splits: int = 5, test_days: int = 30) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Generate expanding-window time series cross-validation splits.

    Unlike sklearn's KFold (which shuffles and leaks future data), each fold
    here trains on all data up to a point and validates on the immediately
    following window, then the training window expands forward in time.
    This is the correct way to cross-validate a forecasting model.
    """
    df = df.sort_values("date").reset_index(drop=True)
    n = len(df)
    splits = []

    fold_size = test_days
    min_train_size = n - (n_splits * fold_size)

    if min_train_size < fold_size:
        raise ValueError("Not enough data for the requested number of splits/test_days.")

    for i in range(n_splits):
        train_end = min_train_size + i * fold_size
        test_end = train_end + fold_size

        train_fold = df.iloc[:train_end]
        test_fold = df.iloc[train_end:test_end]
        splits.append((train_fold, test_fold))

    return splits
