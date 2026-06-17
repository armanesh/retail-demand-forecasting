"""Unit tests for feature engineering."""
import pandas as pd
import numpy as np
import pytest
from src.features.engineer import add_calendar_features, add_lag_features, add_rolling_features, build_features


@pytest.fixture
def sample_series():
    dates = pd.date_range("2020-01-01", periods=60, freq="D")
    return pd.DataFrame({"date": dates, "sales": np.arange(60, dtype=float)})


def test_calendar_features_added(sample_series):
    result = add_calendar_features(sample_series)
    assert "day_of_week" in result.columns
    assert "is_weekend" in result.columns
    assert result["day_of_week"].between(0, 6).all()


def test_lag_features_shift_correctly(sample_series):
    result = add_lag_features(sample_series, lags=[1])
    # lag_1 at row i should equal sales at row i-1
    assert result["lag_1"].iloc[5] == sample_series["sales"].iloc[4]


def test_rolling_features_use_shifted_data(sample_series):
    result = add_rolling_features(sample_series, windows=[7])
    # rolling mean should not include the current row's own value
    assert pd.isna(result["rolling_mean_7"].iloc[0])


def test_build_features_drops_na_rows(sample_series):
    result = build_features(sample_series)
    assert not result.isnull().any().any()
    assert len(result) < len(sample_series)
