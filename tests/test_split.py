"""Unit tests for time-aware splitting — the most critical correctness check
in this project, since a leaky split would invalidate every downstream result."""
import pandas as pd
import pytest
from src.data.split import train_test_split_by_date, expanding_window_splits


@pytest.fixture
def sample_series():
    dates = pd.date_range("2020-01-01", periods=200, freq="D")
    return pd.DataFrame({"date": dates, "sales": range(200)})


def test_split_respects_chronological_order(sample_series):
    train, test = train_test_split_by_date(sample_series, test_days=30)
    assert train["date"].max() < test["date"].min()


def test_split_sizes(sample_series):
    train, test = train_test_split_by_date(sample_series, test_days=30)
    assert len(test) == 30
    assert len(train) == 170


def test_no_overlap_between_train_and_test(sample_series):
    train, test = train_test_split_by_date(sample_series, test_days=30)
    overlap = set(train["date"]) & set(test["date"])
    assert len(overlap) == 0


def test_expanding_window_splits_are_chronological(sample_series):
    splits = expanding_window_splits(sample_series, n_splits=3, test_days=20)
    for train_fold, test_fold in splits:
        assert train_fold["date"].max() < test_fold["date"].min()


def test_expanding_window_train_size_grows(sample_series):
    splits = expanding_window_splits(sample_series, n_splits=3, test_days=20)
    train_sizes = [len(train_fold) for train_fold, _ in splits]
    assert train_sizes == sorted(train_sizes)
    assert train_sizes[0] < train_sizes[-1]
