"""
Classical forecasting approach using Prophet.
Prophet decomposes a series into trend + seasonality + holiday effects,
and handles missing data and outliers gracefully — well suited to series
with strong, regular seasonal patterns and limited history.
"""
import pandas as pd
from prophet import Prophet


def prepare_prophet_data(df: pd.DataFrame, date_col: str = "date", target_col: str = "sales") -> pd.DataFrame:
    """Prophet requires columns named exactly 'ds' and 'y'."""
    return df[[date_col, target_col]].rename(columns={date_col: "ds", target_col: "y"})


def train_prophet(train_df: pd.DataFrame, weekly_seasonality: bool = True, yearly_seasonality: bool = True) -> Prophet:
    """
    Fit a Prophet model.

    weekly_seasonality: captures day-of-week patterns (e.g. weekend spikes)
    yearly_seasonality: captures annual patterns (e.g. holiday season demand)
    """
    prophet_train = prepare_prophet_data(train_df)

    model = Prophet(
        weekly_seasonality=weekly_seasonality,
        yearly_seasonality=yearly_seasonality,
        daily_seasonality=False,
        changepoint_prior_scale=0.05,
    )
    model.fit(prophet_train)
    return model


def predict_prophet(model: Prophet, periods: int) -> pd.DataFrame:
    """Generate a forecast for the next `periods` days."""
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]
