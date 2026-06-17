"""
Forecast evaluation: MAE, RMSE, MAPE, and forecast-vs-actual visualization.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

FIGURES_DIR = Path(__file__).resolve().parents[2] / "outputs" / "figures"


def evaluate_forecast(y_true: np.ndarray, y_pred: np.ndarray, name: str = "Model") -> dict:
    """Compute standard forecasting accuracy metrics."""
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mape = np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1e-8, None))) * 100

    results = {"model": name, "MAE": mae, "RMSE": rmse, "MAPE": mape}
    print(f"\n{name} Forecast Accuracy:")
    print(f"  MAE:  {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  MAPE: {mape:.2f}%")
    return results


def plot_forecast_comparison(
    dates, actual, prophet_pred=None, xgboost_pred=None,
    title: str = "Forecast vs Actual", save_name: str = "forecast_comparison.png"
):
    """Plot actual sales against model forecasts for visual comparison."""
    plt.figure(figsize=(12, 5))
    plt.plot(dates, actual, label="Actual", color="black", linewidth=1.5)

    if prophet_pred is not None:
        plt.plot(dates, prophet_pred, label="Prophet", color="steelblue", linestyle="--")
    if xgboost_pred is not None:
        plt.plot(dates, xgboost_pred, label="XGBoost", color="coral", linestyle="--")

    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.title(title)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURES_DIR / save_name, dpi=150)
    print(f"Saved plot to {FIGURES_DIR / save_name}")
    plt.close()
