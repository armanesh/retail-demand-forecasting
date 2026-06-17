"""
End-to-end pipeline: load -> split -> train (Prophet + XGBoost) -> evaluate -> compare.
Run with: python -m src.pipeline
"""
import pandas as pd

from src.data.load import load_raw, filter_series
from src.data.split import train_test_split_by_date
from src.features.engineer import build_features, get_feature_columns
from src.models.prophet_model import train_prophet, predict_prophet
from src.models.xgboost_model import train_xgboost, save_model
from src.evaluation.metrics import evaluate_forecast, plot_forecast_comparison


def main():
    print("=== Retail Demand Forecasting Pipeline ===\n")

    # 1. Load and select a representative series
    raw_df = load_raw()
    series = filter_series(raw_df, store=1, item=1)

    # 2. Time-aware split (never random for time series)
    train_df, test_df = train_test_split_by_date(series, test_days=90)

    # 3. Prophet (classical approach)
    print("\n--- Training Prophet ---")
    prophet_model = train_prophet(train_df)
    prophet_forecast = predict_prophet(prophet_model, periods=len(test_df))
    prophet_test_pred = prophet_forecast.tail(len(test_df))["yhat"].values

    prophet_metrics = evaluate_forecast(test_df["sales"].values, prophet_test_pred, name="Prophet")

    # 4. XGBoost (ML approach with engineered features)
    print("\n--- Training XGBoost ---")
    featured = build_features(series)
    feature_cols = get_feature_columns(featured)

    train_feat = featured[featured["date"] <= train_df["date"].max()]
    test_feat = featured[featured["date"] > train_df["date"].max()]

    xgb_model = train_xgboost(train_feat[feature_cols], train_feat["sales"])
    xgb_pred = xgb_model.predict(test_feat[feature_cols])

    xgb_metrics = evaluate_forecast(test_feat["sales"].values, xgb_pred, name="XGBoost")

    # 5. Compare
    print("\n--- Model Comparison ---")
    comparison = pd.DataFrame([prophet_metrics, xgb_metrics]).set_index("model")
    print(comparison.to_string())

    plot_forecast_comparison(
        dates=test_df["date"].values[:len(xgb_pred)],
        actual=test_df["sales"].values[:len(xgb_pred)],
        prophet_pred=prophet_test_pred[:len(xgb_pred)],
        xgboost_pred=xgb_pred,
    )

    save_model(xgb_model)
    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
