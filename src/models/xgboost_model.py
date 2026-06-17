"""
ML forecasting approach using XGBoost on engineered lag/calendar features.
Treats forecasting as a supervised regression problem — typically stronger
than classical methods when there are multiple related series or rich
exogenous features, though less interpretable than Prophet's decomposition.
"""
import joblib
import numpy as np
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "models"


def train_xgboost(X_train, y_train, params: dict = None) -> XGBRegressor:
    """Train an XGBoost regressor for demand forecasting."""
    default_params = {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "tree_method": "hist",
    }
    if params:
        default_params.update(params)

    model = XGBRegressor(**default_params)
    model.fit(X_train, y_train)
    return model


def recursive_forecast(model: XGBRegressor, history_df, feature_fn, horizon: int, target_col: str = "sales"):
    """
    Generate a multi-step forecast by recursively feeding predictions back in
    as lag features for the next step. This is necessary because lag features
    for future dates depend on values the model itself is predicting.
    """
    import pandas as pd

    extended = history_df.copy()
    predictions = []

    last_date = extended["date"].max()

    for step in range(horizon):
        next_date = last_date + pd.Timedelta(days=step + 1)
        new_row = pd.DataFrame({"date": [next_date], target_col: [np.nan]})
        extended = pd.concat([extended, new_row], ignore_index=True)

        features_df = feature_fn(extended.fillna(method="ffill"))
        feature_cols = [c for c in features_df.columns if c not in ["date", target_col]]
        last_features = features_df.iloc[[-1]][feature_cols]

        pred = model.predict(last_features)[0]
        extended.loc[extended["date"] == next_date, target_col] = pred
        predictions.append({"date": next_date, "yhat": pred})

    return pd.DataFrame(predictions)


def save_model(model, name: str = "xgboost_forecast"):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{name}.joblib"
    joblib.dump(model, path)
    print(f"Saved model to {path}")
    return path


def load_model(name: str = "xgboost_forecast"):
    return joblib.load(OUTPUT_DIR / f"{name}.joblib")
