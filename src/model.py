from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.features import feature_columns


@dataclass
class ForecastArtifacts:
    model: RandomForestRegressor
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    y_pred: np.ndarray
    baseline_24h: np.ndarray
    baseline_168h: np.ndarray


def train_and_forecast(df_feat: pd.DataFrame, horizon_hours: int = 24, random_state: int = 42) -> ForecastArtifacts:
    n_test = max(horizon_hours, int(len(df_feat) * 0.2))
    n_test = min(n_test, len(df_feat) - 1)

    train_df = df_feat.iloc[:-n_test].copy()
    test_df = df_feat.iloc[-n_test:].copy()

    X_train = train_df[feature_columns()]
    y_train = train_df["load_mw"]
    X_test = test_df[feature_columns()]

    model = RandomForestRegressor(
        n_estimators=250,
        max_depth=12,
        random_state=random_state,
        n_jobs=-1,
        min_samples_leaf=2,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    baseline_24h = test_df["lag_24h"].values
    baseline_168h = test_df["lag_168h"].values

    return ForecastArtifacts(
        model=model,
        train_df=train_df,
        test_df=test_df,
        y_pred=y_pred,
        baseline_24h=baseline_24h,
        baseline_168h=baseline_168h,
    )


def scenario_temperature_adjustment(test_df: pd.DataFrame, model: RandomForestRegressor, delta_temp: float) -> np.ndarray:
    scen = test_df.copy()
    scen["temperature"] = scen["temperature"] + delta_temp
    scen["temp_sq"] = scen["temperature"] ** 2
    scen["cooling_degree"] = np.maximum(scen["temperature"] - 65, 0)
    scen["heating_degree"] = np.maximum(65 - scen["temperature"], 0)
    return model.predict(scen[feature_columns()])
