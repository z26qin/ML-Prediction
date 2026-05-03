from __future__ import annotations

import numpy as np
import pandas as pd


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["hour"] = out["datetime"].dt.hour
    out["day_of_week"] = out["datetime"].dt.dayofweek
    out["month"] = out["datetime"].dt.month
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)
    return out


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["lag_1h"] = out["load_mw"].shift(1)
    out["lag_24h"] = out["load_mw"].shift(24)
    out["lag_168h"] = out["load_mw"].shift(168)
    out["rolling_mean_24h"] = out["load_mw"].shift(1).rolling(window=24, min_periods=1).mean()
    return out


def add_temperature_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    temp = out.get("temperature")
    if temp is None:
        out["temperature"] = np.nan
        temp = out["temperature"]

    out["temp_sq"] = temp ** 2
    out["cooling_degree"] = np.maximum(temp - 65, 0)
    out["heating_degree"] = np.maximum(65 - temp, 0)
    return out


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = add_time_features(df)
    out = add_lag_features(out)
    out = add_temperature_features(out)

    # If weather columns are fully missing, safe-fill with neutral defaults
    for col, default in [("temperature", 65.0), ("humidity", 50.0), ("wind_speed", 5.0), ("temp_sq", 65.0**2), ("cooling_degree", 0.0), ("heating_degree", 0.0)]:
        if col in out.columns:
            out[col] = out[col].fillna(default)
        else:
            out[col] = default

    out = out.dropna(subset=["load_mw", "lag_1h", "lag_24h", "lag_168h"]).reset_index(drop=True)
    return out


def feature_columns() -> list[str]:
    return [
        "hour",
        "day_of_week",
        "month",
        "is_weekend",
        "lag_1h",
        "lag_24h",
        "lag_168h",
        "rolling_mean_24h",
        "temperature",
        "humidity",
        "wind_speed",
        "temp_sq",
        "cooling_degree",
        "heating_degree",
    ]
