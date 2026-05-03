from __future__ import annotations

from io import StringIO
from typing import Optional

import pandas as pd
import requests

EXPECTED_COLS = ["datetime", "load_mw", "zone", "temperature", "humidity", "wind_speed"]


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = {c: c.strip().lower() for c in df.columns}
    df = df.rename(columns=renamed)

    aliases = {
        "timestamp": "datetime",
        "time": "datetime",
        "load": "load_mw",
        "demand": "load_mw",
        "temp": "temperature",
        "wind": "wind_speed",
    }
    for old, new in aliases.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})

    return df


def load_csv(uploaded_file) -> pd.DataFrame:
    df = pd.read_csv(uploaded_file)
    return prepare_dataframe(df)


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = _normalize_columns(df.copy())

    for col in EXPECTED_COLS:
        if col not in df.columns:
            if col == "zone":
                df[col] = "PJM"
            else:
                df[col] = pd.NA

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)

    numeric_cols = ["load_mw", "temperature", "humidity", "wind_speed"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["zone"] = df["zone"].fillna("PJM").astype(str)

    # Fill load carefully for robustness
    df["load_mw"] = df["load_mw"].interpolate(limit_direction="both")
    df["load_mw"] = df["load_mw"].fillna(method="ffill").fillna(method="bfill")

    # Weather fields can be missing entirely; keep nullable and fill with medians if possible
    for col in ["temperature", "humidity", "wind_speed"]:
        if df[col].notna().any():
            df[col] = df[col].interpolate(limit_direction="both")
            df[col] = df[col].fillna(df[col].median())
        else:
            # Keep as NaN for downstream fallback handling
            df[col] = df[col].astype(float)

    return df


def load_from_url(url: str, timeout: int = 20) -> pd.DataFrame:
    """Optional lightweight loader for a CSV URL."""
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return prepare_dataframe(pd.read_csv(StringIO(resp.text)))


def default_sample_data(path: str = "sample_data.csv") -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(path)
        return prepare_dataframe(df)
    except Exception:
        return None
