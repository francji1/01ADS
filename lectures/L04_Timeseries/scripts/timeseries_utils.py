"""
Shared utilities for the bike sharing time-series forecasting scripts.

Run any of the standalone training scripts from the repository root, e.g.:
    python scripts/run_xgboost.py
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_URL = "https://raw.githubusercontent.com/francji1/01DAS/refs/heads/main/data/bike_share_hour3.csv"
RAW_COLUMNS_TO_DROP = ["casual", "registered", "instant", "date"]
CALENDAR_FEATURES = {
    "season",
    "holiday",
    "workingday",
    "weather",
    "hour",
    "dayofweek",
    "month",
    "weekofyear",
    "is_weekend",
}
CANDIDATE_CATEGORICALS = {
    "season",
    "holiday",
    "workingday",
    "weather",
    "hour",
    "dayofweek",
    "month",
    "weekofyear",
    "is_weekend",
}


def load_bike_data(url: str = DATA_URL) -> pd.DataFrame:
    """Load the bike sharing dataset and prepare a UTC hourly index."""

    df = pd.read_csv(url)
    if "datetime" in df.columns:
        dt_series = pd.to_datetime(df["datetime"], utc=True)
    elif {"date", "hour"}.issubset(df.columns):
        dt_series = pd.to_datetime(df["date"], utc=True) + pd.to_timedelta(df["hour"], unit="h")
    else:
        raise KeyError("Expected 'datetime' or ['date', 'hour'] columns in the dataset.")

    df["datetime"] = dt_series
    df = df.set_index("datetime").sort_index()
    rename_map = {"weathersit": "weather", "weekday": "dayofweek"}
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    drop_cols = [col for col in RAW_COLUMNS_TO_DROP if col in df.columns]
    df = df.drop(columns=drop_cols)
    df = df.asfreq("H")  # ensures a regular hourly grid
    df = df.interpolate(method="time").ffill().bfill()
    return df


def _add_calendar_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame["hour"] = frame.index.hour
    frame["dayofweek"] = frame.index.dayofweek
    frame["month"] = frame.index.month
    frame["weekofyear"] = frame.index.isocalendar().week.astype(int)
    frame["is_weekend"] = (frame["dayofweek"] >= 5).astype(int)
    return frame


def create_time_features(df: pd.DataFrame, horizon: int) -> Tuple[pd.DataFrame, List[str], str]:
    """
    Create lagged, seasonal, EMA, and calendar features for the specified horizon.
    """

    feat = df.copy()
    feat = _add_calendar_features(feat)
    feat["hour_sin"] = np.sin(2 * math.pi * feat["hour"] / 24)
    feat["hour_cos"] = np.cos(2 * math.pi * feat["hour"] / 24)

    for lag in [1, 2, 3, 4]:
        feat[f"lag_{lag}"] = feat["y"].shift(lag)
    feat["lag_24"] = feat["y"].shift(24)
    feat["lag_24_mean4"] = feat["y"].shift(24).rolling(window=4).mean()
    feat["lag_168"] = feat["y"].shift(168)
    feat["lag_168_mean4"] = feat["y"].shift(168).rolling(window=4).mean()

    for span in [6, 12, 24]:
        feat[f"ema_{span}"] = feat["y"].ewm(span=span, adjust=False).mean()

    feature_cols = sorted(
        set(CALENDAR_FEATURES)
        | {
            "hour_sin",
            "hour_cos",
            "lag_1",
            "lag_2",
            "lag_3",
            "lag_4",
            "lag_24",
            "lag_24_mean4",
            "lag_168",
            "lag_168_mean4",
            "ema_6",
            "ema_12",
            "ema_24",
            "temp",
            "atemp",
            "hum",
            "windspeed",
            "y",
        }
    )
    feature_cols = [col for col in feature_cols if col in feat.columns and col != "y"]

    target_col = f"target_t+{horizon}"
    feat[target_col] = feat["y"].shift(-horizon)
    feat = feat.dropna(subset=[target_col] + feature_cols)
    return feat, feature_cols, target_col


@dataclass
class SplitData:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: List[str]


def split_time_series(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> SplitData:
    """Chronologically split the feature frame into train/validation/test arrays."""

    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    X = df[feature_cols]
    y = df[target_col]

    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
    X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]

    return SplitData(
        X_train=X_train.values,
        y_train=y_train.values,
        X_val=X_val.values,
        y_val=y_val.values,
        X_test=X_test.values,
        y_test=y_test.values,
        feature_names=feature_cols,
    )


def evaluate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Return RMSE, MAE, and R^2 metrics."""

    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"rmse": rmse, "mae": mae, "r2": r2}


def get_categorical_columns(feature_names: Sequence[str]) -> List[str]:
    """Return the list of feature names to treat as categorical."""

    return [name for name in feature_names if name in CANDIDATE_CATEGORICALS]


def prepare_catboost_frame(
    array: np.ndarray,
    feature_names: Sequence[str],
    categorical_columns: Sequence[str],
) -> pd.DataFrame:
    """Convert a NumPy array to DataFrame, coercing categorical columns to integers."""

    df = pd.DataFrame(array, columns=feature_names)
    for col in categorical_columns:
        if col in df.columns:
            df[col] = df[col].round().astype(int)
    return df


def create_sequence_data(
    series: np.ndarray,
    window: int,
    horizons: Iterable[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Transform a univariate series into windowed samples suitable for sequence models."""

    horizons = list(horizons)
    max_h = max(horizons)
    X, y = [], []
    for i in range(len(series) - window - max_h + 1):
        X.append(series[i : i + window])
        y.append([series[i + window + h - 1] for h in horizons])
    return np.array(X), np.array(y)
