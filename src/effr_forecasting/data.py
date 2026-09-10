"""Validate monthly observations and construct explicitly past-only features."""
from pathlib import Path
import numpy as np
import pandas as pd

TARGET = "Effective Federal Funds Rate"


def load_data(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if not {"date", TARGET}.issubset(frame.columns):
        raise ValueError(f"CSV must contain date and {TARGET}")
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    frame = frame.sort_values("date").set_index("date")
    months = frame.index.to_period("M")
    if months.has_duplicates or frame.index.hasnans:
        raise ValueError("Dates must be nonmissing and unique by month")
    if len(frame) < 60 or not months.equals(pd.period_range(months.min(), months.max(), freq="M")):
        raise ValueError("At least 60 consecutive monthly observations are required")
    frame[TARGET] = pd.to_numeric(frame[TARGET], errors="raise")
    if not np.isfinite(frame[TARGET]).all():
        raise ValueError("EFFR observations must be finite; targets are never imputed")
    return frame


def make_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """For target month t, use only EFFR observations through t-1.

    The first 12 months are a fixed warm-up. No macro columns are used:
    their release timing, revisions and original interpolation are unverifiable.
    """
    y = frame[TARGET]
    features = pd.DataFrame(index=frame.index)
    for lag in (1, 2, 3, 6, 12):
        features[f"effr_lag_{lag}"] = y.shift(lag)
    past = y.shift(1)
    for window in (3, 6, 12):
        features[f"effr_mean_{window}"] = past.rolling(window).mean()
        features[f"effr_std_{window}"] = past.rolling(window).std()
    features["effr_change"] = y.diff().shift(1)
    return features.iloc[12:], y.iloc[12:]
