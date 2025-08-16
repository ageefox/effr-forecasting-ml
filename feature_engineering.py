# training the model
# === Feature expansion for EFFR modeling (robust, no-leakage) ===
# - Builds on your existing script
# - Creates lags, rolling stats, change/momentum, seasonality, and safe interactions
# - Only uses columns that actually exist in your file
#
# Output: features_effr_data_extended.csv

# working with dataframes
import pandas as pd
# importing numpy for numerical operations
import numpy as np
from itertools import combinations

# ---------- Load ----------
# loading the dataset
df = pd.read_csv("cleaned_effr_data.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

# ---------- Helper ----------
def cols_present(candidates):
    return [c for c in candidates if c in df.columns]

# Core columns you already have
base_cols = cols_present(["CPI", "Inflation Rate", "Unemployment Rate", "Real GDP (Percent Change)"])

# If your dataset also contains any of these, they’ll be used automatically:
optional_cols = cols_present([
    "SOFR", "2Y Yield", "10Y Yield", "Federal Funds Target Rate",
    "Federal Funds Upper Target", "Federal Funds Lower Target",
    "Industrial Production", "ISM Manufacturing PMI", "ISM Services PMI",
    "Retail Sales", "PCE", "Core PCE", "PPI"
])

use_cols = base_cols + optional_cols
if not use_cols:
    raise ValueError("No known numeric macro columns found. Check column names in cleaned_effr_data.csv")

# Align numeric-only features (avoid object columns sneaking in)
for c in use_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# To be conservative about data availability (and reporting lags), we shift most derived features by 1 period
SHIFT = 1

# ---------- 1) Lags ----------
lags = [1, 3, 6, 12]
for col in use_cols:
    for L in lags:
        df[f"{col}_lag{L}"] = df[col].shift(L)

# ---------- 2) Rolling means / volatility (past-only) ----------
windows = [3, 6, 12]
for col in use_cols:
    s = df[col]
    for w in windows:
        df[f"{col}_ma{w}"]  = s.rolling(w, min_periods=w).mean().shift(SHIFT)
        df[f"{col}_std{w}"] = s.rolling(w, min_periods=w).std().shift(SHIFT)

# ---------- 3) Change-based features (momentum) ----------
for col in use_cols:
    s = df[col]
    # Level changes
    df[f"{col}_diff1"]  = s.diff(1).shift(SHIFT)     # MoM change, shifted
    df[f"{col}_diff12"] = s.diff(12).shift(SHIFT)    # YoY change, shifted
    # Percent changes (safe with small eps)
    eps = 1e-9
    df[f"{col}_pct1"]  = (s.pct_change(1)).shift(SHIFT)
    df[f"{col}_pct12"] = (s.pct_change(12)).shift(SHIFT)
    # Momentum: short MA minus long MA
    df[f"{col}_mom_ma3_12"] = (s.rolling(3, min_periods=3).mean()
                                - s.rolling(12, min_periods=12).mean()).shift(SHIFT)

# ---------- 4) Rolling z-scores (level vs local mean) ----------
for col in use_cols:
    roll_mean = df[col].rolling(12, min_periods=12).mean()
    roll_std  = df[col].rolling(12, min_periods=12).std()
    df[f"{col}_z12"] = ((df[col] - roll_mean) / (roll_std.replace(0, np.nan))).shift(SHIFT)

# ---------- 5) Safe interactions (lagged to avoid leakage) ----------
# Interact a small subset to control dimensionality:
interaction_pool = cols_present(["CPI", "Inflation Rate", "Unemployment Rate"]) or use_cols[:3]
for a, b in combinations(interaction_pool, 2):
    df[f"{a}_x_{b}"] = (df[a].shift(SHIFT) * df[b].shift(SHIFT))

# ---------- 6) Seasonality & calendar dummies ----------
df["month"]   = df["date"].dt.month
df["quarter"] = df["date"].dt.quarter
# Cyclical encoding
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)
# Quarter-end liquidity pressures
df["is_qtr_end"] = (df["date"].dt.month.isin([3, 6, 9, 12])).astype(int)
# Year-end dummy
df["is_december"] = (df["date"].dt.month == 12).astype(int)

# ---------- 7) Yield curve (if yields available) ----------
if {"10Y Yield", "2Y Yield"}.issubset(df.columns):
    df["yc_10y_2y"] = (pd.to_numeric(df["10Y Yield"], errors="coerce")
                       - pd.to_numeric(df["2Y Yield"], errors="coerce")).shift(SHIFT)

# ---------- Finalize ----------
# Keep rows where at least one of our engineered features exists and target is present
target_col = "Effective Federal Funds Rate"
if target_col not in df.columns:
    raise ValueError("Target column 'Effective Federal Funds Rate' not found in cleaned_effr_data.csv")

# Build a list of engineered columns to enforce non-NA rows
engineered_prefixes = [
    "_lag", "_ma", "_std", "_diff", "_pct", "_mom_ma3_12", "_z12", "_x_", "month_", "yc_10y_2y"
]
engineered_cols = [c for c in df.columns if any(p in c for p in engineered_prefixes)]
# Also include simple calendar flags
engineered_cols += ["month", "quarter", "is_qtr_end", "is_december"]

# Drop rows with NA from engineered features or target
df_out = df.dropna(subset=engineered_cols + [target_col]).reset_index(drop=True)

# Save
out_path = "features_effr_data_extended.csv"
df_out.to_csv(out_path, index=False)

print(f"Saved {out_path} with {len(df_out)} rows and {df_out.shape[1]} columns.")
print(f"Engineered {len(engineered_cols)} feature columns (plus calendar features).")

