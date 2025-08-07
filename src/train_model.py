# working with dataframes
import pandas as pd
# importing numpy for numerical operations
import numpy as np
# training the model
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

# ---------------- Config ----------------
IN_PATH  = "features_effr_data_extended.csv"
TARGET   = "Effective Federal Funds Rate"
TOP_N    = 25  # change as needed
OUT_IMPORTANCES = "feature_importances_full.csv"
OUT_TOP_RAW     = f"features_effr_top{TOP_N}.csv"
OUT_TOP_CLEAN   = f"features_effr_top{TOP_N}_clean.csv"

# ---------------- Load ----------------
# loading the dataset
df = pd.read_csv(IN_PATH)
if TARGET not in df.columns:
    raise ValueError(f"Target column '{TARGET}' not found in {IN_PATH}")

# Keep only numeric features for ranking
exclude = ["date", TARGET]
X_raw = df.drop(columns=[c for c in exclude if c in df.columns], errors="ignore").select_dtypes(include=[np.number])
y = pd.to_numeric(df[TARGET], errors="coerce")

# Align and drop rows with missing target
mask = y.notna()
X_raw, y = X_raw.loc[mask], y.loc[mask]

# training the model
# ---------------- Clean numeric X for modeling ----------------
# 1) Replace +/-inf with NaN
X = X_raw.replace([np.inf, -np.inf], np.nan)

# 2) Optional: clip extreme outliers per column (0.1%–99.9%) before imputation
q_low  = X.quantile(0.001)
q_high = X.quantile(0.999)
X = X.clip(lower=q_low, upper=q_high, axis=1)

# 3) Median impute remaining NaNs
imp = SimpleImputer(strategy="median")
X_imp = pd.DataFrame(imp.fit_transform(X), columns=X.columns, index=X.index).astype(np.float64)

# 4) Drop zero-variance columns (if any)
var = X_imp.var(axis=0)
keep_cols = var[var > 0].index.tolist()
X_imp = X_imp[keep_cols]

# Safety check: ensure finiteness
if not np.isfinite(X_imp.to_numpy()).all():
    raise ValueError("Non-finite values remain after cleaning. Inspect the source data.")

# ---------------- Rank features with RF ----------------
# training the model
rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_imp, y)
importances = pd.Series(rf.feature_importances_, index=X_imp.columns).sort_values(ascending=False)
importances.to_csv(OUT_IMPORTANCES, header=["importance"])
print(f"Saved feature importances to {OUT_IMPORTANCES}")

# ---------------- Select top N & save datasets ----------------
top_features = importances.head(TOP_N).index.tolist()
print(f"Top {TOP_N} features:\n{top_features}")

# RAW version (original values)
cols_raw = [c for c in ["date", TARGET] if c in df.columns] + top_features
df_top_raw = df.loc[mask, cols_raw]  # same rows as used for ranking
df_top_raw.to_csv(OUT_TOP_RAW, index=False)
print(f"Saved raw top-{TOP_N} dataset to {OUT_TOP_RAW}")

# training the model
# CLEAN version (imputed, clipped, finite) — ready for modeling
# Rebuild a clean frame with date/target plus cleaned top features
df_top_clean = pd.DataFrame(index=X_imp.index)
if "date" in df.columns:
    df_top_clean["date"] = pd.to_datetime(df.loc[mask, "date"]).values
df_top_clean[TARGET] = y.values
df_top_clean[top_features] = X_imp[top_features].values
df_top_clean.to_csv(OUT_TOP_CLEAN, index=False)
print(f"Saved CLEAN top-{TOP_N} dataset to {OUT_TOP_CLEAN}")

# Train & compare RF / LightGBM / CatBoost, with XGBoost as a best‑effort optional add.
# Designed to work even with very old xgboost builds (no eval_metric / no early_stopping in fit).

# working with dataframes
import pandas as pd
# importing numpy for numerical operations
import numpy as np
from pathlib import Path

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# training the model
from sklearn.ensemble import RandomForestRegressor

# ---------------- Optional libs (skip silently if missing) ----------------
HAS_XGB = True
HAS_LGBM = True
HAS_CAT  = True
xgb_ver = None
try:
    import xgboost as xgb
    xgb_ver = getattr(xgb, "__version__", "unknown")
except Exception:
    HAS_XGB = False
try:
    import lightgbm as lgb
except Exception:
    HAS_LGBM = False
try:
    from catboost import CatBoostRegressor, Pool
except Exception:
    HAS_CAT = False

# ---------------- Load & prep ----------------
IN_PATH = "features_effr_top25_clean.csv"   # use the CLEAN file you created earlier
TARGET  = "Effective Federal Funds Rate"

# loading the dataset
df = pd.read_csv(IN_PATH)
if TARGET not in df.columns:
    raise ValueError(f"Target '{TARGET}' not found in {IN_PATH}")

# Keep only numeric features (drop date if present)
exclude = ["date", TARGET]
X = df.drop(columns=[c for c in exclude if c in df.columns], errors="ignore").select_dtypes(include=[np.number])
y = pd.to_numeric(df[TARGET], errors="coerce")

# Safety: drop any remaining NaN/Inf rows in target or features
mask = y.notna()
X, y = X.loc[mask], y.loc[mask]
finite_mask = np.isfinite(X.values).all(axis=1)
X, y = X.loc[finite_mask], y.loc[finite_mask]

# Time-based split 80/20
n = len(X)
split_idx = int(n * 0.80)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# Small validation tail from train (for early stopping where supported)
val_size = max(1, int(len(X_train) * 0.10))
X_tr, X_val = X_train.iloc[:-val_size], X_train.iloc[-val_size:]
y_tr, y_val = y_train.iloc[:-val_size], y_train.iloc[-val_size:]

def eval_preds(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {"RMSE": float(np.sqrt(mse)),
            "MAE":  float(mean_absolute_error(y_true, y_pred)),
            "R2":   float(r2_score(y_true, y_pred))}

metrics = []
preds = pd.DataFrame(index=X_test.index)
preds["Actual"] = y_test.values

# ---------------- Random Forest ----------------
# training the model
rf = RandomForestRegressor(
    n_estimators=600,
    max_depth=12,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
# making predictions
rf_pred = rf.predict(X_test)
# training the model
m = eval_preds(y_test, rf_pred); m["Model"] = "RandomForest"
metrics.append(m)
preds["RF_Pred"] = rf_pred
pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)\
  .to_csv("rf_importance.csv", header=["importance"])

# ---------------- XGBoost (best-effort, fully version-safe) ----------------
if HAS_XGB:
    # Keep params minimal to maximize compatibility with older versions
    xgbr = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=2000,
        learning_rate=0.01,
        max_depth=5,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42
    )

    # Try with eval_set + early stopping; progressively fall back if unsupported
    trained = False
    try:
        # Some old versions reject eval_metric / early_stopping_rounds / verbose in fit(),
        # so DO NOT pass eval_metric; only pass eval_set & early_stopping_rounds first.
        xgbr.fit(X_tr, y_tr,
                 eval_set=[(X_val, y_val)],
                 early_stopping_rounds=200)
        trained = True
    except TypeError:
        try:
            # No early stopping
            xgbr.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])
            trained = True
        except TypeError:
            # Plain fit on all train
            xgbr.fit(X_train, y_train)
            trained = True

    if trained:
# making predictions
        xgb_pred = xgbr.predict(X_test)
        m = eval_preds(y_test, xgb_pred); m["Model"] = f"XGBoost({xgb_ver})"
        metrics.append(m)
        preds["XGB_Pred"] = xgb_pred
        # importance (fallback if attribute missing)
        try:
            imp = pd.Series(xgbr.feature_importances_, index=X.columns)
        except Exception:
            try:
                booster = xgbr.get_booster()
                score = booster.get_score(importance_type="weight")
                imp = pd.Series({col: score.get(f"f{idx}", 0.0) for idx, col in enumerate(X.columns)})
            except Exception:
                imp = pd.Series(dtype=float)
        if not imp.empty:
            imp.sort_values(ascending=False).to_csv("xgb_importance.csv", header=["importance"])

# ---------------- LightGBM ----------------
if HAS_LGBM:
    lgbm = lgb.LGBMRegressor(
        n_estimators=4000,
        learning_rate=0.01,
        num_leaves=96,
        min_data_in_leaf=10,
        feature_fraction=0.9,
        bagging_fraction=0.9,
        bagging_freq=1,
        random_state=42
    )
    # Early stopping via callbacks (works on old & new LightGBM)
    try:
        lgbm.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            eval_metric="rmse",
            callbacks=[lgb.early_stopping(300, verbose=False)]
        )
    except TypeError:
        lgbm.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            eval_metric="rmse",
            callbacks=[lgb.early_stopping(300)]
        )
# making predictions
    lgb_pred = lgbm.predict(X_test)
    m = eval_preds(y_test, lgb_pred); m["Model"] = "LightGBM"
    metrics.append(m)
    preds["LGBM_Pred"] = lgb_pred
    pd.Series(lgbm.feature_importances_, index=X.columns).sort_values(ascending=False)\
      .to_csv("lgbm_importance.csv", header=["importance"])

# ---------------- CatBoost ----------------
if HAS_CAT:
    cat = CatBoostRegressor(
        iterations=4000,
        learning_rate=0.01,
        depth=6,
        l2_leaf_reg=3.0,
        loss_function="RMSE",
        random_seed=42,
        verbose=False
    )
    # Older CatBoost accepts early_stopping_rounds; if not, fallback
    try:
        cat.fit(
            Pool(X_tr, y_tr),
            eval_set=Pool(X_val, y_val),
# training the model
            use_best_model=True,
            early_stopping_rounds=300
        )
    except TypeError:
# training the model
        cat.fit(Pool(X_tr, y_tr), eval_set=Pool(X_val, y_val), use_best_model=True)
# making predictions
    cat_pred = cat.predict(X_test)
    m = eval_preds(y_test, cat_pred); m["Model"] = "CatBoost"
    metrics.append(m)
    preds["CAT_Pred"] = cat_pred
    try:
        fi = cat.get_feature_importance(Pool(X_tr, y_tr))
        pd.Series(fi, index=X.columns).sort_values(ascending=False)\
          .to_csv("cat_importance.csv", header=["importance"])
    except Exception:
        pass

# ---------------- Save & print summary ----------------
metrics_df = pd.DataFrame(metrics).set_index("Model").sort_values("RMSE")
# training the model
metrics_df.to_csv("model_compare_metrics.csv")
# training the model
preds.to_csv("model_compare_predictions.csv", index=False)

print("\n=== Test Set Metrics ===")
# training the model
print(metrics_df if not metrics_df.empty else "No models ran (check installs).")
print("\nSaved:")
# training the model
print(" - model_compare_metrics.csv")
# training the model
print(" - model_compare_predictions.csv")
for p in ["rf_importance.csv", "xgb_importance.csv", "lgbm_importance.csv", "cat_importance.csv"]:
    if Path(p).exists():
        print(f" - {p}")

# ---- BEST MODEL
# 
# 
# # Random Forest tuned for small dataset (no early stopping anywhere)
# - Loads features_effr_top25_clean.csv (falls back to features_effr_top25.csv)
# - TimeSeries CV + RandomizedSearch over depth/leaves/max_features with many trees
# - 80/20 time-based split for final test
# making predictions
# - Saves metrics, predictions, best params, and importance; plots actual vs predicted

import json
# importing numpy for numerical operations
import numpy as np
# working with dataframes
import pandas as pd
# for plotting and visualizing trends
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# training the model
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
# training the model
from sklearn.ensemble import RandomForestRegressor

RANDOM_STATE = 42
IN_PATHS = ["features_effr_top25_clean.csv", "features_effr_top25.csv",
            "features_effr_data_extended.csv", "features_effr_data.csv"]
TARGET = "Effective Federal Funds Rate"

# ---------- Load ----------
for p in IN_PATHS:
    if Path(p).exists():
# loading the dataset
        df = pd.read_csv(p)
        print(f"Loaded: {p}")
        break
else:
    raise FileNotFoundError("Could not find any features CSV.")

if TARGET not in df.columns:
    raise ValueError(f"Target '{TARGET}' not in dataframe.")

# Keep numeric features only; drop 'date' if present
exclude = ["date", TARGET]
X = df.drop(columns=[c for c in exclude if c in df.columns], errors="ignore").select_dtypes(include=[np.number])
y = pd.to_numeric(df[TARGET], errors="coerce")
mask = y.notna()
X, y = X.loc[mask], y.loc[mask]

# Safety: finite values only
X = X.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
y = y.loc[X.index]

# ---------- Time-based split ----------
n = len(X)
split_idx = int(n * 0.80)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

print(f"Train size: {len(X_train)}, Test size: {len(X_test)}, Features: {X.shape[1]}")

# ---------- CV setup ----------
tscv = TimeSeriesSplit(n_splits=5)

# ---------- Search space ----------
param_dist = {
    "n_estimators": np.arange(800, 3001, 200),     # many trees for stability on small data
    "max_depth": np.append(np.arange(4, 25, 2), [None]),
    "min_samples_leaf": np.arange(1, 8),           # smaller leaves capture nonlinearity
    "min_samples_split": np.arange(2, 10),
    "max_features": ["sqrt", "log2", 0.5, 0.7, None],
    "bootstrap": [True],                           # OOB-style sampling tends to help
}

# training the model
rf = RandomForestRegressor(
    random_state=RANDOM_STATE,
    n_jobs=-1
)

search = RandomizedSearchCV(
    rf,
    param_distributions=param_dist,
    n_iter=120,                # bump if you want more thorough search
    cv=tscv,
    scoring="neg_mean_squared_error",
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=1
)

search.fit(X_train, y_train)
best_rf = search.best_estimator_
print("\nBest RF params:")
print(search.best_params_)

# ---------- Evaluate on test ----------
# making predictions
pred = best_rf.predict(X_test)

rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
mae  = float(mean_absolute_error(y_test, pred))
r2   = float(r2_score(y_test, pred))

# training the model
print("\n=== Test Set (RandomForest Tuned) ===")
print(f"RMSE: {rmse:.4f}  MAE: {mae:.4f}  R2: {r2:.4f}")

# ---------- Save artifacts ----------
# training the model
metrics = {"Model": "RandomForest_Tuned",
           "RMSE": rmse, "MAE": mae, "R2": r2}
pd.DataFrame([metrics]).to_csv("rf_tuned_metrics.csv", index=False)

preds = pd.DataFrame({
    "Actual": y_test.values,
    "RF_Tuned_Pred": pred
}, index=X_test.index).reset_index(drop=True)
# making predictions
preds.to_csv("rf_tuned_predictions.csv", index=False)

imp = pd.Series(best_rf.feature_importances_, index=X.columns).sort_values(ascending=False)
imp.to_csv("rf_tuned_feature_importance.csv", header=["importance"])

with open("rf_tuned_best_params.json", "w") as f:
    # ensure JSON-serializable types
    bp = {k: (int(v) if isinstance(v, (np.integer,)) else float(v) if isinstance(v, (np.floating,))
              else (None if v is None else v))
          for k, v in search.best_params_.items()}
    json.dump(bp, f, indent=2)

print("\nSaved:")
print(" - rf_tuned_metrics.csv")
# making predictions
print(" - rf_tuned_predictions.csv")
print(" - rf_tuned_feature_importance.csv")
print(" - rf_tuned_best_params.json")

# ---------- Plot Actual vs Predicted ----------
plt.figure(figsize=(8, 6))
plt.scatter(preds["Actual"], preds["RF_Tuned_Pred"], alpha=0.65)
a_min, a_max = preds["Actual"].min(), preds["Actual"].max()
# making predictions
plt.plot([a_min, a_max], [a_min, a_max], linestyle="--", label="Perfect prediction")
plt.xlabel("Actual EFFR")
plt.ylabel("Predicted EFFR (RF Tuned)")
# training the model
plt.title("Actual vs Predicted EFFR — RandomForest (Tuned)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

