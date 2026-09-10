"""Run a fixed-horizon chronological benchmark, with training-only tuning."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import load_data, make_features


def score(actual, predicted):
    return {"RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
            "MAE": float(mean_absolute_error(actual, predicted)),
            "R2": float(r2_score(actual, predicted))}


def stable_cv_results(search):
    """Return model-selection evidence without machine-dependent timing data."""
    raw = pd.DataFrame(search.cv_results_)
    split_scores = sorted(
        column for column in raw if column.startswith("split") and column.endswith("_test_score")
    )
    stable = pd.DataFrame({
        "params": raw["params"].map(lambda value: json.dumps(value, sort_keys=True)),
    })
    for column in split_scores:
        stable[column.replace("_test_score", "_RMSE")] = -raw[column]
    stable["mean_RMSE"] = -raw["mean_test_score"]
    stable["std_RMSE"] = raw["std_test_score"]
    stable["rank"] = raw["rank_test_score"]
    return stable


def run(data: Path, output: Path, test_fraction=0.2, seed=42):
    if not 0.1 <= test_fraction <= 0.4:
        raise ValueError("test_fraction must be between 0.1 and 0.4")
    frame = load_data(data)
    X, y = make_features(frame)
    split = int(len(X) * (1 - test_fraction))
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    cv = list(TimeSeriesSplit(n_splits=5).split(X_train))
    candidates = {
        "Ridge": (make_pipeline(StandardScaler(), Ridge()),
                  {"ridge__alpha": np.logspace(-2, 3, 12).tolist()}),
        "RandomForest": (RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=1),
                         {"max_depth": [4, 8, None], "min_samples_leaf": [1, 3, 8]}),
    }
    predictions = pd.DataFrame({"Actual": y_test, "Persistence": X_test["effr_lag_1"]})
    baseline_cv = np.mean([np.sqrt(mean_squared_error(y_train.iloc[v], X_train.iloc[v]["effr_lag_1"])) for _, v in cv])
    rows = [{"Model": "Persistence", "CV_RMSE": float(baseline_cv), **score(y_test, predictions.Persistence)}]
    params = {}
    searches = {}
    for name, (estimator, grid) in candidates.items():
        search = GridSearchCV(estimator, grid, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=1)
        search.fit(X_train, y_train)
        searches[name] = search
        predictions[name] = search.predict(X_test)
        rows.append({"Model": name, "CV_RMSE": float(-search.best_score_), **score(y_test, predictions[name])})
        params[name] = search.best_params_
    metrics = pd.DataFrame(rows)
    selected = str(metrics.loc[metrics.CV_RMSE.idxmin(), "Model"])
    output.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output / "metrics.csv", index=False)
    predictions.to_csv(output / "predictions.csv", index_label="date")
    for name, search in searches.items():
        stable_cv_results(search).to_csv(output / f"{name.lower()}_cv.csv", index=False)
    importance = pd.Series(searches["RandomForest"].best_estimator_.feature_importances_, index=X.columns).sort_values()
    importance.to_csv(output / "feature_importance.csv", index_label="feature", header=["importance"])
    metadata = {
        "data_sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
        "data_source_series": "H15/H15/RIFSPFF_N.M",
        "python": platform.python_version(),
        "packages": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scikit-learn", "matplotlib")},
        "seed": seed, "test_fraction": test_fraction, "horizon_months": 1,
        "raw_rows": len(frame), "warmup_rows": 12, "train_rows": len(X_train), "test_rows": len(X_test),
        "train_start": str(y_train.index.min().date()), "train_end": str(y_train.index.max().date()),
        "test_start": str(y_test.index.min().date()), "test_end": str(y_test.index.max().date()),
        "selected_by_training_cv": selected, "best_parameters": params,
        "features": list(X.columns),
        "cv_folds": [{"train_end": str(X_train.index[t[-1]].date()), "validation_start": str(X_train.index[v[0]].date()), "validation_end": str(X_train.index[v[-1]].date())} for t, v in cv],
        "evaluation": "Fixed fitted models; rolling one-month forecasts using previous observed EFFR. Not recursive multi-month forecasting.",
    }
    (output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(11, 5))
    predictions.plot(ax=ax, linewidth=1.5)
    ax.set(title="One-month EFFR forecasts · chronological holdout", ylabel="EFFR (%)", xlabel="Target month")
    fig.tight_layout(); fig.savefig(output / "forecast.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5))
    importance.plot.barh(ax=ax, color="#426a9c")
    ax.set(title="Random Forest · training impurity importance", xlabel="Relative importance")
    fig.tight_layout(); fig.savefig(output / "feature_importance.png", dpi=160); plt.close(fig)
    print(metrics.to_string(index=False))
    print(f"Selected by training CV: {selected}; artifacts: {output.resolve()}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Path to monthly CSV")
    parser.add_argument("--output", type=Path, default=Path("reports"))
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(args.data, args.output, args.test_fraction, args.seed)


if __name__ == "__main__":
    main()
