"""Evaluate the frozen forecasting design on post-February 2017 data."""
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
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import TARGET, load_data, make_features
from .train import score


def load_frozen_design(path: Path) -> dict:
    design = json.loads(path.read_text())
    required = {"features", "best_parameters", "selected_by_training_cv"}
    if not required.issubset(design):
        raise ValueError(f"{path} does not contain the recorded model design")
    if design["selected_by_training_cv"] != "Persistence":
        raise ValueError("The recorded training-CV model choice has changed")
    return design


def paired_block_intervals(
    actual: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    *,
    samples: int = 10_000,
    block_length: int = 12,
    seed: int = 42,
) -> dict[str, tuple[float, float, float]]:
    """Return paired moving-block intervals for improvement over persistence."""
    if not (len(actual) == len(baseline) == len(candidate)):
        raise ValueError("Actual and prediction arrays must have equal lengths")
    if samples < 1 or not 1 <= block_length <= len(actual):
        raise ValueError("Invalid bootstrap settings")

    rng = np.random.default_rng(seed)
    blocks = int(np.ceil(len(actual) / block_length))
    offsets = np.arange(block_length)
    draws = np.empty((samples, len(actual)), dtype=int)
    for row in range(samples):
        starts = rng.integers(0, len(actual), size=blocks)
        draws[row] = ((starts[:, None] + offsets) % len(actual)).ravel()[:len(actual)]

    actual_draws = actual[draws]
    baseline_errors = baseline[draws] - actual_draws
    candidate_errors = candidate[draws] - actual_draws
    differences = {
        "RMSE": (
            np.sqrt(np.mean(baseline_errors ** 2, axis=1))
            - np.sqrt(np.mean(candidate_errors ** 2, axis=1))
        ),
        "MAE": (
            np.mean(np.abs(baseline_errors), axis=1)
            - np.mean(np.abs(candidate_errors), axis=1)
        ),
    }
    point_errors = {
        "RMSE": (
            np.sqrt(np.mean((baseline - actual) ** 2))
            - np.sqrt(np.mean((candidate - actual) ** 2))
        ),
        "MAE": np.mean(np.abs(baseline - actual)) - np.mean(np.abs(candidate - actual)),
    }
    return {
        metric: (
            float(point_errors[metric]),
            float(np.quantile(values, 0.025)),
            float(np.quantile(values, 0.975)),
        )
        for metric, values in differences.items()
    }


def run_external(
    development_data: Path,
    external_data: Path,
    frozen_design: Path,
    output: Path,
    *,
    seed: int = 42,
    bootstrap_samples: int = 10_000,
    block_length: int = 12,
) -> pd.DataFrame:
    development = load_data(development_data)
    external = load_data(external_data)
    expected_start = development.index.max() + pd.offsets.MonthBegin(1)
    if external.index.min() != expected_start:
        raise ValueError("External data must begin one month after the development data")
    if development.index.intersection(external.index).size:
        raise ValueError("Development and external periods must not overlap")

    combined = pd.concat([development[[TARGET]], external[[TARGET]]])
    X, y = make_features(combined)
    training = X.index <= development.index.max()
    evaluation = X.index >= external.index.min()
    X_train, y_train = X.loc[training], y.loc[training]
    X_external, y_external = X.loc[evaluation], y.loc[evaluation]

    design = load_frozen_design(frozen_design)
    if list(X.columns) != design["features"]:
        raise ValueError("Feature definitions differ from the recorded model design")
    ridge_alpha = float(design["best_parameters"]["Ridge"]["ridge__alpha"])
    forest_params = design["best_parameters"]["RandomForest"]
    models = {
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=ridge_alpha)),
        "RandomForest": RandomForestRegressor(
            n_estimators=200,
            max_depth=forest_params["max_depth"],
            min_samples_leaf=forest_params["min_samples_leaf"],
            random_state=seed,
            n_jobs=1,
        ),
    }

    predictions = pd.DataFrame({
        "Actual": y_external,
        "Persistence": X_external["effr_lag_1"],
    })
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions[name] = model.predict(X_external)

    metrics = pd.DataFrame([
        {"Model": name, **score(predictions["Actual"], predictions[name])}
        for name in ("Persistence", "Ridge", "RandomForest")
    ])
    uncertainty_rows = []
    actual = predictions["Actual"].to_numpy()
    baseline = predictions["Persistence"].to_numpy()
    for model_number, name in enumerate(("Ridge", "RandomForest")):
        intervals = paired_block_intervals(
            actual,
            baseline,
            predictions[name].to_numpy(),
            samples=bootstrap_samples,
            block_length=block_length,
            seed=seed + model_number,
        )
        for metric, (difference, lower, upper) in intervals.items():
            uncertainty_rows.append({
                "Model": name,
                "Metric": metric,
                "Difference_vs_Persistence": difference,
                "CI_Lower": lower,
                "CI_Upper": upper,
            })
    uncertainty = pd.DataFrame(uncertainty_rows)

    output.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output / "metrics.csv", index=False)
    predictions.to_csv(output / "predictions.csv", index_label="date")
    uncertainty.to_csv(output / "uncertainty.csv", index=False)
    metadata = {
        "development_data_sha256": hashlib.sha256(development_data.read_bytes()).hexdigest(),
        "external_data_sha256": hashlib.sha256(external_data.read_bytes()).hexdigest(),
        "frozen_design_sha256": hashlib.sha256(frozen_design.read_bytes()).hexdigest(),
        "data_source_series": "H15/H15/RIFSPFF_N.M",
        "python": platform.python_version(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "pandas", "scikit-learn", "matplotlib")
        },
        "seed": seed,
        "horizon_months": 1,
        "training_rows": len(X_train),
        "external_rows": len(X_external),
        "training_start": str(y_train.index.min().date()),
        "training_end": str(y_train.index.max().date()),
        "external_start": str(y_external.index.min().date()),
        "external_end": str(y_external.index.max().date()),
        "frozen_training_cv_choice": design["selected_by_training_cv"],
        "fixed_parameters": {
            "Ridge": {"ridge__alpha": ridge_alpha},
            "RandomForest": {
                "n_estimators": 200,
                "max_depth": forest_params["max_depth"],
                "min_samples_leaf": forest_params["min_samples_leaf"],
                "random_state": seed,
            },
        },
        "features": list(X.columns),
        "uncertainty": {
            "method": "paired circular moving-block bootstrap",
            "confidence_level": 0.95,
            "samples": bootstrap_samples,
            "block_length_months": block_length,
            "interpretation": "Positive metric differences favor the candidate over persistence.",
        },
        "evaluation": (
            "Frozen models refitted through February 2017; rolling one-month forecasts "
            "use the previous observed EFFR. No external observations were used for model selection."
        ),
    }
    (output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")

    rolling_mae = predictions.drop(columns="Actual").sub(predictions["Actual"], axis=0).abs().rolling(12).mean()
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True, height_ratios=(2, 1))
    predictions.plot(ax=axes[0], linewidth=1.4)
    axes[0].set(title="EFFR forecasts on the external period", ylabel="EFFR (%)", xlabel="")
    rolling_mae.plot(ax=axes[1], linewidth=1.4)
    axes[1].set(title="Trailing 12-month mean absolute error", ylabel="Percentage points", xlabel="Target month")
    fig.tight_layout()
    fig.savefig(output / "forecast.png", dpi=160)
    plt.close(fig)

    print(metrics.to_string(index=False))
    print(uncertainty.to_string(index=False))
    print(f"External artifacts: {output.resolve()}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-data", type=Path, required=True)
    parser.add_argument("--external-data", type=Path, required=True)
    parser.add_argument("--frozen-design", type=Path, default=Path("reports/run.json"))
    parser.add_argument("--output", type=Path, default=Path("reports/external"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--block-length", type=int, default=12)
    args = parser.parse_args()
    run_external(
        args.development_data,
        args.external_data,
        args.frozen_design,
        args.output,
        seed=args.seed,
        bootstrap_samples=args.bootstrap_samples,
        block_length=args.block_length,
    )


if __name__ == "__main__":
    main()
