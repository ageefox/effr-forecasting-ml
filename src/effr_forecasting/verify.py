"""Regenerate the benchmark and detect stale published artifacts."""
import argparse
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from .train import run


CSV_TOLERANCES = {
    "metrics.csv": (0.01, 0.001),
    "predictions.csv": (0.01, 0.01),
    "ridge_cv.csv": (0.01, 0.01),
    "randomforest_cv.csv": (0.01, 0.01),
    "feature_importance.csv": (0.10, 0.005),
}


def compare_csv(expected_path: Path, actual_path: Path, rtol: float, atol: float):
    expected = pd.read_csv(expected_path)
    actual = pd.read_csv(actual_path)
    if list(expected.columns) != list(actual.columns):
        raise AssertionError(f"Columns differ for {expected_path.name}")
    if expected.shape != actual.shape:
        raise AssertionError(f"Shape differs for {expected_path.name}: {expected.shape} != {actual.shape}")
    numeric = expected.select_dtypes(include=[np.number]).columns
    text = expected.columns.difference(numeric)
    pd.testing.assert_frame_equal(expected[text], actual[text], check_dtype=False)
    np.testing.assert_allclose(expected[numeric], actual[numeric], rtol=rtol, atol=atol)


def compare_metadata(expected_path: Path, actual_path: Path):
    expected = json.loads(expected_path.read_text())
    actual = json.loads(actual_path.read_text())
    expected_python = tuple(map(int, expected.pop("python").split(".")[:2]))
    actual_python = tuple(map(int, actual.pop("python").split(".")[:2]))
    if expected_python != (3, 12) or actual_python != (3, 12):
        raise AssertionError("Published and regenerated artifacts must use Python 3.12")
    if expected != actual:
        raise AssertionError("Stable run metadata differs from reports/run.json")


def verify(data: Path, expected: Path):
    with tempfile.TemporaryDirectory(prefix="effr-verify-") as directory:
        actual = Path(directory)
        run(data, actual)
        for filename, (rtol, atol) in CSV_TOLERANCES.items():
            compare_csv(expected / filename, actual / filename, rtol, atol)
        compare_metadata(expected / "run.json", actual / "run.json")
    print("Published benchmark artifacts match a fresh run.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Path to the monthly EFFR CSV")
    parser.add_argument("--expected", type=Path, default=Path("reports"))
    args = parser.parse_args()
    verify(args.data, args.expected)


if __name__ == "__main__":
    main()
