from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from effr_forecasting.data import TARGET, load_data, make_features
from effr_forecasting.external import paired_block_intervals, run_external
from effr_forecasting.train import run
from effr_forecasting.verify import compare_csv

DATA = Path(__file__).resolve().parents[1] / "data/effr_monthly.csv"
EXTERNAL_DATA = Path(__file__).resolve().parents[1] / "data/effr_external.csv"
FROZEN_DESIGN = Path(__file__).resolve().parents[1] / "reports/run.json"


def test_features_cannot_see_current_or_future_targets():
    frame = load_data(DATA)
    before, _ = make_features(frame)
    cutoff = frame.index[200]
    changed = frame.copy()
    changed.loc[cutoff:, TARGET] += 1000
    after, _ = make_features(changed)
    pd.testing.assert_frame_equal(before.loc[:cutoff], after.loc[:cutoff])
    assert before.loc[cutoff, "effr_lag_1"] == frame[TARGET].iloc[199]
    assert before.loc[cutoff, "effr_mean_3"] == pytest.approx(frame[TARGET].iloc[197:200].mean())


def test_unrelated_columns_are_excluded():
    frame = load_data(DATA)
    before, _ = make_features(frame)
    frame["unavailable_macro_value"] = np.arange(len(frame))
    after, _ = make_features(frame)
    pd.testing.assert_frame_equal(before, after)
    assert np.isfinite(before.to_numpy()).all()


@pytest.mark.parametrize("problem", ["duplicate", "gap", "missing", "infinite"])
def test_invalid_data_rejected(tmp_path, problem):
    frame = pd.read_csv(DATA)
    if problem == "duplicate": frame = pd.concat([frame, frame.iloc[:1]])
    if problem == "gap": frame = frame.drop(index=20)
    if problem == "missing": frame.loc[20, TARGET] = np.nan
    if problem == "infinite": frame.loc[20, TARGET] = np.inf
    path = tmp_path / "bad.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError): load_data(path)


def test_sorting_is_chronological(tmp_path):
    path = tmp_path / "shuffled.csv"
    pd.read_csv(DATA).sample(frac=1, random_state=7).to_csv(path, index=False)
    pd.testing.assert_frame_equal(load_data(path), load_data(DATA))


def test_invalid_split_rejected(tmp_path):
    with pytest.raises(ValueError): run(DATA, tmp_path, test_fraction=0.8)


def test_holdout_labels_do_not_affect_tuning(tmp_path):
    import json
    frame = pd.read_csv(DATA).iloc[:100].copy()
    original = tmp_path / "original.csv"
    altered = tmp_path / "altered.csv"
    frame.to_csv(original, index=False)
    split = 12 + int((len(frame) - 12) * 0.8)
    frame.loc[split:, TARGET] += 50
    frame.to_csv(altered, index=False)
    first = run(original, tmp_path / "first")
    second = run(altered, tmp_path / "second")
    np.testing.assert_allclose(first.CV_RMSE, second.CV_RMSE)
    a = json.loads((tmp_path / "first/run.json").read_text())
    b = json.loads((tmp_path / "second/run.json").read_text())
    assert a['best_parameters'] == b['best_parameters']
    assert a['selected_by_training_cv'] == b['selected_by_training_cv']
    assert a['data_source_series'] == 'H15/H15/RIFSPFF_N.M'
    assert a['train_end'] < a['test_start']
    assert all(f['train_end'] < f['validation_start'] for f in a['cv_folds'])
    assert not any('time' in column for column in pd.read_csv(tmp_path / "first/ridge_cv.csv").columns)


def test_artifact_comparison_uses_numeric_tolerance(tmp_path):
    expected = tmp_path / "expected.csv"
    actual = tmp_path / "actual.csv"
    pd.DataFrame({"Model": ["Persistence"], "RMSE": [0.15255]}).to_csv(expected, index=False)
    pd.DataFrame({"Model": ["Persistence"], "RMSE": [0.15260]}).to_csv(actual, index=False)
    compare_csv(expected, actual, rtol=0.01, atol=0.001)
    pd.DataFrame({"Model": ["Persistence"], "RMSE": [0.20]}).to_csv(actual, index=False)
    with pytest.raises(AssertionError):
        compare_csv(expected, actual, rtol=0.01, atol=0.001)


def test_external_run_uses_the_recorded_design_and_dates(tmp_path):
    import json

    development = load_data(DATA)
    external = load_data(EXTERNAL_DATA)
    assert development.index.max() + pd.offsets.MonthBegin(1) == external.index.min()

    metrics = run_external(
        DATA,
        EXTERNAL_DATA,
        FROZEN_DESIGN,
        tmp_path,
        bootstrap_samples=100,
    )
    metadata = json.loads((tmp_path / "run.json").read_text())
    frozen = json.loads(FROZEN_DESIGN.read_text())
    predictions = pd.read_csv(tmp_path / "predictions.csv", parse_dates=["date"])

    assert list(metrics.Model) == ["Persistence", "Ridge", "RandomForest"]
    assert metadata["training_end"] == "2017-02-01"
    assert metadata["external_start"] == "2017-03-01"
    assert metadata["external_end"] == "2026-08-01"
    assert metadata["external_rows"] == len(external) == len(predictions)
    assert metadata["features"] == frozen["features"]
    assert metadata["fixed_parameters"]["Ridge"] == frozen["best_parameters"]["Ridge"]
    assert predictions.date.tolist() == external.index.tolist()


def test_paired_block_intervals_are_reproducible():
    actual = np.arange(24, dtype=float)
    persistence = actual + np.tile([1.0, -1.0], 12)
    candidate = actual + 0.25
    first = paired_block_intervals(actual, persistence, candidate, samples=200, block_length=4, seed=7)
    second = paired_block_intervals(actual, persistence, candidate, samples=200, block_length=4, seed=7)

    assert first == second
    assert first["RMSE"][0] > 0
    assert first["MAE"][0] > 0
