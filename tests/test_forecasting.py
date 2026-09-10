from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from effr_forecasting.data import TARGET, load_data, make_features
from effr_forecasting.train import run

DATA = Path(__file__).resolve().parents[1] / "data/cleaned_effr_data.csv"


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


def test_macro_data_is_excluded():
    frame = load_data(DATA)
    before, _ = make_features(frame)
    for col in frame.columns.difference([TARGET]):
        frame[col] = np.nan
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
    assert a['train_end'] < a['test_start']
    assert all(f['train_end'] < f['validation_start'] for f in a['cv_folds'])
