"""Train/export/load and causal input contract with synthetic labels."""
import numpy as np
import pandas as pd
import pytest

from paper.model import Predictor, train
from research.config import CONFIG


@pytest.fixture
def exported(tmp_path):
    rng = np.random.default_rng(42)
    frame = pd.DataFrame({f: rng.normal(size=260) for f in CONFIG.ml_features})
    frame["fwd_ret_adj_20d"] = frame["momentum_20d"] * .01
    frame["exit_date_adj_20d"] = pd.Timestamp("2024-09-01")
    frame.loc[250:, "exit_date_adj_20d"] = pd.Timestamp("2026-10-01")
    frame["actual_disclosure_date"] = pd.Timestamp("2024-07-01")
    frame["report_date"] = pd.Timestamp("2024-06-30")
    path = tmp_path / "panel.parquet"
    frame.to_parquet(path)
    directory = tmp_path / "model"
    metadata = train(path, directory, "2026-09-13")
    return frame, directory, metadata


def test_training_excludes_unmatured_labels(exported):
    _, _, metadata = exported
    assert metadata["rows"] == 250
    assert metadata["max_label_date"] == "2024-09-01"


def test_predictions_need_no_future_returns(exported):
    frame, directory, _ = exported
    model = Predictor(directory)
    live = frame[list(CONFIG.ml_features)].iloc[:5]
    prediction = model.predict(live, "2026-09-14")
    assert len(prediction) == 5 and np.isfinite(prediction).all()
    assert np.allclose(prediction, Predictor(directory).predict(live, "2026-09-14"))


def test_empty_features_not_scored(exported):
    frame, directory, _ = exported
    live = frame[list(CONFIG.ml_features)].iloc[:2].copy()
    live.iloc[0] = np.nan
    assert list(Predictor(directory).predict(live, "2026-09-14").index) == [1]


def test_future_trained_model_rejected(exported):
    frame, directory, _ = exported
    with pytest.raises(ValueError, match="precede"):
        Predictor(directory).predict(frame, "2026-09-13")


def test_model_tampering_rejected(exported):
    _, directory, _ = exported
    with (directory / "model.txt").open("a") as stream:
        stream.write("tamper")
    with pytest.raises(ValueError, match="checksum"):
        Predictor(directory)
