"""Export the existing LightGBM recipe with frozen training transforms.

Native LightGBM text + JSON metadata, never pickle. Prediction requires no labels.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.config import CONFIG
from paper.report import atomic_text


def train(panel_path: Path, directory: Path, cutoff: str):
    import lightgbm as lgb

    if directory.exists() and any(directory.iterdir()):
        raise ValueError("Model directory is not empty; choose a new version directory")
    panel = pd.read_parquet(panel_path)
    features = list(CONFIG.ml_features)
    required = {*features, "fwd_ret_adj_20d", "exit_date_adj_20d", "actual_disclosure_date"}
    if required - set(panel.columns):
        raise ValueError(f"Training panel missing: {sorted(required - set(panel.columns))}")
    cutoff_ts = pd.Timestamp(cutoff).normalize()
    # A report_date cutoff alone does not guarantee that its return label has matured.
    exits = pd.to_datetime(panel["exit_date_adj_20d"])
    announced = pd.to_datetime(panel["actual_disclosure_date"])
    target = pd.to_numeric(panel["fwd_ret_adj_20d"], errors="coerce")
    usable = (exits <= cutoff_ts) & (announced <= cutoff_ts) & np.isfinite(target)
    panel, target = panel.loc[usable], target.loc[usable]
    if len(panel) < 200:
        raise ValueError("Need at least 200 events with matured labels")
    X = panel[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    medians = X.median()
    if medians.isna().any():
        raise ValueError("Training feature is entirely missing")
    X = X.fillna(medians)
    center = X.mean()
    scale = X.std(ddof=1).replace(0, 1).fillna(1)
    model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31,
                              subsample=0.8, colsample_bytree=0.8, random_state=42,
                              verbosity=-1, n_jobs=2)
    model.fit((X - center) / scale, target)
    model_text = model.booster_.model_to_string()
    metadata = dict(schema_version=1, features=features, medians=medians.to_dict(),
                    center=center.to_dict(), scale=scale.to_dict(), trained_through=cutoff,
                    max_label_date=exits.loc[usable].max().date().isoformat(), rows=len(panel),
                    max_report_date=pd.to_datetime(panel["report_date"]).max().date().isoformat(),
                    model_sha256=hashlib.sha256(model_text.encode()).hexdigest(),
                    source_sha256=hashlib.sha256(panel_path.read_bytes()).hexdigest(),
                    lightgbm_version=lgb.__version__,
                    limitations="Research panel is not certified point-in-time. Online sue_neutral uses only currently available events; historical full-quarter neutralization may differ.")
    directory.mkdir(parents=True, exist_ok=True)
    atomic_text(directory / "model.txt", model_text)
    atomic_text(directory / "metadata.json", json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False))
    return metadata


class Predictor:
    def __init__(self, directory: Path):
        import lightgbm as lgb

        meta_path, model_path = directory / "metadata.json", directory / "model.txt"
        if not meta_path.exists() or not model_path.exists():
            raise FileNotFoundError(f"缺少模型：{directory}/model.txt 与 metadata.json。请先导出研究模型。")
        self.meta = json.loads(meta_path.read_text())
        model_bytes = model_path.read_bytes()
        if hashlib.sha256(model_bytes).hexdigest() != self.meta["model_sha256"]:
            raise ValueError("Model checksum mismatch")
        self.model_id = hashlib.sha256(model_bytes + meta_path.read_bytes()).hexdigest()[:16]
        self.booster = lgb.Booster(model_file=str(model_path))
        if self.booster.feature_name() != self.meta["features"]:
            raise ValueError("Model feature order differs from metadata")

    def predict(self, frame, asof):
        if self.meta["trained_through"] >= asof or self.meta["max_label_date"] >= asof:
            raise ValueError("Model training must precede the prediction date")
        features = self.meta["features"]
        if set(features) - set(frame.columns):
            raise ValueError("Prediction frame is missing required feature columns")
        X = frame[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        # Reject uninformative rows instead of silently ranking all-median replacements.
        valid = X.notna().sum(axis=1) >= max(1, len(features) // 2)
        X = X.loc[valid].fillna(pd.Series(self.meta["medians"]))
        X = (X - pd.Series(self.meta["center"])) / pd.Series(self.meta["scale"])
        values = self.booster.predict(X) if len(X) else []
        return pd.Series(values, index=X.index, dtype="float64")
