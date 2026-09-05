"""机器学习扩展: LightGBM / XGBoost."""
from __future__ import annotations

import pandas as pd


def fit_gbm(X_train: pd.DataFrame, y_train: pd.Series, model: str = "lightgbm"):
    """训练树模型, 支持 'lightgbm' 或 'xgboost'."""
    raise NotImplementedError


def feature_importance(model) -> pd.Series:
    """特征重要性."""
    raise NotImplementedError


def shap_explain(model, X: pd.DataFrame):
    """SHAP 解释, 用于分析 interaction (如 SUE × Liquidity)."""
    raise NotImplementedError
