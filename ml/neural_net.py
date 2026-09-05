"""机器学习扩展: MLP (多层感知机).

用于检验非线性模型是否在简单因子之上产生稳定增量 Alpha。
"""
from __future__ import annotations

import pandas as pd


def fit_mlp(X_train: pd.DataFrame, y_train: pd.Series, **kwargs):
    """训练 MLP."""
    raise NotImplementedError


def predict(model, X: pd.DataFrame) -> pd.Series:
    """预测未来收益."""
    raise NotImplementedError
