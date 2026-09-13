"""机器学习扩展: MLP (多层感知机)。

用于检验非线性模型是否在简单因子之上产生稳定增量 Alpha。
"""
from __future__ import annotations

import pandas as pd
from sklearn.neural_network import MLPRegressor

#: MLP 默认超参数。
MLP_DEFAULTS: dict[str, object] = {
    "hidden_layer_sizes": (64, 32),
    "activation": "relu",
    "alpha": 1e-4,
    "learning_rate_init": 1e-3,
    "max_iter": 500,
    "early_stopping": True,
    "n_iter_no_change": 20,
    "random_state": 42,
}


def fit_mlp(X_train: pd.DataFrame, y_train: pd.Series, **kwargs: object) -> MLPRegressor:
    """训练 MLP。

    Parameters
    ----------
    X_train:
        特征矩阵; **不允许含 NaN** (调用方需先做缺失值填充)。
    y_train:
        目标收益; 同样不允许含 NaN。
    **kwargs:
        覆盖默认超参数, 透传给 :class:`~sklearn.neural_network.MLPRegressor`。

    Returns
    -------
    已拟合的 ``MLPRegressor``。

    Raises
    ------
    ValueError
        X 或 y 含 NaN。sklearn 的 MLP 不接受缺失值, 请调用方先 impute。
    """
    if X_train.isna().to_numpy().any() or y_train.isna().to_numpy().any():
        raise ValueError("MLP 不接受 NaN, 请先对 X/y 做 impute 缺失值填充")

    params = {**MLP_DEFAULTS, **kwargs}
    model = MLPRegressor(**params)
    return model.fit(X_train, y_train.reindex(X_train.index))


def predict(model, X: pd.DataFrame) -> pd.Series:
    """预测未来收益。

    Returns
    -------
    Series, 索引与 ``X`` 一致。
    """
    return pd.Series(model.predict(X), index=X.index)
