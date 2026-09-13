"""机器学习扩展: LightGBM / XGBoost."""
from __future__ import annotations

import lightgbm
import numpy as np
import pandas as pd
import xgboost

#: LightGBM 默认超参数。
LIGHTGBM_DEFAULTS: dict[str, object] = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "verbose": -1,
}

#: XGBoost 默认超参数。
XGBOOST_DEFAULTS: dict[str, object] = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "verbosity": 0,
}


def fit_gbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model: str = "lightgbm",
    **kwargs: object,
):
    """训练树模型, 支持 'lightgbm' 或 'xgboost'。

    Parameters
    ----------
    X_train:
        特征矩阵。
    y_train:
        目标收益。
    model:
        ``"lightgbm"`` 或 ``"xgboost"``。
    **kwargs:
        覆盖对应默认超参数, 直接透传给估计器构造函数。

    Returns
    -------
    已拟合的估计器。X / y 为 NaN 的行在拟合前被剔除。

    Raises
    ------
    ValueError
        ``model`` 不是受支持的值。
    """
    if model == "lightgbm":
        params = {**LIGHTGBM_DEFAULTS, **kwargs}
        estimator = lightgbm.LGBMRegressor(**params)
    elif model == "xgboost":
        params = {**XGBOOST_DEFAULTS, **kwargs}
        estimator = xgboost.XGBRegressor(**params)
    else:
        raise ValueError(
            f"model 只能是 'lightgbm' 或 'xgboost', 收到 {model!r}"
        )

    y_aligned = y_train.reindex(X_train.index)
    mask = X_train.notna().all(axis=1) & y_aligned.notna()
    return estimator.fit(X_train.loc[mask], y_aligned.loc[mask])


def feature_importance(model) -> pd.Series:
    """特征重要性。

    Returns
    -------
    Series, 索引为特征名, 按重要性降序排列。

    特征名优先取 ``model.feature_names_in_``, 其次 ``model.feature_name_``,
    最后回退为 ``f0, f1, ...``。
    """
    importance = np.asarray(model.feature_importances_, dtype="float64")
    names = getattr(model, "feature_names_in_", None)
    if names is None:
        names = getattr(model, "feature_name_", None)
    if names is None or len(names) != len(importance):
        names = [f"f{i}" for i in range(len(importance))]
    return pd.Series(importance, index=list(names)).sort_values(ascending=False)


def shap_explain(model, X: pd.DataFrame) -> pd.DataFrame:
    """SHAP 解释, 用于分析 interaction (如 SUE × Liquidity)。

    ``shap`` 在函数内部惰性导入, 未安装时本模块仍可导入。

    Returns
    -------
    DataFrame, 形状 ``(len(X), n_features)``, 索引与列名同 ``X``。
    """
    import shap

    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(X)
    if isinstance(values, list):
        values = values[0]
    return pd.DataFrame(np.asarray(values), index=X.index, columns=X.columns)
