"""ml 层测试: 线性基线 / GBM / MLP。

全部离线、确定性; 树模型与 MLP 均使用极小数据与极小迭代数。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, Ridge

from ml.lightgbm import feature_importance, fit_gbm, shap_explain
from ml.linear import DEFAULT_FEATURES, build_feature_matrix, fit_linear
from ml.neural_net import fit_mlp, predict


# --------------------------------------------------------------------------
# 数据构造
# --------------------------------------------------------------------------
def _feature_frame(n: int = 60, seed: int = 0) -> pd.DataFrame:
    """带非默认索引的合成特征面板, 含 id / 日期 / 目标列."""
    rng = np.random.RandomState(seed)
    idx = pd.Index([f"e{i:03d}" for i in range(n)])
    return pd.DataFrame(
        {
            "code": [f"{i:06d}" for i in range(n)],
            "report_date": pd.date_range("2023-01-01", periods=n, freq="D"),
            "sue": rng.normal(size=n),
            "roe": rng.normal(size=n),
            "momentum": rng.normal(size=n),
            "fwd_ret_20d": rng.normal(size=n),
            "exit_date_20d": pd.date_range("2023-06-01", periods=n, freq="D"),
        },
        index=idx,
    )


# --------------------------------------------------------------------------
# build_feature_matrix
# --------------------------------------------------------------------------
def test_build_feature_matrix_filters_ids_dates_targets() -> None:
    panel = _feature_frame()
    X = build_feature_matrix(panel)

    assert list(X.columns) == ["sue", "roe", "momentum"]
    assert "code" not in X.columns
    assert "report_date" not in X.columns
    assert not any(c.startswith("fwd_ret_") for c in X.columns)
    assert not any(c.startswith("exit_date_") for c in X.columns)
    assert X.index.equals(panel.index)
    # DEFAULT_FEATURES 是首选特征与面板列的有序交集
    assert set(X.columns) <= set(DEFAULT_FEATURES)


def test_build_feature_matrix_feature_cols_intersect_and_coerce() -> None:
    panel = _feature_frame(n=5)
    panel["roe"] = ["1.5", "2.5", "bad", None, "5.0"]
    X = build_feature_matrix(panel, feature_cols=["roe", "sue", "not_present"])

    assert list(X.columns) == ["roe", "sue"]
    assert X["roe"].dtype == np.float64
    assert X["roe"].iloc[0] == pytest.approx(1.5)
    assert pd.isna(X["roe"].iloc[2])
    assert pd.isna(X["roe"].iloc[3])


def test_build_feature_matrix_no_available_column_raises() -> None:
    panel = pd.DataFrame({"code": ["000001"], "fwd_ret_5d": [0.1]})
    with pytest.raises(ValueError):
        build_feature_matrix(panel)


# --------------------------------------------------------------------------
# fit_linear
# --------------------------------------------------------------------------
def test_fit_linear_recovers_known_relationship() -> None:
    rng = np.random.RandomState(1)
    X = pd.DataFrame(
        {"a": rng.normal(size=200), "b": rng.normal(size=200)}
    )
    y = 2.0 * X["a"] - 3.0 * X["b"] + 1.0

    model = fit_linear(X, y)

    assert isinstance(model, LinearRegression)
    assert model.coef_[0] == pytest.approx(2.0, abs=1e-8)
    assert model.coef_[1] == pytest.approx(-3.0, abs=1e-8)
    assert model.intercept_ == pytest.approx(1.0, abs=1e-8)
    assert list(model.feature_names_in_) == ["a", "b"]


def test_fit_linear_drops_nan_rows() -> None:
    X = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [1.0, 2.0, 3.0]})
    y = pd.Series([2.0, 5.0, np.nan])
    model = fit_linear(X, y)
    # 只剩第 1 行可用时 sklearn 拟合退化, 这里只需保证不抛错且可预测
    assert hasattr(model, "coef_")


def test_fit_linear_ridge_when_alpha_positive() -> None:
    X, y = _xy(n=80, seed=2)
    model = fit_linear(X, y, alpha=1.0)
    assert isinstance(model, Ridge)
    assert model.predict(X).shape == (len(X),)


# --------------------------------------------------------------------------
# fit_gbm
# --------------------------------------------------------------------------
def _xy(n: int = 80, seed: int = 3) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.RandomState(seed)
    X = pd.DataFrame(
        {
            "sue": rng.normal(size=n),
            "roe": rng.normal(size=n),
            "momentum": rng.normal(size=n),
        }
    )
    y = pd.Series(0.5 * X["sue"] - 0.3 * X["roe"] + rng.normal(scale=0.1, size=n))
    return X, y


@pytest.mark.parametrize("model_name", ["lightgbm", "xgboost"])
def test_fit_gbm_trains_and_predicts(model_name: str) -> None:
    X, y = _xy()
    model = fit_gbm(X, y, model_name, n_estimators=10)

    pred = np.asarray(model.predict(X))
    assert pred.shape == (len(X),)
    assert np.isfinite(pred).all()


def test_fit_gbm_unknown_model_raises() -> None:
    X, y = _xy(n=20)
    with pytest.raises(ValueError, match="lightgbm"):
        fit_gbm(X, y, "catboost")


def test_fit_gbm_drops_nan_rows() -> None:
    X, y = _xy(n=30)
    X.loc[X.index[0], "sue"] = np.nan
    y.iloc[1] = np.nan
    model = fit_gbm(X, y, "lightgbm", n_estimators=10)
    assert model.n_features_in_ == 3


# --------------------------------------------------------------------------
# feature_importance
# --------------------------------------------------------------------------
def test_feature_importance_sorted_series() -> None:
    X, y = _xy()
    model = fit_gbm(X, y, "xgboost", n_estimators=20)
    imp = feature_importance(model)

    assert isinstance(imp, pd.Series)
    assert list(imp.index) == ["sue", "roe", "momentum"]
    assert imp.is_monotonic_decreasing


# --------------------------------------------------------------------------
# shap_explain
# --------------------------------------------------------------------------
def test_shap_explain_shape_and_index() -> None:
    X, y = _xy(n=40)
    model = fit_gbm(X, y, "lightgbm", n_estimators=10)
    sv = shap_explain(model, X)

    assert isinstance(sv, pd.DataFrame)
    assert sv.shape == X.shape
    assert list(sv.columns) == list(X.columns)
    assert sv.index.equals(X.index)
    assert np.isfinite(sv.to_numpy()).all()


# --------------------------------------------------------------------------
# fit_mlp / predict
# --------------------------------------------------------------------------
def test_fit_mlp_tiny_frame() -> None:
    X, y = _xy(n=30)
    model = fit_mlp(
        X,
        y,
        hidden_layer_sizes=(8,),
        max_iter=20,
        early_stopping=False,
    )
    assert np.isfinite(np.asarray(model.predict(X))).all()


def test_fit_mlp_rejects_nan() -> None:
    X, y = _xy(n=20)
    X.loc[X.index[0], "sue"] = np.nan
    with pytest.raises(ValueError, match="impute"):
        fit_mlp(X, y)

    X2, y2 = _xy(n=20)
    y2.iloc[0] = np.nan
    with pytest.raises(ValueError, match="impute"):
        fit_mlp(X2, y2)


def test_predict_preserves_index() -> None:
    X, y = _xy(n=30)
    model = fit_mlp(X, y, hidden_layer_sizes=(4,), max_iter=20, early_stopping=False)
    pred = predict(model, X)

    assert isinstance(pred, pd.Series)
    assert pred.index.equals(X.index)
    assert len(pred) == len(X)
