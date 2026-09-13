"""组合优化: 在暴露/换手约束下求解权重."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def _as_float(values: pd.Series) -> pd.Series:
    """转 float Series, 无法解析的值 -> NaN, 保留原索引."""
    return pd.to_numeric(pd.Series(values), errors="coerce").astype("float64")


def _bounds(
    values: object | None, index: pd.Index, default: float, fill: float
) -> pd.Series:
    """把标量/序列形式的上下界展开成与 ``index`` 对齐的 Series."""
    if values is None:
        return pd.Series(default, index=index, dtype="float64")
    if np.isscalar(values):
        return pd.Series(float(values), index=index, dtype="float64")
    return _as_float(pd.Series(values)).reindex(index).fillna(fill)


def risk_parity(cov: pd.DataFrame, constraints: dict | None = None) -> pd.Series:
    """逆波动率 ("naive") 风险平价权重.

    ``w_i ∝ 1 / sigma_i``, ``sigma_i = sqrt(cov_ii)``。非有限或 ``<= 0``
    的波动率对应权重 0; 总和 <= 0 时返回全 0。

    Parameters
    ----------
    cov:
        协方差矩阵, 索引即标的。
    constraints:
        可选 ``{"lower": x, "upper": y, "max_weight": z}``; 先裁剪后归一,
        ``max_weight`` 只收紧上界。标量或与 ``cov.index`` 对齐的序列均可。

    Returns
    -------
    Series, 与 ``cov.index`` 对齐, 和为 1 (或全 0)。

    Notes
    -----
    这是逆波动率近似, 不是完整 ERC 数值解。
    """
    matrix = pd.DataFrame(cov)
    index = matrix.index
    diagonal = np.diag(matrix.to_numpy(dtype="float64"))
    sigma = np.sqrt(np.clip(diagonal, 0.0, None))
    valid = np.isfinite(sigma) & (sigma > 0.0) & np.isfinite(diagonal)

    weights = np.zeros(len(index), dtype="float64")
    weights[valid] = 1.0 / sigma[valid]
    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0.0:
        return pd.Series(0.0, index=index, dtype="float64")

    lower = _bounds((constraints or {}).get("lower"), index, 0.0, 0.0)
    upper = _bounds((constraints or {}).get("upper"), index, np.inf, np.inf)
    max_weight = (constraints or {}).get("max_weight")
    if max_weight is not None:
        upper = np.minimum(upper, _bounds(max_weight, index, np.inf, np.inf))

    series = pd.Series(weights / total, index=index, dtype="float64")
    series = series.clip(lower=lower, upper=upper)
    total = float(series.sum())
    if not np.isfinite(total) or total <= 0.0:
        return pd.Series(0.0, index=index, dtype="float64")
    return series / total


def mean_variance(
    expected_return: pd.Series,
    cov: pd.DataFrame,
    constraints: dict | None = None,
    risk_aversion: float = 5.0,
) -> pd.Series:
    """均值-方差优化 (long-only, 满仓).

    最小化 ``risk_aversion/2 * w'Σw - w'μ``, 约束 ``sum(w) = 1`` 及逐票上下界,
    用 SLSQP 从等权起点求解。

    Parameters
    ----------
    expected_return:
        预期收益, 其索引定义资产顺序与返回索引。
    cov:
        协方差矩阵, 按 ``expected_return.index`` 重排。
    constraints:
        可选 ``{"lower": x, "upper": y, "max_weight": z}``; 默认
        ``lower=0, upper=1``, ``max_weight`` 只收紧上界。标量或序列均可。
    risk_aversion:
        风险厌恶系数, 越大越偏向最小方差。

    Returns
    -------
    Series, 与 ``expected_return`` 同索引; 优化失败时退回等权/边界解。
    """
    mu = _as_float(expected_return)
    index = mu.index
    n = len(index)
    if n == 0:
        return pd.Series(0.0, index=index, dtype="float64")

    matrix = pd.DataFrame(cov)
    if matrix.shape == (n, n) and not bool(index.isin(matrix.index).all()):
        # 标签对不上但形状一致时按位置解释
        sigma = matrix.to_numpy(dtype="float64")
    else:
        sigma = matrix.reindex(index=index, columns=index).to_numpy(dtype="float64")
    sigma = np.nan_to_num(sigma, nan=0.0, posinf=0.0, neginf=0.0)
    sigma = 0.5 * (sigma + sigma.T)
    mu_values = np.nan_to_num(mu.to_numpy(dtype="float64"), nan=0.0)

    lower = _bounds((constraints or {}).get("lower"), index, 0.0, 0.0).to_numpy()
    upper = _bounds((constraints or {}).get("upper"), index, 1.0, 1.0)
    max_weight = (constraints or {}).get("max_weight")
    if max_weight is not None:
        upper = np.minimum(upper, _bounds(max_weight, index, np.inf, np.inf))
    upper = upper.to_numpy()
    upper = np.maximum(upper, lower)

    start = np.clip(np.full(n, 1.0 / n), lower, upper)
    if start.sum() > 0.0:
        start = start / start.sum()
    start = np.clip(start, lower, upper)

    def objective(x: np.ndarray) -> float:
        return float(risk_aversion / 2.0 * x @ sigma @ x - x @ mu_values)

    def gradient(x: np.ndarray) -> np.ndarray:
        return risk_aversion * sigma @ x - mu_values

    result = minimize(
        objective,
        start,
        jac=gradient,
        method="SLSQP",
        bounds=list(zip(lower.tolist(), upper.tolist())),
        constraints=[
            {
                "type": "eq",
                "fun": lambda x: float(x.sum() - 1.0),
                "jac": lambda x: np.ones(n),
            }
        ],
        options={"maxiter": 500, "ftol": 1e-12},
    )

    weights = result.x if bool(result.success) else start
    weights = np.clip(np.asarray(weights, dtype="float64"), lower, upper)
    total = float(weights.sum())
    if np.isfinite(total) and total > 0.0:
        weights = weights / total
    return pd.Series(weights, index=index, dtype="float64")
