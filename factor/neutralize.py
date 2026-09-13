"""行业/市值中性化, 剥离信号对风格的暴露."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

#: cross_sectional_regression 的输出列
REGRESSION_COLUMNS: tuple[str, ...] = (
    "coef",
    "std_err",
    "t_stat",
    "p_value",
    "n_obs",
)


def industry_neutralize(signal: pd.Series, industry: pd.Series) -> pd.Series:
    """按行业去均值, 剥离行业暴露.

    Parameters
    ----------
    signal:
        待中性化的信号。
    industry:
        行业标签, 按索引与 ``signal`` 对齐。

    Returns
    -------
    同索引序列 ``signal - 行业均值``; 行业标签缺失的样本为 ``NaN``。
    """
    values = pd.to_numeric(signal, errors="coerce")
    labels = industry.reindex(values.index)
    demeaned = values - values.groupby(labels).transform("mean")
    return demeaned.where(labels.notna()).rename(signal.name)


def neutralize_cross_section(
    signal: pd.Series,
    exposures: pd.DataFrame,
    *,
    add_constant: bool = True,
) -> pd.Series:
    """对风格暴露做横截面 OLS, 返回残差.

    Parameters
    ----------
    signal:
        待中性化的信号。
    exposures:
        暴露矩阵, 按索引与 ``signal`` 对齐; 缺暴露的样本不参与回归。
    add_constant:
        是否加入截距项。

    Returns
    -------
    残差序列, 索引为 ``signal.index``; 非完整样本为 ``NaN``。
    """
    y = pd.to_numeric(signal, errors="coerce")
    x = exposures.reindex(y.index).apply(pd.to_numeric, errors="coerce")
    mask = y.notna().to_numpy() & x.notna().all(axis=1).to_numpy()

    out = pd.Series(np.nan, index=signal.index, dtype="float64")
    if not mask.any():
        return out

    yv = y.to_numpy(dtype="float64")[mask]
    xv = x.to_numpy(dtype="float64")[mask]
    if add_constant:
        xv = np.column_stack([np.ones(len(xv)), xv])
    if xv.shape[1] == 0:
        residual = yv
    else:
        beta, *_ = np.linalg.lstsq(xv, yv, rcond=None)
        residual = yv - xv @ beta

    out.iloc[np.flatnonzero(mask)] = residual
    return out


def cross_sectional_regression(
    returns: pd.Series,
    features: pd.DataFrame,
) -> pd.DataFrame:
    """横截面回归: R = α + β·features + ε, 只使用完整样本.

    Parameters
    ----------
    returns:
        横截面收益。
    features:
        特征矩阵, 按索引与 ``returns`` 对齐, 列序即输出行序。

    Returns
    -------
    每个特征一行, 索引为特征名, 列为
    ``[coef, std_err, t_stat, p_value, n_obs]``; 截距参与估计但不返回。
    标准误由 OLS 残差方差 ``s² = SSR / (n - k - 1)`` 与 ``(X'X)⁻¹`` 得到,
    双侧 p 值用 t 分布。完整样本少于 ``n_features + 2`` 时返回同列空表。
    """
    y = pd.to_numeric(returns, errors="coerce")
    x = features.reindex(y.index).apply(pd.to_numeric, errors="coerce")
    mask = y.notna().to_numpy() & x.notna().all(axis=1).to_numpy()
    n_obs = int(mask.sum())
    n_features = features.shape[1]

    if n_features == 0 or n_obs < n_features + 2:
        return pd.DataFrame(columns=list(REGRESSION_COLUMNS), dtype="float64")

    yv = y.to_numpy(dtype="float64")[mask]
    xv = x.to_numpy(dtype="float64")[mask]
    design = np.column_stack([np.ones(n_obs), xv])
    beta, *_ = np.linalg.lstsq(design, yv, rcond=None)

    residual = yv - design @ beta
    dof = n_obs - design.shape[1]
    sigma2 = float(residual @ residual) / dof
    xtx_inv = np.linalg.pinv(design.T @ design)
    std_err = np.sqrt(np.maximum(np.diag(xtx_inv) * sigma2, 0.0))

    with np.errstate(invalid="ignore", divide="ignore"):
        t_stat = np.divide(
            beta, std_err, out=np.full_like(beta, np.nan), where=std_err > 0
        )
    p_value = 2.0 * stats.t.sf(np.abs(t_stat), dof)

    return pd.DataFrame(
        {
            "coef": beta[1:],
            "std_err": std_err[1:],
            "t_stat": t_stat[1:],
            "p_value": p_value[1:],
            "n_obs": np.full(n_features, n_obs, dtype="int64"),
        },
        index=features.columns,
    )
