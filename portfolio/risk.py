"""风险分析指标: 年化收益 / Sharpe / MaxDD / Turnover / TrackingError / IR."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _as_float(values: pd.Series) -> pd.Series:
    """转 float Series, 无法解析的值 -> NaN, 保留原索引."""
    return pd.to_numeric(pd.Series(values), errors="coerce").astype("float64")


def annualized_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    """几何年化收益 ``(prod(1+r)) ** (periods_per_year / n) - 1``.

    样本数不足或存在 ``1 + r <= 0`` (净值归零/穿仓) 时返回 ``NaN``。
    """
    r = _as_float(returns).dropna()
    n = len(r)
    if n == 0:
        return float("nan")
    growth = 1.0 + r
    if bool((growth <= 0).any()):
        return float("nan")
    return float(growth.prod() ** (periods_per_year / n) - 1.0)


def sharpe(returns: pd.Series, rf: float = 0.0, periods_per_year: int = 252) -> float:
    """年化 Sharpe.

    分子为 ``mean(r - rf / periods_per_year)``, 分母为样本标准差 (``ddof=1``)。
    样本数 < 2 或标准差为 0 时返回 ``NaN``。
    """
    r = _as_float(returns).dropna()
    if len(r) < 2:
        return float("nan")
    excess = r - rf / periods_per_year
    std = float(r.std(ddof=1))
    if not np.isfinite(std) or std == 0.0:
        return float("nan")
    return float(excess.mean() / std * np.sqrt(periods_per_year))


def max_drawdown(nav: pd.Series) -> float:
    """最大回撤, 取 ``nav / nav.cummax() - 1`` 的最小值 (<= 0).

    序列为空 (或全 ``NaN``) 时返回 ``NaN``。
    """
    values = _as_float(nav).dropna()
    if values.empty:
        return float("nan")
    drawdown = values / values.cummax() - 1.0
    return float(drawdown.min())


def tracking_error(active_returns: pd.Series, periods_per_year: int = 252) -> float:
    """跟踪误差: 主动收益标准差 (``ddof=1``) 年化."""
    r = _as_float(active_returns).dropna()
    if len(r) < 2:
        return float("nan")
    return float(r.std(ddof=1) * np.sqrt(periods_per_year))


def information_ratio(
    active_returns: pd.Series, periods_per_year: int = 252
) -> float:
    """信息比率: 年化主动收益均值 / 跟踪误差; 跟踪误差为 0 -> ``NaN``."""
    r = _as_float(active_returns).dropna()
    if len(r) < 2:
        return float("nan")
    te = tracking_error(r, periods_per_year)
    if not np.isfinite(te) or te == 0.0:
        return float("nan")
    return float(r.mean() * periods_per_year / te)


def turnover(weights: pd.DataFrame) -> pd.Series:
    """组合单期换手率 ``0.5 * sum_i |w_t,i - w_{t-1},i|``.

    Parameters
    ----------
    weights:
        行 = 调仓日, 列 = 标的。

    Returns
    -------
    Series, 与 ``weights.index`` 对齐。

    Notes
    -----
    初始持仓约定 ``w_{-1} = 0``, 因此首行是单边建仓换手: 满仓建仓为 0.5,
    而不是 1.0。``NaN`` 按 0 (无持仓) 处理。
    """
    w = weights.astype("float64").fillna(0.0)
    previous = w.shift(1).fillna(0.0)
    return 0.5 * (w - previous).abs().sum(axis=1)
