"""冲击成本建模.

MarketImpact = f(OrderSize/ADV, Volatility)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _as_float(values: pd.Series) -> pd.Series:
    """转 float Series, 无法解析的值 -> NaN, 保留原索引."""
    return pd.to_numeric(pd.Series(values), errors="coerce").astype("float64")


def market_impact(
    order_size: pd.Series,
    adv: pd.Series,
    volatility: pd.Series,
    participation_cap: float = 0.10,
    *,
    coeff: float = 0.5,
) -> pd.Series:
    """平方根冲击模型.

    ``coeff * volatility * sqrt(min(order_size / adv, participation_cap))``。

    Parameters
    ----------
    order_size:
        委托金额, 非负。
    adv:
        日均成交额; ``adv <= 0`` 或 ``NaN`` 时该名字为 ``NaN``。
    volatility:
        日波动率; ``NaN`` 时该名字为 ``NaN``。
    participation_cap:
        参与率上限。
    coeff:
        冲击系数。

    Returns
    -------
    Series, 索引为三个输入的并集。
    """
    order = _as_float(order_size)
    volume = _as_float(adv)
    vol = _as_float(volatility)
    index = order.index.union(volume.index).union(vol.index)
    order = order.reindex(index)
    volume = volume.reindex(index)
    vol = vol.reindex(index)

    valid = volume.gt(0) & order.notna() & order.ge(0) & vol.notna()
    participation = (order / volume).clip(lower=0.0, upper=participation_cap)
    impact = coeff * vol * np.sqrt(participation)
    return impact.where(valid)


def capacity_analysis(
    adv: pd.Series, weight: pd.Series, max_participation: float = 0.10
) -> pd.Series:
    """基于 ADV 与权重估算单票可承载 AUM.

    ``capacity_i = max_participation * adv_i / weight_i``; ``weight_i <= 0``
    或 ``NaN`` 时该名字为 ``NaN``。

    Returns
    -------
    Series, 索引为两输入的并集。
    """
    volume = _as_float(adv)
    w = _as_float(weight)
    index = volume.index.union(w.index)
    volume = volume.reindex(index)
    w = w.reindex(index)

    valid = w.gt(0) & volume.notna()
    capacity = max_participation * volume / w
    return capacity.where(valid & np.isfinite(capacity))


def portfolio_capacity(
    adv: pd.Series,
    weight: pd.Series,
    max_participation: float = 0.10,
    *,
    quantile: float = 0.05,
) -> float:
    """组合容量: 单票容量序列的给定分位数.

    默认 5% 分位 (最紧的瓶颈票); 无有效值 -> ``NaN``。
    """
    capacity = capacity_analysis(adv, weight, max_participation).dropna()
    if capacity.empty:
        return float("nan")
    return float(capacity.quantile(quantile))
