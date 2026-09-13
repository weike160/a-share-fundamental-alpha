"""滑点建模: 半价差 + 参与率线性冲击."""
from __future__ import annotations

import pandas as pd


def _as_float(values: pd.Series) -> pd.Series:
    """转 float Series, 无法解析的值 -> NaN, 保留原索引."""
    return pd.to_numeric(pd.Series(values), errors="coerce").astype("float64")


def slippage(
    order_size: pd.Series,
    adv: pd.Series,
    participation_cap: float = 0.10,
    *,
    half_spread: float = 0.0005,
    impact_coeff: float = 0.05,
) -> pd.Series:
    """单位成交额的滑点成本 (占成交额比例).

    ``p = order_size / adv``, 生效参与率 ``p_eff = min(p, participation_cap)``,
    ``cost = half_spread + impact_coeff * p_eff``。

    Parameters
    ----------
    order_size:
        委托金额。
    adv:
        日均成交额; ``adv <= 0`` 或 ``NaN`` 时该名字为 ``NaN``。
    participation_cap:
        参与率上限, 超过后滑点不再随规模上升。
    half_spread:
        买卖价差的一半 (固定成本项)。
    impact_coeff:
        参与率的线性冲击系数。

    Returns
    -------
    Series, 索引为两输入的并集。
    """
    order = _as_float(order_size)
    volume = _as_float(adv)
    index = order.index.union(volume.index)
    order = order.reindex(index)
    volume = volume.reindex(index)

    valid = volume.gt(0) & order.notna()
    participation = (order / volume).clip(upper=participation_cap)
    cost = half_spread + impact_coeff * participation
    return cost.where(valid)
