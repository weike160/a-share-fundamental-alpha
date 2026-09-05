"""IC Decay: 研究不同预测期 (1/5/10/20/40/60 天) 的 IC(T).

用来回答: 市场需要多长时间吸收财报信息? Alpha 半衰期有多长?
"""
from __future__ import annotations

import pandas as pd


def ic_decay(
    signal: pd.Series,
    returns: pd.DataFrame,
    horizons: tuple[int, ...] = (1, 5, 10, 20, 40, 60),
) -> pd.Series:
    """返回各预测期对应的 IC 值."""
    raise NotImplementedError
