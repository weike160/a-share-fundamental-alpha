"""数据预处理: 去极值 / 填充 / 标准化, 以及训练/回测的时序划分."""
from __future__ import annotations

import pandas as pd


def winsorize(series: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    """分位数去极值."""
    raise NotImplementedError


def zscore_cross_section(frame: pd.DataFrame, date: pd.Timestamp) -> pd.Series:
    """横截面标准化 (z-score)."""
    raise NotImplementedError


def time_series_split(df: pd.DataFrame, n_splits: int = 5):
    """时序交叉验证划分, 保证无未来信息泄漏."""
    raise NotImplementedError
