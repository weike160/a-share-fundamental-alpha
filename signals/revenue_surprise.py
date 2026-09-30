"""营收超预期信号."""
from __future__ import annotations

import pandas as pd

from signals.earnings_growth import yoy_growth


def revenue_surprise(actual_rev: pd.Series, expected_rev: pd.Series) -> pd.Series:
    """按索引计算 (实际营收 - 预期营收) / |预期营收|；预期为零或缺失时返回 NaN。"""
    actual = pd.to_numeric(actual_rev, errors="coerce")
    expected = pd.to_numeric(expected_rev, errors="coerce")
    denominator = expected.abs().where(expected != 0)
    return ((actual - expected) / denominator).rename("revenue_surprise")


def revenue_surprise_yoy(actual_rev: pd.Series, lag: int = 4) -> pd.Series:
    """用单只股票按时间升序的营收同比作为超预期代理，缺失规则同 yoy_growth。"""
    return yoy_growth(actual_rev, lag=lag).rename("revenue_surprise_yoy")
