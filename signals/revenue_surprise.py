"""营收超预期信号."""
from __future__ import annotations

import pandas as pd


def revenue_surprise(
    actual_rev: pd.Series, expected_rev: pd.Series
) -> pd.Series:
    """营收超预期 = (实际营收 - 预期营收) / 预期营收."""
    raise NotImplementedError


def revenue_surprise_yoy(actual_rev: pd.Series, lag: int = 4) -> pd.Series:
    """无分析师数据时的营收超预期代理 (同比)."""
    raise NotImplementedError
