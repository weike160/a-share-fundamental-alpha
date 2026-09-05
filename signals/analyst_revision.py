"""分析师业绩修正信号 (需一致预期数据)."""
from __future__ import annotations

import pandas as pd


def analyst_revision(
    forecast_prev: pd.Series, forecast_now: pd.Series
) -> pd.Series:
    """分析师一致性预期修正幅度."""
    raise NotImplementedError


def revision_breadth(corrections: pd.DataFrame, window: int) -> pd.Series:
    """一段时间内上调 vs 下调的广度."""
    raise NotImplementedError
