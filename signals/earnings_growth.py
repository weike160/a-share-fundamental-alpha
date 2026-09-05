"""增长类基本面信号: 净利润 / 营收 / EPS 同比变化, ROE 变化, 毛利率变化."""
from __future__ import annotations

import pandas as pd


def yoy_growth(financial: pd.Series, lag: int = 4) -> pd.Series:
    """同比变化率, 默认与 4 个季度前的值比较."""
    raise NotImplementedError


def earnings_growth(financials: pd.DataFrame) -> pd.DataFrame:
    """净利润同比 / 营收同比 / EPS 同比."""
    raise NotImplementedError


def roe_change(financials: pd.DataFrame) -> pd.Series:
    """ROE 的同比变化."""
    raise NotImplementedError
