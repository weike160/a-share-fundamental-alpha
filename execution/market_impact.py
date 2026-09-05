"""冲击成本建模.

MarketImpact = f(OrderSize/ADV, Volatility)
"""
from __future__ import annotations

import pandas as pd


def market_impact(
    order_size: pd.Series,
    adv: pd.Series,
    volatility: pd.Series,
    participation_cap: float = 0.10,
) -> pd.Series:
    """冲击成本估计."""
    raise NotImplementedError


def capacity_analysis(
    adv: pd.Series, weight: pd.Series, max_participation: float = 0.10
) -> pd.Series:
    """基于 ADV 与权重估算策略总容量."""
    raise NotImplementedError
