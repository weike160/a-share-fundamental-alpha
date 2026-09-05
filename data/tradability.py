"""可交易性处理: 停牌 / 涨跌停 / T+1 / 新股期 / ADV.

用于判断某信号在何时、以何种规模真正可以执行。
"""
from __future__ import annotations

import pandas as pd


class TradingCalendar:
    """A 股交易日历, 提供可交易时点判定."""

    def __init__(self, trading_days: pd.DatetimeIndex) -> None:
        self.days = trading_days


def is_tradable(
    code: str,
    date: pd.Timestamp,
    *,
    suspended: pd.DataFrame,
    limit_up: pd.DataFrame,
    limit_down: pd.DataFrame,
    adv: pd.Series,
) -> bool:
    """判定某股票在某时点是否可交易 (未停牌、未封板、非新股期)."""
    raise NotImplementedError


def order_size_cap(adv: float, max_participation: float = 0.1) -> float:
    """以 ADV 估算最大可下单规模, 用于容量分析."""
    raise NotImplementedError
