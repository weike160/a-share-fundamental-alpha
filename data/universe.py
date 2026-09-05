"""Point-in-Time 股票池构造.

构造历史时点 t 当时真实存在且已上市的股票集合 Universe_t,
并保留后来发生退市 / 被收购 / 私有化 / 破产 / ST 的公司,
以避免 survivorship bias.

TODO(Phase-2): 接入数据源, 生成 `Universe_t` 与股票池事件表。
"""
from __future__ import annotations

import pandas as pd

from data.tradability import TradingCalendar


def build_universe(
    date: pd.Timestamp,
    listing_table: pd.DataFrame,
    delisting_table: pd.DataFrame,
) -> set[str]:
    """返回在历史时点 ``date`` 已经上市且尚未退市的股票代码集合。

    Parameters
    ----------
    date:
        历史观察时点, 用作截止日期.
    listing_table:
        上市日期表, 至少包含 ``[code, list_date]``.
    delisting_table:
        退市日期表, 至少包含 ``[code, delist_date]``.
    """
    raise NotImplementedError


def point_in_time_alignment(
    event_ts: pd.Series, calendar: TradingCalendar
) -> pd.Series:
    """将事件时间对齐到下一个可交易时间, 保证 InformationTime < TradingTime."""
    raise NotImplementedError
