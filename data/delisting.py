"""退市处理.

保留并标注所有退市/被收购/破产公司, 用于构造无幸存者偏差的股票池。
"""
from __future__ import annotations

import pandas as pd


def load_delistings(source: str) -> pd.DataFrame:
    """载入退市表: ``[code, delist_date, delist_type]``.

    delist_type 包括: 退市 / 被收购 / 私有化 / 破产 / ST.
    """
    raise NotImplementedError


def flag_delisted(
    universe: pd.DataFrame, delistings: pd.DataFrame
) -> pd.DataFrame:
    """标记在历史时点已退市的股票, 供 universe 过滤使用."""
    raise NotImplementedError
