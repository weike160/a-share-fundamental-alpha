"""财报数据接入与口径处理.

提取净利润 / 营收 / EPS / ROE / 毛利率等基本面字段,
并确保按真实公告时间 (而非财报所属季度) 进入模型。
"""
from __future__ import annotations

import pandas as pd


def load_financials(source: str) -> pd.DataFrame:
    """载入财报数据, 返回宽表: index 为 (code, report_date)."""
    raise NotImplementedError


def attach_announcement_time(
    financials: pd.DataFrame, announcement_table: pd.DataFrame
) -> pd.DataFrame:
    """给每条财报附加真实公告时间 ``ann_date``。

    用于保证 look-ahead 规避: 交易发生在信息真正公开之后。
    """
    raise NotImplementedError
