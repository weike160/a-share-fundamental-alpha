"""真实公告时间对齐.

财报所属季度 ≠ 信息真正公开的时间。本模块负责将
财报事件与真实公告时点 (包含精确到时分秒) 对齐。
"""
from __future__ import annotations

import pandas as pd


def load_announcements(source: str) -> pd.DataFrame:
    """载入公告表: ``[code, report_date, ann_ts]`` (ann_ts 精确到秒)."""
    raise NotImplementedError


def to_tradable_ts(ann_ts: pd.Series, calendar: pd.DatetimeIndex) -> pd.Series:
    """把公告时点对齐到下一个可交易时点, 保证信号在可交易后才生效."""
    raise NotImplementedError
