"""Point-in-Time 数据层.

对外入口按「下载 -> 对齐 -> 组装」分层::

    data.source        原始响应下载与缓存 (唯一允许访问 akshare 的地方)
    data.tradability   交易日历 / 停牌 / 涨跌停 / ADV
    data.announcement  实际披露日 + 可交易时点对齐
    data.financials    财报字段与公告时间连接
    data.universe      point-in-time 股票池
    data.delisting     退市名单 (幸存者偏差)
    data.corporate_action  分红送配与复权
    data.panel         事件面板组装 (对外主入口)
"""
from __future__ import annotations

from data.announcement import load_announcements, to_tradable_ts
from data.corporate_action import (
    apply_adjustment,
    load_adjusted_prices,
    load_corporate_actions,
)
from data.delisting import flag_delisted, load_delistings
from data.financials import attach_announcement_time, audit_coverage, load_financials
from data.panel import (
    DEFAULT_HORIZONS,
    attach_forward_returns,
    audit_panel,
    audit_survivorship,
    build_event_panel,
    load_panel,
    save_panel,
)
from data.source import (
    DataSourceError,
    cache_paths,
    fetch,
    list_cache,
)
from data.st_status import (
    audit_st,
    build_name_timeline,
    drop_st,
    flag_st,
    is_st_name,
    load_sz_name_changes,
    name_as_of,
)
from data.tradability import (
    TradingCalendar,
    detect_limit_moves,
    infer_suspensions,
    is_tradable,
    order_size_cap,
    price_limit_ratio,
)
from data.universe import (
    build_universe,
    is_listed,
    load_listing_table,
)

__all__ = [
    "DEFAULT_HORIZONS",
    "DataSourceError",
    # tradability
    "TradingCalendar",
    "apply_adjustment",
    "attach_announcement_time",
    "attach_forward_returns",
    "audit_coverage",
    "audit_panel",
    # ST status
    "audit_st",
    "audit_survivorship",
    # panel
    "build_event_panel",
    "build_name_timeline",
    "build_universe",
    "cache_paths",
    "detect_limit_moves",
    "drop_st",
    # source
    "fetch",
    "flag_delisted",
    "flag_st",
    "infer_suspensions",
    "is_listed",
    "is_st_name",
    "is_tradable",
    "list_cache",
    "load_adjusted_prices",
    # announcement
    "load_announcements",
    # corporate action
    "load_corporate_actions",
    # delisting
    "load_delistings",
    # financials
    "load_financials",
    # universe
    "load_listing_table",
    "load_panel",
    "load_sz_name_changes",
    "name_as_of",
    "order_size_cap",
    "price_limit_ratio",
    "save_panel",
    "to_tradable_ts",
]
