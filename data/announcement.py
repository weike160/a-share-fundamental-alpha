"""真实公告时间对齐.

财报所属季度 ≠ 信息真正公开的时间。本模块负责把财报事件对齐到
真实公告时点, 再对齐到可交易时点, 保证 ``InformationTime < TradingTime``。

口径说明 (实测)
----------------
AKShare 的 ``stock_yysj_em`` 只提供**日期** (``实际披露时间`` 形如
``2024-04-03``), 没有盘中时分秒, 而 A 股财报多数在收盘后披露。因此本模块
统一采取保守假设:

    信号在披露日**收盘之后**才可用, 最早在**披露日之后的第一个交易日**入场。

这样不会把披露当天的收益算进信号, 代价是会损失一部分「盘前披露」样本的
当日反应, 属于偏保守而非偏乐观的偏差。

另一条重要口径: **绝不使用** ``stock_yjbb_em`` 的 ``最新公告日期`` 字段。
实测查询 2024Q1 时该字段只有 73/5896 行落在 2024-04, 92% 落在 2025-04,
与所查报告期脱钩, 用它定信号时点会引入未来函数。
"""
from __future__ import annotations

import pandas as pd

from data.source import fetch
from data.tradability import TradingCalendar

#: ``stock_yysj_em`` 的默认市场范围
DEFAULT_MARKET = "沪深A股"


def _as_calendar(calendar: TradingCalendar | pd.DatetimeIndex) -> TradingCalendar:
    """允许传入 ``TradingCalendar`` 或裸的 ``DatetimeIndex``."""
    if isinstance(calendar, TradingCalendar):
        return calendar
    return TradingCalendar(pd.DatetimeIndex(calendar))


def load_announcements(
    source: str, *, market: str = DEFAULT_MARKET, force: bool = False
) -> pd.DataFrame:
    """载入某报告期的实际披露日期表.

    Parameters
    ----------
    source:
        报告期, 形如 ``"20240331"`` (AKShare 约定用报告期末日)。
    market:
        传给 ``stock_yysj_em`` 的市场范围。
    force:
        忽略缓存强制重新抓取。

    Returns
    -------
    DataFrame, 列为:

    ``code``                  6 位证券代码
    ``report_date``           报告期 (来自 ``source``)
    ``first_scheduled``       首次预约披露日
    ``actual_disclosure_date`` 实际披露日 —— **信号可用时间的唯一依据**
    ``schedule_changes``      预约日期变更次数
    ``ann_ts``                等于 ``actual_disclosure_date`` (无盘中时刻, 见模块说明)
    """
    report_date = pd.Timestamp(str(source))
    raw = fetch("stock_yysj_em", params={"symbol": market, "date": str(source)}, force=force)

    required = {"股票代码", "实际披露时间"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"stock_yysj_em 返回缺少字段: {sorted(missing)}")

    out = pd.DataFrame(
        {
            "code": raw["股票代码"].astype(str).str.zfill(6),
            "report_date": report_date,
            "first_scheduled": pd.to_datetime(
                raw.get("首次预约时间"), errors="coerce"
            ),
            "actual_disclosure_date": pd.to_datetime(
                raw["实际披露时间"], errors="coerce"
            ).dt.normalize(),
        }
    )

    change_cols = [c for c in raw.columns if "变更日期" in str(c)]
    out["schedule_changes"] = (
        raw[change_cols].notna().sum(axis=1).astype(int) if change_cols else 0
    )

    # 无实际披露日的记录无法确定信号时点, 不能进入样本
    out = out.dropna(subset=["actual_disclosure_date"]).reset_index(drop=True)

    # 无盘中时刻: ann_ts 就是披露日当天 00:00, 语义是「该日收盘后可用」
    out["ann_ts"] = out["actual_disclosure_date"]

    return out.sort_values(["code", "report_date"]).reset_index(drop=True)


def to_tradable_ts(
    ann_ts: pd.Series,
    calendar: TradingCalendar | pd.DatetimeIndex,
    *,
    lag: int = 1,
) -> pd.Series:
    """把公告时点对齐到可交易时点, 保证信号在可交易后才生效.

    Parameters
    ----------
    ann_ts:
        公告时点序列。
    calendar:
        交易日历。
    lag:
        入场延迟, 以**交易日**计。

        * ``lag=0`` —— 披露日当天入场 (仅在前市披露的假设下成立, 偏激进)。
        * ``lag=1`` —— 披露日**之后第一个**交易日入场 (默认, 见模块说明)。
        * ``lag=k`` —— 披露日之后第 ``k`` 个交易日入场。

    Returns
    -------
    与输入等长的时点序列, 无法对齐的为 ``NaT``。
    """
    if lag < 0:
        raise ValueError(f"lag 必须 >= 0, 收到 {lag}")
    cal = _as_calendar(calendar)

    def _map(value: object) -> pd.Timestamp | None:
        if value is None or pd.isna(value):
            return None
        if lag == 0:
            # 当天 (含) 最近交易日
            return cal.next_trading_day(value, inclusive=True)
        base = cal.next_trading_day(value, inclusive=False)
        return cal.shift(base, lag - 1) if base is not None else None

    result = [ _map(v) for v in ann_ts ]
    return pd.Series(
        pd.to_datetime(pd.Series(result, index=ann_ts.index), errors="coerce"),
        index=ann_ts.index,
        name="tradable_ts",
    )
