"""财报事件面板组装.

把财务、公告、股票池、交易日历合成一张 **事件面板**: 每行一个
``(code, report_date)`` 事件, 并显式给出三个时间::

    report_date             财报所属期间       (不能用于定信号时点)
    actual_disclosure_date  实际披露日         (信息真正公开)
    tradable_ts             首个可交易日       (回测入场日)

不变式: ``report_date <= actual_disclosure_date < tradable_ts``。

面板还会附上未来收益列 (``fwd_ret_<h>d``), 供因子检验直接使用。
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import pandas as pd

from data.announcement import load_announcements, to_tradable_ts
from data.delisting import load_delistings
from data.financials import attach_announcement_time, audit_coverage, load_financials
from data.st_status import audit_st, drop_st, flag_st
from data.tradability import TradingCalendar
from data.universe import build_universe, load_listing_table

PROCESSED_DIR = Path(__file__).resolve().parent / "processed"

#: 默认预测期 (交易日)
DEFAULT_HORIZONS: tuple[int, ...] = (5, 10, 20, 40)


def build_event_panel(
    periods: Iterable[str],
    *,
    calendar: TradingCalendar | None = None,
    lag: int = 1,
    force: bool = False,
    filter_universe: bool = True,
    st_policy: str = "drop",
) -> pd.DataFrame:
    """组装事件面板.

    Parameters
    ----------
    periods:
        报告期列表, 形如 ``["20240331", "20240630"]``。
    calendar:
        交易日历; 缺省时从 AKShare 拉取 (带缓存)。
    lag:
        入场延迟 (交易日), 默认 T+1。
    force:
        忽略缓存强制重新抓取。
    filter_universe:
        是否用 point-in-time 股票池过滤。为 ``True`` 时会加载上市 / 退市表,
        剔除**在披露日尚未上市或已经退市**的事件。这是一道防幸存者偏差的
        保险, 但会增加两次网络请求。
    st_policy:
        ST/*ST 处理方式, 见 :mod:`data.st_status`。

        * ``"drop"`` (默认) —— 剔除披露日当天为 ST 的事件。
        * ``"keep"`` —— 保留, 但仍在面板上标记 ``is_st``。
        * ``"flag"`` —— 同 ``keep``, 语义上表示「只标记不处理」。

    Returns
    -------
    事件面板 DataFrame。
    """
    if calendar is None:
        calendar = TradingCalendar.from_akshare(force=force)

    period_list = [str(p) for p in periods]
    if not period_list:
        raise ValueError("periods 不能为空")

    frames: list[pd.DataFrame] = []
    audits: list[pd.DataFrame] = []
    for period in period_list:
        fin = load_financials(period, force=force)
        ann = load_announcements(period, force=force)
        audits.append(audit_coverage(fin, ann))
        merged = attach_announcement_time(fin, ann, how="inner")
        frames.append(merged)

    panel = pd.concat(frames, ignore_index=True)
    panel["report_date"] = pd.to_datetime(panel["report_date"]).dt.normalize()
    panel["actual_disclosure_date"] = pd.to_datetime(
        panel["actual_disclosure_date"]
    ).dt.normalize()

    # 核心: 把披露日映射到可交易入场日
    panel["tradable_ts"] = to_tradable_ts(
        panel["actual_disclosure_date"], calendar, lag=lag
    )
    panel = panel.dropna(subset=["tradable_ts"]).reset_index(drop=True)

    if filter_universe:
        listing = load_listing_table(force=force)
        delistings = load_delistings("all", force=force)
        keep = []
        # 按披露日逐日构造股票池, 避免对每个事件重复扫描全表
        for date, grp in panel.groupby("actual_disclosure_date", sort=True):
            universe = build_universe(date, listing, delistings)
            keep.append(grp[grp["code"].isin(universe)])
        if keep:
            panel = pd.concat(keep, ignore_index=True)
        else:
            panel = panel.iloc[0:0]

    panel["report_date"] = pd.to_datetime(panel["report_date"])
    panel["actual_disclosure_date"] = pd.to_datetime(panel["actual_disclosure_date"])
    panel["tradable_ts"] = pd.to_datetime(panel["tradable_ts"])

    # ST 标记必须在**股票池过滤之后**做, 且基于披露日当时的简称 (见 data.st_status)
    if st_policy not in {"drop", "keep", "flag"}:
        raise ValueError(f"st_policy 只能是 drop/keep/flag, 收到 {st_policy!r}")
    panel = flag_st(panel, force=force)
    st_audit = audit_st(panel)
    if st_policy == "drop":
        panel = drop_st(panel)

    # attrs 会被 pandas 序列化进 parquet, 必须是 JSON 安全类型
    coverage = pd.concat(audits, ignore_index=True)
    coverage["report_date"] = coverage["report_date"].astype(str)
    panel.attrs["coverage_audit"] = coverage.to_dict("records")
    if st_audit is not None:
        panel.attrs["st_audit"] = st_audit.to_dict("records")

    return panel.sort_values(["tradable_ts", "code"]).reset_index(drop=True)


def attach_forward_returns(
    panel: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    calendar: TradingCalendar | None = None,
    price_col: str = "close",
) -> pd.DataFrame:
    """给事件面板附加未来收益.

    入场价为 ``tradable_ts`` 当日收盘价, 出场价为入场后第 ``h`` 个交易日
    收盘价; ``fwd_ret_<h>d = exit / entry - 1``。

    Parameters
    ----------
    panel:
        :func:`build_event_panel` 的输出。
    prices:
        长表 ``[code, date, <price_col>]``, 建议使用**后复权**价。
    horizons:
        预测期 (交易日) 列表。
    calendar:
        用于计算出场日; 缺省时从 AKShare 拉取。
    """
    if {"code", "date", price_col} - set(prices.columns):
        raise ValueError(f"prices 至少需要 [code, date, {price_col}]")
    if calendar is None:
        calendar = TradingCalendar.from_akshare()

    px = prices[["code", "date", price_col]].copy()
    px["code"] = px["code"].astype(str).str.zfill(6)
    px["date"] = pd.to_datetime(px["date"]).dt.normalize()
    lookup = {
        (c, d): v
        for c, d, v in zip(px["code"], px["date"], pd.to_numeric(px[price_col], errors="coerce"))
    }

    out = panel.copy()
    out["code"] = out["code"].astype(str).str.zfill(6)
    entry_dates = pd.to_datetime(out["tradable_ts"]).dt.normalize()
    out["entry_date"] = entry_dates
    out["entry_price"] = [lookup.get((c, d)) for c, d in zip(out["code"], entry_dates)]

    for h in horizons:
        exit_dates = [calendar.shift(d, h) for d in entry_dates]
        out[f"exit_date_{h}d"] = pd.to_datetime(pd.Series(exit_dates, index=out.index))
        exit_px = [
            lookup.get((c, d)) if d is not None and not pd.isna(d) else None
            for c, d in zip(out["code"], exit_dates)
        ]
        out[f"fwd_ret_{h}d"] = (
            pd.Series(exit_px, index=out.index, dtype="float64") / out["entry_price"] - 1.0
        )
    return out


def audit_survivorship(panel: pd.DataFrame, delistings: pd.DataFrame) -> pd.DataFrame:
    """检查样本中「后来退市」的公司数量, 用于暴露数据源层面的幸存者偏差.

    实测发现: ``stock_yjbb_em`` **不会返回已退市公司**的历史财报。例如
    ``000023`` (*ST深天, 2024-09-02 退市) 在 2024Q1 业绩报表里完全不存在,
    尽管它在 2024 年 4 月仍处于上市状态并应披露一季报。

    后果是: 财务样本在**数据源层面**就已经带上了幸存者偏差, 股票池过滤
    无法修复 (没有可过滤的记录)。因此本函数的结果正常情况下会接近 0,
    这本身就是一个需要写进报告的数据限制, 而不是「股票池干净」的证据。
    """
    if len(panel) == 0:
        return pd.DataFrame(
            [
                {
                    "panel_events": 0,
                    "codes_later_delisted": 0,
                    "events_later_delisted": 0,
                    "survivorship_flag": "空样本",
                }
            ]
        )

    if delistings is None or len(delistings) == 0:
        return pd.DataFrame(
            [
                {
                    "panel_events": len(panel),
                    "codes_later_delisted": 0,
                    "events_later_delisted": 0,
                    "survivorship_flag": "无退市表, 无法判断",
                }
            ]
        )

    codes = panel["code"].astype(str).str.zfill(6)
    delisted_codes = set(delistings["code"].astype(str).str.zfill(6))
    hit = codes.isin(delisted_codes).sum()

    return pd.DataFrame(
        [
            {
                "panel_events": len(panel),
                "codes_later_delisted": int(codes[codes.isin(delisted_codes)].nunique()),
                "events_later_delisted": int(hit),
                "survivorship_flag": (
                    "样本内无后来退市公司: 数据源疑似剔除退市股, 存在幸存者偏差"
                    if hit == 0
                    else "样本包含后来退市公司"
                ),
            }
        ]
    )


def audit_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """输出事件面板的审计统计, 供 ``data_audit.csv`` 使用."""
    if len(panel) == 0:
        return pd.DataFrame(
            [
                {
                    "events": 0,
                    "codes": 0,
                    "report_periods": 0,
                    "first_disclosure": pd.NaT,
                    "last_disclosure": pd.NaT,
                    "violations": 0,
                }
            ]
        )
    rd = pd.to_datetime(panel["report_date"])
    ad = pd.to_datetime(panel["actual_disclosure_date"])
    tt = pd.to_datetime(panel["tradable_ts"])
    return pd.DataFrame(
        [
            {
                "events": len(panel),
                "codes": int(panel["code"].nunique()),
                "report_periods": int(pd.Series(rd).nunique()),
                "first_disclosure": ad.min(),
                "last_disclosure": ad.max(),
                # 不变式检查: 报告期 <= 披露日 < 入场日
                "violations": int(((rd > ad) | (ad >= tt)).sum()),
                "industry_null": int(
                    panel.get("industry", pd.Series(dtype=object)).isna().sum()
                ),
            }
        ]
    )


def save_panel(panel: pd.DataFrame, name: str = "event_panel") -> Path:
    """把面板写入 ``data/processed/`` (已被 .gitignore 忽略)."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_DIR / f"{name}.parquet"
    out = panel.copy()
    # parquet 会把 attrs 一并 JSON 序列化, 这里清掉以免非安全类型导致写入失败
    out.attrs = {}
    out.to_parquet(path, index=False)
    return path


def load_panel(name: str = "event_panel") -> pd.DataFrame:
    """读取已保存的面板."""
    path = PROCESSED_DIR / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"面板不存在: {path}")
    return pd.read_parquet(path)
