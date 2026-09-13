"""财报数据接入与口径处理.

提取净利润 / 营收 / EPS / ROE / 毛利率等基本面字段,
并确保按真实公告时间 (而非财报所属季度) 进入模型。

⚠️ 关于 ``最新公告日期``
------------------------
``stock_yjbb_em`` 返回一个 ``最新公告日期`` 字段, 但它**不能**用来确定信号
可用时间。实测查询 2024Q1 (``date="20240331"``) 时, 该字段分布为::

    2024-04      73      <- 本应全部落在这里
    2025-04    5448      <- 92% 落在一年之后
    其余        375

说明它与所查报告期脱钩, 更像是公司最近一次公告的日期。本模块只在
``akshare_latest_notice_date`` 名下**原样存档**该字段供人工审计,
信号时点一律来自 :mod:`data.announcement` 的 ``actual_disclosure_date``。
"""
from __future__ import annotations

import pandas as pd

from data.source import fetch

#: AKShare 中文字段 -> 项目内部字段
FIELD_MAP: dict[str, str] = {
    "股票代码": "code",
    "股票简称": "name",
    "每股收益": "eps",
    "营业总收入-营业总收入": "revenue",
    "营业总收入-同比增长": "revenue_yoy",
    "净利润-净利润": "net_profit",
    "净利润-同比增长": "net_profit_yoy",
    "净资产收益率": "roe",
    "销售毛利率": "gross_margin",
    "每股净资产": "bps",
    "每股经营现金流量": "ocf_per_share",
    "所处行业": "industry",
    # 仅存档审计, 严禁用于信号时点 (见模块 docstring)
    "最新公告日期": "akshare_latest_notice_date",
}

#: 需要转成数值的列
_NUMERIC = [
    "eps",
    "revenue",
    "revenue_yoy",
    "net_profit",
    "net_profit_yoy",
    "roe",
    "gross_margin",
    "bps",
    "ocf_per_share",
]


def load_financials(source: str, *, force: bool = False) -> pd.DataFrame:
    """载入某报告期的财报数据.

    Parameters
    ----------
    source:
        报告期, 形如 ``"20240331"``。
    force:
        忽略缓存强制重新抓取。

    Returns
    -------
    DataFrame, 每行一个 ``(code, report_date)``, 含 ``FIELD_MAP`` 中的字段。
    """
    raw = fetch("stock_yjbb_em", params={"date": str(source)}, force=force)
    report_date = pd.Timestamp(str(source))

    if "股票代码" not in raw.columns:
        raise ValueError("stock_yjbb_em 返回缺少 '股票代码' 字段")

    present = {k: v for k, v in FIELD_MAP.items() if k in raw.columns}
    out = raw[list(present)].rename(columns=present).copy()
    out["code"] = out["code"].astype(str).str.zfill(6)
    out["report_date"] = report_date

    for col in _NUMERIC:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    if "akshare_latest_notice_date" in out.columns:
        out["akshare_latest_notice_date"] = pd.to_datetime(
            out["akshare_latest_notice_date"], errors="coerce"
        ).dt.normalize()

    ordered = ["code", "report_date"] + [
        c for c in out.columns if c not in {"code", "report_date"}
    ]
    return out[ordered].drop_duplicates(subset=["code", "report_date"]).reset_index(
        drop=True
    )


def attach_announcement_time(
    financials: pd.DataFrame,
    announcement_table: pd.DataFrame,
    *,
    how: str = "inner",
) -> pd.DataFrame:
    """给每条财报附加真实公告时间.

    用于保证 look-ahead 规避: 交易发生在信息真正公开之后。

    Parameters
    ----------
    financials:
        :func:`load_financials` 的输出。
    announcement_table:
        :func:`data.announcement.load_announcements` 的输出。
    how:
        默认 ``"inner"`` —— 匹配不到公告日的财报会被**丢弃**。这是刻意的:
        保留它们只能靠 ``最新公告日期`` 之类不可信的字段补时间, 那会引入
        未来函数。需要审计丢失量时用 :func:`audit_coverage`。

    Returns
    -------
    ``financials`` 左表加上 ``actual_disclosure_date`` / ``ann_ts`` /
    ``schedule_changes`` 三列。
    """
    ann_cols = ["code", "report_date", "actual_disclosure_date", "ann_ts"]
    for optional in ("first_scheduled", "schedule_changes"):
        if optional in announcement_table.columns:
            ann_cols.append(optional)

    missing = set(ann_cols) - set(announcement_table.columns)
    if missing:
        raise ValueError(f"announcement_table 缺少字段: {sorted(missing)}")

    merged = financials.merge(
        announcement_table[ann_cols], on=["code", "report_date"], how=how
    )
    return merged.sort_values(["report_date", "code"]).reset_index(drop=True)


def audit_coverage(
    financials: pd.DataFrame, announcement_table: pd.DataFrame
) -> pd.DataFrame:
    """生成财务表与公告表的匹配审计, 供 ``data_audit.csv`` 使用.

    Returns
    -------
    单行 DataFrame: 报告期、两侧行数、交集、单边行数与匹配率。
    """
    f_codes = set(financials["code"].astype(str).str.zfill(6))
    a_codes = set(announcement_table["code"].astype(str).str.zfill(6))
    inter = f_codes & a_codes

    period = (
        pd.Timestamp(financials["report_date"].iloc[0])
        if len(financials)
        else pd.NaT
    )
    return pd.DataFrame(
        [
            {
                "report_date": period,
                "financials_rows": len(financials),
                "announcement_rows": len(announcement_table),
                "financials_codes": len(f_codes),
                "announcement_codes": len(a_codes),
                "matched_codes": len(inter),
                "financials_only": len(f_codes - a_codes),
                "announcement_only": len(a_codes - f_codes),
                "match_rate": (len(inter) / len(f_codes)) if f_codes else float("nan"),
            }
        ]
    )
