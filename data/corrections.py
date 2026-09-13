"""财报「后续更正 / 修订」检测.

为什么需要这个
--------------
人工抽查发现了一个真实案例: ``688084 晶品特装`` 的 2024 一季报在
**2024-04-25** 首次披露, 但因为交易所问询函, 公司又在 **2024-07-11**
发布了「2024年第一季度报告(更正版)」。

问题在于 ``stock_yjbb_em`` 返回的财务数值**可能已经是更正后的版本**,
而我们的入场日是 2024-04-26 —— 那天市场看到的是**原始版**。

这就构成一处轻微但真实的**未来函数**::

    披露日 2024-04-25  →  入场 2024-04-26  →  更正 2024-07-11
                            ↑
                    此刻可用信息 = 原始版
                    但我们用的可能是 7 月那版

计划文档对此早有预警:

    「AKShare 返回的历史财务报表可能包含后续修订……无法恢复历史版本时,
      结果中必须保留对应限制。」

本模块用**巨潮资讯网**(交易所指定披露平台)的完整公告历史, 把这件事从
「理论风险」变成**可量化、可标记**的事实, 供后续做敏感性分析。

判定口径
--------
对每个事件, 检索披露日之后 ``window_days`` 天内的全部公告, 命中同时满足
两个条件的记录:

1. 标题含「更正」或「修订」等字样;
2. 标题含该报告期对应的报告名称 (如 ``2024-03-31`` → 「第一季度报告」)。

这样能区分「年报被修订」(不影响一季报) 与「我们正在用的那份报告被更正」。
"""
from __future__ import annotations

import re
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from data.source import fetch

warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

#: 默认检索窗口 (自然日)
DEFAULT_WINDOW_DAYS = 120
DEFAULT_WORKERS = 4

#: 更正 / 修订 关键词
_CORRECTION_RE = re.compile(r"更正|修订|勘误|补充公告")

#: 报告期 -> 报告名称
_REPORT_LABELS: dict[tuple[int, int], str] = {
    (3, 31): "第一季度报告",
    (6, 30): "半年度报告",
    (9, 30): "第三季度报告",
    (12, 31): "年度报告",
}


def report_label(report_date: object) -> str:
    """把报告期映射成报告名称, 例如 ``2024-03-31`` → 「第一季度报告」."""
    ts = pd.Timestamp(report_date)
    return _REPORT_LABELS.get((ts.month, ts.day), "报告")


def _fetch_announcements(
    code: str, start_date: str, end_date: str, *, force: bool
) -> pd.DataFrame:
    """取某股票某区间的公告历史 (带缓存)."""
    return fetch(
        "stock_zh_a_disclosure_report_cninfo",
        params={
            "symbol": str(code).zfill(6),
            "market": "沪深京",
            "keyword": "",
            "category": "",
            "start_date": str(start_date),
            "end_date": str(end_date),
        },
        force=force,
    )


def scan_corrections(
    panel: pd.DataFrame,
    *,
    window_days: int = DEFAULT_WINDOW_DAYS,
    max_workers: int = DEFAULT_WORKERS,
    force: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """扫描每个事件在披露日之后是否发生过报告更正.

    Parameters
    ----------
    panel:
        至少包含 ``[code, report_date, actual_disclosure_date]``。
    window_days:
        披露日之后检索多少自然日。
    max_workers:
        并发线程数。
    force:
        忽略缓存强制重新抓取。

    Returns
    -------
    DataFrame, 每行对应一个事件:

    ``code``, ``report_date``
        事件标识
    ``announcements_total``
        窗口内公告总数 (用于判断接口是否真的返回了数据)
    ``correction_count``
        窗口内命中「更正/修订」的公告数 (不限报告期)
    ``report_correction_count``
        其中标题还命中该报告期名称的条数 —— **这是真正要紧的数字**
    ``report_corrected``
        bool, 我们正在用的那份报告是否被更正过
    ``correction_titles``
        命中的公告标题 (用 `` | `` 连接), 便于人工复核
    """
    need = {"code", "report_date", "actual_disclosure_date"}
    missing = need - set(panel.columns)
    if missing:
        raise ValueError(f"panel 缺少必需列: {sorted(missing)}")

    events = panel[["code", "report_date", "actual_disclosure_date"]].copy()
    events["code"] = events["code"].astype(str).str.zfill(6)
    events["report_date"] = pd.to_datetime(events["report_date"])
    events["actual_disclosure_date"] = pd.to_datetime(events["actual_disclosure_date"])
    events = events.reset_index(drop=True)
    events["_row"] = events.index

    # 相同 (code, 窗口) 只请求一次
    events["_start"] = (events["actual_disclosure_date"] + pd.Timedelta(days=1)).dt.strftime("%Y%m%d")
    events["_end"] = (events["actual_disclosure_date"] + pd.Timedelta(days=window_days)).dt.strftime("%Y%m%d")
    keys = events[["code", "_start", "_end"]].drop_duplicates()

    results: dict[tuple[str, str, str], pd.DataFrame | None] = {}
    errors: dict[tuple[str, str, str], str] = {}

    if verbose:
        print(f"    公告扫描: {len(keys)} 个请求 (共 {len(events)} 个事件)", file=sys.stderr)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                _fetch_announcements, row["code"], row["_start"], row["_end"], force=force
            ): (row["code"], row["_start"], row["_end"])
            for _, row in keys.iterrows()
        }
        for done, fut in enumerate(as_completed(futures), start=1):
            key = futures[fut]
            try:
                results[key] = fut.result()
            except Exception as exc:  # noqa: BLE001 - 单只失败不应中断整体
                results[key] = None
                errors[key] = f"{type(exc).__name__}: {exc}"
            if verbose and (done % 250 == 0 or done == len(futures)):
                print(
                    f"    公告扫描 {done}/{len(futures)} (失败 {len(errors)})",
                    file=sys.stderr,
                    flush=True,
                )

    records: list[dict[str, object]] = []
    for _, ev in events.iterrows():
        key = (ev["code"], ev["_start"], ev["_end"])
        raw = results.get(key)
        label = report_label(ev["report_date"])

        if raw is None or len(raw) == 0:
            records.append(
                {
                    "code": ev["code"],
                    "report_date": ev["report_date"],
                    "announcements_total": 0 if raw is not None else pd.NA,
                    "correction_count": 0,
                    "report_correction_count": 0,
                    "report_corrected": False,
                    "correction_titles": "",
                    "scan_error": errors.get(key, "" if raw is not None else "无数据"),
                }
            )
            continue

        titles = raw["公告标题"].astype(str)
        is_corr = titles.str.contains(_CORRECTION_RE, na=False)
        hit = titles[is_corr]
        report_hit = hit[hit.str.contains(label, na=False)]

        records.append(
            {
                "code": ev["code"],
                "report_date": ev["report_date"],
                "announcements_total": len(raw),
                "correction_count": int(is_corr.sum()),
                "report_correction_count": len(report_hit),
                "report_corrected": bool(len(report_hit) > 0),
                "correction_titles": " | ".join(hit.tolist()),
                "scan_error": "",
            }
        )

    out = pd.DataFrame(records)
    out["report_date"] = pd.to_datetime(out["report_date"])
    return out


def audit_corrections(scan: pd.DataFrame) -> pd.DataFrame:
    """更正扫描的汇总审计, 供 ``data_audit.csv`` 使用."""
    if scan is None or len(scan) == 0:
        return pd.DataFrame(
            [{"correction_events": 0, "report_corrected": 0, "correction_rate": float("nan")}]
        )
    corrected = int(scan["report_corrected"].sum())
    return pd.DataFrame(
        [
            {
                "correction_events": len(scan),
                "announcements_scanned": int(
                    pd.to_numeric(scan["announcements_total"], errors="coerce").fillna(0).sum()
                ),
                "report_corrected": corrected,
                "correction_rate": corrected / len(scan) if len(scan) else float("nan"),
                "scan_errors": int((scan.get("scan_error", "") != "").sum())
                if "scan_error" in scan.columns
                else 0,
            }
        ]
    )
