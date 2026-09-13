"""ST / *ST 风险警示状态的 **point-in-time** 判定.

为什么不能直接用股票当前的名字
------------------------------
实测证明 ``stock_yjbb_em`` 返回的 ``股票简称`` 是**当前**名字, 不是报告期当时的
名字。例如 ``002168`` 的历史是::

    惠程科技 → ST惠程 → 惠程科技 → ST惠程 → *ST惠程 → ST惠程

它在 2024 年 4 月未必叫 ST。若用「当前名字里含 ST」去筛 2024 年的样本, 就是用
未来信息做决策 —— 正是本项目要避免的 look-ahead bias。

数据可得性 (实测)
------------------
* **深市**: ``stock_info_sz_change_name(symbol="简称变更")`` 提供 **7479 条带日期**
  的简称变更记录 (变更日期 / 变更前简称 / 变更后简称), 可以精确还原任意历史
  时点的简称。覆盖 000/001/002/003/200/300/301, **完全不含沪市 6xx**。
* **沪市**: AKShare **没有**带日期的简称变更接口。``stock_info_change_name``
  只返回一个有序名字列表, **没有日期**, 无法定位到具体某一天。

因此本模块的能力边界是:

===========  ==========================  ====================
市场         判定方式                     精度
===========  ==========================  ====================
深市 (SZ)   历史简称时间线                精确 (point-in-time)
沪市 (SH)   退回当前简称                  近似, 存在双向误差
===========  ==========================  ====================

沪市近似的两个方向的误差都要知道:

* **多剔除** —— 2024 年 4 月不是 ST、但今天变成 ST 的公司, 会被误删。
* **漏剔除** —— 2024 年 4 月是 ST、但今天已经摘帽的公司, 会被留下。

后者更危险, 因为它会让 ST 样本混进结果。这项限制必须写进报告。
"""
from __future__ import annotations

import pandas as pd

from data.source import fetch

#: ``st_basis`` 取值
BASIS_SZ_HISTORY = "sz_name_history"      # 深市: 由历史简称精确还原
BASIS_SH_CURRENT = "sh_current_name"      # 沪市: 退回当前简称 (近似)
BASIS_UNKNOWN = "unknown"                 # 无任何依据

#: 风险警示前缀
_ST_PREFIXES = ("*ST", "ST")


def _normalise_name(value: object) -> str:
    """去掉空格与全角空格, 便于前缀匹配."""
    if value is None or pd.isna(value):
        return ""
    return str(value).replace(" ", "").replace("\u3000", "").strip()


def is_st_name(value: object) -> bool:
    """判断一个股票简称是否为风险警示股 (*ST / ST).

    ``S深发展A`` 这类「S」开头表示未完成股改, **不算** ST。
    """
    name = _normalise_name(value)
    return name.startswith(_ST_PREFIXES)


def load_sz_name_changes(force: bool = False) -> pd.DataFrame:
    """载入深市简称变更历史 (带日期).

    Returns
    -------
    DataFrame``[code, change_date, name_before, name_after]``, 仅深市。
    """
    raw = fetch(
        "stock_info_sz_change_name", params={"symbol": "简称变更"}, force=force
    )
    required = {"变更日期", "证券代码", "变更前简称", "变更后简称"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"stock_info_sz_change_name 缺少字段: {sorted(missing)}")

    out = pd.DataFrame(
        {
            "code": raw["证券代码"].astype(str).str.zfill(6),
            "change_date": pd.to_datetime(raw["变更日期"], errors="coerce").dt.normalize(),
            "name_before": raw["变更前简称"],
            "name_after": raw["变更后简称"],
        }
    )
    return (
        out.dropna(subset=["change_date"])
        .drop_duplicates(subset=["code", "change_date"], keep="last")
        .sort_values(["code", "change_date"])
        .reset_index(drop=True)
    )


def build_name_timeline(changes: pd.DataFrame) -> dict[str, list[tuple[pd.Timestamp, str]]]:
    """把变更记录整理成 ``{code: [(生效日, 该日之后的简称), ...]}``.

    每个列表的第一项是该股票**已知最早**的简称 (取自第一条变更记录的
    ``变更前简称``), 生效日取一个极早的时间戳, 保证任何查询日期都能命中。
    """
    timeline: dict[str, list[tuple[pd.Timestamp, str]]] = {}
    if changes is None or len(changes) == 0:
        return timeline

    for code, grp in changes.sort_values("change_date").groupby("code", sort=False):
        entries: list[tuple[pd.Timestamp, str]] = []
        first = grp.iloc[0]
        if pd.notna(first["name_before"]):
            entries.append((pd.Timestamp("1900-01-01"), _normalise_name(first["name_before"])))
        for _, row in grp.iterrows():
            entries.append((row["change_date"], _normalise_name(row["name_after"])))
        timeline[str(code).zfill(6)] = entries
    return timeline


def name_as_of(
    code: str, date: pd.Timestamp, timeline: dict[str, list[tuple[pd.Timestamp, str]]]
) -> str | None:
    """还原某股票在某历史时点的简称; 无记录返回 ``None``."""
    entries = timeline.get(str(code).zfill(6))
    if not entries:
        return None
    date = pd.Timestamp(date).normalize()
    name = None
    for effective, candidate in entries:
        if effective <= date:
            name = candidate
        else:
            break
    return name or entries[0][1]


def flag_st(
    panel: pd.DataFrame,
    *,
    changes: pd.DataFrame | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """给事件面板标记 point-in-time 的 ST 状态.

    Parameters
    ----------
    panel:
        至少包含 ``[code, actual_disclosure_date, name]``。
    changes:
        深市简称变更表; 缺省时自动下载。
    force:
        忽略缓存强制重新抓取。

    Returns
    -------
    ``panel`` 副本, 追加:

    ``name_at_disclosure``  披露日当时的简称 (沪市为当前简称)
    ``is_st``               披露日是否为 ST/*ST
    ``st_basis``            判定依据, 见 ``BASIS_*`` 常量
    """
    if {"code", "actual_disclosure_date"} - set(panel.columns):
        raise ValueError("panel 至少需要 [code, actual_disclosure_date]")

    if changes is None:
        changes = load_sz_name_changes(force=force)
    timeline = build_name_timeline(changes)

    out = panel.copy()
    out["code"] = out["code"].astype(str).str.zfill(6)
    disclosure = pd.to_datetime(out["actual_disclosure_date"]).dt.normalize()
    current_name = (
        out["name"] if "name" in out.columns else pd.Series([None] * len(out), index=out.index)
    )

    names: list[str] = []
    bases: list[str] = []
    for code, date, cur in zip(out["code"], disclosure, current_name):
        hist = name_as_of(code, date, timeline)
        if hist is not None:
            names.append(hist)
            bases.append(BASIS_SZ_HISTORY)
        elif not pd.isna(cur):
            # 沪市 (或无变更记录的深市): 退回当前简称, 标记为近似
            names.append(_normalise_name(cur))
            bases.append(BASIS_SH_CURRENT)
        else:
            names.append("")
            bases.append(BASIS_UNKNOWN)

    out["name_at_disclosure"] = names
    out["st_basis"] = bases
    out["is_st"] = [is_st_name(n) if b != BASIS_UNKNOWN else pd.NA for n, b in zip(names, bases)]
    return out


def drop_st(panel: pd.DataFrame) -> pd.DataFrame:
    """剔除 ``is_st`` 为 True 的事件 (需先调用 :func:`flag_st`)."""
    if "is_st" not in panel.columns:
        raise ValueError("请先调用 flag_st() 生成 is_st 列")
    keep = panel[panel["is_st"] != True]
    return keep.reset_index(drop=True)


def audit_st(panel: pd.DataFrame) -> pd.DataFrame:
    """ST 标记的覆盖率审计, 供 ``data_audit.csv`` 使用.

    需先调用 :func:`flag_st` (与 :func:`drop_st` 同一约定); 空面板自然得到全 0。
    """
    if "is_st" not in panel.columns:
        raise ValueError("请先调用 flag_st() 生成 is_st 列")
    basis = panel["st_basis"].value_counts().to_dict()
    return pd.DataFrame(
        [
            {
                "st_total": len(panel),
                "st_flagged": int((panel["is_st"] == True).sum()),
                "st_basis_sz_history": int(basis.get(BASIS_SZ_HISTORY, 0)),
                "st_basis_sh_current": int(basis.get(BASIS_SH_CURRENT, 0)),
                "st_basis_unknown": int(basis.get(BASIS_UNKNOWN, 0)),
            }
        ]
    )
