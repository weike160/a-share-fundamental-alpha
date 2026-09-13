"""退市处理.

保留并标注所有退市/被收购/破产公司, 用于构造无幸存者偏差的股票池。

口径限制 (实测)
----------------
AKShare 只能给出**两个交易所各自的退市名单**, 且字段语义不同:

* 上交所 ``stock_info_sh_delist()`` 给的是 ``暂停上市日期`` (159 行)。
* 深交所 ``stock_info_sz_delist(symbol="终止上市公司")`` 给的是
  ``终止上市日期`` (208 行)。

两个接口都**无法区分**退市 / 被收购 / 私有化 / 破产, 因此本模块统一把
``delist_type`` 标为 ``"退市"``; 被收购、私有化这两类会整体缺失。
这一点必须写进报告的数据限制, 不能声称股票池已完全消除幸存者偏差。
"""
from __future__ import annotations

import pandas as pd

from data.source import fetch

VALID_SOURCES = ("all", "SH", "SZ")


def _load_sh(force: bool) -> pd.DataFrame:
    raw = fetch("stock_info_sh_delist", force=force)
    return pd.DataFrame(
        {
            "code": raw["公司代码"].astype(str).str.zfill(6),
            "name": raw.get("公司简称", pd.Series(dtype=str)).astype(str),
            "list_date": pd.to_datetime(raw.get("上市日期"), errors="coerce"),
            "delist_date": pd.to_datetime(raw.get("暂停上市日期"), errors="coerce"),
            "exchange": "SH",
        }
    )


def _load_sz(force: bool) -> pd.DataFrame:
    raw = fetch(
        "stock_info_sz_delist", params={"symbol": "终止上市公司"}, force=force
    )
    return pd.DataFrame(
        {
            "code": raw["证券代码"].astype(str).str.zfill(6),
            "name": raw.get("证券简称", pd.Series(dtype=str)).astype(str),
            "list_date": pd.to_datetime(raw.get("上市日期"), errors="coerce"),
            "delist_date": pd.to_datetime(raw.get("终止上市日期"), errors="coerce"),
            "exchange": "SZ",
        }
    )


def load_delistings(source: str = "all", *, force: bool = False) -> pd.DataFrame:
    """载入退市表.

    Parameters
    ----------
    source:
        ``"all"`` (默认) / ``"SH"`` / ``"SZ"``, 指定取哪个交易所的名单。
    force:
        忽略缓存强制重新抓取。

    Returns
    -------
    DataFrame``[code, name, list_date, delist_date, exchange, delist_type]``。

    Notes
    -----
    上交所的 ``暂停上市日期`` 被当作 ``delist_date`` 使用, 它是「暂停上市」
    而非严格的终止上市日, 可能略早于实际摘牌日。这是一处已知近似。
    """
    if source not in VALID_SOURCES:
        raise ValueError(f"source 只能是 {VALID_SOURCES}, 收到 {source!r}")

    frames: list[pd.DataFrame] = []
    if source in ("all", "SH"):
        frames.append(_load_sh(force))
    if source in ("all", "SZ"):
        frames.append(_load_sz(force))

    out = pd.concat(frames, ignore_index=True)
    out["list_date"] = pd.to_datetime(out["list_date"]).dt.normalize()
    out["delist_date"] = pd.to_datetime(out["delist_date"]).dt.normalize()
    # 两个接口都无法区分退市原因, 统一标注 (见模块 docstring)
    out["delist_type"] = "退市"

    return (
        out.dropna(subset=["delist_date"])
        .drop_duplicates(subset=["code"], keep="first")
        .sort_values("code")
        .reset_index(drop=True)
    )


def flag_delisted(
    universe: pd.DataFrame, delistings: pd.DataFrame
) -> pd.DataFrame:
    """标记在历史时点已退市的股票, 供 universe 过滤使用.

    Parameters
    ----------
    universe:
        长表, 至少包含 ``[code, date]``。
    delistings:
        :func:`load_delistings` 的输出, 至少包含 ``[code, delist_date]``。

    Returns
    -------
    ``universe`` 的副本, 追加 ``is_delisted`` 与 ``delist_date`` 两列。
    """
    if {"code", "date"} - set(universe.columns):
        raise ValueError("universe 至少需要 [code, date]")

    out = universe.copy()
    out["code"] = out["code"].astype(str).str.zfill(6)
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()

    if delistings is None or len(delistings) == 0:
        out["is_delisted"] = False
        out["delist_date"] = pd.NaT
        return out

    if {"code", "delist_date"} - set(delistings.columns):
        raise ValueError("delistings 至少需要 [code, delist_date]")

    d = delistings[["code", "delist_date"]].copy()
    d["code"] = d["code"].astype(str).str.zfill(6)
    d["delist_date"] = pd.to_datetime(d["delist_date"], errors="coerce").dt.normalize()
    d = d.drop_duplicates(subset=["code"], keep="first")

    out = out.merge(d, on="code", how="left")
    out["is_delisted"] = out["delist_date"].notna() & (
        out["delist_date"] <= out["date"]
    )
    return out
