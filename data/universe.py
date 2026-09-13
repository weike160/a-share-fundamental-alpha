"""Point-in-Time 股票池构造.

构造历史时点 t 当时真实存在且已上市的股票集合 Universe_t,
并保留后来发生退市 / 被收购 / 私有化 / 破产 / ST 的公司,
以避免 survivorship bias.

关键点: 股票池必须由**上市日 + 退市日**两个日期共同决定, 而不能用今天
还在交易的代码表。实测 ``stock_info_a_code_name()`` 只返回 5562 只**当前**
上市股票, 退市公司完全不在其中; 用它构造历史股票池会直接产生幸存者偏差。
"""
from __future__ import annotations

import pandas as pd

from data.source import fetch

#: 上交所 A 股板块 -> ``stock_info_sh_name_code`` 的 symbol
SH_BOARDS: tuple[tuple[str, str], ...] = (
    ("主板A股", "主板"),
    ("科创板", "科创板"),
)


def load_listing_table(force: bool = False) -> pd.DataFrame:
    """合并沪深两市的上市日期表.

    Returns
    -------
    DataFrame[code, name, list_date, board, exchange]
    """
    frames: list[pd.DataFrame] = []

    for symbol, board in SH_BOARDS:
        raw = fetch("stock_info_sh_name_code", params={"symbol": symbol}, force=force)
        if "证券代码" not in raw.columns or "上市日期" not in raw.columns:
            raise ValueError(f"stock_info_sh_name_code({symbol}) 字段异常")
        frames.append(
            pd.DataFrame(
                {
                    "code": raw["证券代码"].astype(str).str.zfill(6),
                    "name": raw.get("证券简称", pd.Series(dtype=str)).astype(str),
                    "list_date": pd.to_datetime(raw["上市日期"], errors="coerce"),
                    "board": board,
                    "exchange": "SH",
                }
            )
        )

    raw = fetch("stock_info_sz_name_code", params={"symbol": "A股列表"}, force=force)
    if "A股代码" not in raw.columns or "A股上市日期" not in raw.columns:
        raise ValueError("stock_info_sz_name_code 字段异常")
    board_map = {"主板": "主板", "创业板": "创业板", "中小板": "中小板"}
    frames.append(
        pd.DataFrame(
            {
                "code": raw["A股代码"].astype(str).str.zfill(6),
                "name": raw.get("A股简称", pd.Series(dtype=str)).astype(str),
                "list_date": pd.to_datetime(raw["A股上市日期"], errors="coerce"),
                "board": raw.get("板块", pd.Series(dtype=str))
                .astype(str)
                .map(lambda b: board_map.get(b, b)),
                "exchange": "SZ",
            }
        )
    )

    out = pd.concat(frames, ignore_index=True)
    out["list_date"] = pd.to_datetime(out["list_date"]).dt.normalize()
    return (
        out.dropna(subset=["list_date"])
        .drop_duplicates(subset=["code"], keep="first")
        .sort_values("code")
        .reset_index(drop=True)
    )


def _normalise_dates(df: pd.DataFrame, col: str, code_col: str = "code") -> pd.DataFrame:
    """统一代码补零与日期归一."""
    out = df[[code_col, col]].copy()
    out[code_col] = out[code_col].astype(str).str.zfill(6)
    out[col] = pd.to_datetime(out[col], errors="coerce").dt.normalize()
    return out


def build_universe(
    date: pd.Timestamp,
    listing_table: pd.DataFrame,
    delisting_table: pd.DataFrame,
) -> set[str]:
    """返回在历史时点 ``date`` 已经上市且尚未退市的股票代码集合。

    Parameters
    ----------
    date:
        历史观察时点, 用作截止日期。
    listing_table:
        上市日期表, 至少包含 ``[code, list_date]``。
    delisting_table:
        退市日期表, 至少包含 ``[code, delist_date]``。可以为空。

    Notes
    -----
    纳入条件为 ``list_date <= date`` 且 (无退市日 或 ``delist_date > date``)。
    ``delist_date == date`` 视为**当日已退市**, 不计入, 以免在退市日仍建仓。
    """
    if {"code", "list_date"} - set(listing_table.columns):
        raise ValueError("listing_table 至少需要 [code, list_date]")

    date = pd.Timestamp(date).normalize()
    listing = _normalise_dates(listing_table, "list_date")
    listed = listing[listing["list_date"] <= date]

    codes = set(listed["code"])

    if delisting_table is not None and len(delisting_table) > 0:
        if {"code", "delist_date"} - set(delisting_table.columns):
            raise ValueError("delisting_table 至少需要 [code, delist_date]")
        delisted = _normalise_dates(delisting_table, "delist_date")
        already_gone = set(delisted[delisted["delist_date"] <= date]["code"])
        codes -= already_gone

    return codes


def is_listed(
    code: str,
    date: pd.Timestamp,
    listing_table: pd.DataFrame,
    delisting_table: pd.DataFrame | None = None,
) -> bool:
    """判断单只股票在某时点是否处于上市状态."""
    return str(code).zfill(6) in build_universe(
        date, listing_table, delisting_table if delisting_table is not None else pd.DataFrame()
    )
