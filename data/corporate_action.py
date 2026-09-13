"""公司行为处理: 分红 / 送股 / 配股 / 拆分 / 复权.

复权方式须与回测用途匹配, 避免把公司行为误当作收益。

复权口径
--------
对每一条除权除息记录, 记除权前收盘价 ``P`` 为 ``prev_close``,
每股现金分红 ``c`` (税前), 每股送转比例 ``r`` (送转总比例 / 10),
则理论除权价为::

    P_ex = (P - c) / (1 + r)

定义单次调整因子 ``k = P_ex / P``。于是:

* **前复权 (qfq)**: ``adj(t) = raw(t) * Π_{ex_date > t} k``
* **后复权 (hfq)**: ``adj(t) = raw(t) / Π_{ex_date <= t} k``

以 10 送 10 (``r=1, c=0``) 检验: ``k = 0.5``, 前复权把除权前价格减半,
与拆股后价格可比; 后复权把除权后价格加倍, 与最初锚点可比。

已知近似
--------
1. 现金分红用**税前**金额, 未扣红利税。
2. **配股 (rights issue) 未处理** —— ``stock_fhps_em`` 不直接给配股比例与
   配股价, 涉及配股的除权日复权会有偏差。
3. 交易所价格按 0.01 取整, 与理论价存在分位误差。
4. 停牌跨越除权日时 ``prev_close`` 取自停牌前最后一个交易日。

因此**回测主口径应使用** :func:`load_adjusted_prices` (直接取 AKShare 的
官方复权价), :func:`apply_adjustment` 用于独立复算与一致性核对。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from data.prices import load_daily_em
from data.source import fetch

#: 有效复权方式
VALID_METHODS = ("qfq", "hfq")


def load_corporate_actions(source: str, *, force: bool = False) -> pd.DataFrame:
    """载入某报告期的分红送配表.

    Parameters
    ----------
    source:
        报告期, 形如 ``"20240331"``。
    force:
        忽略缓存强制重新抓取。

    Returns
    -------
    DataFrame, 列为:

    ``code``                     6 位证券代码
    ``report_date``              报告期
    ``ex_date``                  除权除息日
    ``record_date``              股权登记日
    ``plan_date``                预案公告日
    ``progress``                 方案进度 (如「实施分配」)
    ``cash_dividend_per_share``  每股现金分红 (税前, 元)
    ``share_ratio``              每股送转比例 (股)
    ``type``                     公司行为类型标签
    """
    raw = fetch("stock_fhps_em", params={"date": str(source)}, force=force)
    report_date = pd.Timestamp(str(source))

    if "代码" not in raw.columns:
        raise ValueError("stock_fhps_em 返回缺少 '代码' 字段")

    cash_per_10 = pd.to_numeric(
        raw.get("现金分红-现金分红比例"), errors="coerce"
    )
    share_per_10 = pd.to_numeric(
        raw.get("送转股份-送转总比例"), errors="coerce"
    )

    out = pd.DataFrame(
        {
            "code": raw["代码"].astype(str).str.zfill(6),
            "report_date": report_date,
            "ex_date": pd.to_datetime(raw.get("除权除息日"), errors="coerce").dt.normalize(),
            "record_date": pd.to_datetime(raw.get("股权登记日"), errors="coerce").dt.normalize(),
            "plan_date": pd.to_datetime(raw.get("预案公告日"), errors="coerce").dt.normalize(),
            "progress": raw.get("方案进度", pd.Series(dtype=str)).astype(str),
            # 接口给出的是「每 10 股」口径, 统一换算成每股
            "cash_dividend_per_share": cash_per_10.fillna(0.0) / 10.0,
            "share_ratio": share_per_10.fillna(0.0) / 10.0,
        }
    )

    out["type"] = np.where(
        out["share_ratio"] > 0,
        np.where(out["cash_dividend_per_share"] > 0, "送转+分红", "送转"),
        np.where(out["cash_dividend_per_share"] > 0, "分红", "其他"),
    )

    return (
        out.dropna(subset=["ex_date"])
        .sort_values(["code", "ex_date"])
        .reset_index(drop=True)
    )


def apply_adjustment(
    prices: pd.DataFrame,
    actions: pd.DataFrame,
    method: str = "qfq",
    *,
    price_col: str = "close",
) -> pd.DataFrame:
    """对价格做前复权(默认)或后复权处理.

    Parameters
    ----------
    prices:
        长表, 至少包含 ``[code, date, <price_col>]``。
    actions:
        :func:`load_corporate_actions` 的输出。
    method:
        ``"qfq"`` (前复权) 或 ``"hfq"`` (后复权)。
    price_col:
        要调整的价格列名。

    Returns
    -------
    ``prices`` 的副本, 追加 ``adj_factor`` 与 ``adjusted`` 两列。
    """
    if method not in VALID_METHODS:
        raise ValueError(f"method 只能是 {VALID_METHODS}, 收到 {method!r}")
    if {"code", "date", price_col} - set(prices.columns):
        raise ValueError(f"prices 至少需要 [code, date, {price_col}]")

    df = prices.copy()
    df["code"] = df["code"].astype(str).str.zfill(6)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.sort_values(["code", "date"]).reset_index(drop=True)
    df["adj_factor"] = 1.0

    if actions is None or len(actions) == 0:
        df["adjusted"] = df[price_col]
        return df

    act = actions.copy()
    act["code"] = act["code"].astype(str).str.zfill(6)
    act["ex_date"] = pd.to_datetime(act["ex_date"]).dt.normalize()
    act = act.dropna(subset=["ex_date"])

    for code, grp in df.groupby("code", sort=False):
        acts = act[act["code"] == code].sort_values("ex_date")
        if len(acts) == 0:
            continue

        dates = grp["date"].to_numpy()

        # 每个除权日的因子 k = (P - c) / ((1 + r) * P)
        ks: list[tuple[np.datetime64, float]] = []
        for _, row in acts.iterrows():
            ex = np.datetime64(row["ex_date"])
            prior = grp[grp["date"] < row["ex_date"]]
            if len(prior) == 0:
                continue  # 除权日早于行情起点, 无法取 prev_close
            prev_close = pd.to_numeric(prior[price_col], errors="coerce").iloc[-1]
            if pd.isna(prev_close) or prev_close <= 0:
                continue
            c = float(row["cash_dividend_per_share"])
            r = float(row["share_ratio"])
            k = (prev_close - c) / ((1.0 + r) * prev_close)
            if k > 0:
                ks.append((ex, k))

        if not ks:
            continue

        factor = np.ones(len(dates), dtype="float64")
        for ex, k in ks:
            if method == "qfq":
                # 除权日之前的价格乘以 k
                factor[dates < ex] *= k
            else:
                # 除权日当天及之后的价格除以 k (即乘以 1/k)
                factor[dates >= ex] /= k

        df.loc[grp.index, "adj_factor"] = factor

    df["adjusted"] = pd.to_numeric(df[price_col], errors="coerce") * df["adj_factor"]
    return df


def load_adjusted_prices(
    code: str,
    start_date: str,
    end_date: str,
    *,
    adjust: str = "hfq",
    force: bool = False,
) -> pd.DataFrame:
    """直接取 AKShare 官方复权价的日频行情 (回测主口径).

    Parameters
    ----------
    code:
        6 位证券代码。
    start_date, end_date:
        ``"YYYYMMDD"``。
    adjust:
        ``"hfq"`` (后复权, 默认) / ``"qfq"`` / ``""`` (不复权)。
    """
    out = load_daily_em(code, start_date, end_date, adjust=adjust, force=force)
    out["adjust"] = adjust
    return out
