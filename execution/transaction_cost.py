"""交易成本: 佣金 + 印花税 (+ 可选滑点/冲击)."""
from __future__ import annotations

import pandas as pd


def _as_float(values: pd.Series) -> pd.Series:
    """转 float Series, 无法解析的值 -> NaN, 保留原索引."""
    return pd.to_numeric(pd.Series(values), errors="coerce").astype("float64")


def commission(turnover: pd.Series, rate: float = 0.0003) -> pd.Series:
    """双向佣金.

    Parameters
    ----------
    turnover:
        成交额占 NAV 的比例, 非负。
    rate:
        单边佣金率, 默认万 3。

    Returns
    -------
    与 ``turnover`` 同索引的成本 (NAV 占比); 负值输入 -> ``NaN``。
    """
    t = _as_float(turnover)
    return (t * rate).mask(t < 0)


def stamp_tax(sell_turnover: pd.Series, rate: float = 0.0005) -> pd.Series:
    """印花税, 仅卖出单边征收.

    Parameters
    ----------
    sell_turnover:
        卖出成交额占 NAV 的比例, 非负。
    rate:
        印花税率, 默认万 5。

    Returns
    -------
    与 ``sell_turnover`` 同索引的成本 (NAV 占比); 负值输入 -> ``NaN``。
    """
    t = _as_float(sell_turnover)
    return (t * rate).mask(t < 0)


def trade_cost(
    buy_turnover: pd.Series,
    sell_turnover: pd.Series,
    *,
    commission_rate: float = 0.0003,
    stamp_tax_rate: float = 0.0005,
    slippage_rate: float = 0.0,
    impact_rate: float = 0.0,
) -> pd.Series:
    """一次调仓的往返总成本.

    = 双边佣金 + 卖出印花税 + (slippage_rate + impact_rate) * 双边换手。

    Parameters
    ----------
    buy_turnover, sell_turnover:
        买入/卖出成交额占 NAV 的比例。索引取并集: 只在一边出现的名字按 0
        处理, 显式 ``NaN`` 保留并传播为 ``NaN``。
    commission_rate, stamp_tax_rate:
        佣金率与印花税率。
    slippage_rate, impact_rate:
        单位换手的滑点/冲击系数, 默认 0 (未建模)。

    Returns
    -------
    Series, 成本占 NAV 的比例。
    """
    buy = _as_float(buy_turnover)
    sell = _as_float(sell_turnover)
    index = buy.index.union(sell.index)
    buy = buy.reindex(index, fill_value=0.0)
    sell = sell.reindex(index, fill_value=0.0)

    per_side = slippage_rate + impact_rate
    return (
        commission(buy, commission_rate)
        + commission(sell, commission_rate)
        + stamp_tax(sell, stamp_tax_rate)
        + per_side * (buy + sell)
    )
