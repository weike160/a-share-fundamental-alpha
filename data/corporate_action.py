"""公司行为处理: 分红 / 送股 / 配股 / 拆分 / 复权.

复权方式须与回测用途匹配, 避免把公司行为误当作收益。
"""
from __future__ import annotations

import pandas as pd


def load_corporate_actions(source: str) -> pd.DataFrame:
    """载入公司行为表: ``[code, ex_date, type, factor]``."""
    raise NotImplementedError


def apply_adjustment(
    prices: pd.DataFrame, actions: pd.DataFrame, method: str = "qfq"
) -> pd.DataFrame:
    """对价格做前复权(默认)或后复权处理, 返回调整后价格."""
    raise NotImplementedError
