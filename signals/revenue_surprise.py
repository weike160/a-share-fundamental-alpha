"""营收超预期信号."""
from __future__ import annotations

import pandas as pd

from signals.earnings_growth import yoy_growth


def revenue_surprise(actual_rev: pd.Series, expected_rev: pd.Series) -> pd.Series:
    """营收超预期 = ``(实际营收 - 预期营收) / |预期营收|``, 按索引对齐.

    Parameters
    ----------
    actual_rev:
        实际营收。
    expected_rev:
        预期营收; 为 0 或缺失时输出 ``NaN`` (不用有符号分母, 负预期不翻转方向)。

    Returns
    -------
    与输入对齐的序列, 名为 ``revenue_surprise``。
    """
    actual = pd.to_numeric(actual_rev, errors="coerce")
    expected = pd.to_numeric(expected_rev, errors="coerce")
    valid = expected.notna() & (expected != 0)
    out = (actual - expected) / expected.abs().where(valid)
    return out.where(valid).rename("revenue_surprise")


def revenue_surprise_yoy(actual_rev: pd.Series, lag: int = 4) -> pd.Series:
    """无分析师数据时的营收超预期代理: 单只标的的营收同比.

    Parameters
    ----------
    actual_rev:
        **单只标的**按时间升序的营收序列。
    lag:
        同比滞后阶数。

    Returns
    -------
    与输入同索引的序列, 名为 ``revenue_surprise_yoy``; 规则同
    :func:`signals.earnings_growth.yoy_growth`。
    """
    return yoy_growth(actual_rev, lag=lag).rename("revenue_surprise_yoy")
