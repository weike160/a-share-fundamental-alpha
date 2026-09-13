"""增长类基本面信号: 净利润 / 营收 / EPS 同比变化, ROE 变化, 毛利率变化."""
from __future__ import annotations

import numpy as np
import pandas as pd

#: 同比默认滞后 (4 个季度 = 1 年)
DEFAULT_LAG = 4

#: earnings_growth 的数值列 -> 输出列名
GROWTH_VALUE_COLS: dict[str, str] = {
    "net_profit": "net_profit_yoy_calc",
    "revenue": "revenue_yoy_calc",
    "eps": "eps_yoy_calc",
}


def yoy_growth(financial: pd.Series, lag: int = 4) -> pd.Series:
    """同比变化率 ``x_t / x_{t-lag} - 1``, 按位置取滞后值.

    与 ``pct_change`` 的差别: 去年同期为 0 时返回 ``NaN`` 而不是 ``inf``。

    Parameters
    ----------
    financial:
        单只标的的时序数值序列。
    lag:
        同比滞后阶数, 必须 >= 1。

    Returns
    -------
    与输入等长、同索引的序列; 当期值缺失、去年同期值缺失或为 0 时为 ``NaN``。
    """
    if lag < 1:
        raise ValueError(f"lag 必须 >= 1, 收到 {lag}")
    values = pd.to_numeric(financial, errors="coerce")
    base = values.shift(lag)
    valid = values.notna() & base.notna() & (base != 0)
    out = (values / base.where(valid)) - 1.0
    return out.where(valid).rename(financial.name)


def _sorted_work(
    financials: pd.DataFrame, required: set[str], value_cols: tuple[str, ...]
) -> pd.DataFrame:
    """校验必需列, 按 (code, report_date) 稳定排序并记录原始行位置.

    ``value_cols`` 中存在的列会先转成数值再一起排序, 避免排序后按位置错位。
    """
    missing = required - set(financials.columns)
    if missing:
        raise ValueError(f"financials 缺少必需列: {sorted(missing)}")

    data: dict[str, object] = {
        "code": financials["code"].astype(str).str.zfill(6).to_numpy(),
        "report_date": pd.to_datetime(financials["report_date"]).to_numpy(),
        "_orig_pos": np.arange(len(financials)),
    }
    for col in value_cols:
        if col in financials.columns:
            data[col] = pd.to_numeric(financials[col], errors="coerce").to_numpy()

    work = pd.DataFrame(data)
    return work.sort_values(
        ["code", "report_date", "_orig_pos"], kind="mergesort"
    ).reset_index(drop=True)


def _original_order(work: pd.DataFrame) -> np.ndarray:
    """排序后各行在调用者原始输入中的行位置."""
    return np.argsort(work["_orig_pos"].to_numpy(), kind="stable")


def earnings_growth(financials: pd.DataFrame) -> pd.DataFrame:
    """净利润 / 营收 / EPS 同比.

    Parameters
    ----------
    financials:
        长表, 至少包含 ``[code, report_date]``; 数值列取
        ``net_profit`` / ``revenue`` / ``eps``, 缺失的数值列对应输出全为 ``NaN``。

    Returns
    -------
    列 ``[net_profit_yoy_calc, revenue_yoy_calc, eps_yoy_calc]``,
    行序与索引与输入完全一致。
    """
    work = _sorted_work(
        financials, {"code", "report_date"}, tuple(GROWTH_VALUE_COLS)
    )

    out: dict[str, pd.Series | float] = {}
    for value_col, out_col in GROWTH_VALUE_COLS.items():
        if value_col not in work.columns:
            out[out_col] = np.nan
            continue
        out[out_col] = work.groupby("code", sort=False)[value_col].transform(yoy_growth)

    result = pd.DataFrame(out, index=work.index)
    result = result.iloc[_original_order(work)]
    result.index = financials.index
    return result


def roe_change(financials: pd.DataFrame) -> pd.Series:
    """ROE 的同比变化 ``roe_t - roe_{t-4}``.

    Parameters
    ----------
    financials:
        长表, 至少包含 ``[code, report_date, roe]``。

    Returns
    -------
    与输入同索引的序列 ``roe_change``, 任一侧缺失时为 ``NaN``。

    Raises
    ------
    ValueError
        缺少 ``code`` / ``report_date`` / ``roe`` 时。
    """
    work = _sorted_work(financials, {"code", "report_date", "roe"}, ("roe",))
    changed = work.groupby("code", sort=False)["roe"].transform(
        lambda s: s - s.shift(DEFAULT_LAG)
    )
    result = changed.iloc[_original_order(work)]
    result.index = financials.index
    return result.rename("roe_change")
