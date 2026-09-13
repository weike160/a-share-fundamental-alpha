"""IC Decay: 研究不同预测期 (1/5/10/20/40/60 天) 的 IC(T).

用来回答: 市场需要多长时间吸收财报信息? Alpha 半衰期有多长?
"""
from __future__ import annotations

import numpy as np
import pandas as pd

#: returns 宽表的列名格式 ``fwd_ret_<h>d``
RETURN_COL_TEMPLATE = "fwd_ret_{horizon}d"


def _spearman(signal: pd.Series, forward_return: pd.Series) -> float:
    """有效配对 >= 2 时的 Spearman 相关, 否则 ``NaN``."""
    pair = pd.DataFrame({"signal": signal, "return": forward_return}).dropna()
    if len(pair) < 2:
        return float("nan")
    value = pair["signal"].corr(pair["return"], method="spearman")
    return float(value) if pd.notna(value) else float("nan")


def ic_decay(
    signal: pd.Series,
    returns: pd.DataFrame,
    horizons: tuple[int, ...] = (1, 5, 10, 20, 40, 60),
    *,
    dates: pd.Series | None = None,
) -> pd.Series:
    """各预测期的 RankIC (Spearman).

    Parameters
    ----------
    signal:
        信号值。
    returns:
        宽表, 列名为 ``fwd_ret_<h>d``, 按索引与 ``signal`` 对齐。
    horizons:
        预测期 (天)。
    dates:
        为 ``None`` 时对全部对齐样本计算一个池化 RankIC; 给定日期序列时先按
        日期分组逐期计算 RankIC, 再对非 ``NaN`` 的期取均值。

    Returns
    -------
    以预测期 (int) 升序为索引的序列, 名为 ``rank_ic``; 对应收益列缺失时为
    ``NaN``。
    """
    values = pd.to_numeric(signal, errors="coerce")
    out: dict[int, float] = {}

    for horizon in horizons:
        key = int(horizon)
        col = RETURN_COL_TEMPLATE.format(horizon=key)
        if col not in returns.columns:
            out[key] = float("nan")
            continue

        forward = pd.to_numeric(returns[col], errors="coerce")
        if dates is None:
            out[key] = _spearman(values, forward)
            continue

        frame = pd.DataFrame(
            {"signal": values, "return": forward, "date": pd.to_datetime(dates)}
        ).dropna()
        per_date = [
            _spearman(grp["signal"], grp["return"])
            for _, grp in frame.groupby("date", sort=True)
        ]
        finite = [value for value in per_date if np.isfinite(value)]
        out[key] = float(np.mean(finite)) if finite else float("nan")

    return pd.Series(out, name="rank_ic", dtype="float64").sort_index()
