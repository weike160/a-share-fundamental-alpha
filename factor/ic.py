"""IC / RankIC / ICIR 计算.

IC 是什么, 为什么它是核心
-------------------------
IC (Information Coefficient) = **信号值与未来收益的横截面相关系数**。

关键在于「横截面」三个字: 每个时点内部, 比较「谁比谁强」, 而不是看绝对涨跌。
这样市场整体涨跌会自动抵消 —— 2024 年 4~7 月大盘跌了 9%, 但如果超预期的
股票跌得比别人少, IC 依然为正。

    IC = corr( SUE(i,t),  FutureReturn(i,t) )   在同一个 t 上, 跨 i 计算

三种口径
--------
============  ====================  ==============================
名称           方法                  特点
============  ====================  ==============================
IC            Pearson 相关          受极端值影响大
**RankIC**    Spearman (排名) 相关  **本项目主口径**, 抗极端值
ICIR          Mean(IC)/Std(IC)      衡量信号**稳定性**, 不是强度
============  ====================  ==============================

为什么主口径用 RankIC: SUE 的分布有厚尾 (盈利暴增/暴亏), Pearson 会被少数
极端值主导。排名相关只看次序, 更稳健。

IC 的量级参考 (行业经验, 不是本项目标准)
----------------------------------------
* ``|IC| < 0.02`` —— 基本没用
* ``0.02 ~ 0.05`` —— 弱但可能可用
* ``> 0.05`` —— 不错
* ICIR ``> 0.5`` —— 稳定性尚可

这些只是参照系。**本项目的完成条件不包含任何一个具体阈值** —— 结论可以是
「得到支持」「证据不足」或「被否定」。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

VALID_METHODS = ("pearson", "spearman")

#: 一个横截面至少要这么多有效样本才计算 IC
MIN_OBS = 5


def _validate_method(method: str) -> None:
    if method not in VALID_METHODS:
        raise ValueError(f"method 只能是 {VALID_METHODS}, 收到 {method!r}")


def ic(
    signal: pd.Series,
    forward_return: pd.Series,
    method: str = "pearson",
    *,
    min_obs: int = MIN_OBS,
) -> float:
    """单个横截面的 IC.

    Parameters
    ----------
    signal, forward_return:
        同一横截面内的信号与未来收益, 按标的对齐。
    method:
        ``"pearson"`` (IC) 或 ``"spearman"`` (RankIC)。
    min_obs:
        有效配对样本下限, 不足则返回 ``NaN``。

    Returns
    -------
    float, 无法计算时返回 ``NaN`` (而不是 0 —— 0 会被误读成「无相关性」)。
    """
    _validate_method(method)

    s = pd.to_numeric(pd.Series(signal), errors="coerce")
    r = pd.to_numeric(pd.Series(forward_return), errors="coerce")
    aligned = pd.DataFrame({"s": s, "r": r}).dropna()

    if len(aligned) < min_obs:
        return float("nan")
    # 常数序列的相关系数无定义 (分母为 0)
    if aligned["s"].nunique() < 2 or aligned["r"].nunique() < 2:
        return float("nan")

    value = aligned["s"].corr(aligned["r"], method=method)
    return float(value) if pd.notna(value) else float("nan")


def ic_series(
    signals: pd.DataFrame,
    forward_returns: pd.DataFrame,
    method: str = "spearman",
    *,
    min_obs: int = MIN_OBS,
) -> pd.Series:
    """逐期 IC 序列, 用于计算 ICIR.

    Parameters
    ----------
    signals, forward_returns:
        **宽表**: 索引是时间, 列是标的。两者索引与列会自动对齐。
    method:
        默认 ``"spearman"`` (即 RankIC)。
    min_obs:
        每期最少有效样本数。

    Returns
    -------
    以时间为索引的 IC 序列, 样本不足的期间为 ``NaN``。
    """
    _validate_method(method)

    common_idx = signals.index.intersection(forward_returns.index)
    common_cols = signals.columns.intersection(forward_returns.columns)
    if len(common_idx) == 0 or len(common_cols) == 0:
        return pd.Series(dtype="float64", name=f"ic_{method}")

    s = signals.loc[common_idx, common_cols]
    r = forward_returns.loc[common_idx, common_cols]

    out = {
        ts: ic(s.loc[ts], r.loc[ts], method=method, min_obs=min_obs)
        for ts in common_idx
    }
    return pd.Series(out, name=f"ic_{method}").sort_index()


def icir(ics: pd.Series, *, min_periods: int = 3) -> float:
    """ICIR = Mean(IC) / Std(IC).

    ⚠️ 注意分母是 **标准差** 而不是标准误 —— ICIR 衡量的是「信噪比」,
    不是统计显著性。要判断显著性请用 :func:`t_stat`。

    Parameters
    ----------
    min_periods:
        最少有效期数, 不足返回 ``NaN``。2~3 期算出来的 ICIR 没有意义。

    Returns
    -------
    float, 标准差为 0 或期数不足时返回 ``NaN``。
    """
    s = pd.to_numeric(pd.Series(ics), errors="coerce").dropna()
    if len(s) < min_periods:
        return float("nan")
    std = s.std(ddof=1)
    if not np.isfinite(std) or std == 0:
        return float("nan")
    return float(s.mean() / std)


def t_stat(
    ics: pd.Series, *, lags: int | None = None, min_periods: int = 3
) -> float:
    """IC 均值的 t 统计量 (可选 Newey-West 调整).

    Parameters
    ----------
    lags:
        ``None`` (默认) 用普通 t 统计量 ``mean/se * sqrt(n)``。
        给一个非负整数则用 Newey-West 异方差自相关稳健标准误 —— IC 序列常有
        自相关 (相邻期间的信号重叠), 此时普通 t 会高估显著性。

    Returns
    -------
    float, 期数不足时返回 ``NaN``。
    """
    s = pd.to_numeric(pd.Series(ics), errors="coerce").dropna()
    n = len(s)
    if n < min_periods:
        return float("nan")

    mean = s.mean()
    demeaned = s - mean

    if lags is None:
        se = s.std(ddof=1) / np.sqrt(n)
    else:
        if lags < 0:
            raise ValueError(f"lags 必须 >= 0, 收到 {lags}")
        # Newey-West: gamma_0 + 2 * sum_{l=1..L} (1 - l/(L+1)) * gamma_l
        gamma0 = float((demeaned**2).sum() / n)
        var = gamma0
        for lag in range(1, min(lags, n - 1) + 1):
            weight = 1.0 - lag / (lags + 1.0)
            cov = float((demeaned.iloc[lag:].to_numpy() * demeaned.iloc[:-lag].to_numpy()).sum() / n)
            var += 2.0 * weight * cov
        se = np.sqrt(var / n) if var > 0 else np.nan

    if not np.isfinite(se) or se == 0:
        return float("nan")
    return float(mean / se)


def ic_summary(ics: pd.Series, *, nw_lags: int | None = None) -> pd.DataFrame:
    """IC 序列的完整汇总, 供结果表与 ``factor_metrics.json`` 使用.

    Returns
    -------
    单行 DataFrame: 期数、IC 均值、标准差、ICIR、t 值、正 IC 占比、最值。
    """
    s = pd.to_numeric(pd.Series(ics), errors="coerce").dropna()
    if len(s) == 0:
        return pd.DataFrame(
            [
                {
                    "periods": 0,
                    "ic_mean": float("nan"),
                    "ic_std": float("nan"),
                    "icir": float("nan"),
                    "t_stat": float("nan"),
                    "positive_rate": float("nan"),
                }
            ]
        )
    return pd.DataFrame(
        [
            {
                "periods": len(s),
                "ic_mean": float(s.mean()),
                "ic_std": float(s.std(ddof=1)) if len(s) > 1 else float("nan"),
                "icir": icir(s),
                "t_stat": t_stat(s, lags=nw_lags),
                "positive_rate": float((s > 0).mean()),
                "ic_min": float(s.min()),
                "ic_max": float(s.max()),
            }
        ]
    )


def ic_by_date(
    panel: pd.DataFrame,
    *,
    signal_col: str = "sue",
    return_col: str = "fwd_ret_20d",
    date_col: str = "tradable_ts",
    method: str = "spearman",
    min_obs: int = MIN_OBS,
) -> pd.Series:
    """事件面板 (长表) 上按入场日分组计算 IC.

    这是本项目 PEAD 研究的**主口径**: 每个入场日构成一个横截面, 截面内比较
    「SUE 高的股票 vs SUE 低的股票, 之后收益谁更强」。

    Parameters
    ----------
    panel:
        事件面板长表。
    signal_col, return_col, date_col:
        信号列、未来收益列、分组日期列。

    Returns
    -------
    以入场日为索引的 IC 序列 (仅包含样本充足的日期)。
    """
    for col in (signal_col, return_col, date_col):
        if col not in panel.columns:
            raise ValueError(f"panel 缺少列 {col!r}")

    df = panel[[date_col, signal_col, return_col]].copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.dropna(subset=[date_col])

    out = {
        ts: ic(grp[signal_col], grp[return_col], method=method, min_obs=min_obs)
        for ts, grp in df.groupby(date_col, sort=True)
    }
    series = pd.Series(out, name=f"ic_{method}").sort_index()
    return series.dropna()
