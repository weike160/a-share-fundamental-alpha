"""数据预处理: 去极值 / 填充 / 标准化, 以及训练/回测的时序划分."""
from __future__ import annotations

import numpy as np
import pandas as pd


def winsorize(series: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    """分位数去极值.

    Parameters
    ----------
    series:
        待处理序列, ``NaN`` 原样保留。
    lower, upper:
        分位点 (0~1), 按**非缺失值**计算; 超出区间的值截断到分位点。

    Returns
    -------
    与输入同索引的序列; 输入为空或全为 ``NaN`` 时返回输入的副本。
    """
    values = pd.to_numeric(series, errors="coerce")
    valid = values.dropna()
    if valid.empty:
        return series.copy()
    return values.clip(valid.quantile(lower), valid.quantile(upper))


def zscore_cross_section(
    frame: pd.DataFrame,
    date: pd.Timestamp,
    *,
    value_col: str,
    date_col: str = "tradable_ts",
) -> pd.Series:
    """横截面标准化 (z-score).

    Parameters
    ----------
    frame:
        长表。
    date:
        目标横截面日期, 只按**日历日**比较 (忽略时分秒)。
    value_col:
        待标准化的数值列。
    date_col:
        日期列。

    Returns
    -------
    与选中行同索引的序列 ``(x - mean) / std``, 标准差用样本口径 (``ddof=1``);
    有效值少于 2 个或标准差为 0 时全为 ``NaN``。
    """
    for col in (value_col, date_col):
        if col not in frame.columns:
            raise ValueError(f"frame 缺少列 {col!r}")

    ts = pd.to_datetime(frame[date_col])
    mask = (ts.dt.normalize() == pd.Timestamp(date).normalize()).to_numpy()
    selected = frame.loc[mask]
    values = pd.to_numeric(selected[value_col], errors="coerce")

    out = pd.Series(np.nan, index=selected.index, dtype="float64")
    valid = values.dropna()
    if len(valid) >= 2:
        std = valid.std(ddof=1)
        if np.isfinite(std) and std > 0:
            positions = np.flatnonzero(values.notna().to_numpy())
            out.iloc[positions] = ((valid - valid.mean()) / std).to_numpy()
    return out


def time_series_split(
    df: pd.DataFrame,
    n_splits: int = 5,
    *,
    date_col: str = "tradable_ts",
    purge_days: int = 0,
    min_train_frac: float = 0.3,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """时序交叉验证划分, 保证无未来信息泄漏.

    唯一日期升序后切成 ``n_splits`` 个连续区间; 第 k 折的测试集是第 k 个区间,
    训练集是日期严格早于 ``区间起点 - purge_days`` 的所有行。第一个区间只做
    训练集, 不产生折。

    Parameters
    ----------
    df:
        长表。
    n_splits:
        区间数, 必须 >= 2。
    date_col:
        日期列。
    purge_days:
        训练集与测试集之间额外空出的日历天数 (防止标签窗口重叠)。
    min_train_frac:
        训练行数低于 ``min_train_frac * len(df)`` 的折被丢弃; 默认会丢掉只含
        第一个区间的折。

    Returns
    -------
    ``(train, test)`` 列表, 均保留原始索引, 且训练集日期严格早于测试集日期。
    """
    if date_col not in df.columns:
        raise ValueError(f"df 缺少列 {date_col!r}")
    if n_splits < 2:
        raise ValueError(f"n_splits 必须 >= 2, 收到 {n_splits}")

    ts = pd.to_datetime(df[date_col])
    dates = np.sort(ts.dropna().unique())
    folds: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    if len(dates) == 0:
        return folds

    min_train = min_train_frac * len(df)
    for block in np.array_split(dates, n_splits)[1:]:
        if len(block) == 0:
            continue
        start = pd.Timestamp(block[0])
        cutoff = start - pd.DateOffset(days=int(purge_days))
        train = df.loc[(ts < cutoff).to_numpy()]
        test = df.loc[((ts >= start) & (ts <= pd.Timestamp(block[-1]))).to_numpy()]
        if len(test) == 0 or len(train) < min_train:
            continue
        folds.append((train, test))
    return folds
