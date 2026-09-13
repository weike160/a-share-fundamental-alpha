"""Q1–Q10 分层收益, 观察信号分组的收益单调性."""
from __future__ import annotations

import pandas as pd

#: 单调性判定的容差
MONOTONIC_TOL = 1e-12


def _assign_quantiles(signal: pd.Series, n_quantiles: int) -> pd.Series:
    """按信号升序分组, 返回 1..n 的组号 (1 = 信号最低)."""
    return pd.qcut(signal, n_quantiles, labels=False, duplicates="drop") + 1


def quantile_returns(
    signals: pd.Series,
    forward_returns: pd.Series,
    n_quantiles: int = 10,
) -> pd.DataFrame:
    """按 signal 分 n 组, 返回各组的平均未来收益.

    Parameters
    ----------
    signals, forward_returns:
        同一横截面的信号与未来收益, 按索引对齐, 缺失配对剔除。
    n_quantiles:
        分组数, 必须 >= 2。

    Returns
    -------
    以分位号 (1 = 信号最低) 为索引的 DataFrame, 列为
    ``[mean_return, median_return, std_return, count]``。

    Raises
    ------
    ValueError
        有效样本少于 ``2 * n_quantiles`` 时 (此时分层没有统计意义)。
    """
    if n_quantiles < 2:
        raise ValueError(f"n_quantiles 必须 >= 2, 收到 {n_quantiles}")

    signal = pd.to_numeric(pd.Series(signals), errors="coerce")
    forward = pd.to_numeric(pd.Series(forward_returns), errors="coerce")
    aligned = pd.DataFrame({"signal": signal, "return": forward}).dropna()
    if len(aligned) < 2 * n_quantiles:
        raise ValueError(
            f"有效样本 {len(aligned)} 少于 2 * n_quantiles = {2 * n_quantiles}"
        )

    quantile = _assign_quantiles(aligned["signal"], n_quantiles)
    grouped = aligned["return"].groupby(quantile)
    out = pd.DataFrame(
        {
            "mean_return": grouped.mean(),
            "median_return": grouped.median(),
            "std_return": grouped.std(ddof=1),
            "count": grouped.count(),
        }
    )
    out.index.name = "quantile"
    return out


def monotonicity_check(
    quantile_returns: pd.DataFrame,
) -> bool:
    """分层收益是否随分位升序单调不减 (容差 ``1e-12``).

    分位少于 3 个, 或收益均值含缺失, 返回 ``False``。
    """
    if "mean_return" not in quantile_returns.columns:
        raise ValueError("quantile_returns 缺少 mean_return 列")
    means = pd.to_numeric(quantile_returns["mean_return"], errors="coerce").sort_index()
    if len(means) < 3:
        return False
    diffs = means.diff().iloc[1:]
    return bool((diffs >= -MONOTONIC_TOL).all())


def quantile_returns_by_date(
    df: pd.DataFrame,
    *,
    signal_col: str,
    return_col: str,
    date_col: str = "actual_disclosure_date",
    n_quantiles: int = 5,
) -> pd.DataFrame:
    """逐日期做横截面分层, 样本不足的日期跳过.

    Parameters
    ----------
    df:
        事件面板长表。
    signal_col, return_col, date_col:
        信号列、未来收益列、日期列。
    n_quantiles:
        每个横截面的分组数。

    Returns
    -------
    长表, 列为 ``[date, quantile, mean_return, count]`` (``date`` 取自
    ``date_col``), 按日期升序。
    """
    for col in (signal_col, return_col, date_col):
        if col not in df.columns:
            raise ValueError(f"df 缺少列 {col!r}")

    work = df[[date_col, signal_col, return_col]].copy()
    work[date_col] = pd.to_datetime(work[date_col])
    work = work.dropna(subset=[date_col])

    columns = ["date", "quantile", "mean_return", "count"]
    rows: list[dict[str, object]] = []
    for dt, grp in work.groupby(date_col, sort=True):
        try:
            table = quantile_returns(grp[signal_col], grp[return_col], n_quantiles)
        except ValueError:
            continue
        for quantile, row in table.iterrows():
            rows.append(
                {
                    "date": dt,
                    "quantile": int(quantile),
                    "mean_return": float(row["mean_return"]),
                    "count": int(row["count"]),
                }
            )

    out = pd.DataFrame(rows, columns=columns)
    if out.empty:
        out = out.astype(
            {
                "date": "datetime64[ns]",
                "quantile": "int64",
                "mean_return": "float64",
                "count": "int64",
            }
        )
    return out
