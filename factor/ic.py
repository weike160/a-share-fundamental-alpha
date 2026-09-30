"""IC、RankIC 及其时序统计。主口径为 Spearman 排名相关。"""
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
    """按索引对齐信号和未来收益，计算横截面相关。有效样本不足或序列为常数时返回 NaN。"""
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
    """宽表逐期计算 IC；只使用共有日期和标的，样本不足的期间保留 NaN。"""
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
    """IC 均值除以样本标准差；有效期数不足或标准差为零时返回 NaN。"""
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
    """汇总 IC 的期数、均值、标准差、ICIR、t 值和正值占比；非空时附最值。"""
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
    """长表按入场日计算横截面 IC，跳过无法计算的日期。"""
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
