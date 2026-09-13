"""Standardized Unexpected Earnings (SUE) —— 财报超预期程度的度量.

核心公式::

    SUE_{i,t} = (ActualEPS_{i,t} − ExpectedEPS_{i,t}) / σ(EarningsSurprise_i)

分子是「实际比预期高多少」, 分母是「这家公司历史上超预期有多不稳定」。
除以分母是为了让不同公司可比 —— 盈利波动大的公司, 超预期 1 毛钱不算什么。

本项目的数据现实
----------------
**没有分析师一致预期数据** (免费渠道拿不到)。因此第一版使用
**季节性随机游走 (seasonal random walk)** 作为预期:

    预期 EPS(i,t) = EPS(i, t−4)      # 去年同一季度

于是未预期盈余 (UE) 与 SUE 为::

    UE(i,t)  = EPS(i,t) − EPS(i,t−4)
    SUE(i,t) = UE(i,t) / std( UE(i, t−k … t−1) )

文档中必须称之为 **「季节性随机游走 SUE」或「SUE proxy」**,
**不能**与基于分析师一致预期的 SUE 混为一谈 (见 research/02_akshare_mvp.md)。

反未来函数要求
--------------
标准差 σ 只能使用**严格早于 t** 的历史 UE。本模块用 ``shift(1)`` 保证这一点,
并有专门的回归测试 (``test_sue_uses_only_past_information``) 守住它。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

#: 默认同比滞后 (4 个季度 = 1 年)
DEFAULT_LAG = 4
#: 默认历史窗口 (用过去多少期 UE 估标准差)
DEFAULT_WINDOW = 8
#: 默认最少有效期数
DEFAULT_MIN_PERIODS = 4
#: sigma 的相对下限 (见 seasonal_random_walk_sue 中的浮点残差说明)
DEFAULT_MIN_SIGMA_REL = 1e-10


def compute_sue(
    actual_eps: pd.Series,
    expected_eps: pd.Series,
    surprise_std: pd.Series,
) -> pd.Series:
    """通用 SUE 计算: ``(actual − expected) / surprise_std``.

    Parameters
    ----------
    actual_eps:
        实际每股收益。
    expected_eps:
        预期每股收益, 可来自分析师一致预期或季节性随机游走代理。
    surprise_std:
        「超预期程度」的历史标准差, 用作分母。

    Returns
    -------
    与输入等长的 SUE 序列。分母为 0、缺失或非正时返回 ``NaN``
    (而不是 inf —— 那会污染后续所有排序与回归)。
    """
    actual = pd.to_numeric(actual_eps, errors="coerce")
    expected = pd.to_numeric(expected_eps, errors="coerce")
    std = pd.to_numeric(surprise_std, errors="coerce")

    ue = actual - expected
    # 分母必须有限且为正; 否则标准化无意义
    valid = std.notna() & np.isfinite(std) & (std > 0)
    sue = ue.where(valid) / std.where(valid)
    return sue.rename("sue")


def sue_from_analyst(
    actual_eps: pd.Series, analyst_forecast: pd.Series
) -> pd.Series:
    """基于分析师一致预期的 SUE.

    ⚠️ **本项目当前不可用**: 免费数据源拿不到分析师一致预期, 因此没有
    历史标准差可作分母。保留此接口是为了数据到位后可以直接接上, 现在调用
    会明确抛错而不是悄悄返回一个错的数。
    """
    raise NotImplementedError(
        "缺少分析师一致预期数据, 无法计算基于分析师预期的 SUE。"
        "请使用 seasonal_random_walk_sue() 作为代理, "
        "并在报告中标注为 'SUE proxy'。"
    )


def seasonal_random_walk_sue(
    panel: pd.DataFrame,
    *,
    eps_col: str = "eps",
    lag: int = DEFAULT_LAG,
    window: int = DEFAULT_WINDOW,
    min_periods: int = DEFAULT_MIN_PERIODS,
    min_sigma_rel: float = DEFAULT_MIN_SIGMA_REL,
    return_components: bool = False,
) -> pd.Series | pd.DataFrame:
    """季节性随机游走 SUE.

    Parameters
    ----------
    panel:
        长表, 至少包含 ``[code, report_date, <eps_col>]``, 且**必须覆盖多个
        报告期** —— 只有单期数据时无法计算 (同比与历史标准差都需要过去)。
    eps_col:
        EPS 列名。
    lag:
        同比滞后, 默认 4 (即去年同期)。
    window:
        估计标准差所用的历史期数, 默认 8。
    min_periods:
        历史 UE 的最少有效期数, 不足则为 ``NaN``, 默认 4。
    return_components:
        为 ``True`` 时额外返回 ``ue`` 与 ``sigma`` 两列, 便于排查。

    Returns
    -------
    ``pd.Series`` (或 ``DataFrame``), 索引与 ``panel`` 对齐。

    Notes
    -----
    每只股票独立计算, 序列按 ``report_date`` 升序:

    1. ``ue = eps − eps.shift(lag)``
    2. ``sigma = ue.shift(1).rolling(window, min_periods).std()`` —— 注意
       ``shift(1)``: 估标准差时**不含当期**, 否则就是用未来信息预测自己。
    3. ``sue = ue / sigma`` (sigma 为 0 或缺失时置 NaN)
    """
    need = {"code", "report_date", eps_col}
    missing = need - set(panel.columns)
    if missing:
        raise ValueError(f"panel 缺少必需列: {sorted(missing)}")
    if lag < 1:
        raise ValueError(f"lag 必须 >= 1, 收到 {lag}")
    if min_periods < 2:
        raise ValueError(f"min_periods 必须 >= 2 才有标准差可言, 收到 {min_periods}")

    df = panel[["code", "report_date", eps_col]].copy()
    df["code"] = df["code"].astype(str).str.zfill(6)
    df["report_date"] = pd.to_datetime(df["report_date"])
    df["_orig"] = df.index
    df = df.sort_values(["code", "report_date"])

    ue_all = pd.Series(np.nan, index=df.index, dtype="float64")
    sigma_all = pd.Series(np.nan, index=df.index, dtype="float64")
    sue_all = pd.Series(np.nan, index=df.index, dtype="float64")

    for _code, grp in df.groupby("code", sort=False):
        eps = pd.to_numeric(grp[eps_col], errors="coerce")
        ue = eps - eps.shift(lag)
        # 关键: shift(1) 保证标准差只用严格过去的信息
        sigma = ue.shift(1).rolling(window=window, min_periods=min_periods).std()

        # 浮点残差防护: 当历史 UE 几乎不变时, rolling.std 会返回 1e-17 这类
        # 「非零但无意义」的值, 直接相除会产生 1e15 量级的 SUE, 彻底破坏
        # Pearson IC 与横截面回归。这里按该股票的量级设一个相对下限。
        scale = float(np.nanmax(np.abs(ue.to_numpy()))) if ue.notna().any() else 0.0
        floor = min_sigma_rel * max(1.0, scale)
        valid = sigma.notna() & np.isfinite(sigma) & (sigma > floor)
        sue = ue.where(valid) / sigma.where(valid)

        ue_all.loc[grp.index] = ue
        sigma_all.loc[grp.index] = sigma
        sue_all.loc[grp.index] = sue

    out = pd.DataFrame(
        {
            "sue": sue_all,
            "ue": ue_all,
            "sigma": sigma_all,
        },
        index=df.index,
    )
    # df.index 保留了调用者传入的标签, reindex 精确还原行序
    out = out.reindex(panel.index)

    if return_components:
        return out
    return out["sue"].rename("sue")


def sue_coverage(panel: pd.DataFrame, sue: pd.Series) -> pd.DataFrame:
    """SUE 覆盖率审计, 供 ``data_audit.csv`` 使用."""
    total = len(panel)
    valid = int(pd.to_numeric(sue, errors="coerce").notna().sum())
    return pd.DataFrame(
        [
            {
                "sue_events": total,
                "sue_valid": valid,
                "sue_coverage": valid / total if total else float("nan"),
                "sue_mean": float(pd.to_numeric(sue, errors="coerce").mean())
                if valid
                else float("nan"),
                "sue_std": float(pd.to_numeric(sue, errors="coerce").std())
                if valid > 1
                else float("nan"),
            }
        ]
    )
