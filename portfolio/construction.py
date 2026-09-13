"""组合构建.

默认先做 Long-only benchmark-relative portfolio,
避免一开始就处理 A 股融券问题。
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


def _as_float(values: pd.Series) -> pd.Series:
    """转 float Series, 无法解析的值 -> NaN, 保留原索引."""
    return pd.to_numeric(pd.Series(values), errors="coerce").astype("float64")


def _normalize(weights: pd.Series) -> pd.Series:
    """归一到和为 1; 和 <= 0 时返回全 0."""
    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0.0:
        return pd.Series(0.0, index=weights.index, dtype="float64")
    return weights / total


def _water_fill(weights: pd.Series, max_weight: float) -> pd.Series:
    """迭代削峰: 超过 ``max_weight`` 的部分按现有权重比例分给其余名字."""
    out = weights.clip(lower=0.0).copy()
    if max_weight <= 0.0 or out.empty:
        return out
    for _ in range(100):
        over = out > max_weight + 1e-15
        if not bool(over.any()):
            break
        excess = float((out[over] - max_weight).sum())
        out[over] = max_weight
        free = ~over
        free_sum = float(out[free].sum())
        if free_sum <= 0.0:
            break
        out[free] = out[free] + excess * out[free] / free_sum
    return out


def long_only_portfolio(
    signal: pd.Series,
    benchmark_weights: pd.Series,
    top_pct: float = 0.10,
) -> pd.Series:
    """从基准成分中选信号最强的头部名字做等权多头.

    Parameters
    ----------
    signal:
        选股信号, 越大越优。
    benchmark_weights:
        基准权重, 同时决定候选域与返回索引。
    top_pct:
        头部比例; 入选数 ``max(1, ceil(top_pct * n))`` (上限为候选数)。

    Returns
    -------
    Series, 索引与 ``benchmark_weights`` 完全一致: 入选名字等权 ``1/k``,
    其余为 0。候选池 = 索引在基准内、信号非 ``NaN`` 且基准权重 > 0 的名字;
    候选为空时返回全 0。
    """
    benchmark = _as_float(benchmark_weights)
    scores = _as_float(signal).reindex(benchmark.index)
    result = pd.Series(0.0, index=benchmark.index, dtype="float64")

    candidates = scores[benchmark.gt(0) & scores.notna()]
    n = len(candidates)
    if n == 0:
        return result

    k = int(min(n, max(1, math.ceil(top_pct * n))))
    top = candidates.sort_values(ascending=False, kind="mergesort").index[:k]
    result.loc[top] = 1.0 / k
    return result


def apply_constraints(
    weights: pd.Series,
    *,
    max_weight: float,
    industry: pd.Series | None = None,
    industry_cap: float | None = None,
    size_cap: pd.Series | None = None,
    turnover_cap: float | None = None,
    prev_weights: pd.Series | None = None,
    liquidity_cap: pd.Series | None = None,
) -> pd.Series:
    """按固定顺序施加单票/行业/市值/换手/流动性约束.

    Parameters
    ----------
    weights:
        目标权重, 负值按 0 处理。
    max_weight:
        单票权重上限。
    industry:
        标的 -> 行业映射, 配合 ``industry_cap`` 使用。
    industry_cap:
        单一行业权重上限。
    size_cap, liquidity_cap:
        逐票上限 (如按市值/流动性折算的最大权重), 同时给定时取较小者。
    turnover_cap, prev_weights:
        单边换手上限与上期权重; 两者都给定时按 ``w = prev + k(w - prev)``
        收缩, ``k`` 取满足 ``0.5 * sum|w - prev| <= turnover_cap`` 的最大
        ``k <= 1``。
    prev_weights:
        上期权重, 缺失名字按 0 处理。

    Returns
    -------
    Series, 索引与 ``weights`` 一致, 非负且和为 1; 输入和 <= 0 时为全 0。

    Notes
    -----
    顺序: (1) 逐票上限裁剪并归一; (2) 行业超限缩放, 超额按比例分给行业外
    名字 (迭代若干轮); (3) ``max_weight`` 迭代削峰; (4) 换手收缩。
    当约束互相冲突 (如 ``max_weight * n < 1``) 时以和为 1 优先。
    """
    target = _as_float(weights).fillna(0.0).clip(lower=0.0)
    index = target.index
    if float(target.sum()) <= 0.0:
        return pd.Series(0.0, index=index, dtype="float64")

    # (1) 逐票上限 (size_cap / liquidity_cap 取交集) 后归一
    cap = pd.Series(np.inf, index=index, dtype="float64")
    for extra in (size_cap, liquidity_cap):
        if extra is not None:
            bound = _as_float(extra).reindex(index)
            cap = np.minimum(cap, bound.where(bound.notna(), np.inf))
    target = _normalize(target.clip(upper=cap))
    if float(target.sum()) <= 0.0:
        return target

    # (2) 行业上限: 超限行业内等比例缩减, 超额按比例分给行业外名字
    if industry is not None and industry_cap is not None:
        groups = pd.Series(industry).reindex(index)
        for _ in range(20):
            sums = target.groupby(groups).sum()
            over = sums[sums > industry_cap]
            if over.empty:
                break
            scale = groups.map(industry_cap / over).fillna(1.0)
            scaled = target * scale
            excess = float(target.sum() - scaled.sum())
            outside = ~groups.isin(over.index)
            base = scaled[outside]
            if not bool(outside.any()) or float(base.sum()) <= 0.0:
                target = scaled
                break
            scaled[outside] = base + excess * base / base.sum()
            target = scaled
        target = _normalize(target)

    # (3) 单票上限削峰
    target = _normalize(_water_fill(target, max_weight))
    if float(target.sum()) <= 0.0:
        return target

    # (4) 换手上限: 朝上期权重线性收缩
    if turnover_cap is not None and prev_weights is not None:
        previous = _normalize(
            _as_float(prev_weights).reindex(index).fillna(0.0).clip(lower=0.0)
        )
        delta = target - previous
        traded = 0.5 * float(delta.abs().sum())
        if traded > 0.0 and traded > turnover_cap:
            k = min(1.0, max(0.0, float(turnover_cap) / traded))
            target = previous + k * delta

    return _normalize(target)


def active_weights(
    portfolio_weights: pd.Series, benchmark_weights: pd.Series
) -> pd.Series:
    """主动权重 = 组合 - 基准, 索引取并集, 缺失一侧按 0 处理."""
    portfolio = _as_float(portfolio_weights)
    benchmark = _as_float(benchmark_weights)
    index = portfolio.index.union(benchmark.index)
    active = portfolio.reindex(index).fillna(0.0) - benchmark.reindex(index).fillna(0.0)
    active.name = "active_weight"
    return active
