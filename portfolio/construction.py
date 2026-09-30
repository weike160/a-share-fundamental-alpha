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
    """在正基准权重且信号非缺失的标的中选前 top_pct，等权持有。

    至少选一只，上限为候选数；返回基准索引，未入选权重为零。"""
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
    """依次施加逐票、行业、单票和换手约束，保留输入索引。

    负权重按零处理，size_cap 与 liquidity_cap 取较小者。行业超额按比例
    分配到行业外，再迭代削峰；换手约束使权重向上期收缩。
    约束冲突时优先归一化，输入权重和不为正时返回全零。"""
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
