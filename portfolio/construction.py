"""组合构建.

默认先做 Long-only benchmark-relative portfolio,
避免一开始就处理 A 股融券问题。
"""
from __future__ import annotations

import pandas as pd


def long_only_portfolio(
    signal: pd.Series,
    benchmark_weights: pd.Series,
    top_pct: float = 0.10,
) -> pd.Series:
    """Long top 10%, 降低/回避 bottom 10%, 相对基准构建多头组合."""
    raise NotImplementedError


def apply_constraints(
    weights: pd.Series,
    *,
    max_weight: float,
    industry_cap: pd.Series | None = None,
    size_cap: pd.Series | None = None,
    turnover_cap: float | None = None,
    liquidity_cap: pd.Series | None = None,
) -> pd.Series:
    """施加单票权重/行业/市值/换手/流动性约束."""
    raise NotImplementedError
