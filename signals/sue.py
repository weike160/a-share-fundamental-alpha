"""Standardized Unexpected Earnings (SUE).

SUE_{i,t} = (ActualEPS - ExpectedEPS) / σ(EarningsSurprise)
"""
from __future__ import annotations

import pandas as pd


def compute_sue(
    actual_eps: pd.Series,
    expected_eps: pd.Series,
    surprise_std: pd.Series,
) -> pd.Series:
    """标准化未预期盈余.

    expected_eps 可来自分析师预期, 或使用季节性随机游走 / 同比代理。
    """
    raise NotImplementedError


def sue_from_analyst(
    actual_eps: pd.Series, analyst_forecast: pd.Series
) -> pd.Series:
    """基于分析师一致预期的 SUE."""
    raise NotImplementedError


def seasonal_random_walk_sue(eps: pd.DataFrame) -> pd.Series:
    """在无分析师数据时, 用季节性随机游走预期 (去年同期 EPS) 构造 SUE."""
    raise NotImplementedError
