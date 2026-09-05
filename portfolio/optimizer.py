"""组合优化: 在暴露/换手约束下求解权重."""
from __future__ import annotations

import pandas as pd


def risk_parity(
    cov: pd.DataFrame, constraints: dict | None = None
) -> pd.Series:
    """风险平价权重."""
    raise NotImplementedError


def mean_variance(
    expected_return: pd.Series,
    cov: pd.DataFrame,
    constraints: dict | None = None,
    risk_aversion: float = 5.0,
) -> pd.Series:
    """均值-方差优化权重."""
    raise NotImplementedError
