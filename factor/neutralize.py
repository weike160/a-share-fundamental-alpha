"""行业/市值中性化, 剥离信号对风格的暴露."""
from __future__ import annotations

import pandas as pd


def industry_neutralize(
    signal: pd.Series, industry: pd.Series
) -> pd.Series:
    """按行业去均值, 剥离行业暴露."""
    raise NotImplementedError


def cross_sectional_regression(
    returns: pd.Series,
    features: pd.DataFrame,
) -> pd.DataFrame:
    """横截面回归: R = α + β1·SUE + β2·Size + β3·Momentum + β4·Vol + ε.

    返回各特征系数与显著性, 重点观察 β1 (SUE) 是否仍显著。
    """
    raise NotImplementedError
