"""机器学习的线性基线。

特征 X = [SUE, RevenueGrowth, ProfitGrowth, Momentum, Volatility,
          Liquidity, Size, Industry, MarketRegime]
预测 Y = FutureReturn (或 FutureReturnRank)。
"""
from __future__ import annotations

import pandas as pd


def build_feature_matrix(signal: pd.DataFrame) -> pd.DataFrame:
    """组织特征矩阵 X."""
    raise NotImplementedError


def fit_linear(X: pd.DataFrame, y: pd.Series):
    """线性回归基线."""
    raise NotImplementedError
