"""IC / RankIC / ICIR 计算."""
from __future__ import annotations

import pandas as pd


def ic(signal: pd.Series, forward_return: pd.Series, method: str = "pearson") -> float:
    """横截面 IC, 返回常数. method: 'pearson' | 'spearman'."""
    raise NotImplementedError


def ic_series(
    signals: pd.DataFrame, forward_returns: pd.DataFrame, method: str = "spearman"
) -> pd.Series:
    """逐期 IC 序列, 用于计算 ICIR."""
    raise NotImplementedError


def icir(ics: pd.Series) -> float:
    """RankICIR = Mean(IC) / Std(IC)."""
    raise NotImplementedError
