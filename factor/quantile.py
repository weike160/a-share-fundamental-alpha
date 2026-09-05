"""Q1–Q10 分层收益, 观察信号分组的收益单调性."""
from __future__ import annotations

import pandas as pd


def quantile_returns(
    signals: pd.Series,
    forward_returns: pd.Series,
    n_quantiles: int = 10,
) -> pd.DataFrame:
    """按 signal 分 n 组, 返回各组的平均未来收益."""
    raise NotImplementedError


def monotonicity_check(
    quantile_returns: pd.DataFrame,
) -> bool:
    """检验分层收益是否单调 (Return(Q1) < ... < Return(Q10))."""
    raise NotImplementedError
