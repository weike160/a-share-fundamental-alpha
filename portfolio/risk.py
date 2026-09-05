"""风险分析指标: 年化收益 / Sharpe / MaxDD / Turnover / TrackingError / IR."""
from __future__ import annotations

import pandas as pd


def annualized_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    raise NotImplementedError


def sharpe(returns: pd.Series, rf: float = 0.0, periods_per_year: int = 252) -> float:
    raise NotImplementedError


def max_drawdown(nav: pd.Series) -> float:
    raise NotImplementedError


def tracking_error(active_returns: pd.Series, periods_per_year: int = 252) -> float:
    raise NotImplementedError


def information_ratio(active_returns: pd.Series, periods_per_year: int = 252) -> float:
    raise NotImplementedError


def turnover(weights: pd.DataFrame) -> pd.Series:
    """组合换手率序列."""
    raise NotImplementedError
