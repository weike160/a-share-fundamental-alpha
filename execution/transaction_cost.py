"""交易成本: 佣金 + 印花税."""
from __future__ import annotations

import pandas as pd


def commission(turnover: pd.Series, rate: float = 0.0003) -> pd.Series:
    """佣金, 默认万 3."""
    raise NotImplementedError


def stamp_tax(sell_turnover: pd.Series, rate: float = 0.0005) -> pd.Series:
    """印花税, 默认 万 5 (卖出单边)."""
    raise NotImplementedError
