"""事件/组合回测引擎.

负责把信号、可交易性、交易成本三者组合成净收益路径,
并输出 Gross vs Net Alpha 拆解。
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class BacktestResult:
    """回测结果容器."""

    gross_nav: pd.Series
    net_nav: pd.Series
    turnover: pd.Series
    cost_breakdown: dict[str, pd.Series]


class EventBacktester:
    """事件驱动回测器.

    TODO(Phase-5): 实现持仓构建、逐期再平衡、成本扣除。
    """

    def run(self, **kwargs) -> BacktestResult:
        raise NotImplementedError
