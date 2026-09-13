"""事件驱动回测引擎.

组合构造
--------
按**入场月份**把事件切成互不重叠的队列 (cohort): 队列 m 包含所有在 m 月
入场的财报事件。每个队列独立建仓:

* 基准 (benchmark) = 该队列全部可交易事件的等权组合
* 组合 (portfolio) = 按信号排序取前 ``top_pct`` 分位的等权组合, long-only
* 持有到各自入场后第 ``horizon`` 个交易日收盘

因此每个队列给出一个组合收益、一个基准收益和一个主动收益。队列收益是
**事件队列收益**, 不是单一可交易账户的净值: 相邻队列的持有期可能重叠,
这点在报告的限制里必须写明。

成本
----
每个队列按"整仓买入 + 到期整仓卖出"计费, 因此买入/卖出换手率都是 1.0::

    佣金      = (1 + 1) * commission_rate
    印花税    = 1 * stamp_tax_rate          # 卖出单边
    滑点      = 2 * slippage(order_size, adv)
    冲击成本  = 2 * market_impact(order_size, adv, volatility)

其中 ``order_size = AUM * 单票权重``, 因此成本随 AUM 上升 —— 这也是容量
分析的来源。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from execution.market_impact import market_impact, portfolio_capacity
from execution.slippage import slippage
from execution.transaction_cost import trade_cost
from portfolio import risk as risk_metrics

#: 队列指数 (年) —— 队列按月
PERIODS_PER_YEAR = 12


@dataclass(frozen=True)
class CostModel:
    """交易成本参数 (费率均为占 NAV 的比例)."""

    commission_rate: float = 0.0003
    stamp_tax_rate: float = 0.0005
    half_spread: float = 0.0005
    slippage_impact_coeff: float = 0.05
    impact_coeff: float = 0.5
    participation_cap: float = 0.10
    aum: float = 1e8

    def scaled(self, *, aum: float | None = None, **overrides: float) -> "CostModel":
        """派生一个新的成本模型 (用于容量/双倍成本情景)."""
        base = {**self.__dict__}
        if aum is not None:
            base["aum"] = aum
        base.update(overrides)
        return CostModel(**base)


@dataclass
class BacktestResult:
    """回测结果容器."""

    gross_nav: pd.Series
    net_nav: pd.Series
    turnover: pd.Series
    cost_breakdown: dict[str, pd.Series]
    cohorts: pd.DataFrame = field(default_factory=pd.DataFrame)
    metrics: dict[str, float] = field(default_factory=dict)


class EventBacktester:
    """财报事件队列回测器.

    Parameters
    ----------
    signal_col:
        排序用的信号列, 默认 ``"sue"``。
    horizon:
        持有期 (交易日), 决定读取哪一列未来收益。
    top_pct:
        组合选股比例 (默认前 20%, 即五分位最高一组)。
    cohort_col:
        队列分组列; 默认按 ``date_col`` 的月份分组。
    return_col:
        未来收益列; 默认优先用顺延可交易后的 ``fwd_ret_adj_<h>d``,
        不存在时退回 ``fwd_ret_<h>d``。
    """

    def __init__(
        self,
        *,
        signal_col: str = "sue",
        horizon: int = 20,
        top_pct: float = 0.2,
        cohort_col: str | None = None,
        date_col: str = "tradable_ts",
        return_col: str | None = None,
        cost: CostModel | None = None,
        min_names: int = 5,
        drop_blocked: bool = True,
    ) -> None:
        self.signal_col = signal_col
        self.horizon = int(horizon)
        self.top_pct = float(top_pct)
        self.cohort_col = cohort_col
        self.date_col = date_col
        self.cost = cost or CostModel()
        self.min_names = int(min_names)
        self.drop_blocked = drop_blocked
        adj = f"fwd_ret_adj_{self.horizon}d"
        raw = f"fwd_ret_{self.horizon}d"
        self.return_col = return_col or adj

    # ------------------------------------------------------------------
    def _prepare(self, panel: pd.DataFrame) -> pd.DataFrame:
        if self.signal_col not in panel.columns:
            raise ValueError(f"panel 缺少信号列: {self.signal_col}")

        return_col = self.return_col
        if return_col not in panel.columns:
            fallback = f"fwd_ret_{self.horizon}d"
            if fallback in panel.columns:
                return_col = fallback
            else:
                raise ValueError(f"panel 缺少收益列: {return_col} / {fallback}")
        self.return_col = return_col

        df = panel.copy()
        if self.drop_blocked and "entry_blocked" in df.columns:
            df = df[~df["entry_blocked"].astype(bool)]

        df[self.signal_col] = pd.to_numeric(df[self.signal_col], errors="coerce")
        df[return_col] = pd.to_numeric(df[return_col], errors="coerce")
        df = df.dropna(subset=[self.signal_col, return_col])

        if self.cohort_col and self.cohort_col in df.columns:
            df["_cohort"] = df[self.cohort_col].astype(str)
        else:
            df["_cohort"] = (
                pd.to_datetime(df[self.date_col]).dt.to_period("M").astype(str)
            )
        return df.reset_index(drop=True)

    # ------------------------------------------------------------------
    def _cohort_costs(
        self,
        selected: pd.DataFrame,
        weights: pd.Series,
    ) -> dict[str, float]:
        """单个队列的成本明细 (占队列 NAV 的比例)."""
        n = len(selected)
        if n == 0:
            return {
                "commission": 0.0,
                "stamp_tax": 0.0,
                "slippage": 0.0,
                "impact": 0.0,
                "total": 0.0,
                "capacity": float("nan"),
                "adv_missing": 0.0,
            }

        buy_turnover = 1.0
        sell_turnover = 1.0

        adv = pd.to_numeric(selected.get("adv20"), errors="coerce")
        if adv is None or adv.isna().all():
            adv = pd.Series(np.nan, index=selected.index)
        adv_missing = int(adv.isna().sum())
        if adv_missing:
            adv = adv.fillna(adv.median())

        vol = pd.to_numeric(selected.get("volatility_20d"), errors="coerce")
        if vol is None:
            vol = pd.Series(np.nan, index=selected.index)
        vol = vol.fillna(vol.median() if vol.notna().any() else 0.0).fillna(0.0)

        order_size = pd.Series(self.cost.aum * weights.to_numpy(), index=selected.index)

        slip = slippage(
            order_size,
            adv,
            participation_cap=self.cost.participation_cap,
            half_spread=self.cost.half_spread,
            impact_coeff=self.cost.slippage_impact_coeff,
        )
        imp = market_impact(
            order_size,
            adv,
            vol,
            participation_cap=self.cost.participation_cap,
            coeff=self.cost.impact_coeff,
        )

        slip_rate = float(np.nanmean(slip.to_numpy(dtype="float64"))) if len(slip) else 0.0
        imp_rate = float(np.nanmean(imp.to_numpy(dtype="float64"))) if len(imp) else 0.0
        if not np.isfinite(slip_rate):
            slip_rate = 0.0
        if not np.isfinite(imp_rate):
            imp_rate = 0.0

        buy = pd.Series([buy_turnover], dtype="float64")
        sell = pd.Series([sell_turnover], dtype="float64")
        total = trade_cost(
            buy,
            sell,
            commission_rate=self.cost.commission_rate,
            stamp_tax_rate=self.cost.stamp_tax_rate,
            slippage_rate=slip_rate,
            impact_rate=imp_rate,
        ).iloc[0]

        capacity = portfolio_capacity(
            adv, weights, max_participation=self.cost.participation_cap
        )

        return {
            "commission": self.cost.commission_rate * (buy_turnover + sell_turnover),
            "stamp_tax": self.cost.stamp_tax_rate * sell_turnover,
            "slippage": slip_rate * (buy_turnover + sell_turnover),
            "impact": imp_rate * (buy_turnover + sell_turnover),
            "total": float(total),
            "capacity": float(capacity),
            "adv_missing": float(adv_missing),
        }

    # ------------------------------------------------------------------
    def run(self, panel: pd.DataFrame) -> BacktestResult:
        df = self._prepare(panel)
        if len(df) == 0:
            raise ValueError("过滤后没有可用事件, 无法回测")

        rows: list[dict[str, object]] = []
        for cohort, grp in df.groupby("_cohort", sort=True):
            n = len(grp)
            if n < self.min_names:
                continue

            k = max(1, int(np.ceil(self.top_pct * n)))
            ranked = grp.sort_values(self.signal_col, ascending=False, kind="mergesort")
            selected = ranked.head(k)

            weights = pd.Series(1.0 / len(selected), index=selected.index)
            gross = float(selected[self.return_col].mean())
            benchmark = float(grp[self.return_col].mean())
            costs = self._cohort_costs(selected, weights)

            rows.append(
                {
                    "cohort": cohort,
                    "n_candidates": n,
                    "n_selected": len(selected),
                    "gross_return": gross,
                    "benchmark_return": benchmark,
                    "active_return": gross - benchmark,
                    "cost_total": costs["total"],
                    "net_return": gross - costs["total"],
                    "net_active_return": gross - costs["total"] - benchmark,
                    "commission": costs["commission"],
                    "stamp_tax": costs["stamp_tax"],
                    "slippage": costs["slippage"],
                    "impact": costs["impact"],
                    "capacity": costs["capacity"],
                    "adv_missing": costs["adv_missing"],
                    "buy_turnover": 1.0,
                    "sell_turnover": 1.0,
                }
            )

        if not rows:
            raise ValueError("没有任何队列达到最少股票数要求")

        cohorts = pd.DataFrame(rows).set_index("cohort").sort_index()

        gross_nav = (1.0 + cohorts["gross_return"]).cumprod()
        net_nav = (1.0 + cohorts["net_return"]).cumprod()
        active = cohorts["active_return"]
        net_active = cohorts["net_active_return"]

        cost_breakdown = {
            key: cohorts[key]
            for key in ("commission", "stamp_tax", "slippage", "impact", "cost_total")
        }

        metrics: dict[str, float] = {
            "n_cohorts": float(len(cohorts)),
            "n_events_used": float(cohorts["n_candidates"].sum()),
            "n_names_mean": float(cohorts["n_selected"].mean()),
            "gross_annual_return": risk_metrics.annualized_return(
                cohorts["gross_return"], PERIODS_PER_YEAR
            ),
            "net_annual_return": risk_metrics.annualized_return(
                cohorts["net_return"], PERIODS_PER_YEAR
            ),
            "gross_sharpe": risk_metrics.sharpe(
                cohorts["gross_return"], periods_per_year=PERIODS_PER_YEAR
            ),
            "net_sharpe": risk_metrics.sharpe(
                cohorts["net_return"], periods_per_year=PERIODS_PER_YEAR
            ),
            "net_max_drawdown": risk_metrics.max_drawdown(net_nav),
            "gross_max_drawdown": risk_metrics.max_drawdown(gross_nav),
            "benchmark_annual_return": risk_metrics.annualized_return(
                cohorts["benchmark_return"], PERIODS_PER_YEAR
            ),
            "benchmark_sharpe": risk_metrics.sharpe(
                cohorts["benchmark_return"], periods_per_year=PERIODS_PER_YEAR
            ),
            "tracking_error": risk_metrics.tracking_error(active, PERIODS_PER_YEAR),
            "information_ratio": risk_metrics.information_ratio(active, PERIODS_PER_YEAR),
            "net_information_ratio": risk_metrics.information_ratio(
                net_active, PERIODS_PER_YEAR
            ),
            "mean_cost": float(cohorts["cost_total"].mean()),
            "cost_drag_annual": float(cohorts["cost_total"].mean() * PERIODS_PER_YEAR),
            "mean_turnover": float(
                (cohorts["buy_turnover"] + cohorts["sell_turnover"]).mean()
            ),
            "hit_rate_gross": float((cohorts["gross_return"] > 0).mean()),
            "hit_rate_active": float((active > 0).mean()),
            "capacity_median": float(np.nanmedian(cohorts["capacity"].to_numpy())),
            "adv_missing_rate": float(
                cohorts["adv_missing"].sum() / max(cohorts["n_selected"].sum(), 1)
            ),
        }

        return BacktestResult(
            gross_nav=gross_nav,
            net_nav=net_nav,
            turnover=cohorts["buy_turnover"],
            cost_breakdown=cost_breakdown,
            cohorts=cohorts,
            metrics=metrics,
        )
