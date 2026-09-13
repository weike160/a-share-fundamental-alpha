"""execution / portfolio 层测试: 成本模型与组合构建、优化.

全部离线、确定性。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from execution.market_impact import (
    capacity_analysis,
    market_impact,
    portfolio_capacity,
)
from execution.slippage import slippage
from execution.transaction_cost import commission, stamp_tax, trade_cost
from portfolio.construction import (
    active_weights,
    apply_constraints,
    long_only_portfolio,
)
from portfolio.optimizer import mean_variance, risk_parity
from portfolio.risk import (
    annualized_return,
    information_ratio,
    max_drawdown,
    sharpe,
    tracking_error,
    turnover,
)


# --------------------------------------------------------------------------
# transaction_cost
# --------------------------------------------------------------------------
def test_commission_and_stamp_tax_values() -> None:
    t = pd.Series([0.1, 0.2], index=["a", "b"])
    assert commission(t).tolist() == pytest.approx([0.00003, 0.00006])
    assert commission(t, rate=0.001).tolist() == pytest.approx([0.0001, 0.0002])
    # 印花税只在卖出侧, 且用卖出换手
    assert stamp_tax(t).tolist() == pytest.approx([0.00005, 0.0001])


def test_commission_negative_is_nan_and_index_preserved() -> None:
    t = pd.Series([0.1, -0.05, np.nan], index=["a", "b", "c"])
    out = commission(t)
    assert out.index.equals(t.index)
    assert out["a"] == pytest.approx(0.00003)
    assert pd.isna(out["b"]), "负换手必须是 NaN"
    assert pd.isna(out["c"])

    taxed = stamp_tax(pd.Series([0.2, -0.1], index=["a", "b"]))
    assert taxed["a"] == pytest.approx(0.0001)
    assert pd.isna(taxed["b"])


def test_trade_cost_composition() -> None:
    buy = pd.Series([0.3], index=["a"])
    sell = pd.Series([0.2], index=["a"])
    cost = trade_cost(
        buy,
        sell,
        commission_rate=0.0003,
        stamp_tax_rate=0.0005,
        slippage_rate=0.001,
        impact_rate=0.0005,
    )
    # 0.5*3e-4 + 0.2*5e-4 + 0.0015*0.5 = 0.00015 + 0.0001 + 0.00075
    assert cost["a"] == pytest.approx(0.001)


def test_trade_cost_defaults_and_side_alignment() -> None:
    buy = pd.Series([0.1], index=["a"])
    sell = pd.Series([0.1], index=["a"])
    assert trade_cost(buy, sell)["a"] == pytest.approx(0.00011)

    # 只在一边出现的名字按 0 处理, 索引取并集
    cost = trade_cost(pd.Series([0.1], index=["a"]), pd.Series([0.1], index=["b"]))
    assert set(cost.index) == {"a", "b"}
    assert cost["a"] == pytest.approx(0.00003)  # 只有买方佣金
    assert cost["b"] == pytest.approx(0.00003 + 0.00005)  # 卖方佣金 + 印花税


def test_trade_cost_negative_propagates_nan() -> None:
    cost = trade_cost(pd.Series([-0.1]), pd.Series([0.1]))
    assert pd.isna(cost.iloc[0])


# --------------------------------------------------------------------------
# slippage
# --------------------------------------------------------------------------
def test_slippage_low_participation() -> None:
    cost = slippage(pd.Series([50.0]), pd.Series([1000.0]))
    # 0.0005 + 0.05 * 0.05
    assert cost.iloc[0] == pytest.approx(0.003)


def test_slippage_is_capped_at_participation_cap() -> None:
    small = slippage(pd.Series([100.0]), pd.Series([1000.0]), participation_cap=0.10)
    large = slippage(pd.Series([200.0]), pd.Series([1000.0]), participation_cap=0.10)
    assert small.iloc[0] == pytest.approx(0.0005 + 0.05 * 0.10)
    assert large.iloc[0] == pytest.approx(small.iloc[0])
    assert large.iloc[0] == pytest.approx(0.0055)


def test_slippage_nan_guards() -> None:
    order = pd.Series([10.0, 10.0, 10.0, np.nan], index=list("abcd"))
    adv = pd.Series([0.0, np.nan, 100.0, 100.0], index=list("abcd"))
    out = slippage(order, adv)
    assert out.index.equals(order.index)
    assert pd.isna(out["a"])  # adv <= 0
    assert pd.isna(out["b"])  # adv NaN
    assert pd.isna(out["d"])  # order_size NaN
    assert np.isfinite(out["c"])


# --------------------------------------------------------------------------
# market_impact
# --------------------------------------------------------------------------
def test_market_impact_sqrt_and_monotone() -> None:
    order = pd.Series([10.0, 20.0, 50.0, 100.0, 500.0])
    adv = pd.Series([1000.0] * 5)
    vol = pd.Series([0.02] * 5)
    impact = market_impact(order, adv, vol, participation_cap=0.10, coeff=0.5)

    assert impact.iloc[0] == pytest.approx(0.5 * 0.02 * np.sqrt(0.01))
    assert impact.iloc[2] == pytest.approx(0.5 * 0.02 * np.sqrt(0.05))
    # 刚好到达参与率上限
    assert impact.iloc[3] == pytest.approx(0.5 * 0.02 * np.sqrt(0.10))
    # 超过参与率上限后饱和
    assert impact.iloc[4] == pytest.approx(impact.iloc[3])
    assert impact.iloc[:4].is_monotonic_increasing
    assert (impact >= 0).all()


def test_market_impact_nan_guards() -> None:
    out = market_impact(
        pd.Series([10.0, 10.0, 10.0], index=list("abc")),
        pd.Series([0.0, np.nan, 100.0], index=list("abc")),
        pd.Series([0.02, 0.02, np.nan], index=list("abc")),
    )
    assert pd.isna(out["a"]) and pd.isna(out["b"]) and pd.isna(out["c"])


def test_capacity_analysis_and_portfolio_capacity() -> None:
    adv = pd.Series([1e8, 2e8, 5e8], index=["a", "b", "c"])
    weight = pd.Series([0.1, 0.05, 0.0], index=["a", "b", "c"])
    cap = capacity_analysis(adv, weight, max_participation=0.10)

    assert cap["a"] == pytest.approx(1e8)
    assert cap["b"] == pytest.approx(4e8)
    assert pd.isna(cap["c"]), "weight <= 0 -> NaN"

    # 默认 5% 分位 (线性插值): 1e8 + 0.05 * (4e8 - 1e8)
    assert portfolio_capacity(adv, weight) == pytest.approx(1.15e8)
    assert portfolio_capacity(adv, weight, quantile=0.5) == pytest.approx(2.5e8)

    all_nan = capacity_analysis(adv, pd.Series([np.nan] * 3, index=adv.index))
    assert all_nan.isna().all()
    assert pd.isna(portfolio_capacity(adv, pd.Series([np.nan] * 3, index=adv.index)))


# --------------------------------------------------------------------------
# risk
# --------------------------------------------------------------------------
def test_annualized_return_hand_computed() -> None:
    r = pd.Series([0.25] * 4)
    assert annualized_return(r, periods_per_year=4) == pytest.approx(1.25**4 - 1.0)

    r2 = pd.Series([0.01, 0.02, -0.01])
    assert annualized_return(r2, periods_per_year=3) == pytest.approx(
        1.01 * 1.02 * 0.99 - 1.0
    )


def test_annualized_return_nan_cases() -> None:
    assert pd.isna(annualized_return(pd.Series([], dtype=float)))
    assert pd.isna(annualized_return(pd.Series([-1.0, 0.5])))


def test_sharpe_hand_computed() -> None:
    r = pd.Series([0.01, 0.02, 0.03])
    # mean 0.02, ddof=1 std 0.01
    assert sharpe(r) == pytest.approx(0.02 / 0.01 * np.sqrt(252))
    assert sharpe(r, rf=0.0252) == pytest.approx(
        (0.02 - 0.0001) / 0.01 * np.sqrt(252)
    )


def test_sharpe_nan_cases() -> None:
    assert pd.isna(sharpe(pd.Series([0.01])))
    assert pd.isna(sharpe(pd.Series([0.01, 0.01, 0.01])))


def test_max_drawdown_hand_computed() -> None:
    nav = pd.Series([1.0, 1.2, 0.9, 1.1])
    assert max_drawdown(nav) == pytest.approx(-0.25)
    assert pd.isna(max_drawdown(pd.Series([], dtype=float)))


def test_tracking_error_and_information_ratio_hand_computed() -> None:
    active = pd.Series([0.01, 0.02, 0.03])
    assert tracking_error(active) == pytest.approx(0.01 * np.sqrt(252))
    # 年化均值 0.02*252 / TE
    assert information_ratio(active) == pytest.approx(
        0.02 * 252 / (0.01 * np.sqrt(252))
    )
    assert information_ratio(active) == pytest.approx(2.0 * np.sqrt(252))


def test_information_ratio_zero_te_is_nan() -> None:
    assert pd.isna(information_ratio(pd.Series([0.01, 0.01])))


def test_turnover_first_row_convention() -> None:
    weights = pd.DataFrame(
        {"a": [0.6, 0.5], "b": [0.4, 0.5]},
        index=["t0", "t1"],
    )
    t = turnover(weights)
    assert t.index.equals(weights.index)
    # 首期 w_{-1}=0: 0.5 * (0.6 + 0.4)
    assert t["t0"] == pytest.approx(0.5)
    # 0.5 * (|0.5-0.6| + |0.5-0.4|)
    assert t["t1"] == pytest.approx(0.1)


# --------------------------------------------------------------------------
# construction
# --------------------------------------------------------------------------
def test_long_only_portfolio_selection_and_equal_weight() -> None:
    names = [f"s{i}" for i in range(10)]
    benchmark = pd.Series(0.1, index=names)
    signal = pd.Series(np.arange(10, dtype=float), index=names)

    out = long_only_portfolio(signal, benchmark, top_pct=0.20)
    assert out.index.equals(benchmark.index)
    assert out.sum() == pytest.approx(1.0)
    selected = out[out > 0]
    assert len(selected) == 2, "ceil(0.2 * 10) = 2"
    assert selected.tolist() == pytest.approx([0.5, 0.5])
    assert set(selected.index) == {"s9", "s8"}


def test_long_only_portfolio_excludes_nan_signal_and_zero_benchmark() -> None:
    benchmark = pd.Series([0.4, 0.3, 0.2, 0.1], index=list("abcd"))
    signal = pd.Series([1.0, np.nan, 3.0, 4.0], index=list("abcd"))
    benchmark["d"] = 0.0  # 基准外

    out = long_only_portfolio(signal, benchmark, top_pct=1.0)
    assert set(out[out > 0].index) == {"a", "c"}
    assert out["a"] == pytest.approx(0.5) and out["c"] == pytest.approx(0.5)
    assert out["b"] == 0.0 and out["d"] == 0.0


def test_long_only_portfolio_min_one_name() -> None:
    benchmark = pd.Series(0.25, index=list("abcd"))
    signal = pd.Series([1.0, 2.0, 3.0, 4.0], index=list("abcd"))
    out = long_only_portfolio(signal, benchmark, top_pct=0.01)
    assert out["d"] == pytest.approx(1.0)
    assert out.sum() == pytest.approx(1.0)


def test_apply_constraints_max_weight() -> None:
    w = pd.Series([0.5, 0.3, 0.2], index=list("abc"))
    out = apply_constraints(w, max_weight=0.4)
    assert out.index.equals(w.index)
    assert out.sum() == pytest.approx(1.0)
    assert (out <= 0.4 + 1e-12).all()
    assert out["a"] == pytest.approx(0.4)
    assert out["b"] == pytest.approx(0.36)
    assert out["c"] == pytest.approx(0.24)


def test_apply_constraints_industry_cap() -> None:
    w = pd.Series([0.5, 0.3, 0.2], index=list("abc"))
    industry = pd.Series(["X", "X", "Y"], index=list("abc"))
    out = apply_constraints(w, max_weight=1.0, industry=industry, industry_cap=0.6)

    assert out.sum() == pytest.approx(1.0)
    assert out[["a", "b"]].sum() == pytest.approx(0.6)
    assert out["c"] == pytest.approx(0.4)
    assert out["a"] == pytest.approx(0.375)
    assert out["b"] == pytest.approx(0.225)


def test_apply_constraints_per_name_caps_and_non_negativity() -> None:
    w = pd.Series([0.6, -0.2, 0.4], index=list("abc"))
    size_cap = pd.Series([0.3, np.inf, np.inf], index=list("abc"))
    out = apply_constraints(w, max_weight=1.0, size_cap=size_cap)
    assert (out >= 0).all()
    assert out.sum() == pytest.approx(1.0)
    # [-0.2 -> 0], clip a to 0.3: [0.3, 0, 0.4] / 0.7
    assert out["a"] == pytest.approx(0.3 / 0.7)
    assert out["c"] == pytest.approx(0.4 / 0.7)


def test_apply_constraints_turnover_cap() -> None:
    w = pd.Series([1.0, 0.0, 0.0], index=list("abc"))
    prev = pd.Series([0.5, 0.5, 0.0], index=list("abc"))
    out = apply_constraints(w, max_weight=1.0, turnover_cap=0.10, prev_weights=prev)

    assert out.sum() == pytest.approx(1.0)
    assert 0.5 * (out - prev).abs().sum() == pytest.approx(0.10)
    assert out["a"] == pytest.approx(0.6)
    assert out["b"] == pytest.approx(0.4)

    # 未触发上限时保持原权重
    loose = apply_constraints(w, max_weight=1.0, turnover_cap=0.9, prev_weights=prev)
    assert loose["a"] == pytest.approx(1.0)


def test_apply_constraints_combined_keeps_all_bounds() -> None:
    index = [f"s{i}" for i in range(6)]
    w = pd.Series([0.40, 0.25, 0.15, 0.10, 0.06, 0.04], index=index)
    prev = pd.Series(1.0 / 6.0, index=index)
    industry = pd.Series(["A", "A", "B", "B", "C", "C"], index=index)
    size_cap = pd.Series(0.30, index=index)

    out = apply_constraints(
        w,
        max_weight=0.35,
        industry=industry,
        industry_cap=0.50,
        size_cap=size_cap,
        turnover_cap=0.20,
        prev_weights=prev,
    )

    assert out.index.equals(w.index)
    assert (out >= 0).all()
    assert out.sum() == pytest.approx(1.0)
    assert out.max() <= 0.35 + 1e-9
    assert out.groupby(industry).sum().max() <= 0.50 + 1e-9
    assert 0.5 * (out - prev).abs().sum() <= 0.20 + 1e-9


def test_apply_constraints_zero_input_returns_zeros() -> None:
    w = pd.Series([0.0, 0.0], index=list("ab"))
    out = apply_constraints(w, max_weight=0.5)
    assert (out == 0).all()


def test_active_weights_alignment() -> None:
    portfolio = pd.Series([0.5, 0.5], index=["a", "b"])
    benchmark = pd.Series([0.3, 0.3], index=["b", "c"])
    active = active_weights(portfolio, benchmark)
    assert set(active.index) == {"a", "b", "c"}
    assert active["a"] == pytest.approx(0.5)
    assert active["b"] == pytest.approx(0.2)
    assert active["c"] == pytest.approx(-0.3)


# --------------------------------------------------------------------------
# optimizer
# --------------------------------------------------------------------------
def test_risk_parity_inverse_vol() -> None:
    cov = pd.DataFrame(
        np.diag([0.04, 0.09, 0.01]), index=list("abc"), columns=list("abc")
    )
    w = risk_parity(cov)
    assert w.index.equals(cov.index)
    assert w.sum() == pytest.approx(1.0)
    # 逆波动率: w_i * sigma_i 常数
    sigma = pd.Series([0.2, 0.3, 0.1], index=list("abc"))
    products = (w * sigma).to_numpy()
    assert products == pytest.approx([products[0]] * 3)
    assert w["a"] == pytest.approx(5.0 / (5.0 + 10.0 / 3.0 + 10.0))


def test_risk_parity_invalid_sigma_and_clipping() -> None:
    cov = pd.DataFrame(
        np.diag([0.04, 0.0, -0.01]), index=list("abc"), columns=list("abc")
    )
    w = risk_parity(cov)
    assert w["a"] == pytest.approx(1.0)
    assert w["b"] == pytest.approx(0.0) and w["c"] == pytest.approx(0.0)

    cov_ok = pd.DataFrame(
        np.diag([0.04, 0.09, 0.01]), index=list("abc"), columns=list("abc")
    )
    raw = risk_parity(cov_ok)
    capped = risk_parity(cov_ok, {"max_weight": 0.4})
    expected = pd.Series([raw["a"], raw["b"], 0.4], index=list("abc"))
    expected = expected / expected.sum()

    assert capped.sum() == pytest.approx(1.0)
    # 简单裁剪后归一: 未触界名字的相对比例保持, 触界名字被重新放大
    assert capped["a"] / capped["b"] == pytest.approx(raw["a"] / raw["b"])
    assert capped["c"] == pytest.approx(expected["c"])


def test_mean_variance_analytic_interior() -> None:
    mu = pd.Series([0.10, 0.05], index=["a", "b"])
    cov = pd.DataFrame(np.diag([0.04, 0.01]), index=["a", "b"], columns=["a", "b"])
    w = mean_variance(mu, cov, risk_aversion=5.0)

    assert w.index.equals(mu.index)
    assert w.sum() == pytest.approx(1.0)
    assert w["a"] == pytest.approx(0.4, abs=1e-6)
    assert w["b"] == pytest.approx(0.6, abs=1e-6)


def test_mean_variance_hits_long_only_bound() -> None:
    mu = pd.Series([0.01, 0.20], index=["a", "b"])
    cov = pd.DataFrame(np.diag([0.04, 0.01]), index=["a", "b"], columns=["a", "b"])
    w = mean_variance(mu, cov, risk_aversion=5.0)

    assert w.sum() == pytest.approx(1.0)
    assert w["a"] == pytest.approx(0.0, abs=1e-6)
    assert w["b"] == pytest.approx(1.0, abs=1e-6)
    assert (w >= -1e-9).all()


def test_mean_variance_respects_upper_constraint() -> None:
    mu = pd.Series([0.10, 0.05, 0.05], index=list("abc"))
    cov = pd.DataFrame(
        np.diag([0.04, 0.04, 0.04]), index=list("abc"), columns=list("abc")
    )
    # 无约束最优 [0.5, 0.25, 0.25], 单票上限 0.4 触界, 余下等分
    w = mean_variance(mu, cov, constraints={"max_weight": 0.4}, risk_aversion=5.0)

    assert w.index.equals(mu.index)
    assert w.sum() == pytest.approx(1.0)
    assert w["a"] == pytest.approx(0.4, abs=1e-6)
    assert w["b"] == pytest.approx(0.3, abs=1e-6)
    assert w["c"] == pytest.approx(0.3, abs=1e-6)
    assert (w <= 0.4 + 1e-9).all()
