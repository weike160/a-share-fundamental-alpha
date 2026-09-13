"""signals 层测试.

全部离线, 重点是守住**反未来函数**这条红线。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from signals.sue import (
    compute_sue,
    seasonal_random_walk_sue,
    sue_coverage,
    sue_from_analyst,
)


# --------------------------------------------------------------------------
# compute_sue
# --------------------------------------------------------------------------
def test_compute_sue_basic() -> None:
    actual = pd.Series([1.2, 0.8, 1.0])
    expected = pd.Series([1.0, 1.0, 1.0])
    std = pd.Series([0.1, 0.1, 0.1])
    got = compute_sue(actual, expected, std)
    assert got.tolist() == pytest.approx([2.0, -2.0, 0.0])


def test_compute_sue_zero_std_is_nan_not_inf() -> None:
    """分母为 0 必须给 NaN, 不能给 inf —— inf 会污染排序和回归."""
    got = compute_sue(
        pd.Series([1.0, 1.0]),
        pd.Series([0.0, 0.0]),
        pd.Series([0.0, 0.1]),
    )
    assert pd.isna(got.iloc[0])
    assert not np.isinf(got).any()


def test_compute_sue_negative_or_missing_std_is_nan() -> None:
    got = compute_sue(
        pd.Series([1.0, 1.0, 1.0]),
        pd.Series([0.0, 0.0, 0.0]),
        pd.Series([-0.1, np.nan, np.inf]),
    )
    assert got.isna().all()


def test_sue_from_analyst_raises_explicitly() -> None:
    """没有分析师数据时必须明确报错, 不能悄悄返回错的数."""
    with pytest.raises(NotImplementedError, match="分析师"):
        sue_from_analyst(pd.Series([1.0]), pd.Series([1.0]))


# --------------------------------------------------------------------------
# seasonal_random_walk_sue
# --------------------------------------------------------------------------
def _eps_panel(values: list[float], code: str = "000001") -> pd.DataFrame:
    """把一串 EPS 变成按季度递增的面板."""
    dates = pd.date_range("2020-03-31", periods=len(values), freq="QE")
    return pd.DataFrame(
        {"code": [code] * len(values), "report_date": dates, "eps": values}
    )


def _varying_values(n: int) -> list[float]:
    """非退化的 EPS 序列.

    ⚠️ 注意 ``1.0 + 0.05*i`` 这类**线性**序列是退化的: 同比 UE 恒为常数,
    sigma=0, SUE 全是 NaN。算 SUE 的测试数据必须有真实波动。
    """
    return [1.0 + 0.3 * np.sin(1.7 * i) + 0.05 * i for i in range(n)]


def test_sue_empty_until_enough_history() -> None:
    """历史不足时必须给 NaN, 不能编一个数出来."""
    panel = _eps_panel([1.0, 1.1, 1.2, 1.3, 1.4, 1.5])
    sue = seasonal_random_walk_sue(panel, window=8, min_periods=4)
    # 前 4 期没有同比 (lag=4), 后面 sigma 又不够 min_periods
    assert sue.isna().all()


def test_sue_produces_values_with_enough_history() -> None:
    panel = _eps_panel(_varying_values(20))
    sue = seasonal_random_walk_sue(panel, window=8, min_periods=4)
    assert sue.notna().sum() > 0, "有足够历史时应能算出 SUE"
    assert sue.std() > 0
    # 不能出现浮点残差造成的天文数字
    assert sue.abs().max() < 1e6


def test_sue_alignment_matches_input_index() -> None:
    """返回序列必须与输入面板逐行对齐."""
    panel = _eps_panel(_varying_values(16))
    panel.index = [f"row{i}" for i in range(len(panel))]
    sue = seasonal_random_walk_sue(panel, window=4, min_periods=3)
    assert list(sue.index) == list(panel.index)


def test_sue_uses_only_past_information() -> None:
    """核心回归: 改动**未来**的 EPS, 不能改变**过去**的 SUE.

    如果 sigma 忘了 shift(1), 当期 UE 会进入自己的分母, 这个测试就会失败。
    """
    values = [1.0, 1.1, 0.9, 1.2, 1.05, 1.15, 0.95, 1.25, 1.1, 1.3, 1.0, 1.4]
    panel = _eps_panel(values)
    base = seasonal_random_walk_sue(panel, window=4, min_periods=3)

    # 只改最后一期的 EPS (未来信息)
    tampered = panel.copy()
    tampered.loc[tampered.index[-1], "eps"] = 99.0
    after = seasonal_random_walk_sue(tampered, window=4, min_periods=3)

    # 除最后一期外, 所有历史 SUE 必须原封不动
    pd.testing.assert_series_equal(
        base.iloc[:-1], after.iloc[:-1], check_names=False
    )
    # 而最后一期本身应当改变 (说明修改确实生效了, 测试不是假通过)
    assert not np.isclose(base.iloc[-1], after.iloc[-1], equal_nan=True)


def test_past_change_does_affect_future_sue() -> None:
    """反向确认: 改动过去, 应该影响未来 (否则说明根本没在算)."""
    values = [1.0, 1.1, 0.9, 1.2, 1.05, 1.15, 0.95, 1.25, 1.1, 1.3, 1.0, 1.4]
    panel = _eps_panel(values)
    base = seasonal_random_walk_sue(panel, window=4, min_periods=3)

    tampered = panel.copy()
    tampered.loc[tampered.index[4], "eps"] = 5.0  # 改一个早期值
    after = seasonal_random_walk_sue(tampered, window=4, min_periods=3)

    tail = slice(6, None)
    assert not np.allclose(
        base.iloc[tail].fillna(-999), after.iloc[tail].fillna(-999)
    ), "过去的异常应当传导到未来的 sigma"


def test_sue_zero_variance_history_is_nan() -> None:
    """历史 UE 完全没有波动时 sigma=0, 必须 NaN 而不是 inf."""
    values = [1.0] * 16  # EPS 完全不变 -> UE 恒为 0 -> sigma 恒为 0
    panel = _eps_panel(values)
    sue = seasonal_random_walk_sue(panel, window=4, min_periods=3)
    assert sue.isna().all()
    assert not np.isinf(sue).any()


def test_sue_handles_multiple_codes_independently() -> None:
    """加入第二只股票, 不应改变第一只股票的 SUE."""
    a = _eps_panel([1.0 + 0.05 * i + 0.1 * (i % 3) for i in range(16)], code="000001")
    b = _eps_panel([2.0 + 0.02 * i - 0.2 * (i % 2) for i in range(16)], code="600000")

    alone = seasonal_random_walk_sue(a, window=4, min_periods=3)
    assert alone.notna().sum() > 0, "测试数据要有非退化的波动"

    panel = pd.concat([a, b], ignore_index=True)
    both = seasonal_random_walk_sue(panel, window=4, min_periods=3)
    pd.testing.assert_series_equal(
        alone, both.iloc[:16], check_names=False, check_index=False
    )


def test_constant_yoy_growth_gives_nan_not_garbage() -> None:
    """EPS 完全线性增长时 UE 恒定, sigma=0 -> 必须 NaN.

    这不是 bug: 没有波动就无法标准化。真实 EPS 不会这样, 但数据里
    可能出现 (如长期停牌补数据), 必须安全降级而不是产生 inf。
    """
    panel = _eps_panel([1.0 + 0.05 * i for i in range(20)])
    sue = seasonal_random_walk_sue(panel, window=6, min_periods=3)
    assert not np.isinf(sue).any()
    assert sue.notna().sum() == 0


def test_sue_rejects_missing_columns() -> None:
    with pytest.raises(ValueError, match="缺少必需列"):
        seasonal_random_walk_sue(pd.DataFrame({"code": ["000001"], "eps": [1.0]}))


def test_sue_rejects_bad_params() -> None:
    panel = _eps_panel([1.0] * 10)
    with pytest.raises(ValueError):
        seasonal_random_walk_sue(panel, lag=0)
    with pytest.raises(ValueError):
        seasonal_random_walk_sue(panel, min_periods=1)


def test_sue_return_components() -> None:
    values = _varying_values(16)
    panel = _eps_panel(values)
    out = seasonal_random_walk_sue(panel, window=4, min_periods=3, return_components=True)
    assert set(out.columns) == {"sue", "ue", "sigma"}
    # ue 应当等于 eps - eps.shift(4)
    eps = panel["eps"]
    assert out["ue"].iloc[4] == pytest.approx(eps.iloc[4] - eps.iloc[0])


def test_sue_coverage() -> None:
    panel = _eps_panel(_varying_values(16))
    sue = seasonal_random_walk_sue(panel, window=4, min_periods=3)
    cov = sue_coverage(panel, sue)
    assert cov["sue_events"].iloc[0] == 16
    assert 0 <= cov["sue_coverage"].iloc[0] <= 1
