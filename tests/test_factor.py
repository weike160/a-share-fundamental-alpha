"""factor 层测试: IC / RankIC / ICIR / t 值.

全部离线。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor.ic import (
    ic,
    ic_by_date,
    ic_series,
    ic_summary,
    icir,
    t_stat,
)


# --------------------------------------------------------------------------
# ic
# --------------------------------------------------------------------------
def test_ic_perfect_positive() -> None:
    s = pd.Series([1, 2, 3, 4, 5], dtype=float)
    r = pd.Series([1, 2, 3, 4, 5], dtype=float)
    assert ic(s, r) == pytest.approx(1.0)
    assert ic(s, r, method="spearman") == pytest.approx(1.0)


def test_ic_perfect_negative() -> None:
    s = pd.Series([1, 2, 3, 4, 5], dtype=float)
    r = pd.Series([5, 4, 3, 2, 1], dtype=float)
    assert ic(s, r) == pytest.approx(-1.0)
    assert ic(s, r, method="spearman") == pytest.approx(-1.0)


def test_rankic_is_robust_to_outlier_that_breaks_pearson() -> None:
    """RankIC 的意义: 一个极端值能把 Pearson 带偏, 但带不偏排名相关."""
    s = pd.Series([1, 2, 3, 4, 5], dtype=float)
    r = pd.Series([1, 2, 3, 4, 1000], dtype=float)
    pearson = ic(s, r, method="pearson")
    spearman = ic(s, r, method="spearman")
    assert spearman == pytest.approx(1.0)
    assert pearson < 1.0, "Pearson 应当被离群值拉低"


def test_ic_constant_signal_is_nan_not_zero() -> None:
    """常数信号无法定义相关 -> 必须 NaN. 返回 0 会被误读成「无相关性」."""
    s = pd.Series([1.0] * 10)
    r = pd.Series(np.arange(10, dtype=float))
    assert pd.isna(ic(s, r))
    assert pd.isna(ic(s, r, method="spearman"))


def test_ic_insufficient_obs_is_nan() -> None:
    s = pd.Series([1.0, 2.0])
    r = pd.Series([2.0, 1.0])
    assert pd.isna(ic(s, r))  # 少于 MIN_OBS
    assert not pd.isna(ic(s, r, min_obs=2))


def test_ic_drops_nan_pairs() -> None:
    s = pd.Series([1, 2, np.nan, 4, 5, 6], dtype=float)
    r = pd.Series([1, 2, 3, np.nan, 5, 6], dtype=float)
    got = ic(s, r, min_obs=3)
    assert got == pytest.approx(1.0)


def test_ic_rejects_bad_method() -> None:
    with pytest.raises(ValueError, match="method"):
        ic(pd.Series([1.0, 2, 3]), pd.Series([1.0, 2, 3]), method="kendall")


def test_ic_aligns_by_index_not_position() -> None:
    """两个序列索引顺序不同时, 必须按索引对齐而不是按位置."""
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=list("abcde"))
    r = pd.Series([5.0, 4.0, 3.0, 2.0, 1.0], index=list("edcba"))
    # 按索引对齐后其实是完全正相关
    assert ic(s, r) == pytest.approx(1.0)


# --------------------------------------------------------------------------
# ic_series
# --------------------------------------------------------------------------
def test_ic_series_row_wise() -> None:
    dates = pd.date_range("2024-01-01", periods=3)
    cols = list("abcde")
    signals = pd.DataFrame(
        [[1, 2, 3, 4, 5], [5, 4, 3, 2, 1], [1, 2, 3, 4, 5]],
        index=dates, columns=cols, dtype=float,
    )
    rets = pd.DataFrame(
        [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [5, 4, 3, 2, 1]],
        index=dates, columns=cols, dtype=float,
    )
    got = ic_series(signals, rets)
    assert got.tolist() == pytest.approx([1.0, -1.0, -1.0])


def test_ic_series_aligns_columns() -> None:
    dates = pd.date_range("2024-01-01", periods=2)
    signals = pd.DataFrame([[1, 2, 3, 4, 5]] * 2, index=dates,
                           columns=list("abcde"), dtype=float)
    rets = pd.DataFrame([[1, 2, 3, 4, 5]] * 2, index=dates,
                        columns=list("edcba"), dtype=float)
    got = ic_series(signals, rets, method="pearson")
    assert got.notna().all()


def test_ic_series_empty_when_no_overlap() -> None:
    signals = pd.DataFrame({"a": [1.0]}, index=pd.date_range("2024-01-01", periods=1))
    rets = pd.DataFrame({"b": [1.0]}, index=pd.date_range("2024-01-01", periods=1))
    got = ic_series(signals, rets)
    assert len(got) == 0


# --------------------------------------------------------------------------
# icir / t_stat / ic_summary
# --------------------------------------------------------------------------
def test_icir_math() -> None:
    s = pd.Series([0.1, 0.2, 0.3, 0.4])
    assert icir(s) == pytest.approx(s.mean() / s.std(ddof=1))


def test_icir_insufficient_periods_is_nan() -> None:
    assert pd.isna(icir(pd.Series([0.1, 0.2]), min_periods=3))


def test_icir_zero_std_is_nan() -> None:
    assert pd.isna(icir(pd.Series([0.1, 0.1, 0.1, 0.1])))


def test_icir_ignores_nan_periods() -> None:
    s = pd.Series([0.1, np.nan, 0.2, 0.3, 0.4])
    assert pd.notna(icir(s))


def test_t_stat_positive_for_consistent_positive_ic() -> None:
    s = pd.Series([0.05] * 12) + pd.Series(np.linspace(-0.01, 0.01, 12))
    assert t_stat(s) > 2


def test_t_stat_newey_west_runs_and_is_finite() -> None:
    rng = np.random.default_rng(0)
    s = pd.Series(rng.normal(0.03, 0.05, 40))
    plain = t_stat(s)
    nw = t_stat(s, lags=4)
    assert np.isfinite(plain) and np.isfinite(nw)


def test_t_stat_rejects_negative_lags() -> None:
    with pytest.raises(ValueError):
        t_stat(pd.Series([0.1, 0.2, 0.3]), lags=-1)


def test_ic_summary_shape_and_values() -> None:
    s = pd.Series([0.05, 0.1, -0.02, 0.08, 0.03])
    out = ic_summary(s)
    row = out.iloc[0]
    assert row["periods"] == 5
    assert row["ic_mean"] == pytest.approx(s.mean())
    assert 0 <= row["positive_rate"] <= 1
    assert row["positive_rate"] == pytest.approx(4 / 5)


def test_ic_summary_empty_series() -> None:
    out = ic_summary(pd.Series([np.nan, np.nan]))
    assert out["periods"].iloc[0] == 0


# --------------------------------------------------------------------------
# ic_by_date (事件面板主口径)
# --------------------------------------------------------------------------
def _panel(n_dates: int = 4, n_per_date: int = 8) -> pd.DataFrame:
    rows = []
    for d in pd.date_range("2024-04-01", periods=n_dates):
        for i in range(n_per_date):
            rows.append(
                {
                    "code": f"{i:06d}",
                    "tradable_ts": d,
                    "sue": float(i),
                    "fwd_ret_20d": float(i) * 0.01,
                }
            )
    return pd.DataFrame(rows)


def test_ic_by_date_groups_and_detects_perfect_signal() -> None:
    got = ic_by_date(_panel())
    assert len(got) == 4
    assert got.tolist() == pytest.approx([1.0] * 4)


def test_ic_by_date_negative_when_signal_reversed() -> None:
    panel = _panel()
    panel["fwd_ret_20d"] = -panel["fwd_ret_20d"]
    got = ic_by_date(panel)
    assert got.tolist() == pytest.approx([-1.0] * 4)


def test_ic_by_date_skips_thin_cross_sections() -> None:
    panel = _panel(n_per_date=8)
    # 把某个日期只剩 2 个样本
    first = panel["tradable_ts"].min()
    panel = panel[~((panel["tradable_ts"] == first) & (panel["code"] > "000001"))]
    got = ic_by_date(panel)
    assert first not in got.index, "样本不足的横截面应被剔除"


def test_ic_by_date_rejects_missing_columns() -> None:
    with pytest.raises(ValueError, match="缺少列"):
        ic_by_date(pd.DataFrame({"code": ["000001"]}))
