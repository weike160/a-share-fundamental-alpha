"""signals / factor 补充测试: 同比, 预处理, 中性化, 分层, IC 衰减.

全部离线、确定性。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor.decay import ic_decay
from factor.neutralize import (
    cross_sectional_regression,
    industry_neutralize,
    neutralize_cross_section,
)
from factor.preprocess import (
    time_series_split,
    winsorize,
    zscore_cross_section,
)
from factor.quantile import (
    monotonicity_check,
    quantile_returns,
    quantile_returns_by_date,
)
from signals.earnings_growth import earnings_growth, roe_change, yoy_growth
from signals.revenue_surprise import revenue_surprise, revenue_surprise_yoy


# --------------------------------------------------------------------------
# signals: yoy_growth
# --------------------------------------------------------------------------
def test_yoy_growth_zero_and_nan_handling() -> None:
    s = pd.Series(
        [10.0, np.nan, 20.0, 0.0, 15.0, 5.0, np.nan, 30.0],
        index=list("abcdefgh"),
    )
    got = yoy_growth(s, lag=4)

    assert list(got.index) == list(s.index)
    assert got.loc["e"] == pytest.approx(0.5)  # 15 / 10 - 1
    assert got.loc["a":"d"].isna().all()  # 去年同期不存在
    assert pd.isna(got.loc["f"])  # 去年同期缺失
    assert pd.isna(got.loc["g"])  # 当期缺失
    assert pd.isna(got.loc["h"])  # 去年同期为 0 -> NaN 而不是 inf
    assert not np.isinf(got).any()


def test_yoy_growth_rejects_bad_lag() -> None:
    with pytest.raises(ValueError):
        yoy_growth(pd.Series([1.0, 2.0]), lag=0)


# --------------------------------------------------------------------------
# signals: earnings_growth / roe_change
# --------------------------------------------------------------------------
def _growth_panel() -> pd.DataFrame:
    """两只股票各 6 个报告期, 行序打乱, 索引为字符串标签."""
    rows = []
    for code, base in (("000001", 100.0), ("600000", 200.0)):
        for i, day in enumerate(pd.date_range("2020-03-31", periods=6, freq="QE")):
            rows.append(
                {
                    "code": code,
                    "report_date": day,
                    "net_profit": base + 50.0 * i,
                    "revenue": base * 10.0 + 100.0 * i,
                    "eps": base / 100.0 + 0.5 * i,
                    "roe": 5.0 + 0.3 * i,
                }
            )
    panel = pd.DataFrame(rows).sample(frac=1.0, random_state=7)
    panel.index = [f"row{i}" for i in range(len(panel))]
    return panel


def test_earnings_growth_restores_index_and_values() -> None:
    panel = _growth_panel()
    dates = pd.date_range("2020-03-31", periods=6, freq="QE")
    out = earnings_growth(panel)

    assert list(out.index) == list(panel.index)
    assert list(out.columns) == [
        "net_profit_yoy_calc",
        "revenue_yoy_calc",
        "eps_yoy_calc",
    ]
    # 前 4 期没有去年同期
    early = panel["report_date"].isin(dates[:4])
    assert out.loc[early].isna().all().all()

    last = (panel["code"] == "000001") & (panel["report_date"] == dates[4])
    assert out.loc[last, "net_profit_yoy_calc"].iloc[0] == pytest.approx(2.0)
    assert out.loc[last, "revenue_yoy_calc"].iloc[0] == pytest.approx(0.4)
    assert out.loc[last, "eps_yoy_calc"].iloc[0] == pytest.approx(2.0)

    other = (panel["code"] == "600000") & (panel["report_date"] == dates[4])
    assert out.loc[other, "net_profit_yoy_calc"].iloc[0] == pytest.approx(1.0)
    assert out.loc[other, "revenue_yoy_calc"].iloc[0] == pytest.approx(0.2)


def test_earnings_growth_missing_value_column_is_nan_not_error() -> None:
    panel = _growth_panel().drop(columns=["revenue", "eps"])
    out = earnings_growth(panel)

    assert out["net_profit_yoy_calc"].notna().any()
    assert out["revenue_yoy_calc"].isna().all()
    assert out["eps_yoy_calc"].isna().all()


def test_earnings_growth_requires_code_and_report_date() -> None:
    with pytest.raises(ValueError, match="缺少必需列"):
        earnings_growth(pd.DataFrame({"net_profit": [1.0, 2.0]}))


def test_roe_change_matches_manual_difference() -> None:
    panel = _growth_panel()
    dates = pd.date_range("2020-03-31", periods=6, freq="QE")
    got = roe_change(panel)

    assert list(got.index) == list(panel.index)
    assert got.name == "roe_change"
    row = (panel["code"] == "000001") & (panel["report_date"] == dates[4])
    assert got.loc[row].iloc[0] == pytest.approx(1.2)  # 0.3 * 4
    early = panel["report_date"].isin(dates[:4])
    assert got.loc[early].isna().all()


def test_roe_change_rejects_missing_columns() -> None:
    with pytest.raises(ValueError, match="缺少必需列"):
        roe_change(_growth_panel().drop(columns=["roe"]))


# --------------------------------------------------------------------------
# signals: revenue_surprise
# --------------------------------------------------------------------------
def test_revenue_surprise_uses_absolute_denominator() -> None:
    actual = pd.Series([1.0, 1.0, 2.0, 3.0])
    expected = pd.Series([0.0, np.nan, 1.0, -2.0])
    got = revenue_surprise(actual, expected)

    assert pd.isna(got.iloc[0])  # 预期为 0
    assert pd.isna(got.iloc[1])  # 预期缺失
    assert got.iloc[2] == pytest.approx(1.0)
    assert got.iloc[3] == pytest.approx((3.0 - (-2.0)) / 2.0)
    assert not np.isinf(got).any()


def test_revenue_surprise_aligns_by_index() -> None:
    actual = pd.Series([3.0, 4.0], index=["x", "y"])
    expected = pd.Series([2.0, 1.0], index=["y", "x"])
    got = revenue_surprise(actual, expected)
    # 按标签对齐: expected["x"] = 1, expected["y"] = 2
    assert got.loc["x"] == pytest.approx((3.0 - 1.0) / 1.0)
    assert got.loc["y"] == pytest.approx((4.0 - 2.0) / 2.0)
    # 若按位置对齐会得到 0.5 / 3.0, 这两个断言可区分两种对齐方式
    assert got.loc["x"] != pytest.approx(0.5)


def test_revenue_surprise_yoy_matches_yoy_growth() -> None:
    revenue = pd.Series([100.0, 110.0, 120.0, 130.0, 150.0, 165.0])
    got = revenue_surprise_yoy(revenue, lag=4)
    assert got.iloc[:4].isna().all()
    assert got.iloc[4] == pytest.approx(0.5)
    assert got.iloc[5] == pytest.approx(0.5)


# --------------------------------------------------------------------------
# factor: winsorize
# --------------------------------------------------------------------------
def test_winsorize_clips_tails() -> None:
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 100.0])
    got = winsorize(s, lower=0.0, upper=0.8)
    assert got.max() == pytest.approx(5.0)
    assert got.min() == pytest.approx(1.0)
    assert got.iloc[:5].tolist() == pytest.approx([1.0, 2.0, 3.0, 4.0, 5.0])


def test_winsorize_ignores_nan_when_computing_quantiles() -> None:
    s = pd.Series([1.0, np.nan, 5.0, 100.0])
    got = winsorize(s, lower=0.0, upper=0.5)
    assert pd.isna(got.iloc[1])
    assert got.iloc[0] == pytest.approx(1.0)
    assert got.iloc[2] == pytest.approx(5.0)
    assert got.iloc[3] == pytest.approx(5.0)


def test_winsorize_all_nan_or_empty_returns_copy() -> None:
    s = pd.Series([np.nan, np.nan], index=["a", "b"])
    got = winsorize(s)
    assert got.isna().all()
    assert list(got.index) == ["a", "b"]

    empty = pd.Series(dtype="float64")
    assert winsorize(empty).empty


# --------------------------------------------------------------------------
# factor: zscore_cross_section
# --------------------------------------------------------------------------
def test_zscore_std_zero_is_all_nan() -> None:
    day = pd.Timestamp("2024-01-02")
    frame = pd.DataFrame(
        {"tradable_ts": [day] * 3, "x": [5.0, 5.0, 5.0]}, index=list("abc")
    )
    got = zscore_cross_section(frame, day, value_col="x")
    assert got.isna().all()
    assert list(got.index) == list(frame.index)


def test_zscore_selects_same_calendar_day_and_standardizes() -> None:
    day = pd.Timestamp("2024-01-02")
    frame = pd.DataFrame(
        {
            "tradable_ts": [
                pd.Timestamp("2024-01-02 00:00:00"),
                pd.Timestamp("2024-01-02 03:00:00"),
                pd.Timestamp("2024-01-03 00:00:00"),
            ],
            "x": [1.0, 3.0, 100.0],
        },
        index=["a", "b", "c"],
    )
    got = zscore_cross_section(frame, day, value_col="x")

    assert list(got.index) == ["a", "b"]
    std = pd.Series([1.0, 3.0]).std(ddof=1)
    assert got.loc["a"] == pytest.approx((1.0 - 2.0) / std)
    assert got.loc["b"] == pytest.approx((3.0 - 2.0) / std)
    assert got.mean() == pytest.approx(0.0)


def test_zscore_fewer_than_two_valid_rows_is_nan() -> None:
    day = pd.Timestamp("2024-01-02")
    frame = pd.DataFrame(
        {"tradable_ts": [day, day], "x": [1.0, np.nan]}, index=["a", "b"]
    )
    got = zscore_cross_section(frame, day, value_col="x")
    assert got.isna().all()


# --------------------------------------------------------------------------
# factor: time_series_split
# --------------------------------------------------------------------------
def _daily_panel() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=20, freq="D")
    return pd.DataFrame(
        {
            "tradable_ts": np.repeat(dates, 5),
            "code": np.tile([f"{i:06d}" for i in range(5)], 20),
            "x": np.arange(100, dtype="float64"),
        },
        index=[f"r{i}" for i in range(100)],
    )


def test_time_series_split_no_leakage_and_first_block_train_only() -> None:
    panel = _daily_panel()
    folds = time_series_split(panel, n_splits=5, min_train_frac=0.0)

    assert len(folds) == 4  # 第一个区间只做训练集
    for train, test in folds:
        assert len(train) > 0 and len(test) > 0
        assert train["tradable_ts"].max() < test["tradable_ts"].min()
        assert train.index.isin(panel.index).all()
        assert test.index.isin(panel.index).all()

    # 最后一折的训练集覆盖前 4 个区间
    train, test = folds[-1]
    assert len(train) == 80 and len(test) == 20


def test_time_series_split_purge_gap() -> None:
    panel = _daily_panel()
    purge = 2
    folds = time_series_split(
        panel, n_splits=5, purge_days=purge, min_train_frac=0.0
    )
    assert len(folds) == 4
    for train, test in folds:
        assert len(train) > 0
        # 训练集必须早于「测试集起点 - purge 天」
        gap = test["tradable_ts"].min() - train["tradable_ts"].max()
        assert gap.days > purge


def test_time_series_split_drops_folds_with_small_train() -> None:
    panel = _daily_panel()
    # 第一个区间只有 20% 的行 < 30%, 应被丢弃
    folds = time_series_split(panel, n_splits=5, min_train_frac=0.3)
    assert len(folds) == 3


# --------------------------------------------------------------------------
# factor: neutralize
# --------------------------------------------------------------------------
def test_industry_neutralize_removes_group_means() -> None:
    signal = pd.Series([1.0, 2.0, 3.0, 10.0, 11.0, 12.0], index=list("abcdef"))
    industry = pd.Series(["A", "A", "A", "B", "B", "B"], index=list("abcdef"))
    got = industry_neutralize(signal, industry)

    assert got.loc[["a", "b", "c"]].tolist() == pytest.approx([-1.0, 0.0, 1.0])
    assert got.loc[["d", "e", "f"]].tolist() == pytest.approx([-1.0, 0.0, 1.0])

    grouped = pd.DataFrame({"signal": got, "industry": industry}).groupby("industry")
    assert grouped["signal"].mean().abs().max() < 1e-12


def test_industry_neutralize_nan_industry_is_nan() -> None:
    signal = pd.Series([1.0, 2.0, 3.0], index=list("abc"))
    industry = pd.Series(["A", np.nan, "A"], index=list("abc"))
    got = industry_neutralize(signal, industry)
    assert pd.isna(got.loc["b"])
    assert got.loc[["a", "c"]].tolist() == pytest.approx([-1.0, 1.0])


def test_neutralize_cross_section_returns_residuals() -> None:
    x = pd.Series([0.0, 1.0, 2.0, 3.0], index=list("abcd"))

    proportional = 2.0 * x
    assert (
        neutralize_cross_section(
            proportional, x.to_frame("x"), add_constant=False
        ).abs().max()
        < 1e-9
    )

    # 带截距时应剥离 5.0 的平移; 不带截距时平移会留在残差里
    shifted = 5.0 + proportional
    with_const = neutralize_cross_section(shifted, x.to_frame("x"), add_constant=True)
    assert with_const.abs().max() < 1e-9
    no_const = neutralize_cross_section(shifted, x.to_frame("x"), add_constant=False)
    assert no_const.abs().max() > 1.0

    exposed = pd.DataFrame({"x": [1.0, np.nan, 3.0]}, index=list("abc"))
    signal = pd.Series([1.0, 2.0, 3.0], index=list("abc"))
    out = neutralize_cross_section(signal, exposed)
    assert pd.isna(out.loc["b"])  # 暴露缺失 -> NaN
    assert list(out.index) == ["a", "b", "c"]


def test_cross_sectional_regression_recovers_known_coefficients() -> None:
    rng = np.random.default_rng(42)
    x = rng.normal(size=200)
    z = rng.normal(size=200)
    y = 0.5 + 2.0 * x - 1.0 * z + rng.normal(scale=0.1, size=200)

    out = cross_sectional_regression(
        pd.Series(y), pd.DataFrame({"x": x, "z": z})
    )

    assert list(out.columns) == ["coef", "std_err", "t_stat", "p_value", "n_obs"]
    assert list(out.index) == ["x", "z"]
    assert out.loc["x", "coef"] == pytest.approx(2.0, abs=0.05)
    assert out.loc["z", "coef"] == pytest.approx(-1.0, abs=0.05)
    assert out.loc["x", "std_err"] > 0
    assert abs(out.loc["x", "t_stat"]) > 5
    assert out.loc["x", "p_value"] < 1e-6
    assert out["n_obs"].tolist() == [200, 200]


def test_cross_sectional_regression_complete_cases_only() -> None:
    returns = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0])
    features = pd.DataFrame({"x": [0.0, 1.0, 2.0, np.nan, 4.0]})
    out = cross_sectional_regression(returns, features)
    assert out.loc["x", "n_obs"] == 3


def test_cross_sectional_regression_insufficient_rows_is_empty() -> None:
    returns = pd.Series([1.0, 2.0])
    features = pd.DataFrame({"x": [1.0, 2.0], "z": [0.0, 1.0]})
    out = cross_sectional_regression(returns, features)
    assert out.empty
    assert list(out.columns) == ["coef", "std_err", "t_stat", "p_value", "n_obs"]


# --------------------------------------------------------------------------
# factor: quantile
# --------------------------------------------------------------------------
def test_quantile_assignment_order_and_counts() -> None:
    signal = pd.Series(np.arange(100, dtype="float64"))
    forward = signal * 0.01
    table = quantile_returns(signal, forward, n_quantiles=10)

    assert list(table.index) == list(range(1, 11))
    assert list(table.columns) == [
        "mean_return",
        "median_return",
        "std_return",
        "count",
    ]
    assert table["count"].tolist() == [10] * 10
    assert table["mean_return"].is_monotonic_increasing
    assert table.loc[1, "mean_return"] < table.loc[10, "mean_return"]


def test_quantile_one_is_lowest_signal() -> None:
    signal = pd.Series(np.arange(100, dtype="float64"))
    forward = -signal * 0.01
    table = quantile_returns(signal, forward, n_quantiles=10)
    assert table["mean_return"].is_monotonic_decreasing
    assert table.loc[1, "mean_return"] > table.loc[10, "mean_return"]


def test_quantile_returns_raises_when_too_few_rows() -> None:
    signal = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    with pytest.raises(ValueError):
        quantile_returns(signal, signal, n_quantiles=3)


def test_monotonicity_check_true_false_and_tolerance() -> None:
    good = pd.DataFrame({"mean_return": [0.0, 0.01, 0.02]}, index=[1, 2, 3])
    bad = pd.DataFrame({"mean_return": [0.0, 0.02, 0.01]}, index=[1, 2, 3])
    tiny = pd.DataFrame({"mean_return": [0.0, -1e-13, 0.0]}, index=[1, 2, 3])
    short = pd.DataFrame({"mean_return": [0.0, 0.01]}, index=[1, 2])

    assert monotonicity_check(good) is True
    assert monotonicity_check(bad) is False
    assert monotonicity_check(tiny) is True
    assert monotonicity_check(short) is False


def test_quantile_returns_by_date_skips_thin_dates() -> None:
    dates = pd.date_range("2024-01-01", periods=3, freq="D")
    rows = []
    for i, day in enumerate(dates):
        n = 10 if i != 1 else 3  # 中间那天样本不足, 应被跳过
        for j in range(n):
            rows.append(
                {
                    "actual_disclosure_date": day,
                    "sue": float(j),
                    "fwd_ret_5d": float(j) * 0.01,
                }
            )
    long = pd.DataFrame(rows)
    out = quantile_returns_by_date(
        long, signal_col="sue", return_col="fwd_ret_5d", n_quantiles=5
    )

    assert list(out.columns) == ["date", "quantile", "mean_return", "count"]
    assert out["date"].nunique() == 2
    assert dates[1] not in set(out["date"])
    assert out["count"].sum() == 20
    assert out.groupby("quantile")["mean_return"].mean().is_monotonic_increasing


def test_quantile_returns_by_date_missing_column_raises() -> None:
    with pytest.raises(ValueError, match="缺少列"):
        quantile_returns_by_date(
            pd.DataFrame({"sue": [1.0]}), signal_col="sue", return_col="fwd_ret_5d"
        )


# --------------------------------------------------------------------------
# factor: ic_decay
# --------------------------------------------------------------------------
def test_ic_decay_positive_ic_and_missing_column_is_nan() -> None:
    n = 50
    signal = pd.Series(np.arange(n, dtype="float64"), index=[f"s{i}" for i in range(n)])
    returns = pd.DataFrame(
        {f"fwd_ret_{h}d": signal.to_numpy() for h in (1, 5, 10)},
        index=signal.index,
    )
    got = ic_decay(signal, returns, horizons=(1, 5, 10, 20))

    assert got.name == "rank_ic"
    assert got.index.tolist() == [1, 5, 10, 20]
    assert (got.loc[[1, 5, 10]] > 0.99).all()
    assert pd.isna(got.loc[20])


def test_ic_decay_negative_when_returns_reversed() -> None:
    n = 40
    signal = pd.Series(np.arange(n, dtype="float64"))
    returns = pd.DataFrame({"fwd_ret_5d": -signal.to_numpy()})
    got = ic_decay(signal, returns, horizons=(5,))
    assert got.iloc[0] < -0.99


def test_ic_decay_per_date_average() -> None:
    n_per_date = 10
    dates = pd.Series(
        np.repeat(pd.date_range("2024-01-01", periods=3, freq="D"), n_per_date)
    )
    signal = pd.Series(np.tile(np.arange(n_per_date, dtype="float64"), 3))
    forward = np.concatenate(
        [
            np.arange(n_per_date, dtype="float64"),
            np.arange(n_per_date, dtype="float64"),
            -np.arange(n_per_date, dtype="float64"),
        ]
    )
    returns = pd.DataFrame({"fwd_ret_5d": forward})

    got = ic_decay(signal, returns, horizons=(5,), dates=dates)
    # 逐期 RankIC = [1, 1, -1] -> 均值 1/3
    assert got.iloc[0] == pytest.approx(1.0 / 3.0)

    pooled = ic_decay(signal, returns, horizons=(5,))
    assert pooled.iloc[0] != pytest.approx(1.0 / 3.0)


def test_ic_decay_drops_nan_pairs() -> None:
    signal = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0])
    returns = pd.DataFrame({"fwd_ret_1d": [1.0, 2.0, 3.0, np.nan, 5.0]})
    got = ic_decay(signal, returns, horizons=(1,))
    assert got.iloc[0] == pytest.approx(1.0)
