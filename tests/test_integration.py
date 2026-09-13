"""集成层测试: 规模/流动性特征、顺延成交、事件回测引擎.

全部离线, 使用合成行情与面板。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.engine import BacktestResult, CostModel, EventBacktester
from data.fills import audit_fills, build_trade_blocks, resolve_fills
from data.liquidity import FEATURE_COLUMNS, attach_price_features, build_price_features
from data.tradability import TradingCalendar

CAL = pd.DatetimeIndex(pd.bdate_range("2024-01-01", periods=40))


@pytest.fixture
def calendar() -> TradingCalendar:
    return TradingCalendar(CAL)


def _bars(code: str, closes: list[float], *, skip: set[int] | None = None) -> pd.DataFrame:
    """构造单只股票的日线, ``skip`` 中的下标表示缺失 (停牌)."""
    skip = skip or set()
    rows = []
    for i, c in enumerate(closes):
        if i in skip:
            continue
        rows.append(
            {
                "code": code,
                "date": CAL[i],
                "close": float(c),
                "amount": 1.0e8 + i,
                "turnover": 2.43,
                "volume": 1.0e6,
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# data/liquidity.py
# --------------------------------------------------------------------------
def test_build_price_features_columns_and_float_mcap() -> None:
    closes = [10.0 + 0.1 * i for i in range(25)]
    px = _bars("600000", closes)
    feats = build_price_features(px, adv_window=5, mom_window=5, min_periods=3)

    assert set(FEATURE_COLUMNS) <= set(feats.columns)
    last = feats.iloc[-1]
    # 成交额 / 换手率(小数) = 流通市值
    expected_mcap = (1.0e8 + 24) / (2.43 / 100.0)
    assert last["float_mcap"] == pytest.approx(expected_mcap)
    assert last["log_mcap"] == pytest.approx(np.log(expected_mcap))
    assert last["adv20"] == pytest.approx(np.mean([1.0e8 + i for i in range(20, 25)]))
    assert last["momentum_20d"] == pytest.approx(closes[24] / closes[19] - 1.0)
    assert last["volatility_20d"] > 0

    shifted = build_price_features(px, adv_window=5, mom_window=5, min_periods=3)
    assert pd.isna(shifted.iloc[0]["adv20"])  # 窗口不足


def test_attach_price_features_uses_only_past_sessions() -> None:
    # 跳空发生在入场日当天 (索引 11), 入场前的观测仍是 10.0
    px = _bars("600000", [10.0] * 11 + [20.0] * 9)
    feats = build_price_features(px, adv_window=3, mom_window=3, min_periods=2)

    panel = pd.DataFrame(
        {
            "code": ["600000"],
            "tradable_ts": [CAL[11]],
        }
    )
    merged = attach_price_features(panel, feats)
    # 特征必须取上一交易日 (close=10) 的观测, 而不是入场日当天的 20
    assert merged.loc[0, "momentum_20d"] == pytest.approx(
        feats.loc[feats["date"] == CAL[10], "momentum_20d"].iloc[0]
    )
    assert merged.loc[0, "momentum_20d"] != pytest.approx(20.0 / 10.0 - 1.0)


def test_attach_price_features_missing_code_gives_nan() -> None:
    feats = build_price_features(_bars("600000", [10.0] * 10), adv_window=3, min_periods=2)
    panel = pd.DataFrame({"code": ["000001"], "tradable_ts": [CAL[5]]})
    merged = attach_price_features(panel, feats)
    assert pd.isna(merged.loc[0, "adv20"])


# --------------------------------------------------------------------------
# data/fills.py
# --------------------------------------------------------------------------
def _limit_prices() -> list[float]:
    """第 10 日涨停, 第 15 日跌停."""
    closes = [10.0] * 10
    closes += [11.0, 11.0, 11.0, 11.5, 12.0]  # 10..14
    closes += [10.8, 10.5, 10.4, 10.3, 10.2]  # 15..19   15 为跌停
    closes += [10.1 + 0.01 * i for i in range(20, 40)]
    return closes


def test_build_trade_blocks_detects_limit_and_suspension(calendar: TradingCalendar) -> None:
    px = _bars("600000", _limit_prices(), skip={12})
    blocks = build_trade_blocks(px, calendar)
    block = blocks["600000"]

    day10 = int(pd.Timestamp(CAL[10]).to_datetime64().astype("datetime64[D]").astype(int))
    day15 = int(pd.Timestamp(CAL[15]).to_datetime64().astype("datetime64[D]").astype(int))
    day12 = int(pd.Timestamp(CAL[12]).to_datetime64().astype("datetime64[D]").astype(int))

    assert day10 in block["limit_up"]
    assert day15 in block["limit_down"]
    assert day12 in block["suspended"]


def test_resolve_fills_shifts_entry_and_exit(calendar: TradingCalendar) -> None:
    px = _bars("600000", _limit_prices())
    panel = pd.DataFrame({"code": ["600000"], "tradable_ts": [CAL[10]]})

    out = resolve_fills(panel, px, calendar, horizons=(2, 4), max_shift=5)
    row = out.iloc[0]

    assert not bool(row["entry_blocked"])
    assert row["entry_shift_days"] == 1
    assert row["entry_date_adj"] == CAL[11]
    assert row["entry_price_adj"] == pytest.approx(11.0)

    # h=2: 入场索引 11 -> 目标 13, 可成交
    assert row["exit_date_adj_2d"] == CAL[13]
    assert row["fwd_ret_adj_2d"] == pytest.approx(11.5 / 11.0 - 1.0)

    # h=4: 目标 15 跌停 -> 顺延到 16
    assert row["exit_date_adj_4d"] == CAL[16]
    assert row["fwd_ret_adj_4d"] == pytest.approx(10.5 / 11.0 - 1.0)
    assert bool(row["exit_blocked_4d"]) is False


def test_resolve_fills_skips_suspension(calendar: TradingCalendar) -> None:
    px = _bars("600000", _limit_prices(), skip={11, 12})
    panel = pd.DataFrame({"code": ["600000"], "tradable_ts": [CAL[10]]})

    out = resolve_fills(panel, px, calendar, horizons=(2,), max_shift=5)
    row = out.iloc[0]
    assert row["entry_date_adj"] == CAL[13]
    assert row["entry_shift_days"] == 3


def test_resolve_fills_marks_blocked_beyond_max_shift(calendar: TradingCalendar) -> None:
    px = _bars("600000", _limit_prices())
    panel = pd.DataFrame({"code": ["600000"], "tradable_ts": [CAL[10]]})

    out = resolve_fills(panel, px, calendar, horizons=(2,), max_shift=0)
    assert bool(out.iloc[0]["entry_blocked"])
    assert pd.isna(out.iloc[0]["entry_date_adj"])
    assert pd.isna(out.iloc[0]["fwd_ret_adj_2d"])


def test_audit_fills_reports_shift_and_blocked(calendar: TradingCalendar) -> None:
    px = _bars("600000", _limit_prices())
    panel = pd.DataFrame(
        {
            "code": ["600000", "600000"],
            "tradable_ts": [CAL[10], CAL[20]],
            "fwd_ret_2d": [0.0, 0.0],
        }
    )
    out = resolve_fills(panel, px, calendar, horizons=(2,), max_shift=5)
    audit = audit_fills(out, horizons=(2,))

    assert int(audit.loc[0, "fill_events"]) == 2
    assert int(audit.loc[0, "entry_blocked"]) == 0
    assert audit.loc[0, "entry_shift_gt0_rate"] == pytest.approx(0.5)


# --------------------------------------------------------------------------
# backtest/engine.py
# --------------------------------------------------------------------------
def _engine_panel() -> pd.DataFrame:
    rows = []
    for cohort, month in enumerate(["2024-01", "2024-02"], start=1):
        for i in range(1, 11):
            rows.append(
                {
                    # 不同队列持有不同股票, 符合月度事件组合的实际情形
                    "code": f"{600000 + cohort * 100 + i:06d}",
                    "tradable_ts": pd.Timestamp(f"{month}-15"),
                    "sue": float(i),
                    "fwd_ret_adj_20d": 0.05 if i >= 9 else 0.01,
                    "fwd_ret_20d": 0.05 if i >= 9 else 0.01,
                    "adv20": 1.0e8,
                    "volatility_20d": 0.02,
                    "entry_blocked": False,
                    "industry": "银行" if i % 2 else "医药",
                }
            )
    return pd.DataFrame(rows)


def test_engine_cohort_returns_and_costs() -> None:
    panel = _engine_panel()
    bt = EventBacktester(signal_col="sue", horizon=20, top_pct=0.2, min_names=3)
    res = bt.run(panel)

    assert isinstance(res, BacktestResult)
    assert len(res.cohorts) == 2
    first = res.cohorts.iloc[0]
    assert first["n_candidates"] == 10
    assert first["n_selected"] == 2
    assert first["gross_return"] == pytest.approx(0.05)
    assert first["benchmark_return"] == pytest.approx(0.018)
    assert first["active_return"] == pytest.approx(0.032)
    assert first["cost_total"] > 0
    assert first["net_return"] == pytest.approx(0.05 - first["cost_total"])
    assert first["commission"] == pytest.approx(0.0003 * 2)
    assert first["stamp_tax"] == pytest.approx(0.0005)

    assert set(res.cost_breakdown) == {
        "commission",
        "stamp_tax",
        "slippage",
        "impact",
        "cost_total",
    }
    assert res.metrics["n_cohorts"] == 2
    assert res.metrics["mean_turnover"] == pytest.approx(2.0)
    assert len(res.net_nav) == 2
    assert res.net_nav.iloc[-1] < res.gross_nav.iloc[-1]


def test_engine_higher_aum_raises_cost() -> None:
    panel = _engine_panel()
    small = EventBacktester(cost=CostModel(aum=1e6), min_names=3).run(panel)
    large = EventBacktester(cost=CostModel(aum=1e10), min_names=3).run(panel)

    assert (
        large.cohorts["cost_total"].mean() > small.cohorts["cost_total"].mean()
    )
    # 容量只由 ADV 与权重决定, 与 AUM 无关
    assert large.metrics["capacity_median"] == pytest.approx(
        small.metrics["capacity_median"]
    )


def test_engine_drops_blocked_entries() -> None:
    panel = _engine_panel()
    panel.loc[panel["sue"] >= 9, "entry_blocked"] = True
    res = EventBacktester(min_names=3).run(panel)
    # 两个高分位事件被剔除后, 组合只能选到次高分位的股票
    assert res.cohorts["gross_return"].max() <= 0.01 + 1e-12


def test_engine_prefers_adjusted_returns() -> None:
    panel = _engine_panel()
    panel["fwd_ret_20d"] = -0.5
    bt = EventBacktester(min_names=3)
    res = bt.run(panel)
    assert bt.return_col == "fwd_ret_adj_20d"
    assert res.cohorts["gross_return"].iloc[0] == pytest.approx(0.05)


def test_engine_uses_custom_cohort_column() -> None:
    panel = _engine_panel()
    panel["report_date"] = pd.to_datetime(["2023-12-31"] * 10 + ["2024-03-31"] * 10)
    res = EventBacktester(cohort_col="report_date", min_names=3).run(panel)
    assert list(res.cohorts.index) == ["2023-12-31", "2024-03-31"]


def test_engine_rejects_empty_panel() -> None:
    panel = _engine_panel()
    panel["sue"] = np.nan
    with pytest.raises(ValueError):
        EventBacktester(min_names=3).run(panel)
