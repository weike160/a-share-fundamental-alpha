"""data 层单元测试.

全部离线: 网络接口一律通过构造数据或注入 ``call`` 钩子替代, 保证测试
快速、确定、可离线运行。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data import prices
from data import source as src
from data.announcement import to_tradable_ts
from data.corporate_action import apply_adjustment
from data.delisting import flag_delisted
from data.financials import attach_announcement_time, audit_coverage
from data.panel import audit_survivorship, build_event_panel
from data.prices import _CircuitBreaker, to_sina_symbol
from data.source import DataSourceError
from data.st_status import (
    BASIS_SH_CURRENT,
    BASIS_SZ_HISTORY,
    audit_st,
    build_name_timeline,
    drop_st,
    flag_st,
    is_st_name,
    name_as_of,
)
from data.tradability import (
    TradingCalendar,
    detect_limit_moves,
    infer_suspensions,
    is_tradable,
    order_size_cap,
    price_limit_ratio,
)
from data.universe import build_universe, is_listed

# --------------------------------------------------------------------------
# 测试用交易日历: 2024-01-01(一) ~ 2024-01-12(五) 的工作日
# --------------------------------------------------------------------------
SESSIONS = pd.DatetimeIndex(
    [
        "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05",
        "2024-01-08", "2024-01-09", "2024-01-10", "2024-01-11", "2024-01-12",
    ]
)


@pytest.fixture()
def cal() -> TradingCalendar:
    return TradingCalendar(SESSIONS)


# --------------------------------------------------------------------------
# 交易日历
# --------------------------------------------------------------------------
def test_calendar_membership(cal: TradingCalendar) -> None:
    assert len(cal) == 9
    assert cal.is_trading_day("2024-01-05")
    assert not cal.is_trading_day("2024-01-06")  # 周六
    assert not cal.is_trading_day("2024-01-07")  # 周日
    assert "2024-01-08" in cal


def test_calendar_next_prev(cal: TradingCalendar) -> None:
    # 周五 -> 下周一
    assert cal.next_trading_day("2024-01-05") == pd.Timestamp("2024-01-08")
    # 周六 -> 下周一
    assert cal.next_trading_day("2024-01-06") == pd.Timestamp("2024-01-08")
    # inclusive 时落在当天
    assert cal.next_trading_day("2024-01-05", inclusive=True) == pd.Timestamp("2024-01-05")
    # 周一 -> 上周五
    assert cal.prev_trading_day("2024-01-08") == pd.Timestamp("2024-01-05")
    assert cal.prev_trading_day("2024-01-06") == pd.Timestamp("2024-01-05")


def test_calendar_shift(cal: TradingCalendar) -> None:
    assert cal.shift("2024-01-02", 1) == pd.Timestamp("2024-01-03")
    assert cal.shift("2024-01-02", -1) is None          # 越界
    assert cal.shift("2024-01-12", 1) is None           # 越界
    assert cal.shift("2024-01-05", 1) == pd.Timestamp("2024-01-08")  # 跨周末


def test_sessions_between(cal: TradingCalendar) -> None:
    got = cal.sessions_between("2024-01-03", "2024-01-09")
    assert list(got) == list(pd.DatetimeIndex(["2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08", "2024-01-09"]))
    assert len(cal.sessions_between("2024-01-06", "2024-01-07")) == 0


# --------------------------------------------------------------------------
# T+1 对齐 (含 off-by-one 回归)
# --------------------------------------------------------------------------
def test_to_tradable_ts_lag1_is_exactly_one_session(cal: TradingCalendar) -> None:
    # 周三披露 -> 周四入场 (不是周五)
    got = to_tradable_ts(pd.Series([pd.Timestamp("2024-01-03")]), cal, lag=1)
    assert got.iloc[0] == pd.Timestamp("2024-01-04")

    # 周五披露 -> 下周一入场
    got = to_tradable_ts(pd.Series([pd.Timestamp("2024-01-05")]), cal, lag=1)
    assert got.iloc[0] == pd.Timestamp("2024-01-08")

    # 周日披露 -> 周一入场
    got = to_tradable_ts(pd.Series([pd.Timestamp("2024-01-07")]), cal, lag=1)
    assert got.iloc[0] == pd.Timestamp("2024-01-08")


def test_to_tradable_ts_lag_ordering(cal: TradingCalendar) -> None:
    ts = pd.Series([pd.Timestamp("2024-01-03")])
    d1 = to_tradable_ts(ts, cal, lag=1).iloc[0]
    d2 = to_tradable_ts(ts, cal, lag=2).iloc[0]
    d3 = to_tradable_ts(ts, cal, lag=3).iloc[0]
    assert d1 == pd.Timestamp("2024-01-04")
    assert d2 == pd.Timestamp("2024-01-05")
    assert d3 == pd.Timestamp("2024-01-08")  # 跨周末


def test_to_tradable_ts_lag0_same_day(cal: TradingCalendar) -> None:
    got = to_tradable_ts(pd.Series([pd.Timestamp("2024-01-03")]), cal, lag=0)
    assert got.iloc[0] == pd.Timestamp("2024-01-03")


def test_to_tradable_ts_rejects_negative_lag(cal: TradingCalendar) -> None:
    with pytest.raises(ValueError):
        to_tradable_ts(pd.Series([pd.Timestamp("2024-01-03")]), cal, lag=-1)


def test_to_tradable_ts_strictly_after_disclosure(cal: TradingCalendar) -> None:
    """不变式: 入场日必须严格晚于披露日 (look-ahead 防护)."""
    disclosures = pd.Series(pd.to_datetime([f"2024-01-{d:02d}" for d in range(1, 13)]))
    mapped = to_tradable_ts(disclosures, cal, lag=1)
    for d, t in zip(disclosures, mapped):
        if pd.notna(t):
            assert t > d


# --------------------------------------------------------------------------
# 股票池 / 幸存者偏差
# --------------------------------------------------------------------------
@pytest.fixture()
def listing() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "code": ["000001", "000002", "000003"],
            "list_date": pd.to_datetime(["2000-01-01", "2024-01-04", "2010-06-01"]),
        }
    )


@pytest.fixture()
def delisting() -> pd.DataFrame:
    return pd.DataFrame(
        {"code": ["000003", "000009"], "delist_date": pd.to_datetime(["2024-01-05", "2020-01-01"])}
    )


def test_build_universe_excludes_not_yet_listed(listing: pd.DataFrame, delisting: pd.DataFrame) -> None:
    u = build_universe("2024-01-02", listing, delisting)
    assert "000001" in u
    assert "000002" not in u          # 2024-01-04 才上市
    assert "000003" in u              # 尚未退市 (01-05 退)


def test_build_universe_excludes_delisted(listing: pd.DataFrame, delisting: pd.DataFrame) -> None:
    # 退市日当天视为已退市, 不计入
    assert "000003" not in build_universe("2024-01-05", listing, delisting)
    assert "000003" not in build_universe("2024-01-06", listing, delisting)
    assert "000003" in build_universe("2024-01-04", listing, delisting)


def test_build_universe_delisted_stock_still_present_historically(
    listing: pd.DataFrame, delisting: pd.DataFrame
) -> None:
    """已退市股票在退市前必须仍在池子里 —— 这正是防幸存者偏差的关键."""
    u = build_universe("2024-01-04", listing, delisting)
    assert "000003" in u


def test_build_universe_handles_empty_delistings(listing: pd.DataFrame) -> None:
    u = build_universe("2024-01-10", listing, pd.DataFrame(columns=["code", "delist_date"]))
    assert u == {"000001", "000002", "000003"}


def test_is_listed(listing: pd.DataFrame, delisting: pd.DataFrame) -> None:
    assert is_listed("000001", "2024-01-02", listing, delisting)
    assert not is_listed("000002", "2024-01-02", listing, delisting)


def test_flag_delisted() -> None:
    universe = pd.DataFrame(
        {"code": ["000003", "000001"], "date": pd.to_datetime(["2024-01-06", "2024-01-06"])}
    )
    delistings = pd.DataFrame(
        {"code": ["000003"], "delist_date": pd.to_datetime(["2024-01-05"])}
    )
    out = flag_delisted(universe, delistings)
    flags = dict(zip(out["code"], out["is_delisted"]))
    assert flags["000003"] is True or bool(flags["000003"]) is True
    assert not flags["000001"]


# --------------------------------------------------------------------------
# 可交易性
# --------------------------------------------------------------------------
def test_is_tradable_suspension_blocks_both_sides() -> None:
    susp = pd.DataFrame({"code": ["000001"], "date": pd.to_datetime(["2024-01-03"])})
    empty = pd.DataFrame(columns=["code", "date"])
    adv = pd.Series({"000001": 1e8})
    kw = {"suspended": susp, "limit_up": empty, "limit_down": empty, "adv": adv}
    assert not is_tradable("000001", "2024-01-03", side="buy", **kw)
    assert not is_tradable("000001", "2024-01-03", side="sell", **kw)
    assert is_tradable("000001", "2024-01-04", side="buy", **kw)


def test_is_tradable_limit_up_blocks_buy_only() -> None:
    up = pd.DataFrame({"code": ["000001"], "date": pd.to_datetime(["2024-01-03"])})
    empty = pd.DataFrame(columns=["code", "date"])
    adv = pd.Series({"000001": 1e8})
    kw = {"suspended": empty, "limit_up": up, "limit_down": empty, "adv": adv}
    assert not is_tradable("000001", "2024-01-03", side="buy", **kw)
    assert is_tradable("000001", "2024-01-03", side="sell", **kw)


def test_is_tradable_limit_down_blocks_sell_only() -> None:
    down = pd.DataFrame({"code": ["000001"], "date": pd.to_datetime(["2024-01-03"])})
    empty = pd.DataFrame(columns=["code", "date"])
    adv = pd.Series({"000001": 1e8})
    kw = {"suspended": empty, "limit_up": empty, "limit_down": down, "adv": adv}
    assert is_tradable("000001", "2024-01-03", side="buy", **kw)
    assert not is_tradable("000001", "2024-01-03", side="sell", **kw)


def test_is_tradable_min_adv_and_missing_adv() -> None:
    empty = pd.DataFrame(columns=["code", "date"])
    adv = pd.Series({"000001": 1e6})
    kw = {"suspended": empty, "limit_up": empty, "limit_down": empty, "adv": adv, "min_adv": 1e7}
    assert not is_tradable("000001", "2024-01-03", **kw)
    assert not is_tradable("999999", "2024-01-03", **kw)  # ADV 缺失


def test_is_tradable_rejects_bad_side() -> None:
    empty = pd.DataFrame(columns=["code", "date"])
    with pytest.raises(ValueError):
        is_tradable(
            "000001", "2024-01-03", suspended=empty, limit_up=empty,
            limit_down=empty, adv=pd.Series(dtype=float), side="hold",
        )


def test_order_size_cap() -> None:
    assert order_size_cap(1e8, 0.1) == pytest.approx(1e7)
    assert order_size_cap(0.0) == 0.0
    assert order_size_cap(float("nan")) == 0.0
    with pytest.raises(ValueError):
        order_size_cap(1e8, 0.0)


def test_price_limit_ratio_boards() -> None:
    assert price_limit_ratio("600000") == pytest.approx(0.10)
    assert price_limit_ratio("000001") == pytest.approx(0.10)
    assert price_limit_ratio("300750") == pytest.approx(0.20)
    assert price_limit_ratio("688001") == pytest.approx(0.20)
    assert price_limit_ratio("600000", is_st=True) == pytest.approx(0.05)
    # 创业板 ST 仍为 20%
    assert price_limit_ratio("300750", is_st=True) == pytest.approx(0.20)


def test_detect_limit_moves_main_board() -> None:
    bars = pd.DataFrame(
        {
            "code": ["600000"] * 3,
            "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
            "close": [10.0, 11.0, 9.9],   # 次日 +10% 涨停
            "prev_close": [np.nan, 10.0, 11.0],
        }
    )
    out = detect_limit_moves(bars)
    row = out[out["date"] == pd.Timestamp("2024-01-03")].iloc[0]
    assert bool(row["limit_up"]) is True
    assert bool(row["limit_down"]) is False


def test_detect_limit_moves_chinext_20pct_not_flagged_as_limit() -> None:
    """创业板 +10% 不是涨停, 不应被误判."""
    bars = pd.DataFrame(
        {
            "code": ["300750"] * 2,
            "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "close": [100.0, 110.0],
            "prev_close": [np.nan, 100.0],
        }
    )
    out = detect_limit_moves(bars)
    row = out[out["date"] == pd.Timestamp("2024-01-03")].iloc[0]
    assert bool(row["limit_up"]) is False


def test_infer_suspensions_detects_missing_session(cal: TradingCalendar) -> None:
    # 000001 缺 2024-01-04; 000002 连续无缺失
    bars = pd.DataFrame(
        {
            "code": ["000001", "000001", "000002", "000002", "000002"],
            "date": pd.to_datetime(
                ["2024-01-03", "2024-01-05", "2024-01-03", "2024-01-04", "2024-01-05"]
            ),
        }
    )
    susp = infer_suspensions(bars, cal)
    assert set(zip(susp["code"], susp["date"])) == {("000001", pd.Timestamp("2024-01-04"))}


def test_infer_suspensions_ignores_outside_observed_span(cal: TradingCalendar) -> None:
    """区间外不推断: 首末观测日之外不应产生停牌记录."""
    bars = pd.DataFrame(
        {"code": ["000001", "000001"], "date": pd.to_datetime(["2024-01-04", "2024-01-05"])}
    )
    susp = infer_suspensions(bars, cal)
    assert len(susp) == 0


# --------------------------------------------------------------------------
# 复权
# --------------------------------------------------------------------------
def test_apply_adjustment_qfq_and_hfq_on_10for10() -> None:
    prices = pd.DataFrame(
        {
            "code": ["000001"] * 4,
            "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]),
            "close": [10.0, 10.0, 5.0, 5.0],
        }
    )
    actions = pd.DataFrame(
        {
            "code": ["000001"],
            "ex_date": pd.to_datetime(["2024-01-04"]),
            "cash_dividend_per_share": [0.0],
            "share_ratio": [1.0],      # 10 送 10
        }
    )

    qfq = apply_adjustment(prices, actions, method="qfq")
    assert qfq["adjusted"].tolist() == pytest.approx([5.0, 5.0, 5.0, 5.0])

    hfq = apply_adjustment(prices, actions, method="hfq")
    assert hfq["adjusted"].tolist() == pytest.approx([10.0, 10.0, 10.0, 10.0])


def test_apply_adjustment_cash_dividend() -> None:
    prices = pd.DataFrame(
        {
            "code": ["000001"] * 2,
            "date": pd.to_datetime(["2024-01-02", "2024-01-04"]),
            "close": [10.0, 9.5],
        }
    )
    actions = pd.DataFrame(
        {
            "code": ["000001"],
            "ex_date": pd.to_datetime(["2024-01-04"]),
            "cash_dividend_per_share": [0.5],
            "share_ratio": [0.0],
        }
    )
    qfq = apply_adjustment(prices, actions, method="qfq")
    # k = (10 - 0.5) / 10 = 0.95 -> 除权前 10 * 0.95 = 9.5
    assert qfq["adjusted"].tolist() == pytest.approx([9.5, 9.5])


def test_apply_adjustment_no_actions_is_identity() -> None:
    prices = pd.DataFrame(
        {"code": ["000001"], "date": pd.to_datetime(["2024-01-02"]), "close": [10.0]}
    )
    out = apply_adjustment(prices, pd.DataFrame(columns=["code", "ex_date", "cash_dividend_per_share", "share_ratio"]))
    assert out["adjusted"].tolist() == pytest.approx([10.0])
    assert out["adj_factor"].tolist() == pytest.approx([1.0])


def test_apply_adjustment_rejects_bad_method() -> None:
    prices = pd.DataFrame({"code": ["000001"], "date": pd.to_datetime(["2024-01-02"]), "close": [10.0]})
    with pytest.raises(ValueError):
        apply_adjustment(prices, pd.DataFrame(), method="none")


# --------------------------------------------------------------------------
# 财报 / 公告连接
# --------------------------------------------------------------------------
def test_attach_announcement_time_inner_drops_unmatched() -> None:
    fin = pd.DataFrame(
        {
            "code": ["000001", "000002"],
            "report_date": pd.to_datetime(["2024-03-31", "2024-03-31"]),
            "eps": [1.0, 2.0],
        }
    )
    ann = pd.DataFrame(
        {
            "code": ["000001"],
            "report_date": pd.to_datetime(["2024-03-31"]),
            "actual_disclosure_date": pd.to_datetime(["2024-04-10"]),
            "ann_ts": pd.to_datetime(["2024-04-10"]),
            "schedule_changes": [0],
        }
    )
    out = attach_announcement_time(fin, ann, how="inner")
    assert list(out["code"]) == ["000001"]
    assert out["actual_disclosure_date"].iloc[0] == pd.Timestamp("2024-04-10")


def test_attach_announcement_time_left_keeps_unmatched() -> None:
    fin = pd.DataFrame(
        {"code": ["000001", "000002"], "report_date": pd.to_datetime(["2024-03-31"] * 2)}
    )
    ann = pd.DataFrame(
        {
            "code": ["000001"],
            "report_date": pd.to_datetime(["2024-03-31"]),
            "actual_disclosure_date": pd.to_datetime(["2024-04-10"]),
            "ann_ts": pd.to_datetime(["2024-04-10"]),
        }
    )
    out = attach_announcement_time(fin, ann, how="left")
    assert len(out) == 2
    assert out["actual_disclosure_date"].isna().sum() == 1


def test_audit_coverage_numbers() -> None:
    fin = pd.DataFrame({"code": ["000001", "000002", "000003"], "report_date": pd.to_datetime(["2024-03-31"] * 3)})
    ann = pd.DataFrame({"code": ["000001", "000002"], "report_date": pd.to_datetime(["2024-03-31"] * 2)})
    audit = audit_coverage(fin, ann)
    row = audit.iloc[0]
    assert row["financials_codes"] == 3
    assert row["matched_codes"] == 2
    assert row["financials_only"] == 1
    assert row["match_rate"] == pytest.approx(2 / 3)


# --------------------------------------------------------------------------
# 缓存层 (注入 call, 完全离线)
# --------------------------------------------------------------------------
@pytest.fixture()
def tmp_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(src, "RAW_DIR", tmp_path / "raw")
    return tmp_path / "raw"


def test_fetch_writes_cache_and_metadata(tmp_cache) -> None:
    calls = {"n": 0}

    def fake_call(**kwargs):
        calls["n"] += 1
        return pd.DataFrame({"a": [1, 2, 3]})

    df = src.fetch("fake_iface", params={"x": "1"}, call=fake_call)
    assert len(df) == 3
    assert calls["n"] == 1

    parquet, meta = src.cache_paths("fake_iface", {"x": "1"})
    assert parquet.exists() and meta.exists()

    import json
    info = json.loads(meta.read_text())
    assert info["interface"] == "fake_iface"
    assert info["rows"] == 3
    assert info["columns"] == ["a"]
    assert info["attempts"] == 1
    assert len(info["sha256"]) == 64


def test_fetch_reads_cache_without_calling_again(tmp_cache) -> None:
    calls = {"n": 0}

    def fake_call(**kwargs):
        calls["n"] += 1
        return pd.DataFrame({"a": [1]})

    src.fetch("fake_iface", params={"x": "1"}, call=fake_call)
    src.fetch("fake_iface", params={"x": "1"}, call=fake_call)
    assert calls["n"] == 1, "第二次应命中缓存"


def test_fetch_force_refetches(tmp_cache) -> None:
    calls = {"n": 0}

    def fake_call(**kwargs):
        calls["n"] += 1
        return pd.DataFrame({"a": [1]})

    src.fetch("fake_iface", params={"x": "1"}, call=fake_call)
    src.fetch("fake_iface", params={"x": "1"}, call=fake_call, force=True)
    assert calls["n"] == 2


def test_fetch_retries_then_succeeds(tmp_cache) -> None:
    state = {"n": 0}

    def flaky(**kwargs):
        state["n"] += 1
        if state["n"] < 3:
            raise ConnectionError("RemoteDisconnected")
        return pd.DataFrame({"a": [1]})

    df = src.fetch("flaky_iface", call=flaky, retries=4, backoff=0.01)
    assert len(df) == 1
    assert state["n"] == 3


def test_fetch_raises_after_retries_exhausted(tmp_cache) -> None:
    def always_fail(**kwargs):
        raise ConnectionError("boom")

    with pytest.raises(src.DataSourceError):
        src.fetch("bad_iface", call=always_fail, retries=2, backoff=0.01)


def test_cache_key_differs_by_params(tmp_cache) -> None:
    p1, _ = src.cache_paths("iface", {"date": "20240331"})
    p2, _ = src.cache_paths("iface", {"date": "20240630"})
    p3, _ = src.cache_paths("iface", {"date": "20240331"})
    assert p1 != p2
    assert p1 == p3


def test_list_cache_empty_on_fresh_dir(tmp_cache) -> None:
    assert len(src.list_cache()) == 0


# --------------------------------------------------------------------------
# 幸存者偏差审计
# --------------------------------------------------------------------------
def test_audit_survivorship_flags_source_level_bias() -> None:
    """样本内没有退市公司时, 必须显式标注数据源层面的偏差."""
    panel = pd.DataFrame(
        {
            "code": ["000001", "000002"],
            "actual_disclosure_date": pd.to_datetime(["2024-04-10", "2024-04-11"]),
        }
    )
    delistings = pd.DataFrame(
        {"code": ["000003"], "delist_date": pd.to_datetime(["2024-09-02"])}
    )
    out = audit_survivorship(panel, delistings)
    assert out["events_later_delisted"].iloc[0] == 0
    assert "幸存者偏差" in out["survivorship_flag"].iloc[0]


def test_audit_survivorship_detects_overlap() -> None:
    panel = pd.DataFrame(
        {
            "code": ["000003", "000001"],
            "actual_disclosure_date": pd.to_datetime(["2024-04-10", "2024-04-11"]),
        }
    )
    delistings = pd.DataFrame(
        {"code": ["000003"], "delist_date": pd.to_datetime(["2024-09-02"])}
    )
    out = audit_survivorship(panel, delistings)
    assert out["events_later_delisted"].iloc[0] == 1
    assert "包含后来退市公司" in out["survivorship_flag"].iloc[0]


def test_audit_survivorship_handles_empty_panel() -> None:
    out = audit_survivorship(
        pd.DataFrame(columns=["code", "actual_disclosure_date"]),
        pd.DataFrame(columns=["code", "delist_date"]),
    )
    assert out["panel_events"].iloc[0] == 0


# --------------------------------------------------------------------------
# ST 判定 (point-in-time)
# --------------------------------------------------------------------------
def test_is_st_name_basic() -> None:
    assert is_st_name("ST龙大")
    assert is_st_name("*ST宁视")
    assert is_st_name("ST晨鸣（ST晨鸣B）")
    assert is_st_name("*ST深天")
    # 「S」是未股改, 不是 ST
    assert not is_st_name("S深发展A")
    assert not is_st_name("惠程科技")
    assert not is_st_name("")
    assert not is_st_name(None)
    # 前后有空白也要能识别
    assert is_st_name("  ST惠程")


def test_is_st_name_does_not_match_st_inside_word() -> None:
    """只有开头才是风险警示标记."""
    assert not is_st_name("东ST科技")


@pytest.fixture()
def name_changes() -> pd.DataFrame:
    """002168 的真实名字轨迹 (简化)."""
    return pd.DataFrame(
        {
            "code": ["002168"] * 3,
            "change_date": pd.to_datetime(["2000-01-01", "2024-10-01", "2025-05-01"]),
            "name_before": ["惠程科技", "惠程科技", "ST惠程"],
            "name_after": ["惠程科技", "ST惠程", "*ST惠程"],
        }
    )


def test_name_as_of_reconstructs_history(name_changes: pd.DataFrame) -> None:
    tl = build_name_timeline(name_changes)
    assert name_as_of("002168", "2024-04-20", tl) == "惠程科技"
    assert name_as_of("002168", "2024-10-15", tl) == "ST惠程"
    assert name_as_of("002168", "2026-01-01", tl) == "*ST惠程"
    # 变更日当天即生效
    assert name_as_of("002168", "2024-10-01", tl) == "ST惠程"
    # 无记录的股票
    assert name_as_of("999999", "2024-04-20", tl) is None


def test_flag_st_uses_point_in_time_not_current_name() -> None:
    """核心回归: 今天叫 ST 不代表披露日就是 ST."""
    changes = pd.DataFrame(
        {
            "code": ["002168"] * 2,
            "change_date": pd.to_datetime(["2000-01-01", "2024-10-01"]),
            "name_before": ["惠程科技", "惠程科技"],
            "name_after": ["惠程科技", "ST惠程"],
        }
    )
    panel = pd.DataFrame(
        {
            "code": ["002168"],
            "name": ["ST惠程"],                       # 当前名 (今天已 ST)
            "actual_disclosure_date": pd.to_datetime(["2024-04-20"]),
        }
    )
    out = flag_st(panel, changes=changes)
    assert out["name_at_disclosure"].iloc[0] == "惠程科技"
    assert bool(out["is_st"].iloc[0]) is False
    assert out["st_basis"].iloc[0] == BASIS_SZ_HISTORY


def test_flag_st_detects_st_at_disclosure() -> None:
    changes = pd.DataFrame(
        {
            "code": ["000005"],
            "change_date": pd.to_datetime(["2023-05-01"]),
            "name_before": ["世纪星源"],
            "name_after": ["ST星源"],
        }
    )
    panel = pd.DataFrame(
        {
            "code": ["000005"],
            "name": ["ST星源"],
            "actual_disclosure_date": pd.to_datetime(["2024-04-20"]),
        }
    )
    out = flag_st(panel, changes=changes)
    assert bool(out["is_st"].iloc[0]) is True


def test_flag_st_shanghai_falls_back_and_is_labelled() -> None:
    """沪市无历史简称数据, 必须显式标注为近似而不是假装精确."""
    changes = pd.DataFrame(columns=["code", "change_date", "name_before", "name_after"])
    panel = pd.DataFrame(
        {
            "code": ["600000"],
            "name": ["浦发银行"],
            "actual_disclosure_date": pd.to_datetime(["2024-04-20"]),
        }
    )
    out = flag_st(panel, changes=changes)
    assert out["st_basis"].iloc[0] == BASIS_SH_CURRENT
    assert bool(out["is_st"].iloc[0]) is False


def test_drop_st_removes_flagged_and_keeps_unknown() -> None:
    panel = pd.DataFrame(
        {
            "code": ["000001", "000002", "600000"],
            "is_st": [True, False, pd.NA],
        }
    )
    out = drop_st(panel)
    assert list(out["code"]) == ["000002", "600000"]


def test_drop_st_requires_flag() -> None:
    with pytest.raises(ValueError):
        drop_st(pd.DataFrame({"code": ["000001"]}))


def test_audit_st_counts_and_requires_flag() -> None:
    panel = pd.DataFrame(
        {
            "code": ["000001", "000002", "600000"],
            "is_st": [True, False, pd.NA],
            "st_basis": [BASIS_SZ_HISTORY, BASIS_SZ_HISTORY, BASIS_SH_CURRENT],
        }
    )
    row = audit_st(panel).iloc[0]
    assert (row["st_total"], row["st_flagged"]) == (3, 1)
    assert (row["st_basis_sz_history"], row["st_basis_sh_current"]) == (2, 1)
    # 空面板走同一路径, 自然得到全 0
    assert audit_st(panel.iloc[0:0])["st_flagged"].iloc[0] == 0
    with pytest.raises(ValueError):
        audit_st(pd.DataFrame({"code": ["000001"]}))


def test_flag_st_requires_columns() -> None:
    with pytest.raises(ValueError):
        flag_st(pd.DataFrame({"code": ["000001"]}), changes=pd.DataFrame())


def test_build_event_panel_rejects_bad_st_policy() -> None:
    with pytest.raises(ValueError):
        build_event_panel(["20240331"], st_policy="nonsense")


# --------------------------------------------------------------------------
# 行情源: 符号转换 / 熔断 / 降级
# --------------------------------------------------------------------------
def test_to_sina_symbol() -> None:
    assert to_sina_symbol("000001") == "sz000001"
    assert to_sina_symbol("002168") == "sz002168"
    assert to_sina_symbol("300076") == "sz300076"
    assert to_sina_symbol("600000") == "sh600000"
    assert to_sina_symbol("688001") == "sh688001"
    assert to_sina_symbol("830799") == "bj830799"
    # 不足 6 位要补零
    assert to_sina_symbol("1") == "sz000001"


def test_circuit_breaker_trips_at_threshold() -> None:
    b = _CircuitBreaker(threshold=3)
    assert not b.tripped
    b.record_failure()
    b.record_failure()
    assert not b.tripped
    b.record_failure()
    assert b.tripped


def test_circuit_breaker_success_resets_counter() -> None:
    b = _CircuitBreaker(threshold=2)
    b.record_failure()
    b.record_success()
    b.record_failure()
    assert not b.tripped, "成功一次应重置连续失败计数"


def test_load_daily_em_normalises_chinese_columns(monkeypatch) -> None:
    def fake_fetch(interface, **_kw):
        assert interface == "stock_zh_a_hist"
        return pd.DataFrame(
            {
                "日期": ["2024-04-01"],
                "开盘": [1.0], "收盘": [2.0], "最高": [3.0], "最低": [0.5],
                "成交量": [100.0], "成交额": [200.0], "涨跌幅": [1.5], "换手率": [0.5],
            }
        )

    monkeypatch.setattr(prices, "fetch", fake_fetch)
    df = prices.load_daily_em("000001", "20240401", "20240430")
    assert list(df["close"]) == [2.0]
    assert df["source"].iloc[0] == "em"


def test_load_daily_falls_back_to_sina_when_em_fails(monkeypatch) -> None:
    def fake_fetch(interface, **_kw):
        if interface == "stock_zh_a_hist":
            raise DataSourceError("blocked")
        return pd.DataFrame(
            {
                "date": ["2024-04-01"],
                "open": [1.0], "high": [3.0], "low": [0.5], "close": [2.0],
                "volume": [100.0], "amount": [200.0],
                "outstanding_share": [1e9], "turnover": [0.01],
            }
        )

    monkeypatch.setattr(prices, "fetch", fake_fetch)
    breaker = _CircuitBreaker(threshold=2)
    df = prices.load_daily("000001", "20240401", "20240430", breaker=breaker)
    assert df["source"].iloc[0] == "sina"
    assert not breaker.tripped


def test_load_daily_skips_em_once_breaker_tripped(monkeypatch) -> None:
    calls: list[str] = []

    def fake_fetch(interface, **_kw):
        calls.append(interface)
        if interface == "stock_zh_a_hist":
            raise DataSourceError("blocked")
        return pd.DataFrame(
            {
                "date": ["2024-04-01"],
                "open": [1.0], "high": [3.0], "low": [0.5], "close": [2.0],
                "volume": [100.0], "amount": [200.0],
                "outstanding_share": [1e9], "turnover": [0.01],
            }
        )

    monkeypatch.setattr(prices, "fetch", fake_fetch)
    breaker = _CircuitBreaker(threshold=1)
    prices.load_daily("000001", "20240401", "20240430", breaker=breaker)
    assert breaker.tripped

    calls.clear()
    prices.load_daily("000002", "20240401", "20240430", breaker=breaker)
    assert "stock_zh_a_hist" not in calls, "熔断后不应再尝试东财"


def test_load_daily_source_em_raises_without_fallback(monkeypatch) -> None:
    def fake_fetch(interface, **_kw):
        raise DataSourceError("blocked")

    monkeypatch.setattr(prices, "fetch", fake_fetch)
    with pytest.raises(DataSourceError):
        prices.load_daily("000001", "20240401", "20240430", source="em")


def test_load_daily_rejects_bad_source() -> None:
    with pytest.raises(ValueError):
        prices.load_daily("000001", "20240401", "20240430", source="bloomberg")
