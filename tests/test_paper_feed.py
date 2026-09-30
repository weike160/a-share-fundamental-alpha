"""Provider units, dates and fallback behaviour, without external requests."""
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from paper import feed
from paper.engine import Settings, initial_state
from research.config import CONFIG


def test_index_falls_back_to_sina(monkeypatch):
    calls = []
    def fetch(name, **kwargs):
        calls.append(name)
        if name.endswith("_em"):
            raise ConnectionError("outage")
        return pd.DataFrame({"date": ["2026-09-14"], "close": [4480.082]})
    monkeypatch.setattr(feed, "fetch", fetch)
    assert feed.benchmark_bars().iloc[0]["close"] == 4480.082
    assert calls == ["stock_zh_index_daily_em", "stock_zh_index_daily"]


def test_spot_fallback_normalizes_exchange_prefix(monkeypatch):
    def fetch(name, **kwargs):
        if name.endswith("_em"):
            raise ConnectionError("outage")
        return pd.DataFrame({"代码": ["sh600000", "sz000001"], "名称": ["浦发", "平安"], "昨收": [10, 11]})
    monkeypatch.setattr(feed, "fetch", fetch)
    assert feed.spot_quotes()["代码"].tolist() == ["600000", "000001"]


@pytest.mark.parametrize("day,error", [("2026-09-12", feed.MarketClosed), ("2026-09-15", ValueError)])
def test_calendar_rejects_holiday_and_unknown_future(monkeypatch, day, error):
    monkeypatch.setattr(feed, "fetch", lambda *a, **k: pd.DataFrame({"trade_date": ["2026-09-11", "2026-09-14"]}))
    with pytest.raises(error):
        feed.market_calendar(day)


@pytest.fixture
def mock_feed(monkeypatch):
    class Clock:
        @classmethod
        def now(cls, tz):
            return datetime(2026, 9, 14, 18, tzinfo=ZoneInfo("Asia/Shanghai"))
    class Predictor:
        meta = dict(trained_through="2026-09-13", features=list(CONFIG.ml_features))
        model_id = "mock-only"
        def __init__(self, path):
            pass
        def predict(self, frame, day):
            assert not any(c.startswith("fwd_ret") for c in frame)
            return pd.Series(.1, index=frame.index)
    monkeypatch.setattr(feed, "datetime", Clock)
    monkeypatch.setattr(feed, "Predictor", Predictor)
    monkeypatch.setattr(feed, "market_calendar", lambda day: "2026-09-11")
    monkeypatch.setattr(feed, "benchmark_bars", lambda: pd.DataFrame({"date": ["2026-09-14"], "close": [4480.082]}))
    monkeypatch.setattr(feed, "spot_quotes", lambda: pd.DataFrame({"代码": ["600000"], "名称": ["浦发银行"], "昨收": [10.]}))
    monkeypatch.setattr(feed, "current_events", lambda day: pd.DataFrame([dict(code="600000", report_date=pd.Timestamp("2026-06-30"),
        actual_disclosure_date=pd.Timestamp("2026-09-11"), eps=3., net_profit_yoy_calc=.1,
        revenue_yoy_calc=.2, eps_yoy_calc=.3, roe_change=.4, industry="银行")]))
    monkeypatch.setattr(feed, "load_eps_history", lambda *a, **k: pd.DataFrame(dict(code="600000", report_date=pd.date_range("2022-03-31", periods=18, freq="QE"), eps_raw=np.arange(18) ** 2 * .01)))
    return monkeypatch


@pytest.mark.parametrize("source,volume", [("em", 10_000), ("sina", 1_000_000)])
def test_feed_causal_snapshot_and_volume_units(mock_feed, source, volume, tmp_path):
    def daily(*a, **k):
        return pd.DataFrame(dict(code="600000", date=pd.bdate_range(end="2026-09-14", periods=80),
                                open=10., close=10., high=10.1, low=9.9, amount=10_000_000.,
                                turnover=1., volume=volume, source=source))
    mock_feed.setattr(feed, "load_daily", daily)
    snap = feed.build_snapshot("2026-09-14", initial_state(Settings(), "live"), tmp_path)
    assert snap["quotes"]["600000"]["volume_shares"] == 1_000_000
    assert snap["candidates"][0]["available_date"] == "2026-09-14"
    assert snap["candidates"][0]["disclosure_date"] == "2026-09-11"
    assert set(snap["candidates"][0]["features"]) == set(CONFIG.ml_features)


def test_feed_rejects_stale_held_price(mock_feed, tmp_path):
    state = initial_state(Settings(), "live")
    state["positions"] = {"600000": {"qty": 100}}
    mock_feed.setattr(feed, "load_daily", lambda *a, **k: pd.DataFrame({"date": [pd.Timestamp("2026-09-11")]}))
    with pytest.raises(ValueError, match="持仓"):
        feed.build_snapshot("2026-09-14", state, tmp_path)


def test_no_historical_live_signal_backfill(mock_feed, tmp_path):
    with pytest.raises(ValueError, match="禁止补造"):
        feed.build_snapshot("2026-09-11", initial_state(Settings(), "live"), tmp_path)


def test_tick_rounding():
    assert feed.limit_price(10.05, 1.1) == 11.06
    assert feed.main_board("600000") and not feed.main_board("688001")


def test_missing_model_still_settles_cash(mock_feed, tmp_path):
    def missing(*a, **k):
        raise FileNotFoundError("missing model")
    mock_feed.setattr(feed, "Predictor", missing)
    snap = feed.build_snapshot("2026-09-14", initial_state(Settings(), "live"), tmp_path)
    assert snap["selection_status"] == "unavailable"
    assert not snap["candidates"] and snap["model_id"] is None


def test_financial_failure_does_not_block_frozen_orders(mock_feed, tmp_path):
    def unavailable(*a, **k):
        raise ConnectionError("financial feed down")
    mock_feed.setattr(feed, "current_events", unavailable)
    mock_feed.setattr(feed, "load_daily", lambda *a, **k: pd.DataFrame(dict(
        code="600000", date=pd.bdate_range(end="2026-09-14", periods=80),
        open=10., close=10., high=10.1, low=9.9, volume=10000., source="em")))
    state = initial_state(Settings(), "live")
    state["pending"] = [dict(code="600000", side="BUY", created="2026-09-11", qty=100, budget=2000)]
    snap = feed.build_snapshot("2026-09-14", state, tmp_path)
    from paper.engine import advance
    result = advance(state, snap)
    assert len(result["trades"]) == 1
    assert result["selection_status"] == "unavailable"
