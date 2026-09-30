"""After-close AKShare adapter. Immutable snapshots freeze first observation.

Only standard Shanghai/Shenzhen main-board stocks are supported in v1. Missing
held-stock bars, stale index dates and corporate action discrepancies stop a day.
"""
from __future__ import annotations

from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from data.announcement import load_announcements
from data.eps_history import load_eps_history
from data.financials import attach_announcement_time, load_financials
from data.liquidity import build_price_features
from data.prices import _CircuitBreaker, load_daily
from data.source import fetch
from factor.neutralize import neutralize_cross_section
from factor.preprocess import winsorize
from paper.model import Predictor
from signals.sue import seasonal_random_walk_sue


class MarketClosed(RuntimeError):
    pass


def benchmark_bars():
    try:
        return fetch("stock_zh_index_daily_em", params={"symbol": "sh000300"}, force=True, retries=1)
    except Exception:
        return fetch("stock_zh_index_daily", params={"symbol": "sh000300"}, force=True, retries=2)


def spot_quotes():
    try:
        frame = fetch("stock_zh_a_spot_em", force=True, retries=1)
    except Exception:
        frame = fetch("stock_zh_a_spot", force=True, retries=2)
    frame = frame.copy()
    frame["代码"] = frame["代码"].astype(str).str.replace(r"^(sh|sz|bj)", "", regex=True).str.zfill(6)
    return frame


def market_calendar(day):
    raw = fetch("tool_trade_date_hist_sina", force=True, retries=2)
    dates = pd.to_datetime(raw["trade_date"]).dt.strftime("%Y-%m-%d").sort_values().tolist()
    if not dates or day > dates[-1]:
        raise ValueError("交易日历未覆盖当前日期")
    if day not in dates:
        raise MarketClosed(f"{day} 非交易日")
    earlier = [d for d in dates if d < day]
    if not earlier:
        raise ValueError("缺少上一交易日")
    return earlier[-1]


def main_board(code):
    return str(code).startswith(("600", "601", "603", "605", "000", "001", "002", "003"))


def limit_price(previous, factor):
    return float((Decimal(str(previous)) * Decimal(str(factor))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def neutralize_available(frame):
    """Apply the research transform only to events actually available today."""
    frame = frame.copy()
    frame["sue_neutral"] = np.nan
    for _, group in frame.groupby("report_date"):
        sue = pd.to_numeric(group["sue"], errors="coerce")
        if sue.notna().sum() < 20:
            continue
        exposures = group[["log_mcap"]].copy()
        if "industry" in group:
            dummies = pd.get_dummies(group["industry"].astype("string").fillna("未知"), prefix="ind", drop_first=True).astype(float)
            exposures = pd.concat([exposures, dummies], axis=1)
        frame.loc[group.index, "sue_neutral"] = neutralize_cross_section(winsorize(sue, .01, .99), exposures)
    return frame


def current_events(day):
    """Fetch the last two completed report periods and exact prior-year quarters."""
    now = pd.Timestamp(day)
    periods = pd.date_range(end=now, periods=3, freq="QE")[-2:]
    frames = []
    for period in periods:
        key = period.strftime("%Y%m%d")
        current = attach_announcement_time(load_financials(key, force=True), load_announcements(key, force=True))
        current = current.loc[(current["actual_disclosure_date"] <= now) &
                              (current["actual_disclosure_date"] >= now - pd.Timedelta(days=35))].copy()
        if current.empty:
            continue
        old_key = (period - pd.DateOffset(years=1)).strftime("%Y%m%d")
        old = attach_announcement_time(load_financials(old_key), load_announcements(old_key))
        old = old.loc[old["actual_disclosure_date"] <= now]
        cols = ["code", "eps", "revenue", "net_profit", "roe"]
        current = current.merge(old[cols], on="code", how="left", suffixes=("", "_prior"))
        for raw, feature in (("eps", "eps_yoy_calc"), ("revenue", "revenue_yoy_calc"), ("net_profit", "net_profit_yoy_calc")):
            current[feature] = current[raw] / current[f"{raw}_prior"].replace(0, np.nan) - 1
        current["roe_change"] = current["roe"] - current["roe_prior"]
        frames.append(current)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _build_selected_snapshot(day: str, state: dict, model_dir: Path):
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    if day != now.date().isoformat() or now.hour < 16:
        raise ValueError("在线采集只允许北京时间当天 16:00 后，禁止补造历史信号")
    previous = market_calendar(day)
    model = Predictor(model_dir)
    if model.meta["trained_through"] >= day:
        raise ValueError("模型训练日期必须早于交易日")
    if state["last_date"] and previous != state["last_date"]:
        raise ValueError("账户存在遗漏交易日，需要用当时保存的快照核对，不能用今天数据补造信号")
    index = benchmark_bars()
    index["date"] = pd.to_datetime(index["date"]).dt.strftime("%Y-%m-%d")
    row = index.loc[index["date"] == day]
    if len(row) != 1:
        raise ValueError("沪深300行情未更新到今天")
    # Spot names identify current ST/delisting flags; actual prices come from dated bars.
    spot = spot_quotes()
    names = dict(zip(spot["代码"].astype(str).str.zfill(6), spot["名称"].astype(str)))
    reference_close = dict(zip(spot["代码"].astype(str).str.zfill(6), pd.to_numeric(spot["昨收"], errors="coerce")))
    events = current_events(day)
    if len(events):
        # Forward test starts now: do not open thousands of stale earnings events
        # from the preceding season on the first run. Include the previous session
        # to capture announcements released after that evening's collection.
        since = pd.Timestamp(state.get("selection_since") or state["last_date"] or previous)
        events = events.loc[events["actual_disclosure_date"] >= since].copy()
        events = events.loc[events["code"].map(main_board)].copy()
        events = events.loc[events["code"].map(lambda c: c in names and not any(t in names[c].upper() for t in ("ST", "退", "N", "C")))].copy()
        events["event_id"] = events["code"] + ":" + events["report_date"].dt.strftime("%Y-%m-%d")
        events = events.loc[~events["event_id"].isin(state["seen"])].copy()
    event_codes = set(events["code"]) if len(events) else set()
    required_codes = set(state["positions"]) | {o["code"] for o in state["pending"]}
    quotes, feature_rows, failures, exclusions = {}, [], [], []
    start = (pd.Timestamp(day) - pd.Timedelta(days=160)).strftime("%Y%m%d")
    end = day.replace("-", "")
    breaker = _CircuitBreaker(threshold=1)
    for number, code in enumerate(sorted(event_codes | required_codes), 1):
        if number % 25 == 0:
            print(f"已采集 {number} / {len(event_codes | required_codes)} 只股票", flush=True)
        try:
            raw = load_daily(code, start, end, adjust="", force=True, breaker=breaker)
            raw = raw.loc[raw["date"] <= pd.Timestamp(day)].copy()
            today = raw.loc[raw["date"] == pd.Timestamp(day)]
            if len(today) != 1:
                raise ValueError("无当日日线，无法区分停牌与数据故障")
            bar = today.iloc[0]
            past = raw.loc[raw["date"] < pd.Timestamp(day)]
            if len(past) == 0:
                raise ValueError("无前收盘价")
            prior = float(past.iloc[-1]["close"])
            ref = reference_close.get(code)
            if not np.isfinite(ref) or ref <= 0:
                raise ValueError("缺少交易所参考前收盘价")
            name = names.get(code)
            if not name:
                raise ValueError("缺少当前证券状态")
            corporate_action = abs(prior - ref) > 0.011
            if corporate_action and code not in state["positions"]:
                exclusions.append(f"{code}: 除权或价格口径异常，跳过新订单")
                continue
            eligible = main_board(code) and not any(t in name.upper() for t in ("ST", "退", "N", "C")) and len(raw) >= 60
            quotes[code] = dict(date=day, open=float(bar["open"]), close=float(bar["close"]),
                                high=float(bar["high"]), low=float(bar["low"]),
                                volume_shares=float(bar["volume"]) * (100 if bar["source"] == "em" else 1),
                                limit_up=limit_price(ref, 1.1), limit_down=limit_price(ref, 0.9),
                                corporate_action=corporate_action, eligible=eligible,
                                suspended=not (float(bar["volume"]) > 0))
            if code not in event_codes or not eligible:
                continue
            adjusted = load_daily(code, start, end, adjust="hfq", force=True, breaker=breaker)
            adjusted = adjusted.loc[adjusted["date"] <= pd.Timestamp(day)]
            if adjusted.empty or adjusted.iloc[-1]["date"] != pd.Timestamp(day):
                raise ValueError("复权行情滞后")
            feature = build_price_features(adjusted).iloc[-1].to_dict()
            eps = load_eps_history(code, force=True)
            # Retain only quarters whose report period does not exceed this event.
            for _, event in events.loc[events["code"] == code].iterrows():
                history = eps.loc[eps["report_date"] <= event["report_date"]].copy()
                history = history.loc[history["report_date"] < event["report_date"]]
                history = pd.concat([history, pd.DataFrame([dict(code=code, report_date=event["report_date"], eps_raw=event["eps"])])], ignore_index=True)
                history = history.sort_values("report_date").drop_duplicates(["code", "report_date"])
                history["sue"] = seasonal_random_walk_sue(history, eps_col="eps_raw")
                feature_rows.append({**event.to_dict(), **feature, "sue": float(history.iloc[-1]["sue"]), "name": name})
        except Exception as exc:
            if code in state["positions"]:
                raise ValueError(f"持仓 {code} 无法核价：{exc}") from exc
            failures.append(f"{code}: {type(exc).__name__}: {exc}")
    if failures:
        # Fail closed rather than alter the ranked universe due to provider outages.
        raise ValueError(f"候选/订单数据不完整（{len(failures)} 只），本日暂不提交：" + "; ".join(failures[:5]))
    candidates = []
    if feature_rows:
        frame = neutralize_available(pd.DataFrame(feature_rows))
        scores = model.predict(frame, day)
        for i, score in scores.items():
            event = frame.loc[i]
            candidates.append(dict(code=event["code"], name=event["name"], score=float(score),
                                   event_id=event["event_id"], available_date=day, feature_date=day,
                                   disclosure_date=event["actual_disclosure_date"].date().isoformat(),
                                   features={key: float(event[key]) if pd.notna(event[key]) and np.isfinite(event[key]) else None
                                             for key in model.meta["features"]}))
    return dict(mode="live", status="ready", date=day, market_date=day, previous_session=previous,
                fetched_at=now.isoformat(), benchmark_close=float(row.iloc[0]["close"]),
                model_id=model.model_id, model_metadata=model.meta, quotes=quotes,
                candidates=candidates, exclusions=exclusions, selection_status="ready",
                warnings=[model.meta.get("limitations", "")])


def settlement_snapshot(day, state, reason):
    """Selection failures must not prevent execution of already frozen orders."""
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    if day != now.date().isoformat() or now.hour < 16:
        raise ValueError("在线采集只允许北京时间当天 16:00 后，禁止补造历史信号")
    previous = market_calendar(day)
    index = benchmark_bars()
    row = index.loc[pd.to_datetime(index["date"]).dt.strftime("%Y-%m-%d") == day]
    if len(row) != 1:
        raise ValueError("沪深300行情未更新到今天")
    codes = set(state["positions"]) | {o["code"] for o in state["pending"]}
    quotes = {}
    if codes:
        spot = spot_quotes().set_index("代码")
        breaker = _CircuitBreaker(threshold=1)
        start = (pd.Timestamp(day) - pd.Timedelta(days=160)).strftime("%Y%m%d")
        for code in sorted(codes):
            raw = load_daily(code, start, day.replace("-", ""), adjust="", force=True, breaker=breaker)
            raw = raw.loc[raw["date"] <= pd.Timestamp(day)]
            today = raw.loc[raw["date"] == pd.Timestamp(day)]
            past = raw.loc[raw["date"] < pd.Timestamp(day)]
            if len(today) != 1 or past.empty or code not in spot.index:
                raise ValueError(f"持仓/待执行订单 {code} 缺少可靠行情，日结暂停")
            bar = today.iloc[0]
            ref = float(spot.loc[code, "昨收"])
            if not np.isfinite(ref) or ref <= 0:
                raise ValueError(f"{code} 缺少参考前收盘价")
            name = str(spot.loc[code, "名称"])
            quotes[code] = dict(date=day, open=float(bar["open"]), close=float(bar["close"]),
                high=float(bar["high"]), low=float(bar["low"]),
                volume_shares=float(bar["volume"]) * (100 if bar["source"] == "em" else 1),
                limit_up=limit_price(ref, 1.1), limit_down=limit_price(ref, .9),
                corporate_action=abs(float(past.iloc[-1]["close"]) - ref) > .011,
                eligible=main_board(code) and len(raw) >= 60 and not any(t in name.upper() for t in ("ST", "退", "N", "C")),
                suspended=not float(bar["volume"]) > 0)
    return dict(mode="live", status="ready", date=day, market_date=day,
        previous_session=previous, fetched_at=now.isoformat(),
        benchmark_close=float(row.iloc[0]["close"]), model_id=state["model_id"],
        quotes=quotes, candidates=[], selection_status="unavailable",
        warnings=[f"今日选股暂停，已有订单与持仓按行情正常结算：{reason}"])


def build_snapshot(day: str, state: dict, model_dir: Path):
    try:
        snap = _build_selected_snapshot(day, state, model_dir)
        if state["model_id"] and state["model_id"] != snap["model_id"]:
            raise ValueError("模型版本变化，暂停新买入；原账户继续结算")
        return snap
    except MarketClosed:
        raise
    except Exception as exc:
        return settlement_snapshot(day, state, f"{type(exc).__name__}: {exc}")
