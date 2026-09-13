"""把入场 / 出场对齐到真实可交易时点.

事件面板给出的 ``tradable_ts`` 只是「披露日之后的首个交易日」, 并不保证那天
真的买得到。本模块在此基础上处理两件事:

1. **停牌** —— 该交易日没有 K 线 (由 :func:`data.tradability.infer_suspensions`
   在观测区间内推断)。
2. **涨跌停** —— 涨停买不进, 跌停卖不掉 (由
   :func:`data.tradability.detect_limit_moves` 判断)。

处理规则是**顺延**: 从目标交易日起逐日向后找第一个可成交的交易日, 最多顺延
``max_shift`` 个交易日; 超过上限则标记为 ``entry_blocked`` / ``exit_blocked``,
该事件的对应收益为 NaN, 而不是拿一个买不到的价格硬算。

输出的 ``fwd_ret_adj_<h>d`` 使用顺延后的入场价与出场价, 与未调整的
``fwd_ret_<h>d`` 并存, 便于比较两者的差异。
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from data.tradability import TradingCalendar, detect_limit_moves, infer_suspensions

#: 默认最多顺延的交易日数
DEFAULT_MAX_SHIFT = 10


def _to_days(values: pd.Series | pd.DatetimeIndex) -> np.ndarray:
    """转成 ``datetime64[D]`` 的整数天数, 便于 O(1) 比较."""
    return pd.to_datetime(pd.Series(values)).to_numpy(dtype="datetime64[D]").astype("int64")


def build_trade_blocks(
    prices: pd.DataFrame,
    calendar: TradingCalendar,
    *,
    st_codes: set[str] | None = None,
) -> dict[str, dict[str, set[int]]]:
    """构造每只股票的「不可交易日」集合.

    Returns
    -------
    ``{code: {"suspended": set[day_int], "limit_up": ..., "limit_down": ...}}``,
    ``day_int`` 为 ``datetime64[D]`` 的整数表示。
    """
    limits = detect_limit_moves(prices, st_codes=st_codes)
    suspended = infer_suspensions(prices, calendar)

    limits["code"] = limits["code"].astype(str).str.zfill(6)
    suspended["code"] = suspended["code"].astype(str).str.zfill(6)

    blocks: dict[str, dict[str, set[int]]] = {}

    def _slot(code: str) -> dict[str, set[int]]:
        return blocks.setdefault(
            code, {"suspended": set(), "limit_up": set(), "limit_down": set()}
        )

    if len(suspended):
        days = _to_days(suspended["date"])
        for code, day in zip(suspended["code"], days):
            _slot(code)["suspended"].add(int(day))

    days = _to_days(limits["date"])
    for code, up, down, day in zip(
        limits["code"], limits["limit_up"], limits["limit_down"], days
    ):
        if bool(up):
            _slot(code)["limit_up"].add(int(day))
        if bool(down):
            _slot(code)["limit_down"].add(int(day))

    return blocks


def _code_bars(prices: pd.DataFrame, price_col: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """``{code: (升序天数数组, 价格数组)}``, 供 searchsorted 定位."""
    px = prices[["code", "date", price_col]].copy()
    px["code"] = px["code"].astype(str).str.zfill(6)
    px["date"] = pd.to_datetime(px["date"])
    px["_day"] = _to_days(px["date"])
    px[price_col] = pd.to_numeric(px[price_col], errors="coerce")
    px = px.dropna(subset=["_day"]).sort_values(["code", "_day"])

    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for code, grp in px.groupby("code", sort=False):
        out[code] = (
            grp["_day"].to_numpy(dtype="int64"),
            grp[price_col].to_numpy(dtype="float64"),
        )
    return out


def _price_on(
    bars: dict[str, tuple[np.ndarray, np.ndarray]], code: str, day: int
) -> float:
    """取某日收盘价; 该日没有 K 线则 NaN."""
    item = bars.get(code)
    if item is None:
        return float("nan")
    days, values = item
    pos = int(np.searchsorted(days, day))
    if pos < len(days) and days[pos] == day:
        return float(values[pos])
    return float("nan")


def _find_session(
    session_days: np.ndarray,
    start_index: int,
    code: str,
    blocks: dict[str, dict[str, set[int]]],
    bars: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    side: str,
    max_shift: int,
) -> tuple[int, int] | None:
    """从 ``start_index`` 起找第一个可成交交易日, 返回 ``(day, shift)``."""
    block = blocks.get(code)
    suspended = block["suspended"] if block else set()
    blocked = block[f"limit_{side}"] if block else set()
    limit = len(session_days)

    for shift in range(max_shift + 1):
        index = start_index + shift
        if index >= limit:
            return None
        day = int(session_days[index])
        if day in suspended or day in blocked:
            continue
        if not np.isfinite(_price_on(bars, code, day)):
            continue
        return day, shift
    return None


def resolve_fills(
    panel: pd.DataFrame,
    prices: pd.DataFrame,
    calendar: TradingCalendar,
    *,
    horizons: Sequence[int] = (5, 10, 20, 40),
    st_codes: set[str] | None = None,
    max_shift: int = DEFAULT_MAX_SHIFT,
    price_col: str = "close",
    date_col: str = "tradable_ts",
) -> pd.DataFrame:
    """顺延入场/出场并重算未来收益.

    Returns
    -------
    面板副本, 追加::

        entry_date_adj, entry_price_adj, entry_shift_days, entry_blocked
        exit_date_adj_<h>d, exit_blocked_<h>d, fwd_ret_adj_<h>d
    """
    if date_col not in panel.columns:
        raise ValueError(f"panel 缺少列: {date_col}")

    session_days = calendar.days.to_numpy(dtype="datetime64[D]").astype("int64")
    blocks = build_trade_blocks(prices, calendar, st_codes=st_codes)
    bars = _code_bars(prices, price_col)

    out = panel.copy()
    out["code"] = out["code"].astype(str).str.zfill(6)
    start_days = _to_days(pd.to_datetime(out[date_col]))

    entry_days = np.full(len(out), -1, dtype="int64")
    entry_prices = np.full(len(out), np.nan, dtype="float64")
    entry_shift = np.full(len(out), -1, dtype="int64")
    exit_days = {h: np.full(len(out), -1, dtype="int64") for h in horizons}
    exit_blocked = {h: np.zeros(len(out), dtype=bool) for h in horizons}

    for i, (code, day) in enumerate(zip(out["code"], start_days)):
        start_index = int(np.searchsorted(session_days, int(day)))
        entry = _find_session(
            session_days, start_index, code, blocks, bars,
            side="up", max_shift=max_shift,
        )
        if entry is None:
            continue
        entry_day, shift = entry
        entry_days[i] = entry_day
        entry_shift[i] = shift
        entry_prices[i] = _price_on(bars, code, entry_day)
        entry_index = start_index + shift

        for h in horizons:
            target = entry_index + h
            if target >= len(session_days):
                exit_blocked[h][i] = True
                continue
            found = _find_session(
                session_days, target, code, blocks, bars,
                side="down", max_shift=max_shift,
            )
            if found is None:
                exit_blocked[h][i] = True
                continue
            exit_days[h][i] = found[0]

    def _days_to_ts(days: np.ndarray) -> pd.Series:
        """``datetime64[D]`` 整数天 -> Timestamp; 负数表示无成交日."""
        values = pd.Series(days, index=out.index).astype("float64")
        return pd.to_datetime(values.where(values >= 0), unit="D")

    entry_na = entry_days < 0
    out["entry_date_adj"] = _days_to_ts(entry_days)
    out["entry_price_adj"] = entry_prices
    out["entry_shift_days"] = (
        pd.Series(entry_shift, index=out.index).where(~entry_na).astype("Int64")
    )
    out["entry_blocked"] = entry_na

    for h in horizons:
        col = f"exit_date_adj_{h}d"
        out[col] = _days_to_ts(exit_days[h])
        out[f"exit_blocked_{h}d"] = exit_blocked[h]
        exit_px = np.array(
            [
                _price_on(bars, c, int(d)) if d >= 0 else np.nan
                for c, d in zip(out["code"], exit_days[h])
            ]
        )
        out[f"fwd_ret_adj_{h}d"] = exit_px / entry_prices - 1.0

    return out


def audit_fills(
    panel: pd.DataFrame, *, horizons: Sequence[int] = (5, 10, 20, 40)
) -> pd.DataFrame:
    """顺延成交的审计统计, 供数据审计与报告使用."""
    row: dict[str, object] = {"fill_events": len(panel)}
    if len(panel) == 0:
        return pd.DataFrame([row])

    if "entry_blocked" in panel.columns:
        row["entry_blocked"] = int(panel["entry_blocked"].sum())
        row["entry_blocked_rate"] = float(panel["entry_blocked"].mean())
    if "entry_shift_days" in panel.columns:
        shift = pd.to_numeric(panel["entry_shift_days"], errors="coerce")
        row["entry_shift_mean"] = float(shift.mean())
        row["entry_shift_gt0_rate"] = float((shift > 0).mean())

    for h in horizons:
        adj = f"fwd_ret_adj_{h}d"
        raw = f"fwd_ret_{h}d"
        if adj in panel.columns:
            row[f"{adj}_coverage"] = float(
                pd.to_numeric(panel[adj], errors="coerce").notna().mean()
            )
        if adj in panel.columns and raw in panel.columns:
            a = pd.to_numeric(panel[adj], errors="coerce")
            b = pd.to_numeric(panel[raw], errors="coerce")
            both = a.notna() & b.notna()
            row[f"fwd_ret_{h}d_shift_impact"] = (
                float((a[both] - b[both]).mean()) if both.any() else float("nan")
            )
    return pd.DataFrame([row])
