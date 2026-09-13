"""事件面板的规模与流动性特征.

所有特征都以**入场前最后一个交易日**为观测点, 严格不使用入场日及之后的
信息 (见 :func:`attach_price_features` 的 ``allow_exact_matches=False``)。

特征定义
--------
``adv20``            过去 20 个交易日平均成交额 (元)
``volatility_20d``   过去 20 个交易日日收益标准差 (小数)
``momentum_20d``     过去 20 个交易日复权收盘价涨幅
``float_mcap``       流通市值 (元) = 成交额 / 换手率, 与复权口径无关
``log_mcap``         ``log(float_mcap)``
``log_adv20``        ``log(adv20)``
"""
from __future__ import annotations

import numpy as np
import pandas as pd

#: :func:`attach_price_features` 附加的列
FEATURE_COLUMNS: tuple[str, ...] = (
    "adv20",
    "volatility_20d",
    "momentum_20d",
    "float_mcap",
    "log_mcap",
    "log_adv20",
)


def build_price_features(
    prices: pd.DataFrame,
    *,
    adv_window: int = 20,
    mom_window: int = 20,
    min_periods: int = 10,
) -> pd.DataFrame:
    """按 ``(code, date)`` 计算滚动特征.

    Parameters
    ----------
    prices:
        长表, 至少含 ``[code, date, close, amount, turnover]``;
        ``turnover`` 用百分数口径 (东财与新浪在本项目内已统一)。
    min_periods:
        滚动窗口内的最少有效观测数, 不足则为 NaN。

    Returns
    -------
    ``[code, date]`` + ``FEATURE_COLUMNS`` + ``prev_close``。
    """
    need = {"code", "date", "close", "amount", "turnover"}
    missing = need - set(prices.columns)
    if missing:
        raise ValueError(f"prices 缺少必需列: {sorted(missing)}")

    px = prices[["code", "date", "close", "amount", "turnover"]].copy()
    px["code"] = px["code"].astype(str).str.zfill(6)
    px["date"] = pd.to_datetime(px["date"]).dt.normalize()
    px["close"] = pd.to_numeric(px["close"], errors="coerce")
    px["amount"] = pd.to_numeric(px["amount"], errors="coerce")
    px["turnover"] = pd.to_numeric(px["turnover"], errors="coerce")
    px = px.sort_values(["code", "date"]).reset_index(drop=True)

    def _group(frame: pd.DataFrame):
        return frame.groupby("code", sort=False)

    px["ret"] = _group(px)["close"].pct_change()
    px["prev_close"] = _group(px)["close"].shift(1)
    px["adv20"] = _group(px)["amount"].transform(
        lambda s: s.rolling(adv_window, min_periods=min_periods).mean()
    )
    px["volatility_20d"] = _group(px)["ret"].transform(
        lambda s: s.rolling(adv_window, min_periods=min_periods).std()
    )
    px["momentum_20d"] = _group(px)["close"].transform(
        lambda s: s / s.shift(mom_window) - 1.0
    )

    # 成交额 / 换手率 = 流通市值, 与复权方式无关 (分子分母都是未复权量)
    turnover_frac = (px["turnover"] / 100.0).replace(0.0, np.nan)
    px["float_mcap"] = px["amount"] / turnover_frac
    px["log_mcap"] = np.log(px["float_mcap"].where(px["float_mcap"] > 0))
    px["log_adv20"] = np.log(px["adv20"].where(px["adv20"] > 0))

    keep = ["code", "date", "prev_close", *FEATURE_COLUMNS]
    return px[keep].reset_index(drop=True)


def attach_price_features(
    panel: pd.DataFrame,
    price_features: pd.DataFrame,
    *,
    date_col: str = "tradable_ts",
    feature_cols: tuple[str, ...] = FEATURE_COLUMNS,
) -> pd.DataFrame:
    """把入场前最后一个交易日的特征并入事件面板.

    ``date_col`` 为事件入场日; 使用 ``direction="backward"`` 且
    ``allow_exact_matches=False``, 因此取到的一定是**严格早于入场日**的
    观测, 不会引入未来信息。
    """
    required = {"code", "date", *feature_cols}
    missing = required - set(price_features.columns)
    if missing:
        raise ValueError(f"price_features 缺少必需列: {sorted(missing)}")
    if date_col not in panel.columns:
        raise ValueError(f"panel 缺少日期列: {date_col}")

    left = panel.copy()
    left["code"] = left["code"].astype(str).str.zfill(6)
    left["_ts"] = pd.to_datetime(left[date_col]).dt.normalize()
    left["_row"] = np.arange(len(left))
    left = left.sort_values("_ts", kind="mergesort")

    right = price_features[["code", "date", *feature_cols]].copy()
    right["code"] = right["code"].astype(str).str.zfill(6)
    right["date"] = pd.to_datetime(right["date"]).dt.normalize()
    right = right.sort_values("date", kind="mergesort")

    merged = pd.merge_asof(
        left,
        right,
        left_on="_ts",
        right_on="date",
        by="code",
        direction="backward",
        allow_exact_matches=False,
    )
    merged = merged.sort_values("_row", kind="mergesort")
    return merged.drop(columns=["_row", "_ts", "date"]).reset_index(drop=True)
