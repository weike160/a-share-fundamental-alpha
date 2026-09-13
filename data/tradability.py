"""可交易性处理: 停牌 / 涨跌停 / T+1 / 新股期 / ADV.

用于判断某信号在何时、以何种规模真正可以执行。

数据来源说明 (实测结论)
------------------------
AKShare 中**没有**可靠的历史停牌数据:

* ``stock_tfp_em(date=...)`` 的 ``date`` 参数不可信 —— 传 ``"20240301"`` 会
  返回 ``停牌时间`` 高达 ``2026-09-15`` 的记录。
* ``stock_zh_a_stop_em()`` 不接受任何参数, 只反映**当前**停牌状态。

因此历史停牌状态由 :func:`infer_suspensions` 从行情缺失推断: 某交易日该股票
没有 K 线, 即视为当日不可交易。这是一个近似, 会把数据缺失一并算作停牌,
使用时应结合数据审计表判断其影响。
"""
from __future__ import annotations

import pandas as pd

from data import source

#: 各板块涨跌幅限制 (代码前缀 -> 比例)
_LIMIT_BY_PREFIX: tuple[tuple[tuple[str, ...], float], ...] = (
    (("688", "689"), 0.20),          # 科创板
    (("300", "301"), 0.20),          # 创业板
    (("8", "4"), 0.30),              # 北交所
)
_DEFAULT_LIMIT = 0.10                # 沪深主板
_ST_LIMIT = 0.05                     # 主板 ST / *ST


class TradingCalendar:
    """A 股交易日历, 提供可交易时点判定."""

    def __init__(self, trading_days: pd.DatetimeIndex) -> None:
        days = pd.DatetimeIndex(pd.to_datetime(trading_days)).normalize()
        self.days = pd.DatetimeIndex(sorted(set(days)))

    # ------------------------------------------------------------- 构造
    @classmethod
    def from_akshare(cls, force: bool = False) -> TradingCalendar:
        """从 AKShare 拉取交易日历 (带缓存)."""
        df = source.fetch("tool_trade_date_hist_sina", force=force)
        col = "trade_date" if "trade_date" in df.columns else df.columns[0]
        return cls(pd.DatetimeIndex(pd.to_datetime(df[col])))

    # ------------------------------------------------------------- 查询
    def __len__(self) -> int:
        return len(self.days)

    def __contains__(self, ts: object) -> bool:
        return self.is_trading_day(ts)

    def __repr__(self) -> str:
        if len(self.days) == 0:
            return "TradingCalendar(empty)"
        return (
            f"TradingCalendar(n={len(self.days)}, "
            f"{self.days[0].date()} .. {self.days[-1].date()})"
        )

    def is_trading_day(self, ts: object) -> bool:
        """判断是否为交易日."""
        if ts is None or pd.isna(ts):
            return False
        return pd.Timestamp(ts).normalize() in self.days

    def next_trading_day(
        self, ts: object, *, inclusive: bool = False
    ) -> pd.Timestamp | None:
        """返回 ``ts`` 之后 (或当天) 的第一个交易日, 无则 ``None``."""
        if ts is None or pd.isna(ts):
            return None
        ts = pd.Timestamp(ts).normalize()
        idx = self.days.searchsorted(ts, side="left" if inclusive else "right")
        return self.days[idx] if idx < len(self.days) else None

    def prev_trading_day(
        self, ts: object, *, inclusive: bool = False
    ) -> pd.Timestamp | None:
        """返回 ``ts`` 之前 (或当天) 的最后一个交易日, 无则 ``None``."""
        if ts is None or pd.isna(ts):
            return None
        ts = pd.Timestamp(ts).normalize()
        idx = self.days.searchsorted(ts, side="right" if inclusive else "left") - 1
        return self.days[idx] if idx >= 0 else None

    def shift(self, ts: object, n: int) -> pd.Timestamp | None:
        """按交易日平移 ``n`` 个 session (``n`` 可为负)."""
        if ts is None or pd.isna(ts):
            return None
        ts = pd.Timestamp(ts).normalize()
        if n >= 0:
            idx = self.days.searchsorted(ts, side="left") + n
        else:
            idx = self.days.searchsorted(ts, side="right") - 1 + n
        return self.days[idx] if 0 <= idx < len(self.days) else None

    def sessions_between(
        self, start: object, end: object, *, inclusive: str = "both"
    ) -> pd.DatetimeIndex:
        """返回 ``[start, end]`` 之间的交易日."""
        if start is None or end is None or pd.isna(start) or pd.isna(end):
            return pd.DatetimeIndex([])
        start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
        if start > end:
            return pd.DatetimeIndex([])
        left = self.days.searchsorted(
            start, side="left" if inclusive in {"both", "left"} else "right"
        )
        right = self.days.searchsorted(
            end, side="right" if inclusive in {"both", "right"} else "left"
        )
        return self.days[left:right]


def price_limit_ratio(code: str, *, is_st: bool = False) -> float:
    """返回该股票当日涨跌幅限制比例.

    按证券代码前缀判断板块。``is_st`` 为主板 ST/*ST 标记 (限制 5%)。
    创业板/科创板 ST 仍为 20%, 因此 ST 收紧只在主板生效。
    """
    code = str(code).zfill(6)
    for prefixes, ratio in _LIMIT_BY_PREFIX:
        if code.startswith(prefixes):
            return ratio
    return _ST_LIMIT if is_st else _DEFAULT_LIMIT


def detect_limit_moves(
    bars: pd.DataFrame,
    *,
    st_codes: set[str] | None = None,
    price_col: str = "close",
    tol: float = 5e-3,
) -> pd.DataFrame:
    """从日频行情识别涨跌停.

    Parameters
    ----------
    bars:
        长表, 至少包含 ``[code, date, close]``; 若已有 ``prev_close`` 则复用,
        否则按 ``code`` 分组用前一日收盘价推算。
    st_codes:
        主板 ST/*ST 代码集合, 用于套用 5% 限制。
    tol:
        与理论涨跌停价的相对容差 (交易所价格按 0.01 取整)。

    Returns
    -------
    DataFrame[code, date, limit_up, limit_down]
    """
    need = {"code", "date", price_col}
    missing = need - set(bars.columns)
    if missing:
        raise ValueError(f"bars 缺少必需列: {sorted(missing)}")

    df = bars.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["code"] = df["code"].astype(str).str.zfill(6)
    df = df.sort_values(["code", "date"]).reset_index(drop=True)

    if "prev_close" in df.columns:
        prev = pd.to_numeric(df["prev_close"], errors="coerce")
    else:
        prev = df.groupby("code")[price_col].shift(1)

    close = pd.to_numeric(df[price_col], errors="coerce")
    st_codes = {str(c).zfill(6) for c in (st_codes or set())}

    ratio = pd.Series(
        [
            price_limit_ratio(c, is_st=c in st_codes)
            for c in df["code"]
        ],
        index=df.index,
        dtype="float64",
    )

    up_price = (prev * (1 + ratio)).round(2)
    down_price = (prev * (1 - ratio)).round(2)

    limit_up = prev.notna() & close.notna() & (close >= up_price * (1 - tol))
    limit_down = prev.notna() & close.notna() & (close <= down_price * (1 + tol))

    return pd.DataFrame(
        {
            "code": df["code"].values,
            "date": df["date"].values,
            "limit_up": limit_up.fillna(False).astype(bool).values,
            "limit_down": limit_down.fillna(False).astype(bool).values,
        }
    )


def infer_suspensions(
    bars: pd.DataFrame,
    calendar: TradingCalendar,
    *,
    code_col: str = "code",
    date_col: str = "date",
) -> pd.DataFrame:
    """从行情缺失推断停牌.

    对每只股票, 取其**首个到最后一个观测交易日**之间的全部交易日, 其中没有
    K 线的日期即为停牌 (或数据缺失)。区间之外不推断, 避免把上市前 / 退市后
    误判成停牌。

    Returns
    -------
    DataFrame[code, date] —— 每行代表一个 (股票, 停牌日)。
    """
    if bars is None or len(bars) == 0:
        return pd.DataFrame(columns=["code", "date"])

    df = bars[[code_col, date_col]].copy()
    df[code_col] = df[code_col].astype(str).str.zfill(6)
    df[date_col] = pd.to_datetime(df[date_col]).dt.normalize()

    records: list[dict[str, object]] = []
    for code, grp in df.groupby(code_col, sort=False):
        observed = pd.DatetimeIndex(sorted(set(grp[date_col])))
        if len(observed) < 2:
            continue
        span = calendar.sessions_between(observed[0], observed[-1], inclusive="both")
        if len(span) == 0:
            continue
        missing = span.difference(observed)
        records.extend({"code": code, "date": d} for d in missing)

    if not records:
        return pd.DataFrame(columns=["code", "date"])
    out = pd.DataFrame(records)
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    return out.sort_values(["code", "date"]).reset_index(drop=True)


def _has(block: pd.DataFrame | None, code: str, date: pd.Timestamp) -> bool:
    """判断 (code, date) 是否出现在某个 blocks 表中."""
    if block is None or len(block) == 0:
        return False
    if not {"code", "date"} <= set(block.columns):
        return False
    c = block["code"].astype(str).str.zfill(6)
    d = pd.to_datetime(block["date"]).dt.normalize()
    return bool(((c == code) & (d == date)).any())


def is_tradable(
    code: str,
    date: pd.Timestamp,
    *,
    suspended: pd.DataFrame,
    limit_up: pd.DataFrame,
    limit_down: pd.DataFrame,
    adv: pd.Series,
    side: str = "buy",
    min_adv: float = 0.0,
) -> bool:
    """判定某股票在某时点是否可交易 (未停牌、未封板).

    Parameters
    ----------
    code, date:
        待判定的标的与时点。
    suspended, limit_up, limit_down:
        长表 ``[code, date]``, 分别列出停牌、涨停、跌停的记录。
    adv:
        以 ``code`` 为索引的日均成交额 (或成交量)。
    side:
        ``"buy"`` 时涨停不可买; ``"sell"`` 时跌停不可卖。停牌对两者都封锁。
    min_adv:
        ADV 下限, 低于此值视为流动性不足, 不可交易。
    """
    code = str(code).zfill(6)
    date = pd.Timestamp(date).normalize()

    if side not in {"buy", "sell"}:
        raise ValueError(f"side 只能是 'buy' 或 'sell', 收到 {side!r}")

    if _has(suspended, code, date):
        return False
    if side == "buy" and _has(limit_up, code, date):
        return False
    if side == "sell" and _has(limit_down, code, date):
        return False

    if adv is not None and len(adv) > 0:
        value = adv.get(code)
        if value is None or pd.isna(value) or float(value) < min_adv:
            return False

    return True


def order_size_cap(adv: float, max_participation: float = 0.1) -> float:
    """以 ADV 估算最大可下单规模, 用于容量分析."""
    if adv is None or pd.isna(adv) or adv < 0:
        return 0.0
    if not 0 < max_participation <= 1:
        raise ValueError(f"max_participation 应在 (0, 1], 收到 {max_participation}")
    return float(adv) * float(max_participation)
