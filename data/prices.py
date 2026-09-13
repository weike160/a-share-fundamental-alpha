"""行情数据下载与未来收益附加.

行情是**逐只**下载的 (AKShare 没有全市场历史日线的批量接口), 所以这里用
线程池并发 + 逐只缓存。缓存由 :mod:`data.source` 负责, 因此:

* 中断后重跑会自动跳过已下载的股票 —— **可断点续传**。
* 单只失败不会中断整体, 失败清单会被收集起来写进审计。

两个行情源
----------
===========  ====================================  ==========  ==========
来源          接口                                   单只耗时     备注
===========  ====================================  ==========  ==========
东财 (em)     ``stock_zh_a_hist``                    ~0.3s      默认, 字段最全
新浪 (sina)   ``stock_zh_a_daily``                   ~0.9s      备用
===========  ====================================  ==========  ==========

**实测教训**: 用 6 线程并发拉东财, 30 只之后整个 ``eastmoney.com`` 域名被
IP 限流, 连裸 HTTP 请求都立即断连 (``RemoteDisconnected``), 且等待数分钟
未恢复; 同一时刻新浪源正常。因此本模块内置**熔断**: 东财连续失败达到阈值
就自动整体切到新浪, 而不是每只股票都白白重试 4 次。

复权口径
--------
默认 **后复权 (hfq)**。算收益必须用复权价, 否则分红送股会被误当成暴跌。
前复权 (qfq) 的历史值会随最新价格变化, 不利于复现。
"""
from __future__ import annotations

import sys
import threading
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from data.panel import DEFAULT_HORIZONS
from data.source import DataSourceError, fetch

#: 并发线程数。两个源都会限流, 不宜过高。
DEFAULT_WORKERS = 4

#: 东财连续失败多少次后熔断切换到新浪
EM_FAILURE_THRESHOLD = 3

VALID_SOURCES = ("auto", "em", "sina")


# --------------------------------------------------------------------------
# 符号与单只下载
# --------------------------------------------------------------------------
def to_sina_symbol(code: str) -> str:
    """把 6 位代码转成新浪要求的 ``sz000001`` / ``sh600000`` 形式."""
    code = str(code).zfill(6)
    if code.startswith(("6", "9")):
        return f"sh{code}"
    if code.startswith(("4", "8")):
        return f"bj{code}"
    return f"sz{code}"


def load_daily_em(
    code: str, start_date: str, end_date: str, *, adjust: str = "hfq", force: bool = False
) -> pd.DataFrame:
    """东财源日线 (字段最全, 但会被限流)."""
    raw = fetch(
        "stock_zh_a_hist",
        params={
            "symbol": str(code).zfill(6),
            "period": "daily",
            "start_date": str(start_date),
            "end_date": str(end_date),
            "adjust": adjust,
        },
        force=force,
    )
    out = pd.DataFrame(
        {
            "code": str(code).zfill(6),
            "date": pd.to_datetime(raw["日期"], errors="coerce").dt.normalize(),
            "open": pd.to_numeric(raw.get("开盘"), errors="coerce"),
            "close": pd.to_numeric(raw.get("收盘"), errors="coerce"),
            "high": pd.to_numeric(raw.get("最高"), errors="coerce"),
            "low": pd.to_numeric(raw.get("最低"), errors="coerce"),
            "volume": pd.to_numeric(raw.get("成交量"), errors="coerce"),
            "amount": pd.to_numeric(raw.get("成交额"), errors="coerce"),
            "pct_change": pd.to_numeric(raw.get("涨跌幅"), errors="coerce"),
            "turnover": pd.to_numeric(raw.get("换手率"), errors="coerce"),
            "source": "em",
        }
    )
    return out.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)


def load_daily_sina(
    code: str, start_date: str, end_date: str, *, adjust: str = "hfq", force: bool = False
) -> pd.DataFrame:
    """新浪源日线 (备用源, 实测在东财被限流时仍可用)."""
    raw = fetch(
        "stock_zh_a_daily",
        params={
            "symbol": to_sina_symbol(code),
            "start_date": str(start_date),
            "end_date": str(end_date),
            "adjust": adjust,
        },
        force=force,
    )
    turnover = pd.to_numeric(raw.get("turnover"), errors="coerce")
    out = pd.DataFrame(
        {
            "code": str(code).zfill(6),
            "date": pd.to_datetime(raw["date"], errors="coerce").dt.normalize(),
            "open": pd.to_numeric(raw.get("open"), errors="coerce"),
            "close": pd.to_numeric(raw.get("close"), errors="coerce"),
            "high": pd.to_numeric(raw.get("high"), errors="coerce"),
            "low": pd.to_numeric(raw.get("low"), errors="coerce"),
            "volume": pd.to_numeric(raw.get("volume"), errors="coerce"),
            "amount": pd.to_numeric(raw.get("amount"), errors="coerce"),
            # 新浪给的是小数比例, 换算成百分数以便与东财口径一致
            "pct_change": pd.to_numeric(raw.get("close"), errors="coerce").pct_change() * 100,
            "turnover": turnover * 100,
            "source": "sina",
        }
    )
    return out.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)


class _CircuitBreaker:
    """东财连续失败到阈值就整体切走, 避免每只股票都空转重试."""

    def __init__(self, threshold: int = EM_FAILURE_THRESHOLD) -> None:
        self.threshold = threshold
        self._consecutive = 0
        self._tripped = False
        self._lock = threading.Lock()

    @property
    def tripped(self) -> bool:
        with self._lock:
            return self._tripped

    def record_success(self) -> None:
        with self._lock:
            self._consecutive = 0

    def record_failure(self) -> None:
        with self._lock:
            self._consecutive += 1
            if self._consecutive >= self.threshold:
                self._tripped = True


def load_daily(
    code: str,
    start_date: str,
    end_date: str,
    *,
    adjust: str = "hfq",
    source: str = "auto",
    breaker: _CircuitBreaker | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """带熔断降级的单只日线下载.

    ``source="auto"`` 时优先东财, 失败 (或熔断后) 自动改用新浪。
    """
    if source not in VALID_SOURCES:
        raise ValueError(f"source 只能是 {VALID_SOURCES}, 收到 {source!r}")

    if source == "sina" or (breaker is not None and breaker.tripped):
        return load_daily_sina(code, start_date, end_date, adjust=adjust, force=force)

    try:
        df = load_daily_em(code, start_date, end_date, adjust=adjust, force=force)
        if breaker is not None:
            breaker.record_success()
        return df
    except DataSourceError:
        if source == "em":
            raise
        if breaker is not None:
            breaker.record_failure()
        return load_daily_sina(code, start_date, end_date, adjust=adjust, force=force)


# --------------------------------------------------------------------------
# 批量下载
# --------------------------------------------------------------------------
def download_price_panel(
    codes: Iterable[str],
    start_date: str,
    end_date: str,
    *,
    adjust: str = "hfq",
    source: str = "auto",
    max_workers: int = DEFAULT_WORKERS,
    force: bool = False,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """并发下载多只股票的日频行情.

    Returns
    -------
    ``(prices, failures)``

    ``prices``   长表 ``[code, date, open, high, low, close, volume, amount, pct_change, turnover, source]``
    ``failures`` ``[code, error]``
    """
    code_list = sorted({str(c).zfill(6) for c in codes})
    total = len(code_list)
    if total == 0:
        return pd.DataFrame(), pd.DataFrame(columns=["code", "error"])

    breaker = _CircuitBreaker()
    frames: list[pd.DataFrame] = []
    failures: list[dict[str, str]] = []

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                load_daily, code, start_date, end_date,
                adjust=adjust, source=source, breaker=breaker, force=force,
            ): code
            for code in code_list
        }
        for done, fut in enumerate(as_completed(futures), start=1):
            code = futures[fut]
            try:
                df = fut.result()
                if df is None or len(df) == 0:
                    failures.append({"code": code, "error": "空数据"})
                else:
                    frames.append(df)
            except Exception as exc:  # noqa: BLE001 - 单只失败不应中断整体
                failures.append({"code": code, "error": f"{type(exc).__name__}: {exc}"})

            if verbose and (done % 250 == 0 or done == total):
                used = sorted({f["source"].iloc[0] for f in frames})
                print(
                    f"    行情下载 {done}/{total} (成功 {len(frames)}, "
                    f"失败 {len(failures)}, 源={used or '?'}"
                    f"{', 东财已熔断' if breaker.tripped else ''})",
                    file=sys.stderr,
                    flush=True,
                )

    prices = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(
            columns=["code", "date", "open", "high", "low", "close", "volume", "amount"]
        )
    )
    return prices, pd.DataFrame(failures, columns=["code", "error"])


def audit_returns(
    panel: pd.DataFrame, horizons: Sequence[int] = DEFAULT_HORIZONS
) -> pd.DataFrame:
    """未来收益的覆盖率与分布审计, 供 ``data_audit.csv`` 使用."""
    row: dict[str, object] = {"return_events": len(panel)}
    for h in horizons:
        col = f"fwd_ret_{h}d"
        if col in panel.columns:
            s = pd.to_numeric(panel[col], errors="coerce")
            row[f"{col}_coverage"] = float(s.notna().mean()) if len(s) else 0.0
            row[f"{col}_mean"] = float(s.mean()) if s.notna().any() else float("nan")
        else:
            row[f"{col}_coverage"] = 0.0
            row[f"{col}_mean"] = float("nan")
    return pd.DataFrame([row])
