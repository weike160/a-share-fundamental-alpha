"""个股 EPS 历史 (用于构造季节性随机游走 SUE).

为什么需要单独一个模块
----------------------
SUE 需要**多个季度**的 EPS 历史::

    UE(i,t)  = EPS(i,t) − EPS(i,t−4)          # 需要去年同期
    SUE(i,t) = UE(i,t) / std(过去 k 期 UE)     # 还需要更长历史估标准差

而 ``stock_yjbb_em`` 是**按报告期整市场**返回的, 要凑够 8 个季度就得拉 8 次
(而且它当前被限流)。同花顺的 ``stock_financial_abstract_ths`` 则**一次返回
一只股票的全部季度历史**, 更直接。

数据源
------
``stock_financial_abstract_ths(symbol, indicator="按报告期")``

* 返回 ``报告期`` + ``基本每股收益`` 等字段
* ``基本每股收益`` 是纯数字字符串 (如 ``"19.1600"``), 缺失值为 ``"False"``
* 实测覆盖到 1989 年, 对 2024Q1 的 SUE 来说绰绰有余

口径校验
--------
:func:`cross_validate` 会把该来源的 EPS 与事件面板里的 ``eps`` (来自东财
业绩报表) 对比。**两个来源的 EPS 口径必须一致**, 否则 SUE 会混入口径噪声。
"""
from __future__ import annotations

import sys
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from data.source import fetch

DEFAULT_WORKERS = 4
#: 只保留这个年份之后的季度 (SUE 用不到更早的)
DEFAULT_SINCE_YEAR = 2015


def _to_float(value: object) -> float:
    """同花顺用字符串 ``"False"`` 表示缺失, 统一转成 NaN."""
    if value is None or value is False:
        return float("nan")
    text = str(value).strip().replace(",", "")
    if text in {"", "False", "None", "nan", "--"}:
        return float("nan")
    try:
        return float(text)
    except ValueError:
        return float("nan")


def load_eps_history(
    code: str, *, since_year: int = DEFAULT_SINCE_YEAR, force: bool = False
) -> pd.DataFrame:
    """取单只股票的季度 EPS 历史.

    Returns
    -------
    DataFrame``[code, report_date, eps_raw]``, 按报告期升序。
    """
    raw = fetch(
        "stock_financial_abstract_ths",
        params={"symbol": str(code).zfill(6), "indicator": "按报告期"},
        force=force,
    )
    if "报告期" not in raw.columns or "基本每股收益" not in raw.columns:
        raise ValueError(f"同花顺返回缺少字段 (code={code}): {list(raw.columns)[:8]}")

    out = pd.DataFrame(
        {
            "code": str(code).zfill(6),
            "report_date": pd.to_datetime(raw["报告期"], errors="coerce").dt.normalize(),
            "eps_raw": raw["基本每股收益"].map(_to_float),
        }
    )
    return (
        out.dropna(subset=["report_date"])
        .loc[lambda d: d["report_date"].dt.year >= since_year]
        .sort_values("report_date")
        .reset_index(drop=True)
    )


def _fetch_with_delay(
    code: str, since_year: int, force: bool, delay: float
) -> pd.DataFrame:
    """取一只股票的 EPS 历史, 可选前置延迟用于限流.

    实测同花顺 (10jqka) 高并发下会限流: 4882 只跑到约 4000 只时, 失败数从
    200 骤增到 905, 且失败**按代码段成块**(003/605 全缺, 603/301 缺一半),
    是典型的服务端限流而非数据缺失 —— 串行重试时 10/10 全部成功。
    """
    if delay > 0:
        time.sleep(delay)
    return load_eps_history(code, since_year=since_year, force=force)


def download_eps_history(
    codes: Iterable[str],
    *,
    since_year: int = DEFAULT_SINCE_YEAR,
    max_workers: int = DEFAULT_WORKERS,
    delay: float = 0.0,
    force: bool = False,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """并发下载多只股票的 EPS 历史.

    Parameters
    ----------
    delay:
        每次请求前的延迟 (秒), 用于规避上游限流。默认 0。
        遇到大面积失败时建议 ``max_workers=2, delay=0.2``。

    Returns
    -------
    ``(eps_panel, failures)``
    """
    code_list = sorted({str(c).zfill(6) for c in codes})
    total = len(code_list)
    if total == 0:
        return pd.DataFrame(columns=["code", "report_date", "eps_raw"]), pd.DataFrame(
            columns=["code", "error"]
        )

    frames: list[pd.DataFrame] = []
    failures: list[dict[str, str]] = []

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_fetch_with_delay, code, since_year, force, delay): code
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
            except Exception as exc:  # noqa: BLE001
                failures.append({"code": code, "error": f"{type(exc).__name__}: {exc}"})
            if verbose and (done % 250 == 0 or done == total):
                print(
                    f"    EPS 历史 {done}/{total} (成功 {len(frames)}, 失败 {len(failures)})",
                    file=sys.stderr,
                    flush=True,
                )

    eps = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=["code", "report_date", "eps_raw"])
    )
    return eps, pd.DataFrame(failures, columns=["code", "error"])


def cross_validate(
    eps_history: pd.DataFrame, panel: pd.DataFrame, *, tol: float = 0.02
) -> pd.DataFrame:
    """把同花顺 EPS 与事件面板的 EPS 做口径校验.

    两者都声称是「基本每股收益」, 但来自不同数据源, 必须实测确认一致。

    Parameters
    ----------
    tol:
        相对容差。EPS 接近 0 时改用绝对容差 ``tol``。

    Returns
    -------
    单行 DataFrame: 可比样本数、一致率、平均绝对差等。
    """
    if "eps" not in panel.columns:
        raise ValueError("panel 需要 eps 列")

    left = panel[["code", "report_date", "eps"]].copy()
    left["code"] = left["code"].astype(str).str.zfill(6)
    left["report_date"] = pd.to_datetime(left["report_date"])
    left["eps"] = pd.to_numeric(left["eps"], errors="coerce")

    right = eps_history.rename(columns={"eps_raw": "eps_ths"})
    merged = left.merge(right, on=["code", "report_date"], how="inner")
    both = merged.dropna(subset=["eps", "eps_ths"])

    if len(both) == 0:
        return pd.DataFrame(
            [{"compared": 0, "match_rate": float("nan"), "mean_abs_diff": float("nan")}]
        )

    diff = (both["eps"] - both["eps_ths"]).abs()
    scale = both["eps"].abs().clip(lower=1.0)
    ok = diff <= (tol * scale)

    return pd.DataFrame(
        [
            {
                "compared": len(both),
                "match_rate": float(ok.mean()),
                "mean_abs_diff": float(diff.mean()),
                "max_abs_diff": float(diff.max()),
                "mismatch_examples": int((~ok).sum()),
            }
        ]
    )
