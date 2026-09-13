#!/usr/bin/env python
"""构造 SUE 信号并合并进事件面板.

流程: 读面板 → 逐只下载 EPS 历史 (同花顺) → 口径校验 → 计算季节性随机游走
SUE → 合并 → 落盘 + 审计。

用法::

    python scripts/build_signal.py                  # 全量
    python scripts/build_signal.py --limit 50       # 试跑
    python scripts/build_signal.py --window 8 --min-periods 4

产出::

    data/processed/event_panel_signal.parquet    含 sue 列
    results/eps_history.parquet                  EPS 历史面板 (供复用)
    results/signal_audit.csv                     SUE 覆盖率与口径校验
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.eps_history import cross_validate, download_eps_history
from data.panel import load_panel, save_panel
from signals.sue import (
    DEFAULT_MIN_PERIODS,
    DEFAULT_WINDOW,
    seasonal_random_walk_sue,
    sue_coverage,
)

RESULTS_DIR = ROOT / "results"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="构造 SUE 信号")
    parser.add_argument("--panel", default="event_panel_returns", help="输入面板名")
    parser.add_argument("--out", default="event_panel_signal", help="输出面板名")
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW, help="标准差窗口")
    parser.add_argument("--min-periods", type=int, default=DEFAULT_MIN_PERIODS, help="最少期数")
    parser.add_argument("--lag", type=int, default=4, help="同比滞后 (季度)")
    parser.add_argument("--limit", type=int, default=None, help="只处理前 N 只股票 (试跑)")
    parser.add_argument("--workers", type=int, default=4, help="并发线程数")
    parser.add_argument(
        "--delay", type=float, default=0.0,
        help="每次请求前的延迟 (秒), 用于规避同花顺限流。大面积失败时用 0.2",
    )
    parser.add_argument("--force", action="store_true", help="忽略缓存重新抓取")
    args = parser.parse_args(argv)

    pd.set_option("display.width", 220)

    panel = load_panel(args.panel)
    codes = sorted(panel["code"].astype(str).str.zfill(6).unique())
    if args.limit:
        codes = codes[: args.limit]
    print(f"[1/4] 面板 {len(panel)} 条事件, {len(codes)} 只股票")

    print("[2/4] 下载 EPS 历史 (同花顺) ...")
    eps, failures = download_eps_history(
        codes, max_workers=args.workers, delay=args.delay, force=args.force
    )
    print(f"      EPS 历史 {len(eps)} 行, {eps['code'].nunique() if len(eps) else 0} 只, 失败 {len(failures)}")
    if len(eps) == 0:
        print("!! 没有拿到任何 EPS 历史, 无法计算 SUE")
        return 1

    print("[3/4] 口径校验 (同花顺 vs 东财业绩报表) ...")
    check = cross_validate(eps, panel)
    print(check.to_string(index=False))

    print("[4/4] 计算 SUE 并合并 ...")
    sue_all = seasonal_random_walk_sue(
        eps.rename(columns={"eps_raw": "eps"}),
        lag=args.lag,
        window=args.window,
        min_periods=args.min_periods,
    )
    eps = eps.assign(sue=sue_all)

    merged = panel.merge(
        eps[["code", "report_date", "sue"]],
        on=["code", "report_date"],
        how="left",
    )
    cov = sue_coverage(merged, merged["sue"])
    print(cov.to_string(index=False))

    path = save_panel(merged, args.out)
    print(f"      面板 -> {path.relative_to(ROOT)}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    eps.to_parquet(RESULTS_DIR / "eps_history.parquet", index=False)
    audit = pd.concat([check, cov], axis=1)
    audit.to_csv(RESULTS_DIR / "signal_audit.csv", index=False, encoding="utf-8-sig")
    print("      EPS 历史 -> results/eps_history.parquet")
    print("      审计     -> results/signal_audit.csv")
    if len(failures):
        failures.to_csv(
            RESULTS_DIR / "eps_failures.csv", index=False, encoding="utf-8-sig"
        )
        print(f"      失败清单 -> results/eps_failures.csv ({len(failures)} 只)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
