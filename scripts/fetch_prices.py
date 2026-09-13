#!/usr/bin/env python
"""给事件面板接上行情, 计算未来收益.

流程: 读面板 → 按面板时间范围确定行情窗口 → 并发下载 (带缓存, 可断点续传)
→ 附加 ``fwd_ret_5d/10d/20d/40d`` → 落盘 + 覆盖率审计。

用法::

    python scripts/fetch_prices.py                      # 全量
    python scripts/fetch_prices.py --limit 50           # 只跑 50 只 (试跑)
    python scripts/fetch_prices.py --horizons 5 10 20 40

产出::

    data/processed/event_panel_returns.parquet
    results/return_audit.csv
    results/price_failures.csv     (若有失败)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.panel import DEFAULT_HORIZONS, attach_forward_returns, load_panel, save_panel
from data.prices import (
    VALID_SOURCES,
    audit_returns,
    download_price_panel,
)
from data.tradability import TradingCalendar

RESULTS_DIR = ROOT / "results"

#: 行情窗口在事件区间之外额外留的交易日缓冲
PAD_SESSIONS = 5


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="下载行情并附加未来收益")
    parser.add_argument("--panel", default="event_panel", help="输入面板名")
    parser.add_argument("--out", default="event_panel_returns", help="输出面板名")
    parser.add_argument(
        "--horizons", nargs="+", type=int, default=list(DEFAULT_HORIZONS),
        help=f"预测期 (交易日), 默认 {list(DEFAULT_HORIZONS)}",
    )
    parser.add_argument("--limit", type=int, default=None, help="只处理前 N 只股票 (试跑)")
    parser.add_argument("--adjust", default="hfq", choices=["hfq", "qfq", ""], help="复权方式")
    parser.add_argument(
        "--source", default="auto", choices=list(VALID_SOURCES),
        help="行情源: auto 优先东财失败降级新浪 (默认) / em / sina",
    )
    parser.add_argument("--workers", type=int, default=4, help="并发线程数")
    parser.add_argument("--force", action="store_true", help="忽略缓存重新抓取")
    parser.add_argument(
        "--save-prices",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="把合并后的日频行情面板写入 data/processed/prices.parquet (回测需要)",
    )
    args = parser.parse_args(argv)

    pd.set_option("display.width", 220)

    panel = load_panel(args.panel)
    codes = sorted(panel["code"].astype(str).str.zfill(6).unique())
    if args.limit:
        codes = codes[: args.limit]
    horizons = sorted(args.horizons)

    calendar = TradingCalendar.from_akshare()
    entry = pd.to_datetime(panel["tradable_ts"])
    first_entry = entry.min()
    last_entry = entry.max()
    last_exit = max(filter(None, (calendar.shift(d, max(horizons)) for d in entry)))

    start = calendar.shift(first_entry, -PAD_SESSIONS) or first_entry
    end = calendar.shift(pd.Timestamp(last_exit), PAD_SESSIONS) or last_exit

    print(f"[1/3] 面板 {len(panel)} 条事件, {len(codes)} 只股票")
    print(f"      入场区间: {first_entry.date()} .. {last_entry.date()}")
    print(f"      行情窗口: {start.date()} .. {end.date()}  (复权={args.adjust or '不复权'})")

    print(f"[2/3] 下载行情 (源={args.source}, 并发 {args.workers}) ...")
    prices, failures = download_price_panel(
        codes,
        start.strftime("%Y%m%d"),
        end.strftime("%Y%m%d"),
        adjust=args.adjust,
        source=args.source,
        max_workers=args.workers,
        force=args.force,
    )
    print(f"      行情: {len(prices)} 行, {prices['code'].nunique() if len(prices) else 0} 只")
    if len(prices) and "source" in prices.columns:
        print(f"      实际使用源: {prices.groupby('source')['code'].nunique().to_dict()}")
    print(f"      失败: {len(failures)} 只")

    print("[3/3] 计算未来收益 ...")
    enriched = attach_forward_returns(panel, prices, horizons=horizons, calendar=calendar)

    audit = audit_returns(enriched, horizons=horizons)
    print(audit.to_string(index=False))

    path = save_panel(enriched, args.out)
    print(f"      面板 -> {path.relative_to(ROOT)}")

    if args.save_prices and len(prices):
        price_path = save_panel(prices, "prices")
        print(f"      行情面板 ({len(prices)} 行) -> {price_path.relative_to(ROOT)}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    audit.to_csv(RESULTS_DIR / "return_audit.csv", index=False, encoding="utf-8-sig")
    print("      审计 -> results/return_audit.csv")
    if len(failures):
        failures.to_csv(
            RESULTS_DIR / "price_failures.csv", index=False, encoding="utf-8-sig"
        )
        print("      失败清单 -> results/price_failures.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
