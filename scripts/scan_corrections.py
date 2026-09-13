#!/usr/bin/env python
"""扫描事件的财报更正情况, 给面板加上数据质量标记.

用法::

    python scripts/scan_corrections.py                     # 全量扫描
    python scripts/scan_corrections.py --limit 30          # 试跑
    python scripts/scan_corrections.py --window-days 180

产出::

    data/processed/event_panel_final.parquet    含 report_corrected 标记
    results/correction_scan.csv                 逐事件明细 (便于人工复核)
    results/correction_audit.csv                汇总
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.corrections import (
    DEFAULT_WINDOW_DAYS,
    audit_corrections,
    scan_corrections,
)
from data.panel import load_panel, save_panel

RESULTS_DIR = ROOT / "results"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="扫描财报更正情况")
    parser.add_argument("--panel", default="event_panel_returns", help="输入面板名")
    parser.add_argument("--out", default="event_panel_final", help="输出面板名")
    parser.add_argument(
        "--window-days", type=int, default=DEFAULT_WINDOW_DAYS,
        help=f"披露日后检索天数, 默认 {DEFAULT_WINDOW_DAYS}",
    )
    parser.add_argument("--limit", type=int, default=None, help="只处理前 N 条 (试跑)")
    parser.add_argument("--workers", type=int, default=4, help="并发线程数")
    parser.add_argument("--force", action="store_true", help="忽略缓存重新抓取")
    args = parser.parse_args(argv)

    pd.set_option("display.width", 220)

    panel = load_panel(args.panel)
    if args.limit:
        panel = panel.head(args.limit).copy()
    print(f"[1/3] 面板 {len(panel)} 条事件, 窗口 {args.window_days} 天")

    print("[2/3] 扫描巨潮公告历史 ...")
    scan = scan_corrections(
        panel,
        window_days=args.window_days,
        max_workers=args.workers,
        force=args.force,
    )

    print("[3/3] 合并与落盘 ...")
    merged = panel.merge(
        scan.drop(columns=["report_date"]),
        on="code",
        how="left",
    )
    merged["report_corrected"] = merged["report_corrected"].fillna(False).astype(bool)

    audit = audit_corrections(scan)
    print(audit.to_string(index=False))

    corrected = merged[merged["report_corrected"]]
    if len(corrected):
        print()
        print(f"  被更正的事件 {len(corrected)} 条, 例:")
        cols = ["code", "name", "actual_disclosure_date", "correction_titles"]
        cols = [c for c in cols if c in corrected.columns]
        print(corrected[cols].head(8).to_string(index=False))

    path = save_panel(merged, args.out)
    print(f"      面板 -> {path.relative_to(ROOT)}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    scan.to_csv(RESULTS_DIR / "correction_scan.csv", index=False, encoding="utf-8-sig")
    audit.to_csv(RESULTS_DIR / "correction_audit.csv", index=False, encoding="utf-8-sig")
    print("      明细 -> results/correction_scan.csv")
    print("      汇总 -> results/correction_audit.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
