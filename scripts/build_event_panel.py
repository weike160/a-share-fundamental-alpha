#!/usr/bin/env python
"""一条命令构建财报事件面板 (data 层的端到端入口).

用法::

    python scripts/build_event_panel.py --periods 20240331
    python scripts/build_event_panel.py --periods 20240331 20240630

产出 (写入 ``results/`` 与 ``data/processed/``)::

    data/processed/event_panel.parquet   事件面板
    results/data_audit.csv               数据审计表
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.delisting import load_delistings
from data.panel import (
    audit_panel,
    audit_survivorship,
    build_event_panel,
    save_panel,
)
from data.source import list_cache

RESULTS_DIR = ROOT / "results"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="构建 A 股财报事件面板")
    parser.add_argument(
        "--periods",
        nargs="+",
        required=True,
        help="报告期, 形如 20240331 (可多个)",
    )
    parser.add_argument("--lag", type=int, default=1, help="入场延迟 (交易日), 默认 1")
    parser.add_argument(
        "--st-policy",
        choices=["drop", "keep", "flag"],
        default="drop",
        help="ST/*ST 处理: drop 剔除 (默认) / keep 保留 / flag 只标记",
    )
    parser.add_argument(
        "--no-universe-filter",
        action="store_true",
        help="跳过 point-in-time 股票池过滤 (调试用, 会引入幸存者偏差)",
    )
    parser.add_argument("--force", action="store_true", help="忽略缓存重新抓取")
    args = parser.parse_args(argv)

    pd.set_option("display.width", 220)

    print(
        f"[1/4] 构建事件面板 periods={args.periods} "
        f"lag={args.lag} st_policy={args.st_policy} ..."
    )
    panel = build_event_panel(
        args.periods,
        lag=args.lag,
        force=args.force,
        filter_universe=not args.no_universe_filter,
        st_policy=args.st_policy,
    )
    print(f"      事件数: {len(panel)}  股票数: {panel['code'].nunique()}")

    print("[2/4] 覆盖率审计 ...")
    records = panel.attrs.get("coverage_audit") or []
    coverage = pd.DataFrame(records)
    if len(coverage):
        print(coverage.to_string(index=False))

    st_records = panel.attrs.get("st_audit") or []
    st_audit = pd.DataFrame(st_records)
    if len(st_audit):
        print("      ST 处理审计:")
        print("      " + st_audit.to_string(index=False).replace("\n", "\n      "))

    print("[3/4] 面板审计与不变式检查 ...")
    audit = audit_panel(panel)
    print(audit.to_string(index=False))
    violations = int(audit["violations"].iloc[0])
    if violations:
        print(f"      !! 发现 {violations} 条违反 report_date <= disclosure < entry 的记录")
    else:
        print("      不变式成立: report_date <= actual_disclosure_date < tradable_ts")

    surv = audit_survivorship(panel, load_delistings("all"))
    print("      幸存者偏差检查:")
    print("      " + surv.to_string(index=False).replace("\n", "\n      "))

    print("[4/4] 落盘 ...")
    path = save_panel(panel)
    print(f"      面板 -> {path.relative_to(ROOT)}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    audit_path = RESULTS_DIR / "data_audit.csv"
    audit_out = audit.copy()
    if len(coverage):
        audit_out = pd.concat([coverage, audit], axis=1)
    audit_out = pd.concat([audit_out, surv], axis=1)
    if len(st_audit):
        audit_out = pd.concat([audit_out, st_audit], axis=1)
    audit_out.to_csv(audit_path, index=False, encoding="utf-8-sig")
    print(f"      审计 -> {audit_path.relative_to(ROOT)}")

    cache = list_cache()
    print(f"      缓存接口数: {len(cache)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
