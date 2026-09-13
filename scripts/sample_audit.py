#!/usr/bin/env python
"""随机抽查事件面板, 导出成可人工核对的表格.

计划文档要求「人工核对至少 20 条事件」, 并对齐三件事:

1. ``actual_disclosure_date`` 是否真的是公告日;
2. ``tradable_ts`` 是否是披露日之后的首个交易日;
3. 财报字段 (EPS / 营收 / 净利润) 是否与原始报告一致。

脚本会给出每只股票的公告查询链接, 并在表格里留出空白列供手工填写。

用法::

    python scripts/sample_audit.py                    # 默认抽 20 条
    python scripts/sample_audit.py --n 30 --seed 7

产出::

    results/sample_audit.csv    可编辑的核对表 (utf-8-sig, Excel 直接打开)
    results/sample_audit.md     便于阅读/粘贴的版本
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.panel import load_panel

RESULTS_DIR = ROOT / "results"

#: 公告查询页 (东方财富), 用于人工核对披露日期
NOTICE_URL = "https://data.eastmoney.com/notices/stock/{code}.html"
#: 巨潮资讯网全文检索 (交易所指定披露平台)
CNINFO_URL = "http://www.cninfo.com.cn/new/fulltextSearch?keyWord={name}"


def build_audit_table(panel: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """随机抽取 ``n`` 条事件, 整理成人工核对表."""
    if len(panel) == 0:
        raise ValueError("面板为空, 无法抽查")
    n = min(n, len(panel))
    sample = panel.sample(n=n, random_state=seed).sort_values("tradable_ts")
    # 抽样后索引是散落的, 必须 reset, 否则与下面新建的 Series 对不齐
    sample = sample.reset_index(drop=True)

    def col(name, default=pd.NA):
        return sample[name] if name in sample.columns else pd.Series([default] * len(sample))

    out = pd.DataFrame(
        {
            "序号": range(1, len(sample) + 1),
            "股票代码": sample["code"].astype(str).str.zfill(6),
            "报告期": pd.to_datetime(sample["report_date"]).dt.strftime("%Y-%m-%d"),
            "首次预约日": pd.to_datetime(col("first_scheduled")).dt.strftime("%Y-%m-%d"),
            "实际披露日": pd.to_datetime(sample["actual_disclosure_date"]).dt.strftime("%Y-%m-%d"),
            "入场日": pd.to_datetime(sample["tradable_ts"]).dt.strftime("%Y-%m-%d"),
            "披露日当时简称": col("name_at_disclosure", ""),
            "当前简称": col("name", ""),
            "当时ST": col("is_st"),
            "每股收益": col("eps"),
            "营业总收入": col("revenue"),
            "净利润": col("net_profit"),
            "公告查询": [
                NOTICE_URL.format(code=str(c).zfill(6)) for c in sample["code"]
            ],
            "巨潮检索": [
                CNINFO_URL.format(name=str(nm)) for nm in col("name", "")
            ],
            # 以下两列留给人工填写
            "①披露日是否正确(填Y/N)": "",
            "②备注/差异说明": "",
        }
    )
    return out.reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="随机抽查事件面板")
    parser.add_argument("--n", type=int, default=20, help="抽查条数, 默认 20")
    parser.add_argument("--seed", type=int, default=42, help="随机种子, 默认 42 (可复现)")
    args = parser.parse_args(argv)

    panel = load_panel()
    print(f"面板: {len(panel)} 条事件, 股票 {panel['code'].nunique()} 只")

    table = build_audit_table(panel, args.n, args.seed)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "sample_audit.csv"
    table.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"抽查表 -> {csv_path.relative_to(ROOT)}  ({len(table)} 条, seed={args.seed})")

    md_path = RESULTS_DIR / "sample_audit.md"
    md_cols = [
        "序号", "股票代码", "报告期", "首次预约日", "实际披露日",
        "入场日", "披露日当时简称", "当时ST", "每股收益",
    ]
    with md_path.open("w", encoding="utf-8") as fh:
        fh.write(f"# 事件面板人工抽查 (n={len(table)}, seed={args.seed})\n\n")
        fh.write("核对要点: ① 实际披露日是否真是公告日; ")
        fh.write("② 入场日是否为披露日之后的首个交易日; ")
        fh.write("③ 财报数值是否与原始报告一致。\n\n")
        fh.write(table[md_cols].to_markdown(index=False))
        fh.write("\n\n## 核对链接\n\n")
        for _, row in table.iterrows():
            fh.write(
                f"- {row['股票代码']} {row['披露日当时简称']}: "
                f"[公告]({row['公告查询']}) · "
                f"[巨潮检索]({row['巨潮检索']})\n"
            )
    print(f"核对清单 -> {md_path.relative_to(ROOT)}")

    print()
    print("预览:")
    print(table[md_cols].head(10).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
