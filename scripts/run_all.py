#!/usr/bin/env python
"""一条命令跑完整条研究链路.

    python scripts/run_all.py                 # 全流程 (使用缓存)
    python scripts/run_all.py --skip-download # 只用已有缓存与面板
    python scripts/run_all.py --force         # 忽略缓存重新下载

步骤::

    1 事件面板      scripts/build_event_panel.py
    2 行情与收益    scripts/fetch_prices.py
    3 SUE 信号      scripts/build_signal.py
    4 研究面板      scripts/build_analysis_panel.py
    5 因子检验      scripts/run_factor.py
    6 回测与稳健性  scripts/run_backtest.py
    7 机器学习      scripts/run_ml.py
    8 研究报告      scripts/make_report.py
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.config import CONFIG  # noqa: E402

RESULTS_DIR = ROOT / "results"
SCRIPTS = ROOT / "scripts"


def run_step(name: str, argv: list[str]) -> None:
    print(f"\n{'=' * 78}\n[{name}] {' '.join(argv)}\n{'=' * 78}", flush=True)
    started = time.time()
    proc = subprocess.run([sys.executable, *argv], cwd=ROOT)
    elapsed = time.time() - started
    if proc.returncode != 0:
        raise SystemExit(f"步骤失败 ({name}), 退出码 {proc.returncode}")
    print(f"[{name}] 完成, 用时 {elapsed / 60:.1f} 分钟", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="运行完整研究链路")
    parser.add_argument("--skip-download", action="store_true", help="跳过数据下载步骤")
    parser.add_argument("--skip-ml", action="store_true", help="跳过机器学习步骤")
    parser.add_argument("--skip-backtest-delay", action="store_true", help="回测跳过延后入场情景")
    parser.add_argument("--skip-tests", action="store_true", help="最后不跑 pytest")
    parser.add_argument(
        "--with-correction-scan",
        action="store_true",
        help="额外跑财报更正扫描 (抽样试点, 较慢)",
    )
    parser.add_argument(
        "--correction-sample", type=int, default=60, help="更正扫描抽样的股票数"
    )
    parser.add_argument("--force", action="store_true", help="忽略缓存重新抓取")
    args = parser.parse_args(argv)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "run_config.json").write_text(CONFIG.to_json(), encoding="utf-8")

    force = ["--force"] if args.force else []
    if not args.skip_download:
        run_step(
            "1/9 事件面板",
            [str(SCRIPTS / "build_event_panel.py"), "--periods", *CONFIG.periods, *force],
        )
        run_step(
            "2/9 人工抽查样本",
            [str(SCRIPTS / "sample_audit.py"), "--n", "30"],
        )
        run_step("3/9 行情与收益", [str(SCRIPTS / "fetch_prices.py"), *force])
        run_step(
            "4/9 SUE 信号",
            [
                str(SCRIPTS / "build_signal.py"),
                "--panel",
                "event_panel",
                "--out",
                "event_panel_signal",
                *force,
            ],
        )
    if args.with_correction_scan:
        run_step(
            "5/9 财报更正扫描 (抽样试点)",
            [
                str(SCRIPTS / "scan_corrections.py"),
                "--limit",
                str(args.correction_sample),
                "--out",
                "event_panel_correction_pilot",
                *force,
            ],
        )
    run_step("6/9 研究面板", [str(SCRIPTS / "build_analysis_panel.py")])
    run_step("7/9 因子检验", [str(SCRIPTS / "run_factor.py")])
    backtest_cmd = [str(SCRIPTS / "run_backtest.py")]
    if args.skip_backtest_delay:
        backtest_cmd.append("--skip-delay")
    run_step("8/9 回测与稳健性", backtest_cmd)
    if not args.skip_ml:
        run_step("机器学习", [str(SCRIPTS / "run_ml.py")])
    run_step("9/9 研究报告", [str(SCRIPTS / "make_report.py")])

    if not args.skip_tests:
        run_step("测试", ["-m", "pytest", "tests/", "-q"])

    print("\n全部完成。结果在 results/, 报告在 report/REPORT.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
