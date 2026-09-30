#!/usr/bin/env python3
"""Daily paper trading, frozen model export, and isolated offline demonstration."""
from __future__ import annotations

import argparse
import fcntl
import json
import signal
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from paper.engine import Account
from paper.report import atomic_text, write_report


class CollectionDeadline(BaseException):
    """Bypass provider retry handlers, then report a clean daily failure."""


def publish(directory, state):
    path = write_report(directory, state)
    status = "ready" if state.get("selection_status", "ready") == "ready" else "settled_without_selection"
    atomic_text(directory / "status.json", json.dumps(dict(
        date=state["last_date"], status=status, report=str(path),
        selection_status=state.get("selection_status", "ready"), warnings=state.get("warnings", [])),
        ensure_ascii=False, indent=2))
    return path


def check(directory, model_dir):
    """Offline startup check; never substitutes a model or changes the account."""
    checks = []
    if (model_dir / "metadata.json").is_file() and (model_dir / "model.txt").is_file():
        try:
            from paper.model import Predictor
            model = Predictor(model_dir)
            checks.append(f"模型可加载：{model.model_id}，训练标签截至 {model.meta['max_label_date']}")
        except Exception as exc:
            checks.append(f"模型不可用：{exc}")
    else:
        checks.append(f"缺少真实模型：{model_dir}")
        panel = ROOT / "data" / "processed" / "analysis_panel.parquet"
        checks.append(f"训练面板：{'已找到' if panel.exists() else '未找到'}（{panel}）")
        checks.append("恢复原研究面板后执行 train；仅查看账户操作可运行 demo，演示不会替代真实模型。")
    if (directory / "account.sqlite3").exists():
        state = Account(directory).read()
        checks.append(f"账户：现金 {state['cash']:,.2f} 元，持仓 {len(state['positions'])} 只，最近结算 {state['last_date'] or '无'}")
    else:
        checks.append("账户尚未创建，daily 首次运行将初始化 100 万元虚拟资金。")
    print("\n".join(checks))
    return 0


def demonstration(directory):
    account = Account(directory, mode="demo")
    days = [date(2026, 8, 3) + timedelta(days=i) for i in range(45)]
    days = [d for d in days if d.weekday() < 5][:30]
    for i, day in enumerate(days):
        value = 10 + i * 0.03
        quotes = {f"600{j:03d}": dict(date=day.isoformat(), open=value, close=value + 0.02,
                    high=value + 0.10, low=value - 0.10, volume_shares=1_000_000,
                    limit_up=round(value * 1.1, 2), limit_down=round(value * .9, 2), eligible=True)
                  for j in range(100)}
        snap = dict(mode="demo", status="ready", date=day.isoformat(), market_date=day.isoformat(),
                    previous_session=days[i - 1].isoformat() if i else "2026-07-31",
                    model_id="SYNTHETIC-DEMO-NOT-LIGHTGBM", benchmark_close=4000 + i * 2,
                    quotes=quotes, candidates=[dict(code=code, name="演示股票", event_id=code + ":demo",
                    available_date=day.isoformat(), feature_date=day.isoformat(), score=j / 100)
                    for j, code in enumerate(quotes)] if i == 0 else [])
        account.commit(snap)
    path = write_report(directory, account.read())
    print(f"演示账户已完成 30 个合成交易日，非真实模型表现：{path}")
    return 0


def daily(directory, model_dir):
    account = Account(directory)
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    day = now.date().isoformat()
    with (directory / "run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("已有每日任务运行中，跳过重复启动")
            return 0
        state = account.read()
        try:
            if state["last_date"] == day:
                print(publish(directory, state))
                return 0
            if now.weekday() >= 5 or now.hour < 16:
                detail = "等待北京时间交易日收盘后运行。模型准备情况可用 scripts/run_paper.py check 检查；尚未接入模型时不会产生新买单。"
                path = write_report(directory, state, status="等待收盘", detail=detail)
                atomic_text(directory / "status.json", json.dumps(dict(date=day, status="waiting_market", report=str(path)), ensure_ascii=False, indent=2))
                print("未到交易日收盘后运行窗口，账户不变")
                return 0
            from paper.feed import MarketClosed, build_snapshot, settlement_snapshot
            snapshot_path = directory / "snapshots" / f"{day}.json"
            if snapshot_path.exists():
                snapshot = json.loads(snapshot_path.read_text())
            else:
                try:
                    def deadline(_signum, _frame):
                        raise CollectionDeadline("数据采集超过 15 分钟，本次停止，稍后可重试")
                    signal.signal(signal.SIGALRM, deadline)
                    signal.alarm(900)
                    if state["last_date"]:
                        from paper.recovery import recover
                        restored = recover(account, directory, (now.date() - timedelta(days=1)).isoformat())
                        if restored:
                            print(f"已恢复 {len(restored)} 个交易日", flush=True)
                        state = account.read()
                    snapshot = build_snapshot(day, state, model_dir)
                except CollectionDeadline:
                    signal.alarm(120)
                    snapshot = settlement_snapshot(day, account.read(), "选股采集超时，改为仅结算账户")
                except MarketClosed as exc:
                    print(str(exc))
                    return 0
                finally:
                    signal.alarm(0)
                atomic_text(snapshot_path, json.dumps(snapshot, ensure_ascii=False, indent=2, allow_nan=False))
            state = account.commit(snapshot)
            path = publish(directory, state)
            print(path)
            return 0
        except (Exception, CollectionDeadline) as exc:
            detail = f"检查时间：{now.isoformat()}\n\n{type(exc).__name__}: {exc}\n\n下表若有数据，仅代表账本中最近一次成功结算；本次失败不补造收益。"
            path = write_report(directory, account.read(), status="等待模型/数据或核对", detail=detail)
            atomic_text(directory / "status.json", json.dumps(dict(date=day, status="blocked", error=str(exc), report=str(path)), ensure_ascii=False, indent=2))
            print(f"未完成日结：{exc}\n诊断日报：{path}", file=sys.stderr)
            return 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("daily", help="收盘后更新真实行情、预测、结算并生成日报")
    run.add_argument("--directory", type=Path, default=ROOT / "paper_output" / "live")
    run.add_argument("--model", type=Path, default=ROOT / "paper_models" / "current")
    status = sub.add_parser("check", help="离线检查模型与账户，列出启动缺项")
    status.add_argument("--directory", type=Path, default=ROOT / "paper_output" / "live")
    status.add_argument("--model", type=Path, default=ROOT / "paper_models" / "current")
    recovery = sub.add_parser("recover", help="恢复保存的日快照或空仓断档，不补造历史信号")
    recovery.add_argument("--directory", type=Path, default=ROOT / "paper_output" / "live")
    recovery.add_argument("--through", required=True, help="恢复截至日期 YYYY-MM-DD，必须早于今天")
    demo = sub.add_parser("demo", help="独立演示账户，使用明确标记的合成数据")
    demo.add_argument("--directory", type=Path, default=ROOT / "paper_output" / "demo")
    train = sub.add_parser("train", help="用现有研究面板导出完整特征 LightGBM")
    train.add_argument("--panel", type=Path, default=ROOT / "data" / "processed" / "analysis_panel.parquet")
    train.add_argument("--model", type=Path, default=ROOT / "paper_models" / "current")
    train.add_argument("--cutoff", required=True, help="最后允许使用的标签完成日期 YYYY-MM-DD")
    args = parser.parse_args(argv)
    if args.command == "check":
        return check(args.directory, args.model)
    if args.command == "recover":
        if date.fromisoformat(args.through) >= datetime.now(ZoneInfo("Asia/Shanghai")).date():
            parser.error("恢复截止日期必须早于今天")
        from paper.recovery import recover
        account = Account(args.directory)
        with (args.directory / "run.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            restored = recover(account, args.directory, args.through)
            print(f"已恢复 {len(restored)} 个交易日")
            print(publish(args.directory, account.read()))
        return 0
    if args.command == "train":
        if not args.panel.is_file():
            parser.error(f"训练面板不存在：{args.panel}")
        if date.fromisoformat(args.cutoff) >= datetime.now(ZoneInfo("Asia/Shanghai")).date():
            parser.error("训练截止日期必须早于今天")
        from paper.model import train as export_model
        print(json.dumps(export_model(args.panel, args.model, args.cutoff), ensure_ascii=False, indent=2))
        return 0
    if args.command == "demo":
        if args.directory.resolve() == (ROOT / "paper_output" / "live").resolve():
            parser.error("演示数据不能写入正式账户目录")
        return demonstration(args.directory)
    return daily(args.directory, args.model)


if __name__ == "__main__":
    raise SystemExit(main())
