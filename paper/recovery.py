"""Resume saved snapshots; fill cash-only gaps without inventing signals or trades."""
import json
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from data.source import fetch
from paper.feed import benchmark_bars
from paper.report import atomic_text, write_report


def recover(account, directory, through):
    state = account.read()
    if not state["last_date"] or through <= state["last_date"]:
        return []
    calendar = fetch("tool_trade_date_hist_sina", force=True, retries=2)
    sessions = sorted(set(pd.to_datetime(calendar["trade_date"]).dt.strftime("%Y-%m-%d")))
    if not sessions or through > sessions[-1]:
        raise ValueError("恢复所需交易日历不完整")
    missing = [d for d in sessions if state["last_date"] < d <= through]
    restored, index = [], None
    for day in missing:
        path = directory / "snapshots" / f"{day}.json"
        if path.exists():
            snap = json.loads(path.read_text())
            if snap["date"] != day:
                raise ValueError("恢复快照日期与文件名不符")
        else:
            if state["positions"] or state["pending"]:
                raise ValueError(f"{day} 有持仓或待执行订单但无已保存快照，不能自动恢复；账本保留，请提供该日可靠快照核对。")
            if index is None:
                index = benchmark_bars()
                index = index.assign(date=pd.to_datetime(index["date"]).dt.strftime("%Y-%m-%d"))
            row = index.loc[index["date"] == day]
            if len(row) != 1:
                raise ValueError(f"缺少恢复日期 {day} 的指数行情")
            snap = dict(mode=state["mode"], status="ready", date=day, market_date=day,
                previous_session=sessions[sessions.index(day) - 1], model_id=state["model_id"],
                fetched_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                benchmark_close=float(row.iloc[0]["close"]), quotes={}, candidates=[],
                selection_status="unavailable", recovery=True,
                warnings=["空仓断档恢复：现金保持不变，只补记历史指数，不生成历史信号或交易。"])
            atomic_text(path, json.dumps(snap, ensure_ascii=False, indent=2, allow_nan=False))
        state = account.commit(snap)
        write_report(directory, state)
        restored.append(day)
    return restored
