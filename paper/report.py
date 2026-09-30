"""Rebuild readable Markdown and CSV reports from the committed account."""
from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path


def atomic_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as f:
        f.write(text)
        temp = f.name
    os.replace(temp, path)


def table(headers, rows):
    def clean(value):
        return str(value).replace("|", "/").replace("\n", " ")
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
                     + ["| " + " | ".join(map(clean, row)) + " |" for row in rows])


def write_report(directory: Path, state, *, status="正常", detail=""):
    mode = "演示数据，非真实模型表现" if state["mode"] == "demo" else "模拟实盘，虚拟资金"
    day = state["last_date"] or "尚未启动"
    lines = [f"# 模拟交易日报 · {day}", "", mode, "", f"运行状态：{status}", "", detail, ""]
    if state.get("selection_status") == "unavailable":
        lines += ["今日已完成账户结算，但未运行有效选股；没有新增模型买单。当前收益不能视为模型已投入运行。", ""]
    if state["history"]:
        d = state["history"][-1]
        lines += [table(["指标", "数值"], [
            ["总资产", f'{d["equity"]:,.2f} 元'], ["可用现金", f'{d["cash"]:,.2f} 元'],
            ["持仓市值", f'{d["market_value"]:,.2f} 元'],
            ["当日盈亏", f'{d["daily_pnl"]:+,.2f} 元（{d["daily_return"]:+.2%}）'],
            ["累计收益", f'{d["total_return"]:+.2%}'],
            ["沪深300价格指数同期收益", f'{d["benchmark_return"]:+.2%}'],
            ["收益差（非风险调整 Alpha）", f'{d["excess_return"]:+.2%}'],
            ["当前回撤", f'{d["drawdown"]:.2%}'], ["今日交易费用", f'{d["fees"]:.2f} 元'],
        ]), ""]
    else:
        lines += [f'初始虚拟资金：{state["settings"]["initial_cash"]:,.2f} 元。尚无有效交易日记录，收益未计算。', ""]
    lines += ["## 当前持仓", "", table(["代码", "名称", "股数", "收盘价", "市值"], [
        [code, p["name"], p["qty"], f'{p["mark"]:.2f}', f'{p["qty"] * p["mark"]:,.2f}']
        for code, p in sorted(state["positions"].items())]) if state["positions"] else "空仓。", "",
        "## 下一交易日订单", "", table(["方向", "代码", "数量", "模型分数"], [
            [o["side"], o["code"], o["qty"], f'{o["score"]:.4f}' if "score" in o else "到期/退出"]
            for o in state["pending"]]) if state["pending"] else "无待执行订单。", "",
        "## 今日执行记录", "", table(["方向", "代码", "成交股数", "状态"], [
            [o["side"], o["code"], o["filled_qty"], o["status"]] for o in state["orders"] if o["execution_date"] == day]), "",
        "## 运行口径", "", f'模型版本：{state["model_id"] or "未加载"}。', "",
        "收盘后产生信号；下一交易日收盘后，用当日未复权开盘价与滑点规则模拟昨日订单的成交。日线撮合不代表真实排队成交，成交额上限使用全日成交量。",
        "", "主板普通股票；排除 ST、退市标记及历史行情不足 60 条的股票。每个财报事件只筛选一次，按当天新事件得分前 20% 入选；不足 20 只时保留现金。持有满 20 个交易日后发出卖单。",
        "", "费用为可配置的模拟假设。沪深300为价格指数参照，不是同风险、同费用的可交易基准。公司行动异常或交易日缺口会暂停记账，等待核对。", ""]
    if state.get("warnings") or state.get("exclusions"):
        lines += ["## 数据与模型提示", "", *state.get("warnings", []), *state.get("exclusions", []), ""]
    text = "\n".join(lines)
    atomic_text(directory / "latest.md", text)
    if state["last_date"] and status == "正常":
        atomic_text(directory / "reports" / f"{day}.md", text)
    for name, rows in (("equity", state["history"]), ("trades", state["trades"]), ("orders", state["orders"]),
                       ("candidates", state.get("candidates", []))):
        import io
        stream = io.StringIO()
        fields = list(dict.fromkeys(k for row in rows for k in row)) or {
            "equity": ["date", "equity"], "trades": ["date", "code", "side", "qty"],
            "orders": ["execution_date", "code", "status"],
            "candidates": ["code", "name", "score", "event_id"]}[name]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        atomic_text(directory / f"{name}.csv", stream.getvalue())
    return directory / "latest.md"
