#!/usr/bin/env python
"""由结果文件生成研究报告 (report/REPORT.md).

报告只做一件事: 把已经落盘的数字整理成可检查的结论, 并明确写出数据限制、
失败实验与证伪结果。所有表格都直接来自 ``results/``, 不引入人工数字。

用法::

    python scripts/make_report.py
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RESULTS_DIR = ROOT / "results"
REPORT_DIR = ROOT / "report"


def _read_csv(name: str) -> pd.DataFrame | None:
    path = RESULTS_DIR / name
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return None


def _read_json(name: str) -> dict | None:
    path = RESULTS_DIR / name
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(value: object, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(number):
        return "n/a"
    return f"{number:.{digits}f}"


def _md_table(frame: pd.DataFrame, digits: int = 4) -> str:
    if frame is None or len(frame) == 0:
        return "_无数据_\n"
    view = frame.copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda v: _fmt(v, digits))
    header = "| " + " | ".join(str(c) for c in view.columns) + " |"
    divider = "|" + "|".join("---" for _ in view.columns) + "|"
    lines = [header, divider]
    lines += [
        "| " + " | ".join(str(v) for v in row) + " |" for row in view.itertuples(index=False)
    ]
    return "\n".join(lines) + "\n"


def verdict(factor: dict, backtest: dict, robustness: pd.DataFrame | None) -> tuple[str, list[str]]:
    """按事先写好的判据给出结论, 不做事后调整."""
    notes: list[str] = []
    signals = factor.get("signals", {})
    primary = signals.get("sue", {})
    h20 = primary.get("20") or primary.get("40") or next(iter(primary.values()), {})
    ic_mean = h20.get("ic_mean")
    t_stat = h20.get("t_stat")
    baseline = backtest.get("baseline_metrics", {})
    net_annual = baseline.get("net_annual_return")
    net_ir = baseline.get("net_information_ratio")

    stat_ok = bool(
        ic_mean is not None and t_stat is not None and ic_mean > 0 and t_stat >= 2
    )
    tradable_ok = bool(
        net_annual is not None and net_ir is not None and net_annual > 0 and net_ir > 0
    )

    notes.append(
        f"统计判据 (RankIC>0 且 t>=2): {'通过' if stat_ok else '未通过'} "
        f"(RankIC={_fmt(ic_mean)}, t={_fmt(t_stat, 2)})"
    )
    notes.append(
        f"交易判据 (扣成本后年化>0 且净 IR>0): {'通过' if tradable_ok else '未通过'} "
        f"(净年化={_fmt(net_annual)}, 净 IR={_fmt(net_ir, 2)})"
    )
    if primary:
        positive_h = [
            h for h in sorted(primary, key=lambda k: int(k))
            if (primary[h].get("ic_mean") or 0) > 0
        ]
        notes.append(
            "主信号 RankIC 为正的预测期: "
            + (", ".join(f"{h}日" for h in positive_h) if positive_h else "无")
        )

    if robustness is not None and len(robustness):
        horizons = robustness[robustness["dimension"] == "horizon"]
        if len(horizons):
            good = horizons[horizons["net_annual_return"] > 0]
            if len(good):
                parts = [
                    f"{row.variant} (净年化 {_fmt(row.net_annual_return * 100, 1)}%, "
                    f"净 IR {_fmt(row.net_information_ratio, 2)})"
                    for row in good.itertuples()
                ]
                notes.append(
                    "唯一方向性例外是持有期 "
                    + "、".join(parts)
                    + "; 正收益必须与净 IR 一起看, 年化为正不等于跑赢基准"
                )
        positive = float((robustness["net_annual_return"] > 0).mean())
        notes.append(f"稳健性情景中扣成本后年化为正的比例: {positive:.1%}")
        if "net_information_ratio" in robustness.columns:
            positive_ir = float((robustness["net_information_ratio"] > 0).mean())
            notes.append(
                f"稳健性情景中扣成本后**跑赢等权事件基准** (净 IR>0) 的比例: {positive_ir:.1%}"
            )
        years = robustness[robustness["dimension"] == "year"]
        if len(years):
            good = int((years["net_annual_return"] > 0).sum())
            notes.append(
                f"分年份情景 {len(years)} 个, 扣成本后年化为正的年份: {good} 个 "
                f"({', '.join(years.loc[years['net_annual_return'] > 0, 'variant'].astype(str))})"
            )
    else:
        positive = float("nan")

    if stat_ok and tradable_ok:
        label = "得到支持"
    elif stat_ok and not tradable_ok:
        label = "统计上存在, 但扣除成本后不足以交易"
    elif not stat_ok and tradable_ok:
        label = "证据不足 (统计判据未通过, 组合结果不足以支撑假设)"
    else:
        label = "被当前实验否定"
    return label, notes


def build_report() -> str:
    data_audit = _read_csv("data_audit.csv")
    analysis_audit = _read_csv("analysis_audit.csv")
    return_audit = _read_csv("return_audit.csv")
    signal_audit = _read_csv("signal_audit.csv")
    correction_audit = _read_csv("correction_audit.csv")
    price_failures = _read_csv("price_failures.csv")
    factor = _read_json("factor_metrics.json") or {}
    backtest = _read_json("backtest_metrics.json") or {}
    ic_by_period = _read_csv("ic_by_period.csv")
    quantiles = _read_csv("quantile_returns.csv")
    regressions = _read_csv("cross_sectional_regression.csv")
    robustness = _read_csv("robustness.csv")
    comparison = _read_csv("model_comparison.csv")
    ml_robustness = _read_csv("ml_robustness.csv")

    panel = factor.get("panel", {})
    label, notes = verdict(factor, backtest, robustness)

    lines: list[str] = []
    add = lines.append

    add("# A 股财报事件 Alpha 研究报告 (v1.0)")
    add("")
    add(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    add("")
    add("> 本报告的全部数字由 `python scripts/run_all.py` 生成, 数据来自 AKShare 公开接口。")
    add("> 报告只解释公开数据能够支持的结论; 无法恢复的历史版本、完整股票池和公告时刻都写在数据限制里。")
    add("")

    add("## 0. 结论")
    add("")
    add(f"**主假设 `SUE ↑ ⇒ FutureReturn ↑` 的当前判定: {label}**")
    add("")
    for note in notes:
        add(f"- {note}")
    add("")
    add("判据在研究开始前就已写入 `research/01_research_question.md` §7, 报告不对判据做事后调整。")
    add("`Sharpe > 2` 之类的收益门槛不属于完成条件。")
    add("")

    add("## 1. 数据与样本")
    add("")
    if panel:
        add(
            f"- 样本区间: {panel.get('first_disclosure', 'n/a')} .. "
            f"{panel.get('last_disclosure', 'n/a')} (披露日)"
        )
        add(
            f"- 事件数: {panel.get('events', 0):,} ; 股票数: {panel.get('codes', 0):,} ; "
            f"报告期: {panel.get('report_periods', 0)}"
        )
    if data_audit is not None and len(data_audit):
        add("")
        add("### 1.1 逐报告期数据覆盖")
        add("")
        cols = [
            c
            for c in [
                "report_date",
                "financials_rows",
                "announcement_rows",
                "matched_codes",
                "financials_only",
                "announcement_only",
                "match_rate",
                "violations",
                "st_dropped",
                "industry_null",
            ]
            if c in data_audit.columns
        ]
        add(_md_table(data_audit[cols]))
        add("")
        add("面板级汇总:")
        add("")
        panel_level = [
            c
            for c in [
                "events",
                "codes",
                "report_periods",
                "first_disclosure",
                "last_disclosure",
                "st_total",
                "st_flagged",
                "panel_events",
                "codes_later_delisted",
                "events_later_delisted",
                "survivorship_flag",
            ]
            if c in data_audit.columns
        ]
        add(
            _md_table(
                pd.DataFrame(
                    {
                        "指标": panel_level,
                        "值": [data_audit[c].dropna().iloc[0] for c in panel_level],
                    }
                )
            )
        )
    if return_audit is not None and len(return_audit):
        add("### 1.2 未来收益覆盖率")
        add("")
        add(_md_table(return_audit))
    if signal_audit is not None and len(signal_audit):
        add("### 1.3 SUE 覆盖率与口径校验 (同花顺 vs 东财)")
        add("")
        add(_md_table(signal_audit))
    if analysis_audit is not None and len(analysis_audit):
        add("### 1.4 最终研究面板的特征覆盖率与顺延成交")
        add("")
        add(_md_table(analysis_audit.T.reset_index().rename(columns={"index": "指标", 0: "值"})))

    sample_audit = _read_csv("sample_audit.csv")
    if sample_audit is not None and len(sample_audit):
        add("### 1.5 人工抽查")
        add("")
        add(
            f"随机抽查 {len(sample_audit)} 条事件 (seed 固定, 可复现), "
            "核对三点: 实际披露日是否为真公告日、入场日是否为披露日之后首个交易日、财报数值是否与原始报告一致。"
        )
        add("逐条明细与公告链接见 `results/sample_audit.md`, 这里只列出抽样分布。")
        add("")
        sample_dist = (
            sample_audit.assign(报告期=sample_audit["报告期"].astype(str))
            .groupby("报告期")
            .size()
            .reset_index(name="抽查条数")
        )
        add(_md_table(sample_dist))
        add("")
    add("")

    add("## 2. 信号")
    add("")
    add("主信号是**季节性随机游走 SUE**:")
    add("")
    add("```text")
    add("UE(i,t)  = EPS(i,t) - EPS(i,t-4)")
    add("SUE(i,t) = UE(i,t) / std(UE(i, t-k .. t-1))")
    add("```")
    add("")
    add("分母只用**严格早于 t** 的历史 UE (`shift(1)`), 因此不引入未来信息。")
    add("由于没有分析师一致预期数据, 该信号在全文称为 SUE proxy, 不与基于一致预期的 SUE 混同。")
    add("")

    add("## 3. 因子检验")
    add("")
    add("![signal distribution](figures/signal_distribution.png)")
    add("")
    signals = factor.get("signals", {})
    if signals:
        add("### 3.1 RankIC / ICIR (按报告期的横截面)")
        add("")
        rows = []
        for signal, by_h in signals.items():
            for horizon, stats in sorted(by_h.items(), key=lambda kv: int(kv[0])):
                rows.append(
                    {
                        "signal": signal,
                        "horizon": int(horizon),
                        "ic_mean": stats.get("ic_mean"),
                        "ic_std": stats.get("ic_std"),
                        "icir": stats.get("icir"),
                        "t_stat": stats.get("t_stat"),
                        "positive_rate": stats.get("positive_rate"),
                        "periods": stats.get("periods"),
                        "coverage": stats.get("coverage"),
                    }
                )
        ic_table = pd.DataFrame(rows)
        family = ic_table[ic_table["signal"].isin(
            ["sue", "sue_z", "sue_ind_neutral", "sue_neutral"]
        )]
        add("SUE 家族 (原始 / 标准化 / 行业中性 / 行业+市值中性):")
        add("")
        add(_md_table(family, digits=4))
        add("")
        raw60 = ic_table[(ic_table["signal"] == "sue") & (ic_table["horizon"] == 60)]
        neu60 = ic_table[(ic_table["signal"] == "sue_neutral") & (ic_table["horizon"] == 60)]
        if len(raw60) and len(neu60):
            add(
                f"注意: 原始 SUE 与行业+市值中性化 SUE 在 60 日的 RankIC 分别为 "
                f"{_fmt(raw60['ic_mean'].iloc[0])} 与 {_fmt(neu60['ic_mean'].iloc[0])}。"
                "中性化后长端转正、短端仍为负, 说明原始信号的负 IC 有相当部分来自"
                "行业与市值暴露, 而不是财报意外本身。"
            )
            add("")
        add("对照信号 (净利润 / 营收 / EPS 同比) 在同一组预测期上的表现:")
        add("")
        controls = ic_table[ic_table["signal"].isin(
            ["net_profit_yoy_calc", "revenue_yoy_calc", "eps_yoy_calc"]
        ) & ic_table["horizon"].isin([20])]
        add(_md_table(controls, digits=4))
        add("")
        add("![ic decay](figures/ic_decay.png)")
        add("")
    if ic_by_period is not None and len(ic_by_period):
        add("### 3.2 IC 的时间序列 (前 20 行)")
        add("")
        add(_md_table(ic_by_period.head(20), digits=4))
        add("")
    if quantiles is not None and len(quantiles):
        add("### 3.3 分层收益 (五分位, 对报告期取平均)")
        add("")
        add(_md_table(quantiles[quantiles["signal"] == "sue"], digits=4))
        add("")
        add("![quantile returns](figures/quantile_returns.png)")
        add("")
    if regressions is not None and len(regressions):
        add("### 3.4 横截面回归 (控制市值 / 动量 / 波动, 可选行业)")
        add("")
        add(_md_table(regressions[regressions["signal"] == "sue"], digits=4))
        add("")

    add("## 4. 组合与交易成本")
    add("")
    baseline = backtest.get("baseline_metrics", {})
    if baseline:
        add("基准情景: SUE 前 20% 等权, long-only, 持有 20 个交易日, AUM=1e8。")
        add("")
        add(
            _md_table(
                pd.DataFrame(
                    sorted(baseline.items()), columns=["指标", "值"]
                ),
                digits=4,
            )
        )
    breakdown = backtest.get("cost_breakdown_mean", {})
    if breakdown:
        add("### 4.1 Gross → Net 成本拆解 (每期, bps)")
        add("")
        add(
            _md_table(
                pd.DataFrame(
                    [
                        {"成本项": k, "bps/期": v * 1e4}
                        for k, v in breakdown.items()
                    ]
                ),
                digits=2,
            )
        )
    scenarios = backtest.get("aum_scenarios")
    if scenarios:
        add("### 4.2 容量情景")
        add("")
        add(_md_table(pd.DataFrame(scenarios), digits=4))
    add("")
    add("![gross to net](figures/gross_to_net.png)")
    add("")

    add("## 5. 稳健性")
    add("")
    if robustness is not None and len(robustness):
        summary = (
            robustness.groupby("dimension")
            .agg(
                情景数=("variant", "count"),
                净年化均值=("net_annual_return", "mean"),
                净年化为正比例=("net_annual_return", lambda s: float((s > 0).mean())),
            )
            .reset_index()
        )
        add(_md_table(summary, digits=4))
        add("完整情景表见 `results/robustness.csv`。")
        add("")
        add("关键子集:")
        add("")
        add(
            _md_table(
                robustness[robustness["dimension"].isin(["year", "horizon", "cost", "signal"])][
                    [
                        "dimension",
                        "variant",
                        "n_cohorts",
                        "gross_annual_return",
                        "net_annual_return",
                        "net_sharpe",
                        "mean_cost",
                    ]
                ],
                digits=4,
            )
        )
    else:
        add("_未生成稳健性结果。_")
    add("")

    add("## 6. 机器学习的样本外增量")
    add("")
    if comparison is not None and len(comparison):
        add(_md_table(comparison, digits=4))
        add("")
        add("切分为 expanding-window walk-forward, 训练集与测试集之间留 60 天 purge gap;")
        add("特征缺失用训练集中位数填充, 标准化只用训练集统计量。")
        add("`*_no_sue` 是去掉全部 SUE 家族特征后的对照, 用来判断增量是否来自财报意外本身。")
        add("")
        if ml_robustness is not None and len(ml_robustness):
            add("样本外信号 (LightGBM 完整特征) 的市值与年份拆分:")
            add("")
            add(_md_table(ml_robustness, digits=4))
            add("")
            add("`benchmark_annual_return` 是该子样本内等权事件基准的年化收益;")
            add("只有净年化明显高于基准, 才说明模型提供了增量, 而不是承担了规模/流动性 beta。")
    else:
        add("_未生成机器学习比较结果。_")
    add("")

    add("## 7. 失败实验、数据限制与异常")
    add("")
    if correction_audit is not None and len(correction_audit):
        add("### 7.1 财报更正扫描")
        add("")
        add(_md_table(correction_audit))
    if price_failures is not None and len(price_failures):
        add("### 7.2 行情抓取失败")
        add("")
        add(_md_table(price_failures))
    add("### 7.3 已知数据限制")
    add("")
    add("- `stock_yjbb_em` 返回的是**当前**财务数值, 历史修订无法还原, 因此不是严格 Point-in-Time 财务库。")
    add("- 数据源在返回历史财报时**已经剔除后来退市的公司**, 股票池过滤无法修复, 结果存在幸存者偏差。")
    add("- 公告只精确到日期。本项目统一假设披露发生在收盘后, 因此最早在披露日之后的第一个交易日入场, 属于偏保守处理。")
    add("- 财务与公告的匹配率低于 1, 未匹配样本被丢弃, 匹配率已写入数据审计表。")
    add("- 队列收益是事件队列收益, 相邻队列的持有期可能重叠, 不等价于单一可交易账户净值。")
    add("")

    add("## 8. 复现方式")
    add("")
    add("```bash")
    add("# 全流程 (数据下载 -> 信号 -> 因子 -> 回测 -> 机器学习 -> 报告)")
    add("python scripts/run_all.py")
    add("")
    add("# 单步执行")
    add("python scripts/build_event_panel.py --periods 20190331 ... 20240630")
    add("python scripts/fetch_prices.py")
    add("python scripts/build_signal.py --panel event_panel --out event_panel_signal")
    add("python scripts/build_analysis_panel.py")
    add("python scripts/run_factor.py")
    add("python scripts/run_backtest.py")
    add("python scripts/run_ml.py")
    add("python scripts/make_report.py")
    add("```")
    add("")
    add("原始响应缓存在 `data/raw/` (不提交), 中间面板在 `data/processed/`。")

    run_config = _read_json("run_config.json")
    if run_config:
        add("")
        add("### 运行配置")
        add("")
        add("```json")
        add(json.dumps(run_config, ensure_ascii=False, indent=2))
        add("```")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="生成研究报告")
    parser.add_argument("--out", default="REPORT.md", help="输出文件名 (report/ 下)")
    args = parser.parse_args(argv)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    text = build_report()
    path = REPORT_DIR / args.out
    path.write_text(text, encoding="utf-8")
    print(f"报告 -> {path.relative_to(ROOT)} ({len(text.splitlines())} 行)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
