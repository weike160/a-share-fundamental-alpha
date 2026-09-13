#!/usr/bin/env python
"""long-only 事件组合回测 + 交易成本拆解 + 稳健性检验.

用法::

    python scripts/run_backtest.py

产出::

    results/backtest_cohorts.csv
    results/backtest_metrics.json
    results/robustness.csv
    results/figures/gross_to_net.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from backtest.engine import CostModel, EventBacktester  # noqa: E402
from data.fills import resolve_fills  # noqa: E402
from data.panel import load_panel  # noqa: E402
from data.tradability import TradingCalendar  # noqa: E402
from research.config import CONFIG  # noqa: E402

RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

BASE_COST = CostModel(
    commission_rate=CONFIG.commission_rate,
    stamp_tax_rate=CONFIG.stamp_tax_rate,
    half_spread=CONFIG.half_spread,
    slippage_impact_coeff=CONFIG.slippage_impact_coeff,
    impact_coeff=CONFIG.impact_coeff,
    participation_cap=CONFIG.participation_cap,
    aum=CONFIG.base_aum,
)

METRIC_KEYS = (
    "n_cohorts",
    "n_events_used",
    "n_names_mean",
    "gross_annual_return",
    "net_annual_return",
    "gross_sharpe",
    "net_sharpe",
    "net_max_drawdown",
    "benchmark_annual_return",
    "tracking_error",
    "information_ratio",
    "net_information_ratio",
    "mean_cost",
    "hit_rate_active",
    "capacity_median",
)


def add_delayed_entry(panel: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """构造「再延后一个交易日入场」的收益列 ``fwd_ret_t2_<h>d``."""
    calendar = TradingCalendar.from_akshare()
    shifted = panel.copy()
    shifted["code"] = shifted["code"].astype(str).str.zfill(6)
    shifted["tradable_ts"] = [
        calendar.shift(ts, 1) for ts in pd.to_datetime(shifted["tradable_ts"])
    ]
    shifted = shifted.dropna(subset=["tradable_ts"]).reset_index(drop=True)

    if "is_st" in shifted.columns:
        st_codes = set(
            shifted.loc[shifted["is_st"].fillna(False).astype(bool), "code"]
            .astype(str)
            .str.zfill(6)
        )
    else:
        st_codes = set()

    out = resolve_fills(
        shifted,
        prices,
        calendar,
        horizons=CONFIG.horizons,
        st_codes=st_codes,
        max_shift=CONFIG.max_shift,
    )
    rename = {f"fwd_ret_adj_{h}d": f"fwd_ret_t2_{h}d" for h in CONFIG.horizons}
    renamed = out[["code", "report_date", "tradable_ts", *rename]].rename(columns=rename)
    merged = panel.merge(
        renamed[["code", "report_date", *rename.values()]],
        on=["code", "report_date"],
        how="left",
    )
    return merged


def run_variant(
    panel: pd.DataFrame,
    *,
    signal: str = CONFIG.primary_signal,
    horizon: int = 20,
    top_pct: float = CONFIG.top_pct,
    cost: CostModel = BASE_COST,
    return_col: str | None = None,
    cohort_col: str | None = None,
) -> tuple[dict[str, float], object]:
    bt = EventBacktester(
        signal_col=signal,
        horizon=horizon,
        top_pct=top_pct,
        cost=cost,
        min_names=CONFIG.min_names,
        return_col=return_col,
        cohort_col=cohort_col,
    )
    res = bt.run(panel)
    return res.metrics, res


def _row(
    dimension: str,
    variant: str,
    metrics: dict[str, float],
    **extra: object,
) -> dict[str, object]:
    row: dict[str, object] = {"dimension": dimension, "variant": variant}
    row.update(extra)
    for key in METRIC_KEYS:
        row[key] = metrics.get(key, np.nan)
    return row


def figure_gross_to_net(scenarios: pd.DataFrame, cohorts: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    x = np.arange(len(scenarios))
    ax.bar(x - 0.2, scenarios["gross_annual_return"] * 100, width=0.4, label="gross")
    ax.bar(x + 0.2, scenarios["net_annual_return"] * 100, width=0.4, label="net")
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios["variant"], rotation=20, fontsize=8)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_ylabel("annualized return (%)")
    ax.set_title("Gross vs net (AUM scenarios)")
    ax.legend(fontsize=8)

    ax = axes[1]
    components = ["commission", "stamp_tax", "slippage", "impact"]
    values = [cohorts[c].mean() * 1e4 for c in components]
    ax.bar([c.replace("_", "\n") for c in components], values, color="#a5523b")
    ax.set_ylabel("mean cost per cohort (bps of NAV)")
    ax.set_title("Cost breakdown (base scenario)")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "gross_to_net.png", dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="事件组合回测")
    parser.add_argument("--panel", default="analysis_panel")
    parser.add_argument("--prices", default="prices")
    parser.add_argument("--skip-delay", action="store_true", help="跳过延后入场情景")
    args = parser.parse_args(argv)

    pd.set_option("display.width", 260)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    panel = load_panel(args.panel)
    print(f"[1/4] 面板 {len(panel)} 事件, {panel['report_date'].nunique()} 个报告期")

    rows: list[dict[str, object]] = []
    scenario_metrics: list[dict[str, object]] = []

    print("[2/4] 基准情景 (SUE, 20 日, 前 20%, AUM=1e8) ...")
    base_metrics, base_result = run_variant(panel)
    base_result.cohorts.to_csv(
        RESULTS_DIR / "backtest_cohorts.csv", encoding="utf-8-sig"
    )
    for key, value in base_metrics.items():
        print(f"      {key:28s} {value:,.4f}")
    rows.append(_row("baseline", "baseline", base_metrics,
                     signal=CONFIG.primary_signal, horizon=20, top_pct=CONFIG.top_pct,
                     aum=CONFIG.base_aum))

    print("[3/4] 稳健性情景 ...")
    # (a) AUM 容量
    for aum in CONFIG.aum_scenarios:
        m, _ = run_variant(panel, cost=BASE_COST.scaled(aum=aum))
        scenario_metrics.append(
            _row("aum", f"AUM={aum:.0e}", m, aum=aum)
        )
        rows.append(_row("aum", f"AUM={aum:.0e}", m, aum=aum))

    # (b) 双倍成本
    doubled = BASE_COST.scaled(
        commission_rate=CONFIG.commission_rate * 2,
        stamp_tax_rate=CONFIG.stamp_tax_rate * 2,
        half_spread=CONFIG.half_spread * 2,
        slippage_impact_coeff=CONFIG.slippage_impact_coeff * 2,
        impact_coeff=CONFIG.impact_coeff * 2,
    )
    m, _ = run_variant(panel, cost=doubled)
    rows.append(_row("cost", "double_cost", m, aum=CONFIG.base_aum))

    # (c) 持有期
    for horizon in CONFIG.horizons:
        m, _ = run_variant(panel, horizon=horizon)
        rows.append(_row("horizon", f"h={horizon}", m, horizon=horizon))

    # (d) 信号
    for signal in ("sue", "sue_z", "sue_ind_neutral", "sue_neutral", *CONFIG.control_signals):
        if signal not in panel.columns or not panel[signal].notna().any():
            continue
        try:
            m, _ = run_variant(panel, signal=signal)
        except ValueError as exc:
            print(f"      跳过 {signal}: {exc}")
            continue
        rows.append(_row("signal", signal, m, signal=signal))

    # (e) 选股比例
    for top_pct in (0.1, 0.2, 0.3):
        m, _ = run_variant(panel, top_pct=top_pct)
        rows.append(_row("top_pct", f"top={top_pct:.0%}", m, top_pct=top_pct))

    # (f) 延后一个交易日入场
    if not args.skip_delay:
        prices = load_panel(args.prices)
        delayed = add_delayed_entry(panel, prices)
        for horizon in (10, 20):
            col = f"fwd_ret_t2_{horizon}d"
            if col not in delayed.columns:
                continue
            try:
                m, _ = run_variant(delayed, horizon=horizon, return_col=col)
            except ValueError as exc:
                print(f"      跳过延后入场 h={horizon}: {exc}")
                continue
            rows.append(_row("entry_delay", f"T+2 h={horizon}", m, horizon=horizon))

    # (g) 分年份
    panel_year = panel.copy()
    panel_year["_year"] = pd.to_datetime(panel_year["tradable_ts"]).dt.year
    for year, grp in panel_year.groupby("_year"):
        if len(grp) < 200:
            continue
        try:
            m, _ = run_variant(grp)
        except ValueError as exc:
            print(f"      跳过 {year}: {exc}")
            continue
        rows.append(_row("year", str(year), m))

    # (h) 分市值 / 流动性
    for col, label in (("log_mcap", "size"), ("log_adv20", "liquidity")):
        if col not in panel.columns:
            continue
        values = pd.to_numeric(panel[col], errors="coerce")
        terciles = pd.qcut(values, 3, labels=["low", "mid", "high"], duplicates="drop")
        for name in ["low", "mid", "high"]:
            grp = panel[terciles == name]
            if len(grp) < 200:
                continue
            try:
                m, _ = run_variant(grp)
            except ValueError as exc:
                print(f"      跳过 {label}={name}: {exc}")
                continue
            rows.append(_row(label, name, m))

    robustness = pd.DataFrame(rows)
    robustness.to_csv(RESULTS_DIR / "robustness.csv", index=False, encoding="utf-8-sig")

    print("[4/4] 图表与指标落盘 ...")
    scenarios = pd.DataFrame(scenario_metrics)
    figure_gross_to_net(scenarios, base_result.cohorts)

    payload = {
        "config": CONFIG.fingerprint(),
        "baseline_metrics": base_metrics,
        "cost_breakdown_mean": {
            key: float(base_result.cohorts[key].mean())
            for key in ("commission", "stamp_tax", "slippage", "impact", "cost_total")
        },
        "aum_scenarios": scenarios.to_dict("records"),
        "robustness_rows": int(len(robustness)),
    }
    (RESULTS_DIR / "backtest_metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

    print("\nGross -> Net 拆解 (基准情景, 单位 bps/期):")
    for key in ("commission", "stamp_tax", "slippage", "impact", "cost_total"):
        print(f"      {key:12s} {base_result.cohorts[key].mean() * 1e4:8.2f}")
    print("\nAUM 情景:")
    print(
        scenarios[
            ["variant", "gross_annual_return", "net_annual_return", "net_sharpe", "capacity_median"]
        ].to_string(index=False)
    )
    print(f"\n稳健性表 {len(robustness)} 行 -> results/robustness.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
