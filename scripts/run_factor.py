#!/usr/bin/env python
"""因子检验: RankIC / ICIR / 分层收益 / IC Decay / 横截面回归.

用法::

    python scripts/run_factor.py

产出::

    results/factor_metrics.json
    results/ic_by_period.csv
    results/quantile_returns.csv
    results/cross_sectional_regression.csv
    results/figures/signal_distribution.png
    results/figures/ic_decay.png
    results/figures/quantile_returns.png
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

from data.panel import load_panel  # noqa: E402
from factor.ic import ic_by_date, ic_summary, icir, t_stat  # noqa: E402
from factor.neutralize import cross_sectional_regression  # noqa: E402
from factor.quantile import (  # noqa: E402
    monotonicity_check,
    quantile_returns_by_date,
)
from research.config import CONFIG  # noqa: E402

RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

#: 参与检验的信号 (主信号 + 中性化变体 + 对照信号)
SIGNALS: tuple[str, ...] = (
    "sue",
    "sue_w",
    "sue_z",
    "sue_ind_neutral",
    "sue_neutral",
    *CONFIG.control_signals,
)

#: 横截面回归的控制变量
REGRESSION_CONTROLS: tuple[str, ...] = (
    "log_mcap",
    "momentum_20d",
    "volatility_20d",
)


def return_column(panel: pd.DataFrame, horizon: int) -> str:
    """优先使用顺延可交易后的收益列."""
    adjusted = f"fwd_ret_adj_{horizon}d"
    return adjusted if adjusted in panel.columns else f"fwd_ret_{horizon}d"


def _available(panel: pd.DataFrame, columns: tuple[str, ...]) -> list[str]:
    return [c for c in columns if c in panel.columns and panel[c].notna().any()]


def ic_tables(
    panel: pd.DataFrame, group_col: str, group_type: str
) -> tuple[dict, list[dict]]:
    """逐 (信号, 预测期, 分组) 计算 IC 序列并汇总."""
    metrics: dict[str, dict] = {}
    long_rows: list[dict] = []

    for signal in _available(panel, SIGNALS):
        metrics[signal] = {}
        for horizon in CONFIG.horizons:
            ret_col = return_column(panel, horizon)
            if ret_col not in panel.columns:
                continue
            data = panel[[group_col, signal, ret_col]].rename(
                columns={signal: "signal", ret_col: "ret"}
            )
            ics = ic_by_date(
                data,
                signal_col="signal",
                return_col="ret",
                date_col=group_col,
                method=CONFIG.ic_method,
                min_obs=CONFIG.min_ic_obs,
            )
            if len(ics) == 0:
                continue

            summary = ic_summary(ics).iloc[0].to_dict()
            summary["icir"] = icir(ics)
            summary["t_stat"] = t_stat(ics)
            summary["coverage"] = float(panel[signal].notna().mean())
            metrics[signal][str(horizon)] = {
                k: (None if pd.isna(v) else float(v)) for k, v in summary.items()
            }

            # ic_by_date 内部对分组列做 pd.to_datetime, 掩码必须用同一口径
            group_dates = pd.to_datetime(panel[group_col], errors="coerce")

            for key, value in ics.items():
                label = pd.Timestamp(key).strftime("%Y-%m-%d")
                mask = group_dates.eq(pd.Timestamp(key))
                long_rows.append(
                    {
                        "signal": signal,
                        "horizon": horizon,
                        "group_type": group_type,
                        "group": label,
                        "ic": float(value),
                        "n_obs": int(
                            (mask & panel[signal].notna() & panel[ret_col].notna()).sum()
                        ),
                    }
                )
    return metrics, long_rows


def quantile_table(panel: pd.DataFrame) -> pd.DataFrame:
    """分层收益: 每个报告期内分组, 再对报告期取平均."""
    rows: list[dict] = []
    for signal in _available(panel, SIGNALS):
        for horizon in CONFIG.horizons:
            ret_col = return_column(panel, horizon)
            if ret_col not in panel.columns:
                continue
            data = panel[["report_date", signal, ret_col]].rename(
                columns={signal: "signal", ret_col: "ret"}
            ).dropna()
            if len(data) == 0:
                continue
            per_date = quantile_returns_by_date(
                data,
                signal_col="signal",
                return_col="ret",
                date_col="report_date",
                n_quantiles=CONFIG.n_quantiles,
            )
            if len(per_date) == 0:
                continue
            agg = per_date.groupby("quantile").agg(
                mean_return=("mean_return", "mean"),
                n_periods=("date", "nunique"),
                n_obs=("count", "sum"),
            )
            mono = monotonicity_check(agg[["mean_return"]])
            for quantile, row in agg.iterrows():
                rows.append(
                    {
                        "signal": signal,
                        "horizon": horizon,
                        "quantile": int(quantile),
                        "mean_return": float(row["mean_return"]),
                        "n_periods": int(row["n_periods"]),
                        "n_obs": int(row["n_obs"]),
                        "monotonic": bool(mono),
                    }
                )
    return pd.DataFrame(rows)


def regression_table(panel: pd.DataFrame) -> pd.DataFrame:
    """横截面回归: 控制市值 / 动量 / 波动 (可选行业) 后的 SUE 系数."""
    rows: list[dict] = []
    controls = _available(panel, REGRESSION_CONTROLS)

    for signal in _available(panel, SIGNALS):
        for horizon in CONFIG.horizons:
            ret_col = return_column(panel, horizon)
            if ret_col not in panel.columns:
                continue

            specs: list[tuple[str, list[str]]] = [("numeric", list(controls))]
            specs.append(("numeric+industry", list(controls)))

            for spec_name, feature_cols in specs:
                per_period: list[pd.DataFrame] = []
                for _period, grp in panel.groupby("report_date", sort=True):
                    features = pd.DataFrame(index=grp.index)
                    features[signal] = pd.to_numeric(grp[signal], errors="coerce")
                    for col in feature_cols:
                        features[col] = pd.to_numeric(grp[col], errors="coerce")
                    if spec_name.endswith("industry") and "industry" in grp.columns:
                        dummies = pd.get_dummies(
                            grp["industry"].astype("string").fillna("未知"),
                            prefix="ind",
                            drop_first=True,
                        ).astype("float64")
                        features = pd.concat([features, dummies], axis=1)
                    y = pd.to_numeric(grp[ret_col], errors="coerce")
                    if features.shape[1] < 2:
                        continue
                    res = cross_sectional_regression(y, features)
                    if len(res) == 0 or signal not in res.index:
                        continue
                    per_period.append(res.loc[[signal]].assign(period=_period))
                if not per_period:
                    continue
                stacked = pd.concat(per_period, ignore_index=True)
                rows.append(
                    {
                        "signal": signal,
                        "horizon": horizon,
                        "spec": spec_name,
                        "n_periods": int(len(stacked)),
                        "mean_coef": float(stacked["coef"].mean()),
                        "mean_t_stat": float(stacked["t_stat"].mean()),
                        "positive_t_rate": float((stacked["t_stat"] > 0).mean()),
                        "share_p_lt_05": float((stacked["p_value"] < 0.05).mean()),
                    }
                )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 图
# --------------------------------------------------------------------------
def figure_signal_distribution(panel: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sue = pd.to_numeric(panel["sue"], errors="coerce").dropna()
    axes[0].hist(sue.clip(sue.quantile(0.01), sue.quantile(0.99)), bins=60, color="#3b6ea5")
    axes[0].set_title(f"SUE distribution (n={len(sue):,})")
    axes[0].set_xlabel("SUE (clipped at 1%/99%)")
    axes[0].set_ylabel("events")

    by_period = panel.dropna(subset=["sue"]).copy()
    by_period["period"] = pd.to_datetime(by_period["report_date"]).dt.strftime("%Y-%m")
    periods = sorted(by_period["period"].unique())
    data = [by_period.loc[by_period["period"] == p, "sue"].to_numpy() for p in periods]
    axes[1].boxplot(data, tick_labels=periods, showfliers=False)
    axes[1].set_title("SUE by report period")
    axes[1].tick_params(axis="x", rotation=90, labelsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "signal_distribution.png", dpi=150)
    plt.close(fig)


def figure_ic_decay(metrics: dict) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for signal in ("sue", "sue_z", "sue_neutral", *CONFIG.control_signals):
        series = metrics.get(signal)
        if not series:
            continue
        horizons = sorted(int(h) for h in series)
        values = [series[str(h)]["ic_mean"] for h in horizons]
        ax.plot(horizons, values, marker="o", label=signal)
    ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("holding horizon (trading days)")
    ax.set_ylabel("mean RankIC")
    ax.set_title("IC decay by signal")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "ic_decay.png", dpi=150)
    plt.close(fig)


def figure_quantile_returns(quantiles: pd.DataFrame, horizon: int = 20) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, signal in zip(axes, ("sue", "sue_neutral")):
        sub = quantiles[(quantiles["signal"] == signal) & (quantiles["horizon"] == horizon)]
        if len(sub) == 0:
            ax.set_visible(False)
            continue
        sub = sub.sort_values("quantile")
        ax.bar(sub["quantile"].astype(str), sub["mean_return"] * 100, color="#4c8c4a")
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_title(f"{signal}: mean {horizon}d return by quintile")
        ax.set_xlabel("signal quintile (1 = lowest)")
        ax.set_ylabel("mean return (%)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "quantile_returns.png", dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="因子检验")
    parser.add_argument("--panel", default="analysis_panel", help="研究面板名")
    args = parser.parse_args(argv)

    pd.set_option("display.width", 240)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    panel = load_panel(args.panel)
    print(f"[1/5] 面板 {len(panel)} 事件, {panel['code'].nunique()} 只股票, "
          f"{panel['report_date'].nunique()} 个报告期")

    print("[2/5] IC / ICIR ...")
    metrics, long_rows = ic_tables(panel, "report_date", "report_period")
    monthly = panel.copy()
    monthly["_entry_month"] = (
        pd.to_datetime(monthly["tradable_ts"]).dt.to_period("M").astype(str)
    )
    _, long_rows_month = ic_tables(monthly, "_entry_month", "entry_month")
    ic_by_period = pd.DataFrame(long_rows + long_rows_month)
    ic_by_period.to_csv(
        RESULTS_DIR / "ic_by_period.csv", index=False, encoding="utf-8-sig"
    )

    print("[3/5] 分层收益 ...")
    quantiles = quantile_table(panel)
    quantiles.to_csv(
        RESULTS_DIR / "quantile_returns.csv", index=False, encoding="utf-8-sig"
    )

    print("[4/5] 横截面回归 ...")
    regressions = regression_table(panel)
    regressions.to_csv(
        RESULTS_DIR / "cross_sectional_regression.csv", index=False, encoding="utf-8-sig"
    )

    print("[5/5] 图表 ...")
    figure_signal_distribution(panel)
    figure_ic_decay(metrics)
    figure_quantile_returns(quantiles)

    payload = {
        "config": CONFIG.fingerprint(),
        "panel": {
            "events": int(len(panel)),
            "codes": int(panel["code"].nunique()),
            "report_periods": int(panel["report_date"].nunique()),
            "first_disclosure": str(panel["actual_disclosure_date"].min())[:10],
            "last_disclosure": str(panel["actual_disclosure_date"].max())[:10],
        },
        "signals": metrics,
    }
    (RESULTS_DIR / "factor_metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

    primary = metrics.get(CONFIG.primary_signal, {})
    if primary:
        table = pd.DataFrame(primary).T[
            ["ic_mean", "ic_std", "icir", "t_stat", "positive_rate", "periods", "coverage"]
        ]
        print(f"\n主信号 {CONFIG.primary_signal} 的 IC:")
        print(table.to_string())
    print("\n横截面回归 (SUE 系数):")
    sel = regressions[regressions["signal"] == CONFIG.primary_signal]
    print(sel.to_string(index=False))
    print("      结果 -> results/factor_metrics.json, ic_by_period.csv, "
          "quantile_returns.csv, cross_sectional_regression.csv, figures/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
