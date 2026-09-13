#!/usr/bin/env python
"""机器学习增量检验: Ridge 与 LightGBM 的 walk-forward 样本外比较.

切分方式与基线一致: 按报告期做 expanding-window, 训练集与测试集之间留
``purge_days`` 天的间隔, 避免持有期重叠造成的信息泄漏。特征缺失用**训练集**
中位数填充并只用训练集统计量标准化 (不用全样本, 否则是未来信息)。

为了判断增量到底来自哪里, 除完整特征集外还跑一个**去掉 SUE 家族特征**的
对照, 并对样本外信号做市值 / 年份拆分。

用法::

    python scripts/run_ml.py

产出::

    results/model_comparison.csv
    results/ml_robustness.csv
    results/ml_predictions.parquet
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

from backtest.engine import CostModel, EventBacktester  # noqa: E402
from data.panel import load_panel  # noqa: E402
from factor.ic import ic_by_date, icir, t_stat  # noqa: E402
from factor.preprocess import time_series_split  # noqa: E402
from ml.lightgbm import fit_gbm  # noqa: E402
from ml.linear import fit_linear  # noqa: E402
from research.config import CONFIG  # noqa: E402

RESULTS_DIR = ROOT / "results"

BASE_COST = CostModel(
    commission_rate=CONFIG.commission_rate,
    stamp_tax_rate=CONFIG.stamp_tax_rate,
    half_spread=CONFIG.half_spread,
    slippage_impact_coeff=CONFIG.slippage_impact_coeff,
    impact_coeff=CONFIG.impact_coeff,
    participation_cap=CONFIG.participation_cap,
    aum=CONFIG.base_aum,
)

HORIZON = 20

METRIC_KEYS = (
    "n_cohorts",
    "gross_annual_return",
    "net_annual_return",
    "benchmark_annual_return",
    "gross_sharpe",
    "net_sharpe",
    "net_max_drawdown",
    "information_ratio",
    "net_information_ratio",
    "hit_rate_active",
    "mean_cost",
    "mean_turnover",
    "capacity_median",
)

PREDICTION_COLUMNS = ("pred_ridge", "pred_gbm")


def _return_col(panel: pd.DataFrame) -> str:
    adjusted = f"fwd_ret_adj_{HORIZON}d"
    return adjusted if adjusted in panel.columns else f"fwd_ret_{HORIZON}d"


def walk_forward(
    panel: pd.DataFrame,
    feature_cols: list[str],
    *,
    target_col: str,
    date_col: str = "report_date",
) -> pd.DataFrame:
    """返回样本外预测 (逐折拼接)."""
    folds = time_series_split(
        panel,
        n_splits=CONFIG.ml_splits,
        date_col=date_col,
        purge_days=CONFIG.ml_purge_days,
        min_train_frac=CONFIG.ml_min_train_frac,
    )
    if not folds:
        raise ValueError("无法构造 walk-forward 折, 样本期太短")

    out: list[pd.DataFrame] = []
    for fold, (train, test) in enumerate(folds, start=1):
        train = train.dropna(subset=[target_col])
        test = test.dropna(subset=[target_col])
        if len(train) < 50 or len(test) == 0:
            continue

        X_train = train[feature_cols].astype("float64")
        X_test = test[feature_cols].astype("float64")
        medians = X_train.median()
        X_train = X_train.fillna(medians)
        X_test = X_test.fillna(medians)

        # 线性模型的惩罚项对量纲敏感, 用训练集统计量做标准化 (测试集不得参与)
        center = X_train.mean()
        scale = X_train.std(ddof=1).replace(0.0, 1.0)
        X_train = (X_train - center) / scale
        X_test = (X_test - center) / scale

        y_train = pd.to_numeric(train[target_col], errors="coerce").astype("float64")
        mask = y_train.notna()
        X_train, y_train = X_train[mask], y_train[mask]

        ridge = fit_linear(X_train, y_train, alpha=1.0)
        gbm = fit_gbm(X_train, y_train, model=CONFIG.ml_model)

        block = test[["code", "report_date", target_col]].copy()
        block["fold"] = fold
        block["pred_ridge"] = ridge.predict(X_test)
        block["pred_gbm"] = gbm.predict(X_test)
        out.append(block)
        print(
            f"      折 {fold}: train={len(X_train):6d} test={len(test):6d} "
            f"期={sorted(test[date_col].astype(str).unique())}"
        )

    if not out:
        raise ValueError("所有折都被跳过")
    return pd.concat(out, ignore_index=True)


def _score(pred: pd.Series, target: pd.Series, dates: pd.Series) -> dict[str, float]:
    data = pd.DataFrame({"signal": pred, "ret": target, "date": dates}).dropna()
    if len(data) == 0:
        return {}
    ics = ic_by_date(
        data,
        signal_col="signal",
        return_col="ret",
        date_col="date",
        method=CONFIG.ic_method,
        min_obs=CONFIG.min_ic_obs,
    )
    return {
        "rank_ic": float(ics.mean()) if len(ics) else float("nan"),
        "icir": icir(ics),
        "t_stat": t_stat(ics),
        "n_periods": float(len(ics)),
        "positive_rate": float((ics > 0).mean()) if len(ics) else float("nan"),
    }


def _backtest(
    panel: pd.DataFrame, signal: pd.Series, *, return_col: str, horizon: int = HORIZON
) -> dict[str, float]:
    frame = panel.copy()
    frame["_signal"] = np.asarray(signal, dtype="float64")
    res = EventBacktester(
        signal_col="_signal",
        horizon=horizon,
        top_pct=CONFIG.top_pct,
        cost=BASE_COST,
        min_names=CONFIG.min_names,
        return_col=return_col,
    ).run(frame)
    return res.metrics


def _merge_predictions(
    merged: pd.DataFrame, preds: pd.DataFrame, suffix: str
) -> pd.DataFrame:
    renamed = preds[["code", "report_date", *PREDICTION_COLUMNS]].rename(
        columns={c: f"{c}{suffix}" for c in PREDICTION_COLUMNS}
    )
    return merged.merge(renamed, on=["code", "report_date"], how="left")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="机器学习增量检验")
    parser.add_argument("--panel", default="analysis_panel")
    args = parser.parse_args(argv)

    pd.set_option("display.width", 240)
    panel = load_panel(args.panel)
    target_col = _return_col(panel)
    panel = panel.dropna(subset=[target_col]).reset_index(drop=True)
    feature_cols = [c for c in CONFIG.ml_features if c in panel.columns]
    if not feature_cols:
        print("!! 没有任何可用特征列")
        return 1
    no_sue = [c for c in feature_cols if not c.startswith("sue")]
    print(f"[1/5] 面板 {len(panel)} 事件, 完整特征 {len(feature_cols)} 个")
    print(f"      去掉 SUE 家族后的特征: {no_sue}")

    print("[2/5] walk-forward 训练与预测 (完整特征集) ...")
    preds_all = walk_forward(panel, feature_cols, target_col=target_col)
    print("[3/5] walk-forward 训练与预测 (去掉 SUE 家族) ...")
    preds_no_sue = walk_forward(panel, no_sue, target_col=target_col)

    merged = preds_all.merge(
        panel[
            [
                "code",
                "report_date",
                "tradable_ts",
                "adv20",
                "volatility_20d",
                "log_mcap",
                "sue",
            ]
        ],
        on=["code", "report_date"],
        how="left",
    )
    merged = _merge_predictions(merged, preds_no_sue, "_nosue")

    print("[4/5] 样本外评分与组合回测 ...")
    candidates = [
        ("linear_ridge", "pred_ridge", "完整特征"),
        (f"gbm_{CONFIG.ml_model}", "pred_gbm", "完整特征"),
        ("linear_ridge_no_sue", "pred_ridge_nosue", "去掉 SUE 家族"),
        (f"gbm_{CONFIG.ml_model}_no_sue", "pred_gbm_nosue", "去掉 SUE 家族"),
    ]
    rows: list[dict[str, object]] = []
    for model_name, pred_col, feature_desc in candidates:
        signal = merged[pred_col]
        score = _score(signal, merged[target_col], merged["report_date"])
        row: dict[str, object] = {"model": model_name, "features": feature_desc, **score}
        try:
            metrics = _backtest(merged, signal, return_col=target_col)
            row.update({k: metrics.get(k, np.nan) for k in METRIC_KEYS})
        except ValueError as exc:
            print(f"      组合回测跳过 {model_name}: {exc}")
        rows.append(row)
        print(
            f"      {model_name:26s} RankIC={score.get('rank_ic', float('nan')):.4f} "
            f"ICIR={score.get('icir', float('nan')):.3f}"
        )

    score = _score(merged["sue"], merged[target_col], merged["report_date"])
    row = {"model": "baseline_sue", "features": "仅 SUE", **score}
    try:
        metrics = _backtest(merged, merged["sue"], return_col=target_col)
        row.update({k: metrics.get(k, np.nan) for k in METRIC_KEYS})
    except ValueError as exc:
        print(f"      组合回测跳过 baseline_sue: {exc}")
    rows.append(row)

    comparison = pd.DataFrame(rows)
    comparison.to_csv(
        RESULTS_DIR / "model_comparison.csv", index=False, encoding="utf-8-sig"
    )

    print("[5/5] 样本外信号的市值 / 年份拆分 (LightGBM 完整特征) ...")
    best_col = "pred_gbm"
    robust_rows: list[dict[str, object]] = []
    tercile = pd.qcut(
        pd.to_numeric(merged["log_mcap"], errors="coerce"),
        3,
        labels=["low", "mid", "high"],
        duplicates="drop",
    )
    subsets: list[tuple[str, str, pd.DataFrame]] = [
        ("size", name, merged[tercile == name]) for name in ["low", "mid", "high"]
    ]
    years = pd.to_datetime(merged["tradable_ts"]).dt.year
    subsets += [
        ("year", str(year), subset) for year, subset in merged.groupby(years)
    ]
    for dimension, variant, subset in subsets:
        if len(subset) < 500:
            continue
        try:
            metrics = _backtest(subset, subset[best_col], return_col=target_col)
        except ValueError:
            continue
        robust_rows.append(
            {
                "dimension": dimension,
                "variant": variant,
                **{k: metrics.get(k, np.nan) for k in METRIC_KEYS},
            }
        )

    ml_robustness = pd.DataFrame(robust_rows)
    ml_robustness.to_csv(
        RESULTS_DIR / "ml_robustness.csv", index=False, encoding="utf-8-sig"
    )

    merged.to_parquet(RESULTS_DIR / "ml_predictions.parquet", index=False)

    payload = {
        "config": CONFIG.fingerprint(),
        "target": target_col,
        "features_all": feature_cols,
        "features_no_sue": no_sue,
        "n_folds": int(preds_all["fold"].nunique()),
        "oos_events": int(len(preds_all)),
    }
    (RESULTS_DIR / "ml_metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(comparison.to_string(index=False))
    if len(ml_robustness):
        print("\n样本外信号的拆分检验 (LightGBM 完整特征):")
        print(ml_robustness.to_string(index=False))
    print("      结果 -> results/model_comparison.csv, ml_robustness.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
