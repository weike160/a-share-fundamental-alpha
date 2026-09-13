#!/usr/bin/env python
"""组装最终研究面板 (数据层的最后一步).

把三样东西并到事件面板上::

    信号      SUE (同花顺 EPS) + 同比对照信号 + 截面标准化/中性化
    特征      入场前最后一个交易日的 ADV / 波动 / 动量 / 流通市值
    可交易性  停牌与涨跌停导致的顺延入场/出场, 以及调整后的未来收益

用法::

    python scripts/build_analysis_panel.py
    python scripts/build_analysis_panel.py --panel event_panel_returns --out analysis_panel

产出::

    data/processed/analysis_panel.parquet
    results/analysis_audit.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.fills import audit_fills, resolve_fills
from data.liquidity import attach_price_features, build_price_features
from data.panel import load_panel, save_panel
from data.tradability import TradingCalendar
from factor.neutralize import (
    industry_neutralize,
    neutralize_cross_section,
)
from factor.preprocess import winsorize
from research.config import CONFIG
from signals.earnings_growth import earnings_growth, roe_change

RESULTS_DIR = ROOT / "results"


def _add_cross_sectional_signals(
    panel: pd.DataFrame, *, group_col: str = "report_date"
) -> pd.DataFrame:
    """按报告期做横截面去极值、标准化与中性化, 生成派生信号列."""
    out = panel.copy()
    for col in ("sue_w", "sue_z", "sue_ind_neutral", "sue_neutral"):
        out[col] = np.nan

    for _period, grp in out.groupby(group_col, sort=True):
        sue_raw = pd.to_numeric(grp["sue"], errors="coerce")
        valid = sue_raw.notna()
        if valid.sum() < CONFIG.min_ic_obs:
            continue

        # 按报告期去极值 (1%/99%), 后续标准化与中性化都基于去极值后的信号
        sue = winsorize(sue_raw, 0.01, 0.99)
        out.loc[grp.index, "sue_w"] = sue

        # 横截面 z-score
        std = sue[valid].std(ddof=1)
        z = (sue - sue[valid].mean()) / std if std and std > 0 else sue * np.nan
        out.loc[grp.index, "sue_z"] = z

        industry = grp.get("industry")
        if industry is not None:
            out.loc[grp.index, "sue_ind_neutral"] = industry_neutralize(sue, industry)

        # 行业 + 市值中性化 (市值用入场前流通市值的对数)
        exposures = pd.DataFrame(index=grp.index)
        if "log_mcap" in grp.columns:
            exposures["log_mcap"] = pd.to_numeric(grp["log_mcap"], errors="coerce")
        if industry is not None:
            dummies = pd.get_dummies(
                industry.astype("string").fillna("未知"), prefix="ind", drop_first=True
            ).astype("float64")
            exposures = pd.concat([exposures, dummies], axis=1)
        if exposures.shape[1] == 0:
            continue
        out.loc[grp.index, "sue_neutral"] = neutralize_cross_section(sue, exposures)

    return out


def _coverage(panel: pd.DataFrame, columns: list[str]) -> dict[str, float]:
    row: dict[str, float] = {"panel_events": float(len(panel))}
    for col in columns:
        if col in panel.columns:
            row[f"{col}_coverage"] = float(
                pd.to_numeric(panel[col], errors="coerce").notna().mean()
            )
        else:
            row[f"{col}_coverage"] = 0.0
    return row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="组装研究面板")
    parser.add_argument("--panel", default="event_panel_returns", help="含未来收益的面板")
    parser.add_argument("--signal-panel", default="event_panel_signal", help="含 sue 的面板")
    parser.add_argument("--prices", default="prices", help="日频行情面板名")
    parser.add_argument("--out", default="analysis_panel", help="输出面板名")
    parser.add_argument("--max-shift", type=int, default=CONFIG.max_shift, help="顺延上限")
    args = parser.parse_args(argv)

    pd.set_option("display.width", 240)

    print("[1/6] 读取面板 ...")
    panel = load_panel(args.panel)
    signal_panel = load_panel(args.signal_panel)
    prices = load_panel(args.prices)
    print(f"      面板 {len(panel)} 事件, 行情 {len(prices)} 行")

    print("[2/6] 合并 SUE ...")
    sue_cols = ["code", "report_date", "sue"]
    sue = signal_panel[sue_cols].copy()
    sue["code"] = sue["code"].astype(str).str.zfill(6)
    panel["code"] = panel["code"].astype(str).str.zfill(6)
    panel = panel.merge(sue, on=["code", "report_date"], how="left", suffixes=("", "_dup"))
    if "sue_dup" in panel.columns:
        panel = panel.drop(columns=["sue_dup"])

    print("[3/6] 对照信号 (净利润/营收/EPS 同比, ROE 变化) ...")
    growth = earnings_growth(panel)
    panel = pd.concat([panel, growth], axis=1)
    panel["roe_change"] = roe_change(panel)

    print("[4/6] 入场前规模与流动性特征 ...")
    features = build_price_features(prices)
    panel = attach_price_features(panel, features)

    print("[5/6] 顺延到真实可交易时点 (停牌 / 涨跌停) ...")
    calendar = TradingCalendar.from_akshare()
    if "is_st" in panel.columns:
        st_codes = set(
            panel.loc[panel["is_st"].fillna(False).astype(bool), "code"]
            .astype(str)
            .str.zfill(6)
        )
    else:
        st_codes = set()
    panel = resolve_fills(
        panel,
        prices,
        calendar,
        horizons=CONFIG.horizons,
        st_codes=st_codes,
        max_shift=args.max_shift,
    )

    print("[6/6] 横截面标准化与中性化 ...")
    panel = _add_cross_sectional_signals(panel)

    path = save_panel(panel, args.out)
    print(f"      面板 -> {path.relative_to(ROOT)}")

    audits = [
        pd.DataFrame(
            [
                _coverage(
                    panel,
                    [
                        "sue",
                        "sue_w",
                        "sue_z",
                        "sue_neutral",
                        "net_profit_yoy_calc",
                        "revenue_yoy_calc",
                        "eps_yoy_calc",
                        "roe_change",
                        "adv20",
                        "volatility_20d",
                        "momentum_20d",
                        "float_mcap",
                        "fwd_ret_adj_20d",
                    ],
                )
            ]
        ),
        audit_fills(panel, horizons=CONFIG.horizons),
    ]
    audit = pd.concat(audits, axis=1)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    audit.to_csv(RESULTS_DIR / "analysis_audit.csv", index=False, encoding="utf-8-sig")
    print(audit.T.to_string())
    print("      审计 -> results/analysis_audit.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
