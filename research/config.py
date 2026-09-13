"""v1.0 研究配置 (单一事实来源).

所有脚本都从这里取参数, 保证同一条命令生成的全部表格与图使用同一套口径。
配置会被写进 ``results/run_config.json``, 结果表里因此可以追溯"这批数字是
用什么参数跑出来的"。
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

#: 样本报告期 (2019Q1 – 2024Q2), 由 ``scripts/probe`` 实测可得性后固定
REPORT_PERIODS: tuple[str, ...] = (
    "20190331", "20190630", "20190930", "20191231",
    "20200331", "20200630", "20200930", "20201231",
    "20210331", "20210630", "20210930", "20211231",
    "20220331", "20220630", "20220930", "20221231",
    "20230331", "20230630", "20230930", "20231231",
    "20240331", "20240630",
)

#: 未来收益窗口 (交易日) —— 覆盖 IC Decay 所需的短/长端
HORIZONS: tuple[int, ...] = (1, 5, 10, 20, 40, 60)

#: 主信号
PRIMARY_SIGNAL = "sue"
#: 对照信号 (无分析师预期数据时的代理变量)
CONTROL_SIGNALS: tuple[str, ...] = (
    "net_profit_yoy_calc",
    "revenue_yoy_calc",
    "eps_yoy_calc",
)
#: 机器学习特征
ML_FEATURES: tuple[str, ...] = (
    "sue",
    "sue_neutral",
    "net_profit_yoy_calc",
    "revenue_yoy_calc",
    "eps_yoy_calc",
    "roe_change",
    "momentum_20d",
    "volatility_20d",
    "log_adv20",
    "log_mcap",
)


@dataclass(frozen=True)
class ResearchConfig:
    """一次完整运行的全部参数."""

    periods: tuple[str, ...] = REPORT_PERIODS
    horizons: tuple[int, ...] = HORIZONS
    primary_signal: str = PRIMARY_SIGNAL
    control_signals: tuple[str, ...] = CONTROL_SIGNALS
    ml_features: tuple[str, ...] = ML_FEATURES

    # 数据时点
    entry_lag: int = 1
    max_shift: int = 10
    st_policy: str = "drop"
    min_adv: float = 0.0

    # 因子检验
    n_quantiles: int = 5
    ic_method: str = "spearman"
    min_ic_obs: int = 20

    # 组合
    top_pct: float = 0.2
    min_names: int = 5
    max_weight: float = 0.05
    industry_cap: float = 0.30

    # 成本
    commission_rate: float = 0.0003
    stamp_tax_rate: float = 0.0005
    half_spread: float = 0.0005
    slippage_impact_coeff: float = 0.05
    impact_coeff: float = 0.5
    participation_cap: float = 0.10
    base_aum: float = 1.0e8
    aum_scenarios: tuple[float, ...] = (1.0e7, 1.0e8, 1.0e9)

    # 机器学习
    ml_splits: int = 5
    ml_min_train_frac: float = 0.15
    ml_purge_days: int = 60
    ml_model: str = "lightgbm"

    random_seed: int = 42
    extra: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2, default=str)

    def fingerprint(self) -> dict:
        """写入结果目录的运行指纹."""
        return asdict(self)


CONFIG = ResearchConfig()
