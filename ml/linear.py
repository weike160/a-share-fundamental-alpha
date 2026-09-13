"""机器学习的线性基线。

特征 X = [SUE, RevenueGrowth, ProfitGrowth, Momentum, Volatility,
          Liquidity, Size, Industry, MarketRegime]
预测 Y = FutureReturn (或 FutureReturnRank)。
"""
from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge

#: 首选特征顺序 (概念口径)。实际可用列由面板提供, 见 DEFAULT_FEATURES。
PREFERRED_FEATURES: list[str] = [
    "sue",
    "revenue_yoy",
    "net_profit_yoy",
    "roe",
    "gross_margin",
    "eps",
    "bps",
    "ocf_per_share",
    "momentum",
    "volatility",
    "turnover",
    "amount",
    "market_cap",
    "log_market_cap",
    "size",
    "market_regime",
]

#: 面板中可能出现、且可直接数值化的特征列 (PREFERRED_FEATURES 的并集超集)。
_PANEL_NUMERIC_COLUMNS: frozenset[str] = frozenset(PREFERRED_FEATURES) | frozenset(
    {"revenue", "net_profit", "entry_price"}
)

#: 默认特征 = 首选特征与面板列的有序交集。
#: 类别/文本列 (industry, name) 需在面板层编码, 不进入默认矩阵。
DEFAULT_FEATURES: list[str] = [
    c for c in PREFERRED_FEATURES if c in _PANEL_NUMERIC_COLUMNS
]

#: 禁止进入特征矩阵的 id / 日期 / 目标列。
_FORBIDDEN: frozenset[str] = frozenset(
    {
        "code",
        "name",
        "report_date",
        "actual_disclosure_date",
        "tradable_ts",
        "entry_date",
        "entry_price",
        "akshare_latest_notice_date",
    }
)
#: 禁止列的前缀 (exit_date_*, fwd_ret_*)。
_FORBIDDEN_PREFIXES: tuple[str, ...] = ("exit_date_", "fwd_ret_")


def _is_forbidden(col: str) -> bool:
    """判断列名是否属于 id / 日期 / 目标等禁止进入特征矩阵的列."""
    return col in _FORBIDDEN or col.startswith(_FORBIDDEN_PREFIXES)


def build_feature_matrix(
    signal: pd.DataFrame, *, feature_cols: list[str] | None = None
) -> pd.DataFrame:
    """组织特征矩阵 X。

    Parameters
    ----------
    signal:
        事件面板 (或含特征列的 DataFrame)。
    feature_cols:
        指定特征列; 为 ``None`` 时使用 :data:`DEFAULT_FEATURES`。

    Returns
    -------
    DataFrame, 仅含 ``signal`` 中实际存在的特征列, 逐列经
    ``pd.to_numeric(errors="coerce")`` 转换, 索引与 ``signal`` 一致。
    ``code`` / ``name`` / 日期列 / ``fwd_ret_*`` 等永不进入矩阵。

    Raises
    ------
    ValueError
        没有任何可用特征列时。
    """
    wanted = DEFAULT_FEATURES if feature_cols is None else list(feature_cols)
    cols = [c for c in wanted if c in signal.columns and not _is_forbidden(c)]
    if not cols:
        raise ValueError("没有可用的特征列")
    return signal[cols].apply(pd.to_numeric, errors="coerce")


def fit_linear(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    alpha: float = 0.0,
    fit_intercept: bool = True,
) -> LinearRegression | Ridge:
    """线性回归基线。

    Parameters
    ----------
    X:
        特征矩阵。
    y:
        目标收益。
    alpha:
        ``> 0`` 时使用 :class:`~sklearn.linear_model.Ridge`, 否则普通最小二乘。
    fit_intercept:
        是否拟合截距。

    Returns
    -------
    已拟合的估计器, 支持 ``.predict``, 并暴露 ``coef_`` / ``feature_names_in_``。
    X 或 y 为 NaN 的行在拟合前被剔除。
    """
    y_aligned = y.reindex(X.index)
    mask = X.notna().all(axis=1) & y_aligned.notna()
    X_clean = X.loc[mask]
    y_clean = y_aligned.loc[mask]

    if alpha > 0:
        estimator: LinearRegression | Ridge = Ridge(
            alpha=alpha, fit_intercept=fit_intercept
        )
    else:
        estimator = LinearRegression(fit_intercept=fit_intercept)
    return estimator.fit(X_clean, y_clean)
