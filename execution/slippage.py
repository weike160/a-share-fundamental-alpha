"""滑点建模."""
from __future__ import annotations

import pandas as pd


def slippage(
    order_size: pd.Series,
    adv: pd.Series,
    participation_cap: float = 0.10,
) -> pd.Series:
    """滑点, 通常随 order_size/ADV 上升."""
    raise NotImplementedError
