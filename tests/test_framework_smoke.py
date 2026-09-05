"""框架冒烟测试占位.

说明：本仓库采用「研究 → 代码」的顺序推进。当前处于研究问题阶段，
核心模块均为空实现的 stub（抛出 NotImplementedError）。
本文件仅在骨架层面验证包可导入。完整的因子/组合/成本测试将在
后续阶段（Level 2 / Level 3）逐模块补齐。
"""
from __future__ import annotations


def test_package_importable() -> None:
    """验证各大包可被导入，框架结构完整。"""
    import backtest  # noqa: F401
    import data  # noqa: F401
    import execution  # noqa: F401
    import factor  # noqa: F401
    import ml  # noqa: F401
    import portfolio  # noqa: F401
    import signals  # noqa: F401

    assert True
