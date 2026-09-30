"""AKShare 原始响应缓存与失败重试。

缓存位于 data/raw/<interface>/，parquet 旁的元数据记录参数、版本、时间和文件摘要。"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd

RAW_DIR = Path(__file__).resolve().parent / "raw"

#: 实测存在偶发断连, 默认重试 4 次
DEFAULT_RETRIES = 4
DEFAULT_BACKOFF = 1.5


class DataSourceError(RuntimeError):
    """接口在重试耗尽后仍然失败."""


@dataclass
class FetchMeta:
    """一次抓取的完整溯源信息."""

    interface: str
    params: dict[str, Any]
    akshare_version: str
    fetched_at: str
    rows: int
    columns: list[str]
    sha256: str
    attempts: int
    from_cache: bool = False

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2, default=str)


def _params_key(interface: str, params: dict[str, Any] | None) -> str:
    """把接口名 + 参数压成稳定、文件系统安全的键."""
    payload = json.dumps(params or {}, sort_keys=True, ensure_ascii=False, default=str)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
    readable = "_".join(f"{k}-{v}" for k, v in sorted((params or {}).items()))
    readable = "".join(c if (c.isalnum() or c in "-_") else "-" for c in readable)[:60]
    return f"{readable}__{digest}" if readable else digest


def cache_paths(
    interface: str, params: dict[str, Any] | None = None
) -> tuple[Path, Path]:
    """返回 ``(parquet 路径, meta 路径)``, 不保证文件存在."""
    key = _params_key(interface, params)
    folder = RAW_DIR / interface
    return folder / f"{key}.parquet", folder / f"{key}.meta.json"


def _akshare_version() -> str:
    try:
        import akshare

        return getattr(akshare, "__version__", "unknown")
    except Exception:  # noqa: BLE001 - 版本号缺失不应该阻断抓取
        return "unknown"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _mixed_object_columns(df: pd.DataFrame) -> list[str]:
    """找出「同一列里混了多种 Python 类型」的 object 列.

    AKShare 有些接口会返回混型列, 例如同花顺的 ``净利润同比增长率`` 里既有
    ``"64.75%"`` (str) 又有 ``False`` (bool)。这类列 pyarrow 无法写入 parquet,
    会抛 ``ArrowInvalid: Could not convert ... tried to convert to boolean``。
    """
    mixed: list[str] = []
    for col in df.columns:
        s = df[col]
        if s.dtype != object:
            continue
        kinds = {type(v).__name__ for v in s.dropna().head(2000)}
        if len(kinds) > 1:
            mixed.append(str(col))
    return mixed


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    """写入 parquet；混型列转为字符串，读取方自行解析。"""
    mixed = _mixed_object_columns(df)
    if mixed:
        df = df.copy()
        for col in mixed:
            df[col] = df[col].astype("string")
    df.to_parquet(path, index=False)


def fetch(
    interface: str,
    *,
    params: dict[str, Any] | None = None,
    force: bool = False,
    retries: int = DEFAULT_RETRIES,
    backoff: float = DEFAULT_BACKOFF,
    call: Callable[..., pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """下载 (或从缓存读取) 一个 AKShare 接口的原始响应.

    Parameters
    ----------
    interface:
        AKShare 里的函数名, 例如 ``"stock_yjbb_em"``.
    params:
        传给该接口的关键字参数.
    force:
        为 ``True`` 时忽略缓存强制重新抓取.
    retries / backoff:
        失败重试次数与退避基数 (秒), 采用指数退避.
    call:
        注入调用函数的测试钩子; 默认从 ``akshare`` 动态取.
    """
    params = params or {}
    parquet_path, meta_path = cache_paths(interface, params)

    if parquet_path.exists() and not force:
        return pd.read_parquet(parquet_path)

    if call is None:
        import akshare

        if not hasattr(akshare, interface):
            raise DataSourceError(f"akshare 中不存在接口 {interface!r}")
        call = getattr(akshare, interface)

    for attempt in range(1, retries + 1):
        try:
            df = call(**params)
            break
        except Exception as exc:
            if attempt == retries:
                raise DataSourceError(
                    f"{interface}({params}) 重试 {retries} 次后仍失败: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            time.sleep(backoff ** attempt)
    else:  # pragma: no cover - 上面的循环要么 break 要么 raise
        raise DataSourceError(f"{interface}: unreachable")

    if df is None:
        raise DataSourceError(f"{interface}({params}) 返回 None")

    df = pd.DataFrame(df)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    _write_parquet(df, parquet_path)

    meta = FetchMeta(
        interface=interface,
        params=params,
        akshare_version=_akshare_version(),
        fetched_at=datetime.now(timezone.utc).isoformat(),
        rows=len(df),
        columns=[str(c) for c in df.columns],
        sha256=_sha256(parquet_path),
        attempts=attempt,
    )
    meta_path.write_text(meta.to_json(), encoding="utf-8")
    return df


def list_cache() -> pd.DataFrame:
    """列出当前缓存内容, 便于写数据审计表."""
    rows = []
    if not RAW_DIR.exists():
        return pd.DataFrame(
            columns=["interface", "params", "rows", "akshare_version", "fetched_at"]
        )
    for meta_path in sorted(RAW_DIR.glob("*/*.meta.json")):
        try:
            with meta_path.open("r", encoding="utf-8") as fh:
                m = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        rows.append(
            {
                "interface": m.get("interface"),
                "params": json.dumps(m.get("params", {}), ensure_ascii=False),
                "rows": m.get("rows"),
                "akshare_version": m.get("akshare_version"),
                "fetched_at": m.get("fetched_at"),
                "sha256": m.get("sha256"),
            }
        )
    return pd.DataFrame(rows)
