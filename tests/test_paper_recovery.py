"""Recovery must conserve cash and never invent historical signals."""
import json

import pandas as pd
import pytest

from paper import recovery
from paper.engine import Account
from paper.report import write_report
from tests.test_paper import snapshot


@pytest.fixture
def sources(monkeypatch):
    monkeypatch.setattr(recovery, "fetch", lambda *a, **k: pd.DataFrame({
        "trade_date": ["2026-09-11", "2026-09-14", "2026-09-15", "2026-09-16", "2026-09-17"]}))
    monkeypatch.setattr(recovery, "benchmark_bars", lambda: pd.DataFrame({
        "date": ["2026-09-15", "2026-09-16"], "close": [4100., 4200.]}))


def test_cash_gap_resumes_without_fabricated_orders(sources, tmp_path):
    account = Account(tmp_path)
    account.commit(snapshot(candidates=False))
    assert recovery.recover(account, tmp_path, "2026-09-16") == ["2026-09-15", "2026-09-16"]
    state = account.read()
    assert state["cash"] == 1_000_000
    assert not state["pending"] and not state["trades"]
    assert state["history"][-1]["benchmark_return"] == pytest.approx(.05)
    assert recovery.recover(account, tmp_path, "2026-09-16") == []


def test_pending_orders_need_original_snapshot(sources, tmp_path):
    account = Account(tmp_path)
    before = account.commit(snapshot())
    with pytest.raises(ValueError, match="无已保存快照"):
        recovery.recover(account, tmp_path, "2026-09-15")
    assert account.read() == before


def test_saved_snapshot_replays_once(sources, tmp_path):
    account = Account(tmp_path)
    account.commit(snapshot())
    folder = tmp_path / "snapshots"
    folder.mkdir()
    (folder / "2026-09-15.json").write_text(json.dumps(snapshot("2026-09-15", "2026-09-14", False)))
    assert recovery.recover(account, tmp_path, "2026-09-15") == ["2026-09-15"]
    assert len(account.read()["trades"]) == 1
    assert recovery.recover(account, tmp_path, "2026-09-15") == []
    assert len(account.read()["trades"]) == 1


def test_unavailable_selection_clears_old_candidate_export(tmp_path):
    account = Account(tmp_path)
    first = account.commit(snapshot())
    write_report(tmp_path, first)
    assert "600000" in (tmp_path / "candidates.csv").read_text()
    snap = snapshot("2026-09-15", "2026-09-14", False)
    snap["selection_status"] = "unavailable"
    result = account.commit(snap)
    path = write_report(tmp_path, result)
    assert len(result["trades"]) == 1
    assert "600000" not in (tmp_path / "candidates.csv").read_text()
    assert "未运行有效选股" in path.read_text()


def test_selection_failure_retains_retry_window(tmp_path):
    account = Account(tmp_path)
    account.commit(snapshot(candidates=False))
    snap = snapshot("2026-09-15", "2026-09-14", False)
    snap["selection_status"] = "unavailable"
    assert account.commit(snap)["selection_since"] == "2026-09-14"
