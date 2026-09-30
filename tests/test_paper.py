"""Forward timing, account conservation, failure rollback and restart tests."""
import copy
import json

import pytest

from paper.engine import Account, Settings, advance, charges, initial_state
from paper.report import write_report


def snapshot(day="2026-09-14", previous="2026-09-11", candidates=True):
    return dict(mode="live", status="ready", date=day, market_date=day,
                previous_session=previous, model_id="test-model", benchmark_close=4000,
                quotes={"600000": dict(date=day, open=10, close=10, high=11, low=9,
                    limit_up=11, limit_down=9, volume_shares=1_000_000, eligible=True)},
                candidates=[dict(code="600000", name="测试", event_id="600000:2026Q2",
                    available_date=day, feature_date=day, score=.2)] if candidates else [])


def bought(cfg=Settings()):
    state = advance(initial_state(cfg, "live"), snapshot())
    return advance(state, snapshot("2026-09-15", "2026-09-14", False))


def test_signal_cannot_fill_same_day():
    state = advance(initial_state(Settings(), "live"), snapshot())
    assert not state["positions"] and not state["trades"]
    assert len(state["pending"]) == 1
    state = advance(state, snapshot("2026-09-15", "2026-09-14", False))
    assert state["positions"]["600000"]["bought"] == "2026-09-15"
    assert state["positions"]["600000"]["qty"] % 100 == 0


def test_cash_conservation_and_no_lookahead_sizing():
    state = advance(initial_state(Settings(), "live"), snapshot())
    snap = snapshot("2026-09-15", "2026-09-14", False)
    snap["quotes"]["600000"]["close"] = 10.8
    high_close = advance(state, snap)
    snap["quotes"]["600000"]["close"] = 9.2
    low_close = advance(state, snap)
    assert high_close["trades"] == low_close["trades"]
    fill = high_close["trades"][0]
    assert high_close["cash"] == pytest.approx(1_000_000 - fill["notional"] - fill["fee"])
    assert fill["notional"] + fill["fee"] <= 50_000
    assert high_close["history"][-1]["equity"] == pytest.approx(high_close["cash"] + fill["qty"] * 10.8)


def test_open_gap_does_not_exceed_order_budget():
    state = advance(initial_state(Settings(), "live"), snapshot())
    snap = snapshot("2026-09-15", "2026-09-14", False)
    snap["quotes"]["600000"]["open"] = 10.9
    result = advance(state, snap)
    fill = result["trades"][0]
    assert fill["notional"] + fill["fee"] <= 50_000


@pytest.mark.parametrize("change,status", [({"open": 11}, "limit_up_at_open"),
    ({"suspended": True}, "no_tradable_quote"), ({"eligible": False}, "ineligible"),
    ({"volume_shares": 1}, "cash_or_volume_insufficient")])
def test_unfilled_buy_cancels_without_spending(change, status):
    state = advance(initial_state(Settings(), "live"), snapshot())
    snap = snapshot("2026-09-15", "2026-09-14", False)
    snap["quotes"]["600000"].update(change)
    result = advance(state, snap)
    assert result["cash"] == 1_000_000
    assert not result["positions"] and not result["pending"]
    assert result["orders"][-1]["status"] == status


def test_sell_expiry_and_t_plus_one():
    state = bought(Settings(hold_sessions=1))
    assert state["pending"][0]["side"] == "SELL"
    assert len(state["trades"]) == 1
    result = advance(state, snapshot("2026-09-16", "2026-09-15", False))
    assert not result["positions"]
    assert result["trades"][-1]["side"] == "SELL"
    assert result["cash"] == pytest.approx(1_000_000 - sum(t["notional"] + t["fee"] for t in result["trades"] if t["side"] == "BUY") + sum(t["notional"] - t["fee"] for t in result["trades"] if t["side"] == "SELL"))


def test_blocked_sell_is_resubmitted():
    state = bought(Settings(hold_sessions=1))
    snap = snapshot("2026-09-16", "2026-09-15", False)
    snap["quotes"]["600000"]["open"] = 9
    result = advance(state, snap)
    assert result["positions"]
    assert result["pending"][0]["side"] == "SELL"
    assert result["orders"][-1]["status"] == "limit_down_at_open"


def test_partial_fill_accounts_only_actual_shares():
    state = advance(initial_state(Settings(), "live"), snapshot())
    snap = snapshot("2026-09-15", "2026-09-14", False)
    snap["quotes"]["600000"]["volume_shares"] = 20_000
    result = advance(state, snap)
    assert result["positions"]["600000"]["qty"] == 200
    assert result["orders"][-1]["status"] == "partial_remainder_cancelled"


def test_seen_event_not_rebought():
    state = bought(Settings(hold_sessions=1))
    result = advance(state, snapshot("2026-09-16", "2026-09-15"))
    assert not result["positions"] and not result["pending"]


@pytest.mark.parametrize("mutation", ["stale", "corporate", "missing", "future", "gap", "model", "mode"])
def test_invalid_day_rolls_back(mutation, tmp_path):
    account = Account(tmp_path)
    account.commit(snapshot())
    before = account.commit(snapshot("2026-09-15", "2026-09-14", False))
    snap = snapshot("2026-09-16", "2026-09-15")
    if mutation == "stale":
        snap["market_date"] = "2026-09-15"
    elif mutation == "corporate":
        snap["quotes"]["600000"]["corporate_action"] = True
    elif mutation == "missing":
        snap["quotes"] = {}
    elif mutation == "future":
        snap["candidates"][0]["available_date"] = "2026-09-17"
    elif mutation == "gap":
        snap["previous_session"] = "2026-09-14"
    elif mutation == "model":
        snap["model_id"] = "different"
    elif mutation == "mode":
        snap["mode"] = "demo"
    with pytest.raises(ValueError):
        account.commit(snap)
    assert Account(tmp_path).read() == before


def test_idempotence_and_frozen_snapshots(tmp_path):
    account = Account(tmp_path)
    first = account.commit(snapshot())
    assert Account(tmp_path).commit(snapshot()) == first
    assert len(account.read()["history"]) == 1
    changed = snapshot()
    changed["benchmark_close"] = 4100
    with pytest.raises(ValueError, match="different data"):
        account.commit(changed)


def test_report_is_recoverable_from_database(tmp_path):
    account = Account(tmp_path)
    account.commit(snapshot())
    path = write_report(tmp_path, account.read())
    content = path.read_text()
    assert "1,000,000.00" in content and "下一交易日订单" in content
    assert write_report(tmp_path, Account(tmp_path).read()).read_text() == content


def test_charges_and_settings():
    assert charges(1000, "BUY", Settings()) == 5.01
    assert charges(1000, "SELL", Settings()) == 5.51
    with pytest.raises(ValueError):
        Settings(initial_cash=-1)
    with pytest.raises(ValueError):
        Settings(max_weight=2)


def test_no_signal_day_is_valid_cash_day():
    state = advance(initial_state(Settings(), "live"), snapshot(candidates=False))
    assert state["history"][-1]["total_return"] == 0
    assert not state["pending"]


def test_demo_cannot_enter_live_directory(tmp_path):
    account = Account(tmp_path)
    account.commit(snapshot())
    snap = snapshot("2026-09-15", "2026-09-14", False)
    snap["mode"] = "demo"
    with pytest.raises(ValueError, match="separate"):
        Account(tmp_path, mode="demo").commit(snap)
