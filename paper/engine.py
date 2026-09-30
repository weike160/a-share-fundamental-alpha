"""Daily paper account with transactional, idempotent SQLite checkpoints."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import sqlite3
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    initial_cash: float = 1_000_000
    max_names: int = 20
    max_weight: float = 0.05
    hold_sessions: int = 20
    top_pct: float = 0.2
    commission: float = 0.0003
    min_commission: float = 5
    stamp_tax: float = 0.0005
    transfer_fee: float = 0.00001
    slippage: float = 0.001
    participation: float = 0.01

    def __post_init__(self):
        for key, value in asdict(self).items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid setting: {key}")
        if self.initial_cash <= 0 or self.max_names < 1 or self.hold_sessions < 1:
            raise ValueError("Cash, max_names and hold_sessions must be positive")
        if not 0 < self.max_weight <= 1 or not 0 < self.top_pct <= 1:
            raise ValueError("Weight and top_pct must be in (0, 1]")
        if not 0 < self.participation <= 1 or self.slippage >= 1:
            raise ValueError("Invalid participation or slippage")


def positive(value):
    return isinstance(value, (int, float)) and math.isfinite(value) and value > 0


def initial_state(settings: Settings, mode: str):
    return dict(mode=mode, settings=asdict(settings), cash=settings.initial_cash,
                positions={}, pending=[], seen=[], history=[], trades=[], orders=[],
                session=0, last_date=None, benchmark_start=None, model_id=None)


def charges(notional, side, cfg):
    return round(max(cfg.min_commission, notional * cfg.commission)
                 + notional * cfg.transfer_fee
                 + (notional * cfg.stamp_tax if side == "SELL" else 0), 2)


def validate_snapshot(snap, state):
    day = snap["date"]
    date.fromisoformat(day)
    if snap["mode"] != state["mode"]:
        raise ValueError("Demo and live accounts must use separate directories")
    if state["last_date"] and day <= state["last_date"]:
        raise ValueError("Cannot rewind an account")
    if snap.get("status") != "ready":
        raise ValueError("Snapshot is not ready")
    if snap.get("selection_status", "ready") != "ready" and snap["candidates"]:
        raise ValueError("Unavailable selection must not generate candidates")
    if snap.get("market_date") != day or not positive(snap.get("benchmark_close")):
        raise ValueError("Missing/stale market date or benchmark")
    if state["last_date"] and snap.get("previous_session") != state["last_date"]:
        raise ValueError("Missed trading session: reconcile the gap before continuing")
    if state["model_id"] and state["model_id"] != snap.get("model_id"):
        raise ValueError("Model changed: start a separate account for the new version")
    for code, quote in snap["quotes"].items():
        if quote.get("date") != day or not positive(quote.get("close")):
            raise ValueError(f"Missing/stale close: {code}")
        if quote.get("corporate_action"):
            raise ValueError(f"Corporate action requires reconciliation: {code}")
        if not quote.get("suspended"):
            if not all(positive(quote.get(k)) for k in ("open", "high", "low", "limit_up", "limit_down")):
                raise ValueError(f"Invalid OHLC/limits: {code}")
            if not quote["low"] <= quote["open"] <= quote["high"] or not quote["low"] <= quote["close"] <= quote["high"]:
                raise ValueError(f"Inconsistent OHLC: {code}")
    for code in state["positions"]:
        if code not in snap["quotes"]:
            raise ValueError(f"Cannot value holding: {code}")
    for candidate in snap["candidates"]:
        available = candidate["available_date"]
        if available > day or candidate["feature_date"] > day:
            raise ValueError("Future information in candidate")
        if not math.isfinite(candidate["score"]):
            raise ValueError("Non-finite model score")


def execute_orders(state, snap, cfg):
    """按当日开盘价处理昨日订单，更新现金和持仓，返回成交及订单状态。"""
    day = snap["date"]
    quotes = snap["quotes"]
    session = state["session"]
    fills, attempts = [], []
    # Yesterday's fixed budgets, never today's close, determine buying power per order.
    for order in sorted(state["pending"], key=lambda x: x["side"] != "SELL"):
        code, side = order["code"], order["side"]
        outcome = {**order, "execution_date": day, "filled_qty": 0}
        quote = quotes.get(code)
        reason = None
        if order["created"] != snap.get("previous_session"):
            reason = "expired"
        elif not quote or quote.get("suspended"):
            reason = "no_tradable_quote"
        elif side == "BUY" and not quote.get("eligible", False):
            reason = "ineligible"
        elif side == "BUY" and quote["open"] >= quote["limit_up"] - 0.005:
            reason = "limit_up_at_open"
        elif side == "SELL" and quote["open"] <= quote["limit_down"] + 0.005:
            reason = "limit_down_at_open"
        elif side == "SELL" and (
            code not in state["positions"]
            or state["positions"][code]["bought"] >= day
        ):
            reason = "T+1_or_no_position"
        if reason:
            outcome["status"] = reason
            attempts.append(outcome)
            continue
        # OHLC open fill approximation, conservative fixed slippage, bounded to observed range.
        direction = 1 if side == "BUY" else -1
        price = round(quote["open"] * (1 + cfg.slippage * direction), 2)
        price = min(quote["high"], quote["limit_up"], max(quote["low"], quote["limit_down"], price))
        volume_cap = int(max(0, quote.get("volume_shares", 0)) * cfg.participation / 100) * 100
        qty = min(order["qty"], volume_cap)
        if side == "BUY":
            budget = min(order["budget"], state["cash"])
            qty = min(qty, int(budget / price / 100) * 100)
            while qty > 0 and qty * price + charges(qty * price, side, cfg) > budget:
                qty -= 100
        else:
            qty = min(qty, state["positions"][code]["qty"])
        if qty <= 0:
            outcome["status"] = "cash_or_volume_insufficient"
            attempts.append(outcome)
            continue
        notional = round(qty * price, 2)
        fee = charges(notional, side, cfg)
        if side == "BUY":
            state["cash"] = round(state["cash"] - notional - fee, 2)
            state["positions"][code] = dict(
                qty=qty, cost=notional + fee, bought=day,
                entry_session=session, mark=quote["close"], name=order.get("name", code),
            )
        else:
            state["cash"] = round(state["cash"] + notional - fee, 2)
            pos = state["positions"][code]
            pos["cost"] *= (pos["qty"] - qty) / pos["qty"]
            pos["qty"] -= qty
            if pos["qty"] == 0:
                del state["positions"][code]
        fills.append(dict(date=day, code=code, side=side, qty=qty, price=price, fee=fee, notional=notional))
        outcome.update(
            status="filled" if qty == order["qty"] else "partial_remainder_cancelled",
            filled_qty=qty,
        )
        attempts.append(outcome)

    return fills, attempts


def plan_orders(state, snap, cfg, equity):
    """收盘后生成到期卖单与新事件买单，并记录已处理的事件。"""
    day = snap["date"]
    quotes = snap["quotes"]
    session = state["session"]
    pending = []
    for code, pos in state["positions"].items():
        holding_sessions = session - pos["entry_session"] + 1
        if holding_sessions >= cfg.hold_sessions or not quotes[code].get("eligible", False):
            pending.append(dict(code=code, side="SELL", qty=pos["qty"], created=day, name=pos["name"]))
    seen = set(state["seen"])
    candidates = sorted(snap["candidates"], key=lambda x: (-x["score"], x["code"]))
    fresh = [candidate for candidate in candidates if candidate["event_id"] not in seen]
    # Every observed event is considered once; no repeated mining of rejected events.
    state["seen"] = sorted(seen | {candidate["event_id"] for candidate in candidates})
    selected = fresh[:math.ceil(len(fresh) * cfg.top_pct)]
    free_slots = cfg.max_names - len(state["positions"])
    reserved = 0.0
    held_or_ordered = set(state["positions"])
    for candidate in selected:
        code = candidate["code"]
        quote = quotes.get(code)
        if free_slots <= 0 or code in held_or_ordered or not quote or not quote.get("eligible"):
            continue
        budget = min(equity * cfg.max_weight, state["cash"] - reserved)
        estimated_price = quote["close"] * (
            1 + cfg.slippage + cfg.commission + cfg.transfer_fee
        )
        qty = int(max(0, budget - cfg.min_commission) / estimated_price / 100) * 100
        if qty <= 0:
            continue
        pending.append(dict(
            code=code, name=candidate.get("name", code), side="BUY", qty=qty,
            budget=round(budget, 2), created=day,
            score=candidate["score"], event_id=candidate["event_id"],
        ))
        reserved += budget
        held_or_ordered.add(code)
        free_slots -= 1
    return pending


def advance(state, snap):
    """Consume yesterday's orders at today's open; produce orders after close."""
    validate_snapshot(snap, state)
    state = copy.deepcopy(state)
    cfg = Settings(**state["settings"])
    day = snap["date"]
    quotes = snap["quotes"]
    state["session"] += 1
    fills, attempts = execute_orders(state, snap, cfg)

    for code, pos in state["positions"].items():
        pos["mark"] = quotes[code]["close"]
    market_value = sum(p["qty"] * p["mark"] for p in state["positions"].values())
    equity = round(state["cash"] + market_value, 2)
    previous_equity = state["history"][-1]["equity"] if state["history"] else cfg.initial_cash
    if state["benchmark_start"] is None:
        state["benchmark_start"] = snap["benchmark_close"]
    benchmark = snap["benchmark_close"] / state["benchmark_start"] - 1
    total = equity / cfg.initial_cash - 1
    peak = max([cfg.initial_cash, equity] + [x["equity"] for x in state["history"]])
    daily = dict(date=day, equity=equity, cash=state["cash"], market_value=round(market_value, 2),
                 daily_pnl=round(equity - previous_equity, 2), daily_return=equity / previous_equity - 1,
                 total_return=total, benchmark_return=benchmark, excess_return=total - benchmark,
                 drawdown=equity / peak - 1, fees=round(sum(f["fee"] for f in fills), 2),
                 holdings=len(state["positions"]), fills=len(fills))
    pending = plan_orders(state, snap, cfg, equity)
    state.update(pending=pending, last_date=day, model_id=snap["model_id"],
                 warnings=snap.get("warnings", []), exclusions=snap.get("exclusions", []),
                 candidates=snap["candidates"], selection_status=snap.get("selection_status", "ready"),
                 selection_since=day if snap.get("selection_status", "ready") == "ready"
                 else state.get("selection_since") or snap["previous_session"])
    state["history"].append(daily)
    state["trades"].extend(fills)
    state["orders"].extend(attempts)
    return state


class Account:
    def __init__(self, directory: Path, settings=Settings(), mode="live"):
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / "account.sqlite3"
        self.settings, self.mode = settings, mode
        with sqlite3.connect(self.path) as db:
            db.execute("CREATE TABLE IF NOT EXISTS checkpoints (day TEXT PRIMARY KEY, digest TEXT, snapshot TEXT, state TEXT)")

    def read(self):
        with sqlite3.connect(self.path) as db:
            row = db.execute("SELECT state FROM checkpoints ORDER BY day DESC LIMIT 1").fetchone()
        return json.loads(row[0]) if row else initial_state(self.settings, self.mode)

    def commit(self, snap):
        payload = json.dumps(snap, ensure_ascii=False, sort_keys=True, allow_nan=False)
        digest = hashlib.sha256(payload.encode()).hexdigest()
        with sqlite3.connect(self.path, timeout=30) as db:
            db.execute("BEGIN IMMEDIATE")
            previous = db.execute("SELECT digest, state FROM checkpoints WHERE day=?", (snap["date"],)).fetchone()
            if previous:
                if previous[0] != digest:
                    raise ValueError("Same date has different data; committed account cannot be overwritten")
                return json.loads(previous[1])
            last = db.execute("SELECT state FROM checkpoints ORDER BY day DESC LIMIT 1").fetchone()
            state = json.loads(last[0]) if last else initial_state(self.settings, self.mode)
            if state["settings"] != asdict(self.settings):
                raise ValueError("Account settings are frozen; use a new directory to change them")
            new = advance(state, snap)
            db.execute("INSERT INTO checkpoints VALUES (?, ?, ?, ?)",
                       (snap["date"], digest, payload, json.dumps(new, ensure_ascii=False, allow_nan=False)))
        return new
