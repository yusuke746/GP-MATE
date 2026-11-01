"""Run the full decision method on other instruments without trading.

``run_shadow_once(instrument_key)`` builds the same reports, debate and
trader decision as production for one instrument, turns the decision into
the geometry the risk manager would enforce, and records it as a shadow plan.
``resolve_shadow_plans()`` later labels open plans from realised H1 bars:
was a pending plan filled before it expired, and did the trade reach TP or
SL first (or time out / get flattened for the weekend). ``scripts/eval_shadow.py``
summarises the result per instrument.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Any, Callable

import pandas as pd

import main
from agents.trader import decide_trade
from analysis.shadow_store import (
    PLAN_MARKET,
    PLAN_PENDING,
    STATUS_OPEN,
    STATUS_RESOLVED,
    append_plan,
    build_plan_record,
    next_slot_utc,
    read_plans,
    write_plans,
)
from config import FRIDAY_FLAT_TIME_NY, MARKET_TZ, NY_RUN_TIMES, SHADOW_JUDGMENT_TIMES_NY, SHADOW_SYMBOLS
from data.confirmed_bars import bar_times_to_utc, drop_forming_bars
from data.mt5_client import get_rates, get_spread_price, resolve_broker_symbol
from instruments import INSTRUMENTS, Instrument
from risk.risk_manager import (
    PENDING_MAX_DISTANCE_ATR,
    PENDING_MIN_DISTANCE_ATR,
    build_risk_plan,
    normalize_horizon,
    select_atr,
)

LOGGER = logging.getLogger(__name__)

# A shadow trade is not held forever: production flattens on Friday and the
# evaluator can close it; here it times out after this many H1 bars.
MAX_HOLD_BARS = 72
SHADOW_BALANCE_JPY = 1_000_000_000.0  # lot sizing is irrelevant for a shadow plan


def shadow_instruments() -> list[Instrument]:
    return [INSTRUMENTS[key] for key in SHADOW_SYMBOLS if key in INSTRUMENTS]


def _recent_context(symbol: str, hours: int = 24, limit: int = 5) -> dict[str, Any]:
    """The trader's own recent shadow decisions for this symbol (no closes)."""
    cutoff = datetime.now(UTC) - timedelta(hours=hours)
    decisions = []
    for row in read_plans():
        if row.get("symbol") != symbol:
            continue
        try:
            ts = datetime.fromisoformat(str(row.get("ts_utc")))
        except ValueError:
            continue
        if ts < cutoff:
            continue
        decisions.append(
            {
                "time_utc": row.get("ts_utc"),
                "action": row.get("trader_action"),
                "directional_bias": row.get("directional_bias", ""),
                "trade_horizon": row.get("trade_horizon", ""),
                "pending_status": row.get("pending_status", ""),
                "reasoning_head": row.get("reasoning_head", ""),
            }
        )
    return {"decisions": decisions[-limit:], "recent_closed": []}


def run_shadow_once(instrument_key: str, now: datetime | None = None) -> dict[str, Any]:
    """One judgment for ``instrument_key`` recorded as a shadow plan. Never raises."""
    now_utc = (now or datetime.now(UTC)).astimezone(UTC)
    now_iso = now_utc.isoformat()
    instrument = INSTRUMENTS.get(instrument_key)
    if instrument is None:
        return {"ok": False, "reason": f"unknown instrument {instrument_key}"}
    try:
        symbol = resolve_broker_symbol(instrument.symbol_candidates)
        if not symbol:
            LOGGER.warning("shadow %s: broker symbol not found", instrument_key)
            return {"ok": False, "reason": "symbol not found"}

        d1, h4, h1, news_items, macro_report, technical_report, sentiment_report, bars_meta = main._build_market_reports(
            now_utc, instrument=instrument, symbol=symbol
        )
        if h4.empty or h1.empty or main._insufficient_bars_reason(bars_meta):
            return {"ok": False, "reason": "insufficient bars", "symbol": symbol}

        gate, debate_report, debate_fallback = main._build_debate_and_decision_reports(
            technical_report, sentiment_report, macro_report, symbol=symbol, instrument=instrument
        )
        spread_usd = get_spread_price(symbol) or 0.0
        trader_report = debate_fallback or decide_trade(
            technical_report,
            sentiment_report,
            debate_report,
            macro_report=macro_report,
            recent_context=_recent_context(symbol),
            spread_usd=spread_usd,
            instrument=instrument,
            symbol=symbol,
        )

        action = str(trader_report.get("action", "HOLD") or "HOLD")
        current_price = float(h1.iloc[-1].get("close", 0.0) or 0.0)
        horizon = normalize_horizon(trader_report.get("trade_horizon"))
        atr, atr_timeframe = select_atr(horizon, float(h1.iloc[-1].get("atr_14", 0.0) or 0.0), float(h4.iloc[-1].get("atr_14", 0.0) or 0.0))

        risk_plan: dict[str, Any] = {}
        pending: dict[str, Any] | None = None
        pending_status = ""
        final_action = "HOLD"
        placed = False
        valid_until: str | None = None
        if action in {"BUY", "SELL"}:
            risk_plan = build_risk_plan(
                action=action,
                entry_price=current_price,
                atr=atr,
                balance_jpy=SHADOW_BALANCE_JPY,
                suggested_tp=trader_report.get("suggested_tp"),
                suggested_sl=trader_report.get("suggested_sl"),
                spread_usd=spread_usd,
                max_risk_pct=0.0,
                structure_buffer=instrument.structure_buffer,
            )
            final_action = str(risk_plan.get("action", "HOLD")) if risk_plan.get("ok") else "HOLD"
            placed = bool(risk_plan.get("ok"))
            pending_status = "" if placed else f"skipped_{risk_plan.get('reason', 'risk_plan')}"
        else:
            pendings = trader_report.get("pending_orders") if isinstance(trader_report.get("pending_orders"), list) else []
            pending = pendings[0] if pendings and isinstance(pendings[0], dict) else None
            if pending is None:
                pending_status = str(trader_report.get("pending_validation", "") or "none_proposed")
            elif main._is_past_pending_placement_cutoff(now_utc):
                pending_status = "skipped_late_placement"
            else:
                price = float(pending.get("price", 0.0) or 0.0)
                distance = abs(price - current_price)
                if atr <= 0 or price <= 0 or not (PENDING_MIN_DISTANCE_ATR * atr <= distance <= PENDING_MAX_DISTANCE_ATR * atr):
                    pending_status = f"skipped_distance:{distance / atr:.2f}atr" if atr > 0 else "skipped_invalid"
                else:
                    direction = "BUY" if str(pending.get("type", "")).startswith("BUY") else "SELL"
                    risk_plan = build_risk_plan(
                        action=direction,
                        entry_price=price,
                        atr=atr,
                        balance_jpy=SHADOW_BALANCE_JPY,
                        suggested_tp=pending.get("tp"),
                        suggested_sl=pending.get("sl"),
                        spread_usd=spread_usd,
                        max_risk_pct=0.0,
                        structure_buffer=instrument.structure_buffer,
                    )
                    placed = bool(risk_plan.get("ok"))
                    pending_status = "placed" if placed else f"skipped_{risk_plan.get('reason', 'risk_plan')}"
                    if placed:
                        slots = SHADOW_JUDGMENT_TIMES_NY or NY_RUN_TIMES
                        valid_until = next_slot_utc(now_utc, slots, MARKET_TZ).isoformat()

        record = build_plan_record(
            symbol=symbol,
            instrument_key=instrument.key,
            ts_utc=now_iso,
            source="shadow",
            trader_report=trader_report,
            risk_plan=risk_plan,
            pending=pending,
            pending_status=pending_status,
            final_action=final_action,
            current_price=current_price,
            atr_used=atr,
            atr_timeframe=atr_timeframe,
            trade_horizon=horizon,
            valid_until_utc=valid_until,
            placed=placed,
            debate_report=debate_report,
        )
        record["usage_total"] = main._sum_token_usage(technical_report, sentiment_report, debate_report, trader_report).get("total_tokens", 0)
        record["news_count"] = len(news_items)
        append_plan(record)
        LOGGER.info(
            "shadow %s: action=%s plan=%s status=%s entry=%s sl=%s tp=%s",
            symbol, action, record["plan_type"], record["status"] if record["status"] == STATUS_OPEN else record["skip_reason"],
            record["entry"], record["sl"], record["tp"],
        )
        return {"ok": True, "symbol": symbol, "record": record}
    except Exception as exc:
        LOGGER.exception("run_shadow_once(%s) failed safely: %s", instrument_key, exc)
        return {"ok": False, "reason": str(exc)}


# --------------------------------------------------------------------------- resolution


def _fill_hit(order_type: str, entry: float, high: float, low: float) -> bool:
    if order_type in {"BUY", "SELL"}:
        return True
    if order_type == "SELL_LIMIT" or order_type == "BUY_STOP":
        return high >= entry
    if order_type == "BUY_LIMIT" or order_type == "SELL_STOP":
        return low <= entry
    return False


def _is_weekend_flat(bar_time_utc: datetime) -> bool:
    ny = bar_time_utc.astimezone(MARKET_TZ)
    return ny.weekday() == 4 and (ny.hour, ny.minute) >= FRIDAY_FLAT_TIME_NY


def resolve_plan(plan: dict[str, Any], bars: list[dict[str, Any]], now_utc: datetime) -> dict[str, Any] | None:
    """Label one OPEN plan from H1 bars after its timestamp (list of dicts with
    time (UTC datetime), high, low, close). None when more bars are needed."""
    order_type = str(plan.get("order_type", "") or "")
    entry = float(plan.get("entry") or 0.0)
    sl = float(plan.get("sl") or 0.0)
    tp = float(plan.get("tp") or 0.0)
    if entry <= 0 or sl <= 0 or tp <= 0 or not bars:
        return None
    side = "BUY" if order_type.startswith("BUY") else "SELL"
    risk = abs(entry - sl)
    if risk <= 0:
        return None
    valid_until = None
    if plan.get("valid_until_utc"):
        valid_until = datetime.fromisoformat(str(plan["valid_until_utc"]))
        if valid_until.tzinfo is None:
            valid_until = valid_until.replace(tzinfo=UTC)

    # 1. fill
    fill_index: int | None = None
    for index, bar in enumerate(bars):
        bar_time = bar["time"]
        if valid_until is not None and bar_time >= valid_until:
            break
        if _fill_hit(order_type, entry, float(bar["high"]), float(bar["low"])):
            fill_index = index
            break
    if fill_index is None:
        if valid_until is not None and now_utc >= valid_until + timedelta(hours=1):
            return {"outcome": "NOT_FILLED", "r_result": 0.0, "filled_at_utc": None, "exited_at_utc": None, "bars_held": 0, "mfe_r": None, "mae_r": None}
        if plan.get("plan_type") == PLAN_MARKET and bars:
            fill_index = 0
        else:
            return None

    # 2. exit
    mfe = 0.0
    mae = 0.0
    held = 0
    for bar in bars[fill_index: fill_index + MAX_HOLD_BARS]:
        held += 1
        high = float(bar["high"])
        low = float(bar["low"])
        fav = (high - entry) if side == "BUY" else (entry - low)
        adv = (entry - low) if side == "BUY" else (high - entry)
        mfe = max(mfe, fav)
        mae = max(mae, adv)
        hit_tp = high >= tp if side == "BUY" else low <= tp
        hit_sl = low <= sl if side == "BUY" else high >= sl
        base = {"filled_at_utc": bars[fill_index]["time"].isoformat(), "exited_at_utc": bar["time"].isoformat(), "bars_held": held, "mfe_r": round(mfe / risk, 3), "mae_r": round(mae / risk, 3)}
        if hit_tp and hit_sl:
            return {"outcome": "AMBIGUOUS_SL", "r_result": -1.0, **base}
        if hit_sl:
            return {"outcome": "SL", "r_result": -1.0, **base}
        if hit_tp:
            return {"outcome": "TP", "r_result": round(abs(tp - entry) / risk, 3), **base}
        if _is_weekend_flat(bar["time"]):
            close = float(bar["close"])
            move = (close - entry) if side == "BUY" else (entry - close)
            return {"outcome": "WEEKEND_FLAT", "r_result": round(move / risk, 3), **base}
    if held >= MAX_HOLD_BARS:
        last = bars[fill_index + MAX_HOLD_BARS - 1]
        close = float(last["close"])
        move = (close - entry) if side == "BUY" else (entry - close)
        return {"outcome": "TIMEOUT", "r_result": round(move / risk, 3), "filled_at_utc": bars[fill_index]["time"].isoformat(), "exited_at_utc": last["time"].isoformat(), "bars_held": held, "mfe_r": round(mfe / risk, 3), "mae_r": round(mae / risk, 3)}
    return None  # still running


def resolve_shadow_plans(now: datetime | None = None, rates_fetcher: Callable[[str, str, int], pd.DataFrame] = get_rates) -> dict[str, int]:
    """Label every OPEN plan that can be labelled from closed H1 bars. Idempotent."""
    now_utc = (now or datetime.now(UTC)).astimezone(UTC)
    rows = read_plans()
    counts = {"total": len(rows), "open": 0, "resolved": 0, "waiting": 0, "errors": 0}
    open_by_symbol: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        if row.get("status") != STATUS_OPEN:
            continue
        open_by_symbol.setdefault(str(row.get("symbol")), []).append(index)
    counts["open"] = sum(len(v) for v in open_by_symbol.values())
    if not open_by_symbol:
        return counts

    changed = False
    for symbol, indexes in open_by_symbol.items():
        oldest = min(datetime.fromisoformat(str(rows[i]["ts_utc"])) for i in indexes)
        hours_back = int((now_utc - oldest.astimezone(UTC)).total_seconds() // 3600) + MAX_HOLD_BARS + 10
        try:
            frame = drop_forming_bars(bar_times_to_utc(rates_fetcher(symbol, "H1", max(50, min(2000, hours_back)))), now_utc, "H1")
        except Exception as exc:
            LOGGER.warning("resolve_shadow_plans: bars unavailable for %s: %s", symbol, exc)
            counts["errors"] += len(indexes)
            continue
        if frame is None or frame.empty:
            counts["waiting"] += len(indexes)
            continue
        frame = frame.sort_values("time").reset_index(drop=True)
        times = pd.to_datetime(frame["time"], utc=True)
        for index in indexes:
            row = rows[index]
            ts = datetime.fromisoformat(str(row["ts_utc"])).astimezone(UTC)
            after = frame[times >= pd.Timestamp(ts)]
            bars = [
                {"time": pd.Timestamp(r["time"]).to_pydatetime().astimezone(UTC), "high": float(r["high"]), "low": float(r["low"]), "close": float(r["close"])}
                for _, r in after.iterrows()
            ]
            try:
                label = resolve_plan(row, bars, now_utc)
            except (KeyError, TypeError, ValueError) as exc:
                LOGGER.warning("resolve_shadow_plans: bad plan %s: %s", row.get("ts_utc"), exc)
                counts["errors"] += 1
                continue
            if label is None:
                counts["waiting"] += 1
                continue
            row.update(label)
            row["status"] = STATUS_RESOLVED
            row["resolved_at_utc"] = now_utc.isoformat()
            counts["resolved"] += 1
            changed = True
    if changed:
        write_plans(rows)
    return counts


def run_shadow_cycle(now: datetime | None = None) -> dict[str, Any]:
    """All configured shadow instruments once, then resolve. Called by the scheduler."""
    results: dict[str, Any] = {}
    for instrument in shadow_instruments():
        results[instrument.key] = run_shadow_once(instrument.key, now=now)
    try:
        results["resolve"] = resolve_shadow_plans(now=now)
    except Exception as exc:
        LOGGER.warning("resolve_shadow_plans failed safely: %s", exc)
    return results
