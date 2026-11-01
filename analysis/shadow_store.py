"""Shadow plans: what the method would have done, per instrument, resolved on
realised bars without placing orders.

One JSON line per judgment and instrument in ``logs/shadow_plans.jsonl``.
Gold's production judgments are recorded here too (from main.run_once, no
extra LLM calls) so the comparison across instruments uses the same code
path. A plan carries the geometry the risk manager produced (entry, SL, TP,
order type, validity) and is resolved later by ``shadow.resolve_shadow_plans``
into TP / SL / TIMEOUT / NOT_FILLED with its result in R.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from config import LOG_DIR

SHADOW_PLANS_FILENAME = "shadow_plans.jsonl"

PLAN_NONE = "NONE"
PLAN_MARKET = "MARKET"
PLAN_PENDING = "PENDING"

STATUS_OPEN = "OPEN"
STATUS_RESOLVED = "RESOLVED"
STATUS_NO_PLAN = "NO_PLAN"


def shadow_plans_path() -> Path:
    return Path(LOG_DIR) / SHADOW_PLANS_FILENAME


def read_plans(path: Path | None = None) -> list[dict[str, Any]]:
    target = path or shadow_plans_path()
    if not target.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in target.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def append_plan(record: dict[str, Any], path: Path | None = None) -> None:
    target = path or shadow_plans_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_plans(rows: list[dict[str, Any]], path: Path | None = None) -> None:
    target = path or shadow_plans_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    tmp.replace(target)


def _f(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed == parsed else None


def build_plan_record(
    *,
    symbol: str,
    instrument_key: str,
    ts_utc: str,
    source: str,
    trader_report: dict[str, Any] | None,
    risk_plan: dict[str, Any] | None,
    pending: dict[str, Any] | None,
    pending_status: str,
    final_action: str,
    current_price: float,
    atr_used: float,
    atr_timeframe: str,
    trade_horizon: str,
    valid_until_utc: str | None,
    placed: bool,
    market_filter_reason: str = "",
    debate_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The plan the method produced this judgment, with or without geometry.

    ``source`` is "production" (gold, from run_once) or "shadow". A plan has
    geometry (status OPEN) when the risk plan passed: a market order plan at
    the current price, or a pending plan at its trigger. Everything else is
    recorded as NO_PLAN with the reason, so skip rates are comparable too.
    """
    report = trader_report if isinstance(trader_report, dict) else {}
    plan = risk_plan if isinstance(risk_plan, dict) else {}
    summary = (debate_report or {}).get("regime_summary") if isinstance(debate_report, dict) else None
    summary = summary if isinstance(summary, dict) else {}

    record: dict[str, Any] = {
        "symbol": symbol,
        "instrument": instrument_key,
        "ts_utc": ts_utc,
        "source": source,
        "trader_action": str(report.get("action", "HOLD") or "HOLD"),
        "final_action": final_action,
        "trade_horizon": trade_horizon,
        "atr_used": round(float(atr_used), 6) if atr_used else None,
        "atr_timeframe": atr_timeframe,
        "current_price": round(float(current_price), 6) if current_price else None,
        "regime": str(summary.get("regime", "") or ""),
        "panel_consensus_type": str(summary.get("panel_consensus_type", "") or ""),
        "direction_if_trend": str(summary.get("direction_if_trend", "") or ""),
        "entry_style": str(summary.get("entry_style", "") or ""),
        "directional_bias": str(report.get("directional_bias", "") or ""),
        "reasoning_head": str(report.get("reasoning", "") or "")[:200],
        "pending_status": pending_status,
        "market_filter_reason": market_filter_reason,
        "placed": bool(placed),
        "plan_type": PLAN_NONE,
        "status": STATUS_NO_PLAN,
        "skip_reason": "",
        "order_type": "",
        "entry": None,
        "sl": None,
        "tp": None,
        "sl_distance": None,
        "tp_distance": None,
        "net_rr": _f(plan.get("net_rr")),
        "sl_source": str(plan.get("sl_source", "") or ""),
        "tp_source": str(plan.get("tp_source", "") or ""),
        "valid_until_utc": valid_until_utc,
        "outcome": None,
    }

    action = record["trader_action"]
    plan_ok = bool(plan.get("ok"))
    sl = _f(plan.get("sl"))
    tp = _f(plan.get("tp"))
    if action in {"BUY", "SELL"}:
        if plan_ok and sl and tp:
            record.update(
                plan_type=PLAN_MARKET,
                status=STATUS_OPEN,
                order_type=action,
                entry=round(float(current_price), 6),
                sl=sl,
                tp=tp,
                sl_distance=round(abs(float(current_price) - sl), 6),
                tp_distance=round(abs(tp - float(current_price)), 6),
                valid_until_utc=None,
            )
        else:
            record["skip_reason"] = str(plan.get("reason", "") or market_filter_reason or "risk_plan_failed")
        return record

    if isinstance(pending, dict) and pending.get("type"):
        entry = _f(pending.get("price"))
        if plan_ok and entry and sl and tp:
            record.update(
                plan_type=PLAN_PENDING,
                status=STATUS_OPEN,
                order_type=str(pending.get("type")),
                entry=entry,
                sl=sl,
                tp=tp,
                sl_distance=round(abs(entry - sl), 6),
                tp_distance=round(abs(tp - entry), 6),
            )
        else:
            record["skip_reason"] = pending_status or str(plan.get("reason", "") or "risk_plan_failed")
            record["order_type"] = str(pending.get("type") or "")
            record["entry"] = entry
        return record

    record["skip_reason"] = pending_status or "none_proposed"
    return record


def next_slot_utc(now_utc: datetime, slots_ny: tuple[tuple[int, int], ...], tz: Any) -> datetime:
    """The next judgment slot (NY wall-clock) strictly after ``now_utc``, in UTC."""
    now_ny = now_utc.astimezone(tz)
    candidates: list[datetime] = []
    for day_offset in (0, 1, 2, 3):
        base = (now_ny + timedelta(days=day_offset)).replace(second=0, microsecond=0)
        for hour, minute in slots_ny:
            slot = base.replace(hour=hour, minute=minute)
            if slot > now_ny:
                candidates.append(slot)
    if not candidates:
        return (now_utc + timedelta(hours=24)).astimezone(UTC)
    return min(candidates).astimezone(UTC)
