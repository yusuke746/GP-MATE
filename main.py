from __future__ import annotations

import csv
import json
import logging
import sys
import time
import warnings
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

# Suppress known non-fatal warning from langchain_core on Python 3.14+.
warnings.filterwarnings(
    "ignore",
    message=r"Core Pydantic V1 functionality isn't compatible with Python 3\.14 or greater\.",
    category=UserWarning,
    module=r"langchain_core\.utils\.pydantic",
)

from agents.debate_graph import build_skipped_debate_report, run_debate_graph, should_execute_debate
from agents.data.fred_client import get_macro_data
from agents.evaluate_position import evaluate_position
from agents.macro_analyst import analyze_macro_environment
from agents.sentiment import analyze_sentiment
from agents.technical import analyze_technical
from agents.trader import decide_trade
from config import (
    BREAKEVEN_BUFFER,
    CLOSE_CONFIDENCE_THRESHOLD,
    CONSECUTIVE_LOSS_LIMIT,
    MARKET_TZ,
    MAX_DAILY_LOSS_PCT,
    MAX_POSITIONS,
    NEWS_FILTER_MINUTES,
    NY_RUN_TIMES,
    SPREAD_SAMPLE_INTERVAL,
    SPREAD_SAMPLES,
    SYMBOL,
)
from data.mt5_client import (
    close_position,
    get_account_info,
    get_baseline_spread,
    get_closed_deals,
    get_position_details,
    get_positions,
    get_rates,
    get_spread,
    modify_sl,
    send_order,
)
from data.news_client import fetch_news, is_high_impact_soon
from indicators.ta_calc import add_indicators
from indicators.horizontal_levels import build_horizontal_levels
from risk.risk_manager import build_risk_plan, check_filters
from risk.breakeven import should_move_to_breakeven

LOGGER = logging.getLogger(__name__)

LOG_DIR = Path(__file__).resolve().parent / "logs"
TRADE_LOG_PATH = LOG_DIR / "trade_log.csv"
CLOSED_DEAL_STATE_PATH = LOG_DIR / "closed_deal_state.json"
SCHEDULER_CATCHUP_WINDOW_SECONDS = 5 * 60

# TP reference proximity in USD. Used only for TP-target confluence labels.
TP_PROXIMITY_ROUND_NUMBER = 5.0
TP_PROXIMITY_PREV_DAY = 8.0
TP_PROXIMITY_MOVING_AVERAGE = 8.0
TP_ROUND_SCAN_RANGE = 300.0
TP_ROUND_STEP_MINOR = 50.0
TP_ROUND_STEP_MAJOR = 100.0
TP_CONFLUENCE_TOUCH_COUNT_MIN = 1

# ADX strength thresholds for direction-context note.
ADX_STRONG_THRESHOLD = 25.0
ADX_MEDIUM_THRESHOLD = 18.0
NY_MARKET_CLOSE_HOUR = 17


def python_runtime_notice() -> str:
    version = sys.version_info
    if version >= (3, 14):
        return (
            "Python 3.14+ detected: LangChain may show compatibility warnings. "
            "Recommended runtime is Python 3.12-3.13 for stable operation."
        )
    return ""

TRADE_LOG_COLUMNS: tuple[str, ...] = (
    "timestamp_utc",
    "deal_id",
    "symbol",
    "action",
    "entry_price",
    "exit_price",
    "holding_seconds",
    "pnl",
    "confidence",
    "reasoning",
    "risk_level",
    "allowed",
    "filter_reason",
    "lot",
    "sl",
    "tp",
    "order_success",
    "retcode",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "analysis_model",
    "decision_model",
    "news_count",
    "error",
    "debate_executed",
    "skip_reason",
    "stronger_side",
    "conflicts",
    "confidence_shift",
    "debate_tokens",
    "judge_parse_ok",
    "judge_error",
    "debate_gate_reason",
    "technical_direction",
    "sentiment_direction",
    "macro_direction",
    "alignment",
    "estimated_confidence",
    "position_direction",
    "technical_signal",
    "debate_direction",
    "evaluate_action",
    "evaluate_confidence",
    "evaluate_reasoning",
    "evaluate_reasoning_len",
    "breakeven_triggered",
    "breakeven_new_sl",
    "breakeven_time",
    "breakeven_ticket",
    "breakeven_entry_price",
    "breakeven_initial_sl",
    "breakeven_trigger_price",
    "breakeven_current_price",
    "breakeven_modify_success",
    "breakeven_modify_retcode",
    "breakeven_reason",
    "directional_bias",
    "bias_strength",
    "trigger_conditions",
)


def _ensure_trade_log_header() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    if not TRADE_LOG_PATH.exists():
        with TRADE_LOG_PATH.open("w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=list(TRADE_LOG_COLUMNS))
            writer.writeheader()
        return

    with TRADE_LOG_PATH.open("r", newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        existing_fields = list(reader.fieldnames or [])
        if existing_fields == list(TRADE_LOG_COLUMNS):
            return
        rows = list(reader)

    with TRADE_LOG_PATH.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(TRADE_LOG_COLUMNS))
        writer.writeheader()
        for row in rows:
            normalized = {key: row.get(key, "") for key in TRADE_LOG_COLUMNS}
            writer.writerow(normalized)


def _append_trade_log(row: dict[str, Any]) -> None:
    _ensure_trade_log_header()
    payload = {col: row.get(col, "") for col in TRADE_LOG_COLUMNS}
    with TRADE_LOG_PATH.open("a", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(TRADE_LOG_COLUMNS))
        writer.writerow(payload)


def manage_breakeven_for_position(
    position_context: dict[str, Any],
    now_iso: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any], str, bool]:
    if now_iso is None:
        now_iso = datetime.now(UTC).isoformat()

    breakeven_result: dict[str, Any] = {
        "success": False,
        "retcode": None,
    }
    breakeven_log: dict[str, Any] = {
        "breakeven_triggered": False,
        "breakeven_new_sl": "",
        "breakeven_time": "",
        "breakeven_ticket": int(position_context.get("ticket", 0) or 0),
        "breakeven_entry_price": float(position_context.get("price_open", 0.0) or 0.0),
        "breakeven_initial_sl": float(position_context.get("sl", 0.0) or 0.0),
        "breakeven_trigger_price": "",
        "breakeven_current_price": float(position_context.get("price_current", 0.0) or 0.0),
        "breakeven_modify_success": "",
        "breakeven_modify_retcode": "",
        "breakeven_reason": "",
    }

    entry_price = float(position_context.get("price_open", 0.0) or 0.0)
    initial_sl = float(position_context.get("sl", 0.0) or 0.0)
    current_price = float(position_context.get("price_current", 0.0) or 0.0)
    side = str(position_context.get("type", "") or "")
    risk_r = abs(entry_price - initial_sl)
    if side == "BUY":
        breakeven_log["breakeven_trigger_price"] = round(entry_price + risk_r, 5)
    elif side == "SELL":
        breakeven_log["breakeven_trigger_price"] = round(entry_price - risk_r, 5)

    should_move, new_sl = should_move_to_breakeven(
        entry=entry_price,
        initial_sl=initial_sl,
        current_price=current_price,
        current_sl=initial_sl,
        side=side,
        buffer=BREAKEVEN_BUFFER,
    )
    if should_move and new_sl is not None:
        breakeven_log["breakeven_triggered"] = True
        breakeven_log["breakeven_new_sl"] = float(new_sl)
        breakeven_log["breakeven_time"] = now_iso
        breakeven_result = modify_sl(int(position_context["ticket"]), float(new_sl))
        breakeven_log["breakeven_modify_success"] = bool(breakeven_result.get("success", False))
        breakeven_log["breakeven_modify_retcode"] = breakeven_result.get("retcode", "")
        if bool(breakeven_result.get("success", False)):
            breakeven_log["breakeven_reason"] = "MOVED"
            return breakeven_result, breakeven_log, "Position hold + breakeven moved", True

        breakeven_log["breakeven_reason"] = "MOVE_FAILED"
        return breakeven_result, breakeven_log, "Breakeven move failed", False

    breakeven_log["breakeven_reason"] = "NOT_TRIGGERED_OR_ALREADY_MOVED"
    return breakeven_result, breakeven_log, "Position hold", True


def _load_closed_deal_state() -> dict[str, Any]:
    if not CLOSED_DEAL_STATE_PATH.exists():
        return {"last_sync_utc": "", "deal_ids": []}
    try:
        payload = json.loads(CLOSED_DEAL_STATE_PATH.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return {"last_sync_utc": "", "deal_ids": []}
        deal_ids = payload.get("deal_ids", [])
        if not isinstance(deal_ids, list):
            deal_ids = []
        return {
            "last_sync_utc": str(payload.get("last_sync_utc", "")),
            "deal_ids": [str(x) for x in deal_ids],
        }
    except Exception:
        return {"last_sync_utc": "", "deal_ids": []}


def _save_closed_deal_state(last_sync_utc: str, deal_ids: set[str]) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "last_sync_utc": last_sync_utc,
        "deal_ids": sorted(deal_ids),
    }
    CLOSED_DEAL_STATE_PATH.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )


def sync_closed_trades() -> int:
    """Sync closed deals into trade_log.csv.

    Returns the number of newly appended closed-deal rows.
    """
    _ensure_trade_log_header()

    state = _load_closed_deal_state()
    seen_ids = set(state.get("deal_ids", []))

    raw_since = str(state.get("last_sync_utc", "") or "")
    if raw_since:
        try:
            since = datetime.fromisoformat(raw_since)
            if since.tzinfo is None:
                since = since.replace(tzinfo=UTC)
            else:
                since = since.astimezone(UTC)
        except Exception:
            since = datetime.now(UTC) - timedelta(days=7)
    else:
        since = datetime.now(UTC) - timedelta(days=7)

    deals = get_closed_deals(SYMBOL, since)
    appended = 0
    for deal in deals:
        deal_id = str(deal.get("deal_id", ""))
        if not deal_id or deal_id in seen_ids:
            continue

        row = {
            "timestamp_utc": str(deal.get("time_utc", datetime.now(UTC).isoformat())),
            "deal_id": deal_id,
            "symbol": str(deal.get("symbol", SYMBOL)),
            "action": str(deal.get("action", "HOLD")),
            "entry_price": deal.get("entry_price", ""),
            "exit_price": deal.get("exit_price", ""),
            "holding_seconds": deal.get("holding_seconds", 0),
            "pnl": float(deal.get("profit", 0.0) or 0.0),
            "confidence": "",
            "reasoning": "closed_trade_sync",
            "risk_level": "",
            "allowed": "",
            "filter_reason": "",
            "lot": float(deal.get("lot", 0.0) or 0.0),
            "sl": "",
            "tp": "",
            "order_success": True,
            "retcode": "",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "news_count": 0,
            "error": "",
        }
        _append_trade_log(row)
        seen_ids.add(deal_id)
        appended += 1

    _save_closed_deal_state(last_sync_utc=datetime.now(UTC).isoformat(), deal_ids=seen_ids)
    return appended


def _extract_latest_features(tf_df: Any) -> dict[str, Any]:
    if tf_df is None or tf_df.empty:
        return {}
    latest = tf_df.iloc[-1]
    return {
        "close": float(latest.get("close", 0.0)),
        "rsi_14": float(latest.get("rsi_14", 50.0)),
        "macd": float(latest.get("macd", 0.0)),
        "macd_signal": float(latest.get("macd_signal", 0.0)),
        "macd_hist": float(latest.get("macd_hist", 0.0)),
        "bb_upper": float(latest.get("bb_upper", 0.0)),
        "bb_mid": float(latest.get("bb_mid", 0.0)),
        "bb_lower": float(latest.get("bb_lower", 0.0)),
        "atr_14": float(latest.get("atr_14", 0.0)),
        "adx_14": float(latest.get("adx_14", 0.0)),
        "recent_high_20": float(latest.get("recent_high_20", 0.0)),
        "recent_low_20": float(latest.get("recent_low_20", 0.0)),
    }


def _safe_float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def _adx_strength_note(adx_value: float | None) -> str:
    if adx_value is None:
        return "トレンド強度: 不明"
    if adx_value >= ADX_STRONG_THRESHOLD:
        return "トレンド強度: 強"
    if adx_value >= ADX_MEDIUM_THRESHOLD:
        return "トレンド強度: 中"
    return "トレンド強度: 弱"


def _calc_latest_ma(frame: Any, period: int) -> float | None:
    try:
        if frame is None or frame.empty or "close" not in frame.columns:
            return None
        series = frame["close"].rolling(window=period, min_periods=period).mean()
        if series.empty:
            return None
        value = series.iloc[-1]
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _is_trading_session_allowed(reference: datetime | None = None) -> tuple[bool, str]:
    """Return whether trading is allowed under NY-time session policy.

    Policy:
    - Block all Monday (weekday=0) trades.
    - Block NY market close window: Fri 17:00 -> Sun 16:59 (America/New_York).
    """
    now_market = (reference or datetime.now(tz=MARKET_TZ)).astimezone(MARKET_TZ)
    weekday = now_market.weekday()

    if weekday == 0:
        return False, "Monday trading paused"

    if weekday == 4 and now_market.hour >= NY_MARKET_CLOSE_HOUR:
        return False, "NY market closed (Friday close)"

    if weekday == 5:
        return False, "NY market closed (Saturday)"

    if weekday == 6 and now_market.hour < NY_MARKET_CLOSE_HOUR:
        return False, "NY market closed (Sunday pre-open)"

    return True, ""


def _build_round_numbers(current_price: float, scan_range: float = TP_ROUND_SCAN_RANGE) -> list[float]:
    if current_price <= 0 or scan_range <= 0:
        return []

    lower = max(current_price - scan_range, 0.0)
    upper = current_price + scan_range
    levels: set[float] = set()

    for step in (TP_ROUND_STEP_MINOR, TP_ROUND_STEP_MAJOR):
        if step <= 0:
            continue
        start = int(lower // step)
        end = int(upper // step)
        for idx in range(start, end + 1):
            level = round(idx * step, 5)
            if lower <= level <= upper:
                levels.add(level)

    return sorted(levels)


def _is_near(value: float, reference: float | None, threshold: float) -> bool:
    if reference is None or threshold <= 0:
        return False
    return abs(value - reference) <= threshold


def _build_tp_reference_only(
    horizontal_levels: dict[str, Any],
    d1_frame: Any,
    h4_frame: Any,
    current_price: float,
) -> dict[str, Any]:
    prev_day = {"high": None, "low": None}
    try:
        if d1_frame is not None and not d1_frame.empty and len(d1_frame.index) >= 2:
            prev_bar = d1_frame.iloc[-2]
            prev_day = {
                "high": _safe_float_or_none(prev_bar.get("high")),
                "low": _safe_float_or_none(prev_bar.get("low")),
            }
    except Exception:
        prev_day = {"high": None, "low": None}

    moving_averages: dict[str, float | None] = {
        "h4_ma20": _calc_latest_ma(h4_frame, 20),
        "h4_ma50": _calc_latest_ma(h4_frame, 50),
        "d1_ma50": _calc_latest_ma(d1_frame, 50),
        "d1_ma200": _calc_latest_ma(d1_frame, 200),
    }

    round_numbers = _build_round_numbers(current_price)

    safe_levels = horizontal_levels if isinstance(horizontal_levels, dict) else {}
    supports_raw = safe_levels.get("supports", []) if isinstance(safe_levels.get("supports", []), list) else []
    resistances_raw = safe_levels.get("resistances", []) if isinstance(safe_levels.get("resistances", []), list) else []

    def _annotate_level(level: dict[str, Any]) -> dict[str, Any]:
        price = _safe_float_or_none(level.get("price"))
        if price is None:
            return {**level, "confluence": [], "confluence_note": "根拠重なりなし"}

        confluence: list[str] = []
        touch_count = int(level.get("touch_count", 0) or 0)
        if touch_count >= TP_CONFLUENCE_TOUCH_COUNT_MIN:
            confluence.append(f"タッチ{touch_count}回")

        near_rounds = [lv for lv in round_numbers if _is_near(price, lv, TP_PROXIMITY_ROUND_NUMBER)]
        if near_rounds:
            nearest_round = min(near_rounds, key=lambda lv: abs(lv - price))
            label = int(nearest_round) if float(nearest_round).is_integer() else nearest_round
            confluence.append(f"キリ番{label}近接")

        prev_high = _safe_float_or_none(prev_day.get("high"))
        prev_low = _safe_float_or_none(prev_day.get("low"))
        if _is_near(price, prev_high, TP_PROXIMITY_PREV_DAY):
            confluence.append("前日高値付近")
        if _is_near(price, prev_low, TP_PROXIMITY_PREV_DAY):
            confluence.append("前日安値付近")

        ma_labels = {
            "h4_ma20": "H4 MA20",
            "h4_ma50": "H4 MA50",
            "d1_ma50": "D1 MA50",
            "d1_ma200": "D1 MA200",
        }
        for key, label in ma_labels.items():
            if _is_near(price, _safe_float_or_none(moving_averages.get(key)), TP_PROXIMITY_MOVING_AVERAGE):
                confluence.append(f"{label}近接")

        note = "、".join(confluence) if confluence else "根拠重なりなし"
        return {
            **level,
            "confluence": confluence,
            "confluence_note": note,
        }

    levels = {
        "supports": [_annotate_level(level) for level in supports_raw if isinstance(level, dict)],
        "resistances": [_annotate_level(level) for level in resistances_raw if isinstance(level, dict)],
    }

    return {
        "levels": levels,
        "round_numbers": round_numbers,
        "prev_day": prev_day,
        "moving_averages": moving_averages,
        "_note": "利確ターゲット選定専用。方向判断には使用しないこと",
    }


def _sum_token_usage(*payloads: dict[str, Any]) -> dict[str, int]:
    prompt = 0
    completion = 0
    total = 0
    for payload in payloads:
        meta = payload.get("_meta", {}) if isinstance(payload, dict) else {}
        usage = meta.get("usage", {}) if isinstance(meta, dict) else {}
        prompt += int(usage.get("prompt_tokens", 0) or 0)
        completion += int(usage.get("completion_tokens", 0) or 0)
        total += int(usage.get("total_tokens", 0) or 0)
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
    }


def _extract_model_name(payload: dict[str, Any]) -> str:
    meta = payload.get("_meta", {}) if isinstance(payload, dict) else {}
    return str(meta.get("model", "")) if isinstance(meta, dict) else ""


def _safe_json_dumps(value: Any, default: str) -> str:
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return default


def _default_debate_log_fields() -> dict[str, Any]:
    return {
        "debate_executed": "",
        "skip_reason": "",
        "stronger_side": "",
        "conflicts": "[]",
        "confidence_shift": "{}",
        "debate_tokens": 0,
        "judge_parse_ok": "",
        "judge_error": "",
        "debate_gate_reason": "",
        "technical_direction": "",
        "sentiment_direction": "",
        "macro_direction": "",
        "alignment": "",
        "estimated_confidence": "",
        "position_direction": "",
        "technical_signal": "",
        "debate_direction": "",
        "evaluate_action": "",
        "evaluate_confidence": "",
        "evaluate_reasoning": "",
        "evaluate_reasoning_len": "",
        "breakeven_triggered": False,
        "breakeven_new_sl": "",
        "breakeven_time": "",
        "breakeven_ticket": "",
        "breakeven_entry_price": "",
        "breakeven_initial_sl": "",
        "breakeven_trigger_price": "",
        "breakeven_current_price": "",
        "breakeven_modify_success": "",
        "breakeven_modify_retcode": "",
        "breakeven_reason": "",
    }


def _debate_direction_from_stronger_side(stronger_side: str) -> str:
    normalized = str(stronger_side or "").strip().lower()
    if normalized == "bull":
        return "BUY"
    if normalized == "bear":
        return "SELL"
    return "NEUTRAL"


def _extract_debate_log_fields(gate: dict[str, Any], debate_report: dict[str, Any]) -> dict[str, Any]:
    fields = _default_debate_log_fields()

    should_debate = bool(gate.get("should_debate", False))
    fields["debate_executed"] = should_debate
    fields["debate_gate_reason"] = str(gate.get("reason", "") or "")
    fields["technical_direction"] = str(gate.get("technical_direction", "") or "")
    fields["sentiment_direction"] = str(gate.get("sentiment_direction", "") or "")
    fields["macro_direction"] = str(gate.get("macro_direction", "") or "")
    fields["alignment"] = str(gate.get("alignment", "") or "")
    estimated_confidence = gate.get("estimated_confidence", "")
    fields["estimated_confidence"] = estimated_confidence if estimated_confidence != "" else ""
    if not should_debate:
        fields["skip_reason"] = str(gate.get("reason", "") or "")

    judge_summary = debate_report.get("judge_summary", {}) if isinstance(debate_report, dict) else {}
    if isinstance(judge_summary, dict):
        fields["stronger_side"] = str(judge_summary.get("stronger_side", "") or "")
        fields["conflicts"] = _safe_json_dumps(judge_summary.get("conflicts", []), default="[]")
        fields["confidence_shift"] = _safe_json_dumps(judge_summary.get("confidence_shift", {}), default="{}")

    debate_meta = debate_report.get("_meta", {}) if isinstance(debate_report, dict) else {}
    if isinstance(debate_meta, dict):
        usage = debate_meta.get("usage", {})
        if isinstance(usage, dict):
            fields["debate_tokens"] = int(usage.get("total_tokens", 0) or 0)

        if "debate_executed" in debate_meta:
            fields["debate_executed"] = bool(debate_meta.get("debate_executed"))
        if "skip_reason" in debate_meta and not should_debate:
            fields["skip_reason"] = str(debate_meta.get("skip_reason", "") or "")
        if "judge_ok" in debate_meta:
            fields["judge_parse_ok"] = bool(debate_meta.get("judge_ok"))
        fields["judge_error"] = str(debate_meta.get("judge_error", "") or "")

    return fields


def _build_market_reports() -> tuple[Any, Any, Any, list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any]]:
    d1 = add_indicators(get_rates(SYMBOL, "D1", 300))
    h4 = add_indicators(get_rates(SYMBOL, "H4", 300))
    h1 = add_indicators(get_rates(SYMBOL, "H1", 300))

    h1_latest = _extract_latest_features(h1) if not h1.empty else {}
    h4_latest = _extract_latest_features(h4) if not h4.empty else {}
    d1_latest = _extract_latest_features(d1) if not d1.empty else {}
    horizontal_levels = build_horizontal_levels(
        d1_frame=d1,
        h4_frame=h4,
        h1_frame=h1,
        current_price=float(h1_latest.get("close", 0.0) or 0.0),
        current_atr=float(h1_latest.get("atr_14", 0.0) or 0.0),
    )

    adx_value = _safe_float_or_none(h4_latest.get("adx_14"))
    direction_context = {
        "d1": d1_latest,
        "h4": h4_latest,
        "h1": h1_latest,
        "technical": {
            "adx": {
                "value": adx_value,
                "note": _adx_strength_note(adx_value),
            }
        },
    }
    tp_reference_only = _build_tp_reference_only(
        horizontal_levels=horizontal_levels,
        d1_frame=d1,
        h4_frame=h4,
        current_price=float(h1_latest.get("close", 0.0) or 0.0),
    )

    news_items = fetch_news(hours=24)
    macro_data = get_macro_data(force_refresh=False)
    macro_report = analyze_macro_environment(macro_data)
    technical_report = analyze_technical(
        {
            "direction_context": direction_context,
            "tp_reference_only": tp_reference_only,
        }
    )
    sentiment_report = analyze_sentiment(news_items)
    return d1, h4, h1, news_items, macro_report, technical_report, sentiment_report


def _build_debate_and_decision_reports(
    technical_report: dict[str, Any],
    sentiment_report: dict[str, Any],
    macro_report: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
    gate = should_execute_debate(technical_report, sentiment_report, macro_report)
    if gate["should_debate"]:
        debate_report = run_debate_graph(technical_report, sentiment_report, macro_report)
        debate_meta = debate_report.get("_meta", {}) if isinstance(debate_report, dict) else {}
        debate_ok = bool(debate_meta.get("ok", False)) if isinstance(debate_meta, dict) else False
        if debate_ok:
            return gate, debate_report, None
        return (
            gate,
            debate_report,
            {
                "action": "HOLD",
                "symbol": SYMBOL,
                "confidence": 0.0,
                "reasoning": "議論エンジン失敗のためHOLD",
                "risk_level": "HIGH",
                "_meta": {
                    "ok": False,
                    "model": "",
                    "error": "debate graph failed",
                    "usage": {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                    },
                },
            },
        )

    debate_report = build_skipped_debate_report(gate["reason"])
    return gate, debate_report, None


def _blocked_hold_result(
    now_iso: str,
    reasoning: str,
    filter_reason: str,
    *,
    error: str = "",
    news_count: int = 0,
    analysis_model: str = "",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build, log, and return a fail-safe HOLD row for blocked/aborted cycles."""
    result: dict[str, Any] = {
        "timestamp_utc": now_iso,
        "deal_id": "",
        "symbol": SYMBOL,
        "action": "HOLD",
        "entry_price": "",
        "exit_price": "",
        "holding_seconds": "",
        "pnl": "",
        "confidence": 0.0,
        "reasoning": reasoning,
        "risk_level": "HIGH",
        "allowed": False,
        "filter_reason": filter_reason,
        "lot": 0.0,
        "sl": 0.0,
        "tp": 0.0,
        "order_success": False,
        "retcode": "",
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "analysis_model": analysis_model,
        "decision_model": "",
        "news_count": news_count,
        "error": error,
    }
    if extra:
        result.update(extra)
    _append_trade_log(result)
    return result


def _position_context_from_details(details: dict[str, Any], position_count: int) -> dict[str, Any]:
    return {
        "ticket": int(details.get("ticket") or 0),
        "symbol": str(details.get("symbol") or SYMBOL),
        "type": str(details.get("type") or "UNKNOWN"),
        "volume": float(details.get("volume") or 0.0),
        "price_open": float(details.get("price_open") or 0.0),
        "price_current": float(details.get("price_current") or 0.0),
        "sl": float(details.get("sl") or 0.0),
        "tp": float(details.get("tp") or 0.0),
        "profit": float(details.get("profit") or 0.0),
        "position_count": int(position_count),
    }


def calc_today_risk_stats() -> tuple[int, float]:
    """Calculate today's consecutive losses and daily loss percentage from trade log.

    - consecutive_losses: count trailing realized losing closures; reset by realized win or HOLD row.
    - daily_loss_pct: today's realized loss percent (loss only) vs current balance.
    - On aggregation failure, return blocking thresholds (safe side).
    """
    try:
        if not TRADE_LOG_PATH.exists():
            return 0, 0.0

        now_utc = datetime.now(UTC)
        today = now_utc.date()
        today_rows: list[dict[str, Any]] = []

        with TRADE_LOG_PATH.open("r", newline="", encoding="utf-8") as fp:
            reader = csv.DictReader(fp)
            for row in reader:
                ts_raw = str(row.get("timestamp_utc", "") or "").strip()
                if not ts_raw:
                    continue

                ts = datetime.fromisoformat(ts_raw)
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=UTC)
                else:
                    ts = ts.astimezone(UTC)

                if ts.date() != today:
                    continue

                action = str(row.get("action", "") or "").strip().upper()
                reasoning = str(row.get("reasoning", "") or "").strip()

                pnl_value: float | None = None
                pnl_raw = str(row.get("pnl", "") or "").strip()
                if pnl_raw:
                    pnl_value = float(pnl_raw)

                today_rows.append(
                    {
                        "timestamp": ts,
                        "action": action,
                        "reasoning": reasoning,
                        "pnl": pnl_value,
                    }
                )

        if not today_rows:
            return 0, 0.0

        today_rows.sort(key=lambda x: x["timestamp"])

        consecutive_losses = 0
        for row in reversed(today_rows):
            action = str(row.get("action", ""))
            if action == "HOLD":
                break

            pnl = row.get("pnl")
            if pnl is None:
                continue

            pnl_value = float(pnl)
            if pnl_value < 0:
                consecutive_losses += 1
                continue
            break

        realized_today_pnl = 0.0
        has_realized_today = False
        for row in today_rows:
            if str(row.get("reasoning", "")) != "closed_trade_sync":
                continue
            pnl = row.get("pnl")
            if pnl is None:
                continue
            has_realized_today = True
            realized_today_pnl += float(pnl)

        if not has_realized_today or realized_today_pnl >= 0:
            return consecutive_losses, 0.0

        account_info = get_account_info()
        if not account_info.get("success") or account_info.get("data") is None:
            LOGGER.warning("calc_today_risk_stats: account info unavailable, using safe fallback")
            return CONSECUTIVE_LOSS_LIMIT, MAX_DAILY_LOSS_PCT

        balance = float(account_info["data"].balance)
        if balance <= 0:
            LOGGER.warning("calc_today_risk_stats: non-positive balance, using safe fallback")
            return CONSECUTIVE_LOSS_LIMIT, MAX_DAILY_LOSS_PCT

        daily_loss_jpy = abs(min(realized_today_pnl, 0.0))
        daily_loss_pct = daily_loss_jpy / balance
        return consecutive_losses, daily_loss_pct
    except Exception as exc:
        LOGGER.exception("calc_today_risk_stats failed; using safe fallback: %s", exc)
        return CONSECUTIVE_LOSS_LIMIT, MAX_DAILY_LOSS_PCT


def _run_scheduler_due_jobs(
    now_local: datetime,
    executed_today: set[str],
    baseline_spread: float | None,
) -> None:
    local_tz = now_local.tzinfo or datetime.now().astimezone().tzinfo or ZoneInfo("UTC")
    current_local = now_local if now_local.tzinfo is not None else now_local.replace(tzinfo=local_tz)
    current_market = current_local.astimezone(MARKET_TZ)

    for hour, minute in NY_RUN_TIMES:
        target_market = current_market.replace(hour=hour, minute=minute, second=0, microsecond=0)
        target_local = target_market.astimezone(local_tz)
        delta = (current_local - target_local).total_seconds()
        execution_key = target_market.strftime("%Y-%m-%d-%H:%M")

        if execution_key in executed_today:
            continue
        if not (0 <= delta <= SCHEDULER_CATCHUP_WINDOW_SECONDS):
            continue

        try:
            consecutive_losses, daily_loss_pct = calc_today_risk_stats()
        except Exception as exc:
            LOGGER.exception("Risk aggregation failed; using safe fallback: %s", exc)
            consecutive_losses = CONSECUTIVE_LOSS_LIMIT
            daily_loss_pct = MAX_DAILY_LOSS_PCT

        try:
            run_once(
                baseline_spread=baseline_spread,
                consecutive_losses=consecutive_losses,
                daily_loss_pct=daily_loss_pct,
            )
        except Exception as exc:
            LOGGER.exception("run_once failed in scheduler loop (continuing): %s", exc)
        finally:
            executed_today.add(execution_key)


def run_once(
    baseline_spread: float | None = None,
    consecutive_losses: int | None = None,
    daily_loss_pct: float | None = None,
) -> dict[str, Any]:
    """Execute one full decision cycle.

    Safety policy:
    - Any exception or external failure must resolve to HOLD.
    """
    now_iso = datetime.now(UTC).isoformat()

    try:
        try:
            sync_closed_trades()
        except Exception as sync_exc:
            LOGGER.warning("sync_closed_trades failed and was skipped: %s", sync_exc)

        trading_allowed, trading_block_reason = _is_trading_session_allowed()
        if not trading_allowed:
            return _blocked_hold_result(
                now_iso,
                reasoning=f"取引停止: {trading_block_reason}",
                filter_reason="NY trading window blocked",
            )

        calibrated_baseline = baseline_spread
        if calibrated_baseline is None:
            calibrated_baseline = get_baseline_spread(
                symbol=SYMBOL,
                samples=SPREAD_SAMPLES,
                interval_sec=SPREAD_SAMPLE_INTERVAL,
            )

        dynamic_consecutive_losses, dynamic_daily_loss_pct = calc_today_risk_stats()
        effective_consecutive_losses = dynamic_consecutive_losses
        if consecutive_losses is not None:
            effective_consecutive_losses = max(consecutive_losses, dynamic_consecutive_losses)

        effective_daily_loss_pct = dynamic_daily_loss_pct
        if daily_loss_pct is not None:
            effective_daily_loss_pct = max(daily_loss_pct, dynamic_daily_loss_pct)

        if calibrated_baseline is None:
            return _blocked_hold_result(
                now_iso,
                reasoning="baseline_spread自動算出に失敗したためHOLD",
                filter_reason="Baseline spread calibration failed",
            )

        if calibrated_baseline <= 0:
            return _blocked_hold_result(
                now_iso,
                reasoning="baseline_spread未設定のためHOLD",
                filter_reason="Missing baseline spread",
            )

        if is_high_impact_soon(minutes=NEWS_FILTER_MINUTES):
            return _blocked_hold_result(
                now_iso,
                reasoning="重要指標前後のため新規取引を停止",
                filter_reason="High impact news window",
            )

        positions = get_positions(SYMBOL)

        d1, h4, h1, news_items, macro_report, technical_report, sentiment_report = _build_market_reports()
        if h4.empty or h1.empty:
            return _blocked_hold_result(
                now_iso,
                reasoning="価格データ取得に失敗",
                filter_reason="Price data unavailable",
            )

        gate, debate_report, debate_fallback_report = _build_debate_and_decision_reports(
            technical_report,
            sentiment_report,
            macro_report,
        )

        try:
            debate_log_fields = _extract_debate_log_fields(gate=gate, debate_report=debate_report)
        except Exception as exc:
            LOGGER.warning("Failed to extract debate log fields; defaults used: %s", exc)
            debate_log_fields = _default_debate_log_fields()

        if len(positions) >= MAX_POSITIONS:
            position_details = get_position_details(SYMBOL)
            if not position_details:
                return _blocked_hold_result(
                    now_iso,
                    reasoning="保有詳細取得に失敗したためHOLD",
                    filter_reason="Position details unavailable",
                    news_count=len(news_items),
                    analysis_model=_extract_model_name(technical_report),
                    extra=debate_log_fields,
                )

            position_context = _position_context_from_details(position_details[0], len(position_details))
            evaluation_report = debate_fallback_report or evaluate_position(
                position_context=position_context,
                technical_report=technical_report,
                sentiment_report=sentiment_report,
                debate_report=debate_report,
                macro_report=macro_report,
                confidence_threshold=CLOSE_CONFIDENCE_THRESHOLD,
            )

            close_result: dict[str, Any] = {
                "success": False,
                "retcode": None,
            }
            breakeven_result: dict[str, Any] = {
                "success": False,
                "retcode": None,
            }
            breakeven_log: dict[str, Any] = {
                "breakeven_triggered": False,
                "breakeven_new_sl": "",
                "breakeven_time": "",
                "breakeven_ticket": int(position_context.get("ticket", 0) or 0),
                "breakeven_entry_price": float(position_context.get("price_open", 0.0) or 0.0),
                "breakeven_initial_sl": float(position_context.get("sl", 0.0) or 0.0),
                "breakeven_trigger_price": "",
                "breakeven_current_price": float(position_context.get("price_current", 0.0) or 0.0),
                "breakeven_modify_success": "",
                "breakeven_modify_retcode": "",
                "breakeven_reason": "",
            }
            evaluation_action = str(evaluation_report.get("action", "HOLD"))
            evaluation_confidence = float(evaluation_report.get("confidence", 0.0) or 0.0)
            evaluation_reasoning = str(evaluation_report.get("reasoning", "") or "")
            position_direction = str(position_context.get("type", "") or "")
            technical_signal = str(technical_report.get("signal", "") or "")
            debate_direction = _debate_direction_from_stronger_side(str(debate_log_fields.get("stronger_side", "")))
            filter_reason = "Position hold"
            allowed = True
            if evaluation_action == "CLOSE":
                close_result = close_position(int(position_context["ticket"]))
                filter_reason = "Position close signal"
                allowed = bool(close_result.get("success", False))
                breakeven_log["breakeven_reason"] = "NOT_HOLD_ACTION"
                if not allowed:
                    filter_reason = "Position close failed"
            elif evaluation_action == "HOLD":
                breakeven_log["breakeven_reason"] = "DELEGATED_TO_MONITOR"
                filter_reason = "Position hold"

            usage = _sum_token_usage(technical_report, sentiment_report, debate_report, evaluation_report)
            execution_result = close_result if evaluation_action == "CLOSE" else breakeven_result
            result = {
                "timestamp_utc": now_iso,
                "deal_id": str(close_result.get("deal", "") or ""),
                "symbol": SYMBOL,
                "action": evaluation_action,
                "entry_price": float(position_context.get("price_open", 0.0) or 0.0),
                "exit_price": float(position_context.get("price_current", 0.0) or 0.0) if evaluation_action == "CLOSE" else "",
                "holding_seconds": "",
                "pnl": float(position_context.get("profit", 0.0) or 0.0),
                "confidence": evaluation_confidence,
                "reasoning": evaluation_reasoning,
                "risk_level": str(evaluation_report.get("risk_level", "MID")),
                "allowed": allowed,
                "filter_reason": filter_reason,
                "lot": float(position_context.get("volume", 0.0) or 0.0),
                "sl": float(position_context.get("sl", 0.0) or 0.0),
                "tp": float(position_context.get("tp", 0.0) or 0.0),
                "order_success": bool(execution_result.get("success", False)),
                "retcode": execution_result.get("retcode", ""),
                "prompt_tokens": usage["prompt_tokens"],
                "completion_tokens": usage["completion_tokens"],
                "total_tokens": usage["total_tokens"],
                "analysis_model": _extract_model_name(technical_report),
                "decision_model": _extract_model_name(evaluation_report),
                "news_count": len(news_items),
                "error": str(execution_result.get("reason", "")),
                **debate_log_fields,
                "position_direction": position_direction,
                "technical_signal": technical_signal,
                "debate_direction": debate_direction,
                "evaluate_action": evaluation_action,
                "evaluate_confidence": evaluation_confidence,
                "evaluate_reasoning": evaluation_reasoning,
                "evaluate_reasoning_len": len(evaluation_reasoning),
                **breakeven_log,
            }
            _append_trade_log(result)
            return result

        trader_report = debate_fallback_report or decide_trade(
            technical_report,
            sentiment_report,
            debate_report,
            macro_report=macro_report,
        )

        spread = get_spread(SYMBOL)
        filter_result = check_filters(
            confidence=float(trader_report.get("confidence", 0.0) or 0.0),
            spread=spread,
            baseline_spread=calibrated_baseline,
            is_news_soon=False,
            consecutive_losses=effective_consecutive_losses,
            daily_loss_pct=effective_daily_loss_pct,
        )

        account_info = get_account_info()
        balance = 0.0
        if account_info.get("success") and account_info.get("data") is not None:
            balance = float(account_info["data"].balance)

        action = str(trader_report.get("action", "HOLD"))
        entry_price = float(h1.iloc[-1].get("close", 0.0))
        atr = float(h1.iloc[-1].get("atr_14", 0.0))

        risk_plan = build_risk_plan(
            action=action,
            entry_price=entry_price,
            atr=atr,
            balance_jpy=balance,
            suggested_tp=trader_report.get("suggested_tp"),
        )

        order_result: dict[str, Any] = {
            "success": False,
            "retcode": None,
        }

        final_action = str(risk_plan.get("action", "HOLD"))
        if filter_result.ok and bool(risk_plan.get("ok")) and final_action in {"BUY", "SELL"}:
            order_result = send_order(
                symbol=SYMBOL,
                action=final_action,
                lot=float(risk_plan["lot"]),
                sl=float(risk_plan["sl"]),
                tp=float(risk_plan["tp"]),
            )
        else:
            final_action = "HOLD"

        usage = _sum_token_usage(technical_report, sentiment_report, debate_report, trader_report)

        result = {
            "timestamp_utc": now_iso,
            "deal_id": "",
            "symbol": SYMBOL,
            "action": final_action,
            "entry_price": "",
            "exit_price": "",
            "holding_seconds": "",
            "pnl": "",
            "confidence": float(trader_report.get("confidence", 0.0) or 0.0),
            "reasoning": str(trader_report.get("reasoning", "")),
            "risk_level": str(trader_report.get("risk_level", "MID")),
            "allowed": bool(filter_result.ok),
            "filter_reason": filter_result.reason,
            "lot": float(risk_plan.get("lot", 0.0) or 0.0),
            "sl": float(risk_plan.get("sl", 0.0) or 0.0),
            "tp": float(risk_plan.get("tp", 0.0) or 0.0),
            "order_success": bool(order_result.get("success", False)),
            "retcode": order_result.get("retcode", ""),
            "prompt_tokens": usage["prompt_tokens"],
            "completion_tokens": usage["completion_tokens"],
            "total_tokens": usage["total_tokens"],
            "analysis_model": _extract_model_name(technical_report),
            "decision_model": _extract_model_name(trader_report),
            "news_count": len(news_items),
            "error": str(order_result.get("reason", "")),
            "directional_bias": str(trader_report.get("directional_bias", "") or ""),
            "bias_strength": trader_report.get("bias_strength", ""),
            "trigger_conditions": _safe_json_dumps(trader_report.get("trigger_conditions", []), default="[]"),
            **debate_log_fields,
        }
        _append_trade_log(result)
        return result
    except Exception as exc:
        LOGGER.exception("run_once failed: %s", exc)
        return _blocked_hold_result(
            now_iso,
            reasoning="例外発生のためHOLD",
            filter_reason="Exception",
            error=str(exc),
        )


def run_scheduler(
    baseline_spread: float | None = None,
    consecutive_losses: int | None = None,
    daily_loss_pct: float | None = None,
) -> None:
    """Run scheduler loop and execute strategy at configured judgment times."""
    _ = (consecutive_losses, daily_loss_pct)
    executed_today: set[str] = set()
    while True:
        now_local = datetime.now()
        # Execution keys are stamped with the market (NY) date, so pruning must
        # use the market date too; the local date can differ for hours around
        # midnight and would evict still-active keys.
        market_today = datetime.now(tz=MARKET_TZ).strftime("%Y-%m-%d")
        executed_today = {x for x in executed_today if x.startswith(market_today)}

        _run_scheduler_due_jobs(
            now_local=now_local,
            executed_today=executed_today,
            baseline_spread=baseline_spread,
        )

        time.sleep(20)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    # baseline_spread can be explicitly passed for tests; None triggers auto calibration.
    run_once()
