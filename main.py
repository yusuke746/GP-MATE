from __future__ import annotations

import csv
import json
import logging
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from agents.debate import run_debate
from agents.sentiment import analyze_sentiment
from agents.technical import analyze_technical
from agents.trader import decide_trade
from config import (
    CONSECUTIVE_LOSS_LIMIT,
    JUDGMENT_TIMES,
    MAX_DAILY_LOSS_PCT,
    MAX_POSITIONS,
    NEWS_FILTER_MINUTES,
    SPREAD_SAMPLE_INTERVAL,
    SPREAD_SAMPLES,
    SYMBOL,
)
from data.mt5_client import (
    get_account_info,
    get_baseline_spread,
    get_closed_deals,
    get_positions,
    get_rates,
    get_spread,
    send_order,
)
from data.news_client import fetch_news, is_high_impact_soon
from indicators.ta_calc import add_indicators
from risk.risk_manager import build_risk_plan, check_filters

LOGGER = logging.getLogger(__name__)

LOG_DIR = Path(__file__).resolve().parent / "logs"
TRADE_LOG_PATH = LOG_DIR / "trade_log.csv"
CLOSED_DEAL_STATE_PATH = LOG_DIR / "closed_deal_state.json"
SCHEDULER_CATCHUP_WINDOW_SECONDS = 5 * 60
SCHEDULER_REFERENCE_BALANCE_JPY = 500000.0

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
        rows = list(reader)

    if existing_fields == list(TRADE_LOG_COLUMNS):
        return

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
        "recent_high_20": float(latest.get("recent_high_20", 0.0)),
        "recent_low_20": float(latest.get("recent_low_20", 0.0)),
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


def _parse_judgment_time(value: str) -> tuple[int, int] | None:
    try:
        hh_str, mm_str = value.split(":", 1)
        hh = int(hh_str)
        mm = int(mm_str)
        if not (0 <= hh <= 23 and 0 <= mm <= 59):
            return None
        return hh, mm
    except Exception:
        return None


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
    now_utc = now_local.astimezone(UTC) if now_local.tzinfo is not None else now_local.replace(tzinfo=UTC)

    for value in JUDGMENT_TIMES:
        parsed = _parse_judgment_time(value)
        if parsed is None:
            LOGGER.warning("Invalid JUDGMENT_TIMES entry skipped: %s", value)
            continue

        hour, minute = parsed
        target = now_local.replace(hour=hour, minute=minute, second=0, microsecond=0)
        delta = (now_local - target).total_seconds()
        execution_key = target.strftime("%Y-%m-%d-%H:%M")

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
            result = {
                "timestamp_utc": now_iso,
                "deal_id": "",
                "symbol": SYMBOL,
                "action": "HOLD",
                "entry_price": "",
                "exit_price": "",
                "holding_seconds": "",
                "pnl": "",
                "confidence": 0.0,
                "reasoning": "baseline_spread自動算出に失敗したためHOLD",
                "risk_level": "HIGH",
                "allowed": False,
                "filter_reason": "Baseline spread calibration failed",
                "lot": 0.0,
                "sl": 0.0,
                "tp": 0.0,
                "order_success": False,
                "retcode": "",
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "analysis_model": "",
                "decision_model": "",
                "news_count": 0,
                "error": "",
            }
            _append_trade_log(result)
            return result

        if calibrated_baseline <= 0:
            result = {
                "timestamp_utc": now_iso,
                "deal_id": "",
                "symbol": SYMBOL,
                "action": "HOLD",
                "entry_price": "",
                "exit_price": "",
                "holding_seconds": "",
                "pnl": "",
                "confidence": 0.0,
                "reasoning": "baseline_spread未設定のためHOLD",
                "risk_level": "HIGH",
                "allowed": False,
                "filter_reason": "Missing baseline spread",
                "lot": 0.0,
                "sl": 0.0,
                "tp": 0.0,
                "order_success": False,
                "retcode": "",
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "news_count": 0,
                "error": "",
            }
            _append_trade_log(result)
            return result

        if is_high_impact_soon(minutes=NEWS_FILTER_MINUTES):
            result = {
                "timestamp_utc": now_iso,
                "deal_id": "",
                "symbol": SYMBOL,
                "action": "HOLD",
                "entry_price": "",
                "exit_price": "",
                "holding_seconds": "",
                "pnl": "",
                "confidence": 0.0,
                "reasoning": "重要指標前後のため新規取引を停止",
                "risk_level": "HIGH",
                "allowed": False,
                "filter_reason": "High impact news window",
                "lot": 0.0,
                "sl": 0.0,
                "tp": 0.0,
                "order_success": False,
                "retcode": "",
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "news_count": 0,
                "error": "",
            }
            _append_trade_log(result)
            return result

        positions = get_positions(SYMBOL)
        if len(positions) >= MAX_POSITIONS:
            result = {
                "timestamp_utc": now_iso,
                "deal_id": "",
                "symbol": SYMBOL,
                "action": "HOLD",
                "entry_price": "",
                "exit_price": "",
                "holding_seconds": "",
                "pnl": "",
                "confidence": 0.0,
                "reasoning": "最大保有ポジション数に到達",
                "risk_level": "MID",
                "allowed": False,
                "filter_reason": "Max positions reached",
                "lot": 0.0,
                "sl": 0.0,
                "tp": 0.0,
                "order_success": False,
                "retcode": "",
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "news_count": 0,
                "error": "",
            }
            _append_trade_log(result)
            return result

        h4 = add_indicators(get_rates(SYMBOL, "H4", 300))
        h1 = add_indicators(get_rates(SYMBOL, "H1", 300))
        if h4.empty or h1.empty:
            result = {
                "timestamp_utc": now_iso,
                "deal_id": "",
                "symbol": SYMBOL,
                "action": "HOLD",
                "entry_price": "",
                "exit_price": "",
                "holding_seconds": "",
                "pnl": "",
                "confidence": 0.0,
                "reasoning": "価格データ取得に失敗",
                "risk_level": "HIGH",
                "allowed": False,
                "filter_reason": "Price data unavailable",
                "lot": 0.0,
                "sl": 0.0,
                "tp": 0.0,
                "order_success": False,
                "retcode": "",
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "news_count": 0,
                "error": "",
            }
            _append_trade_log(result)
            return result

        news_items = fetch_news(hours=24)

        technical_report = analyze_technical(
            {
                "h4": _extract_latest_features(h4),
                "h1": _extract_latest_features(h1),
            }
        )
        sentiment_report = analyze_sentiment(news_items)
        debate_report = run_debate(technical_report, sentiment_report)
        trader_report = decide_trade(technical_report, sentiment_report, debate_report)

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
        }
        _append_trade_log(result)
        return result
    except Exception as exc:
        LOGGER.exception("run_once failed: %s", exc)
        fallback = {
            "timestamp_utc": now_iso,
            "deal_id": "",
            "symbol": SYMBOL,
            "action": "HOLD",
            "entry_price": "",
            "exit_price": "",
            "holding_seconds": "",
            "pnl": "",
            "confidence": 0.0,
            "reasoning": "例外発生のためHOLD",
            "risk_level": "HIGH",
            "allowed": False,
            "filter_reason": "Exception",
            "lot": 0.0,
            "sl": 0.0,
            "tp": 0.0,
            "order_success": False,
            "retcode": "",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "news_count": 0,
            "error": str(exc),
        }
        _append_trade_log(fallback)
        return fallback


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
        today = now_local.strftime("%Y-%m-%d")

        # Keep executed keys bounded per day.
        executed_today = {x for x in executed_today if x.startswith(today)}

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
