"""Forecast-only pipeline: log LLM probabilities every confirmed H1 bar, then
label them against realised bars. No orders are ever sent from here.

Leak rule: every input is built from bars that had CLOSED at the forecast
time. The forming bar MT5 returns as the last row is dropped before any
indicator is computed, and labels use only bars that start at or after the
forecast time.
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any, Callable

import pandas as pd

import main as trading_main  # helpers only; main.py itself is untouched
from agents.data.fred_client import get_macro_data
from agents.data.macro_inputs import build_macro_inputs
from agents.data.releases import releases_as_news_items
from agents.forecaster import build_forecast_payload
from agents.forecaster import forecast as run_forecaster
from agents.macro_analyst import analyze_macro_environment
from agents.sentiment import analyze_sentiment
from agents.technical import analyze_technical, calc_extension_atr
from analysis.forecast_labels import resolve_outcome
from analysis.forecast_store import append_forecast, read_forecasts, save_inputs, write_forecasts
from config import (
    FORECAST_HORIZONS,
    FORECAST_K_DOWN,
    FORECAST_K_UP,
    FORECAST_SAMPLES,
    FORECAST_SESSION_FILTER,
    FORECAST_USE_DEBATE,
    MARKET_TZ,
    MODEL_FORECAST,
    SYMBOL,
)
from data import mt5_client
from data.mt5_client import get_rates
from data.news_client import fetch_news_with_meta
from indicators.horizontal_levels import build_horizontal_levels
from indicators.ta_calc import add_indicators

LOGGER = logging.getLogger(__name__)

TIMEFRAME_HOURS = {"D1": 24, "H4": 4, "H1": 1}
RATES_BARS = 300
MAX_RESOLVE_BARS = 2000


# --------------------------------------------------------------------------- #
# Bars
# --------------------------------------------------------------------------- #
def bar_times_to_utc(frame: pd.DataFrame) -> pd.DataFrame:
    """MT5 bar times are server wall-clock stamped as UTC; reinterpret them.

    Uses the same rule as closed-deal timestamps (MT5_SERVER_TIMEZONE). With
    no timezone configured the frame is returned unchanged (legacy: as UTC).
    """
    if frame is None or frame.empty or "time" not in frame.columns:
        return frame
    if mt5_client.SERVER_TZ is None:
        return frame
    converted = frame.copy()
    converted["time"] = pd.to_datetime(
        [mt5_client._deal_epoch_to_utc(int(pd.Timestamp(t).timestamp())) for t in converted["time"]],
        utc=True,
    )
    return converted


def drop_forming_bars(frame: pd.DataFrame, now_utc: datetime, timeframe: str) -> pd.DataFrame:
    """Keep only bars whose close time (start + timeframe) is <= now."""
    if frame is None or frame.empty or "time" not in frame.columns:
        return frame
    hours = TIMEFRAME_HOURS.get(timeframe.upper(), 1)
    close_times = pd.to_datetime(frame["time"], utc=True) + pd.Timedelta(hours=hours)
    return frame[close_times <= pd.Timestamp(now_utc)].reset_index(drop=True)


def load_confirmed_frames(now_utc: datetime) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for timeframe in ("D1", "H4", "H1"):
        raw = get_rates(SYMBOL, timeframe, RATES_BARS)
        confirmed = drop_forming_bars(bar_times_to_utc(raw), now_utc, timeframe)
        frames[timeframe] = add_indicators(confirmed) if not confirmed.empty else confirmed
    return frames


# --------------------------------------------------------------------------- #
# Reports (mirrors main._build_market_reports on confirmed bars)
# --------------------------------------------------------------------------- #
def build_reports(frames: dict[str, pd.DataFrame], use_debate: bool = FORECAST_USE_DEBATE) -> dict[str, Any]:
    d1, h4, h1 = frames["D1"], frames["H4"], frames["H1"]
    h1_latest = trading_main._extract_latest_features(h1) if not h1.empty else {}
    h4_latest = trading_main._extract_latest_features(h4) if not h4.empty else {}
    d1_latest = trading_main._extract_latest_features(d1) if not d1.empty else {}
    horizontal_levels = build_horizontal_levels(
        d1_frame=d1,
        h4_frame=h4,
        h1_frame=h1,
        current_price=float(h1_latest.get("close", 0.0) or 0.0),
        current_atr=float(h1_latest.get("atr_14", 0.0) or 0.0),
    )
    adx_value = trading_main._safe_float_or_none(h4_latest.get("adx_14"))
    d1_extension = calc_extension_atr(
        close=float(d1_latest.get("close", 0.0) or 0.0),
        bb_mid=float(d1_latest.get("bb_mid", 0.0) or 0.0),
        atr=float(d1_latest.get("atr_14", 0.0) or 0.0),
    )
    direction_context = {
        "d1": d1_latest,
        "h4": h4_latest,
        "h1": h1_latest,
        "technical": {
            "adx": {"value": adx_value, "note": trading_main._adx_strength_note(adx_value)},
            "extension": {
                "d1_close_vs_mid_atr": round(d1_extension, 2) if d1_extension is not None else None,
                "note": trading_main._extension_note(d1_extension),
            },
        },
    }
    tp_reference_only = trading_main._build_tp_reference_only(
        horizontal_levels=horizontal_levels,
        d1_frame=d1,
        h4_frame=h4,
        current_price=float(h1_latest.get("close", 0.0) or 0.0),
    )

    news_items, feed_meta = fetch_news_with_meta(hours=24)
    macro_data = build_macro_inputs(get_macro_data(force_refresh=False))
    macro_report = analyze_macro_environment(macro_data)
    release_items = releases_as_news_items(macro_data.get("recent_releases", []) or [])
    if release_items:
        news_items = release_items + list(news_items)
    technical_report = analyze_technical({"direction_context": direction_context, "tp_reference_only": tp_reference_only})
    sentiment_report = analyze_sentiment(news_items)
    if isinstance(sentiment_report, dict):
        sentiment_report["feed_meta"] = feed_meta

    debate_report: dict[str, Any] | None = None
    if use_debate:
        from agents.debate_graph import run_debate_graph  # optional, heavier

        debate_report = run_debate_graph(technical_report, sentiment_report, macro_report)

    return {
        "technical": technical_report,
        "sentiment": sentiment_report,
        "macro": macro_report,
        "debate": debate_report,
        "news_count": len(news_items),
    }


# --------------------------------------------------------------------------- #
# Session / schedule helpers
# --------------------------------------------------------------------------- #
def parse_session_filter(value: str) -> tuple[tuple[int, int], tuple[int, int]] | None:
    text = (value or "").strip()
    if not text:
        return None
    try:
        start_text, end_text = text.split("-")
        sh, sm = (int(x) for x in start_text.strip().split(":"))
        eh, em = (int(x) for x in end_text.strip().split(":"))
    except ValueError:
        LOGGER.warning("Invalid FORECAST_SESSION_FILTER %r; ignoring", value)
        return None
    return (sh, sm), (eh, em)


def session_allows(now_utc: datetime, session_filter: str | None = None) -> bool:
    # Read the module global at call time so tests / reloads can override it.
    window = parse_session_filter(FORECAST_SESSION_FILTER if session_filter is None else session_filter)
    if window is None:
        return True
    start, end = window
    local = now_utc.astimezone(MARKET_TZ)
    current = (local.hour, local.minute)
    if start <= end:
        return start <= current < end
    return current >= start or current < end  # overnight window


def _floor_hour(now_utc: datetime) -> datetime:
    return now_utc.replace(minute=0, second=0, microsecond=0)


def _momentum_3(h1: pd.DataFrame) -> float | None:
    if h1 is None or len(h1) < 4:
        return None
    return round(float(h1["close"].iloc[-1]) - float(h1["close"].iloc[-4]), 5)


def _sum_usage(*reports: Any) -> dict[str, int]:
    total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for report in reports:
        if not isinstance(report, dict):
            continue
        usage = report.get("_meta", {}).get("usage") if isinstance(report.get("_meta"), dict) else report.get("usage")
        if isinstance(usage, dict):
            for key in total:
                total[key] += int(usage.get(key, 0) or 0)
    return total


def _existing_keys(rows: list[dict[str, Any]]) -> set[tuple[str, int]]:
    keys: set[tuple[str, int]] = set()
    for row in rows:
        try:
            keys.add((str(row.get("bar_time_utc")), int(row.get("horizon_bars"))))
        except (TypeError, ValueError):
            continue
    return keys


# --------------------------------------------------------------------------- #
# Forecast cycle
# --------------------------------------------------------------------------- #
def run_once(
    now: datetime | None = None,
    *,
    client: Any = None,
    horizons: tuple[int, ...] = FORECAST_HORIZONS,
    use_debate: bool = FORECAST_USE_DEBATE,
    samples: int = FORECAST_SAMPLES,
    model: str = MODEL_FORECAST,
) -> dict[str, Any]:
    """One forecast cycle. Never raises; returns a summary dict."""
    now_utc = (now or datetime.now(UTC)).astimezone(UTC)
    summary: dict[str, Any] = {"ok": False, "written": 0, "skipped": "", "error": ""}

    if trading_main._is_market_closed_for_weekend(now_utc):
        summary.update(ok=True, skipped="market_closed")
        return summary
    if not session_allows(now_utc):
        summary.update(ok=True, skipped="outside_session_filter")
        return summary

    try:
        frames = load_confirmed_frames(now_utc)
        h1 = frames["H1"]
        if h1.empty or "atr_14" not in h1.columns:
            summary["error"] = "no confirmed H1 bars"
            return summary
        last = h1.iloc[-1]
        bar_time = pd.Timestamp(last["time"]).to_pydatetime().astimezone(UTC)
        ts_utc = bar_time + timedelta(hours=1)
        if ts_utc < _floor_hour(now_utc) - timedelta(hours=3):
            LOGGER.warning("Confirmed H1 bar is stale (bar close %s, now %s)", ts_utc, now_utc)
        p0 = float(last["close"])
        atr = float(last.get("atr_14", 0.0) or 0.0)
        if p0 <= 0 or atr <= 0:
            summary["error"] = f"invalid p0/atr ({p0}, {atr})"
            return summary

        existing = _existing_keys(read_forecasts())
        pending_horizons = [h for h in horizons if (bar_time.isoformat(), int(h)) not in existing]
        if not pending_horizons:
            summary.update(ok=True, skipped="already_logged")
            return summary

        reports = build_reports(frames, use_debate=use_debate)
    except Exception as exc:
        LOGGER.exception("forecast cycle failed before the LLM call: %s", exc)
        summary["error"] = str(exc)
        return summary

    momentum = _momentum_3(h1)
    local = ts_utc.astimezone(MARKET_TZ)
    for horizon in pending_horizons:
        forecast_id = str(uuid.uuid4())
        barrier_up = round(p0 + FORECAST_K_UP * atr, 5)
        barrier_down = round(p0 - FORECAST_K_DOWN * atr, 5)
        payload = build_forecast_payload(
            symbol=SYMBOL,
            bar_time_utc=bar_time.isoformat(),
            ts_utc=ts_utc.isoformat(),
            p0=p0,
            atr_h1=atr,
            k_up=FORECAST_K_UP,
            k_down=FORECAST_K_DOWN,
            horizon_bars=int(horizon),
            barrier_up=barrier_up,
            barrier_down=barrier_down,
            technical_report=reports["technical"],
            sentiment_report=reports["sentiment"],
            macro_report=reports["macro"],
            debate_report=reports["debate"],
        )
        payload_json = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
        input_hash = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
        try:
            inputs_path = save_inputs(forecast_id, {"payload": payload, "reports": reports})
        except Exception as exc:
            LOGGER.warning("could not archive forecast inputs: %s", exc)
            inputs_path = ""

        try:
            result = run_forecaster(payload, model=model, samples=samples, client=client)
        except Exception as exc:  # forecaster is itself fail-safe; belt and braces
            result = {"ok": False, "error": str(exc), "model": model, "p_up": None, "p_down": None,
                      "p_timeout": None, "p_raw": {}, "samples": [], "n_samples": 0, "n_valid_samples": 0,
                      "probs_valid": False, "invalid_reason": "exception", "key_reason": "",
                      "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}

        record = {
            "forecast_id": forecast_id,
            "ts_utc": ts_utc.isoformat(),
            "bar_time_utc": bar_time.isoformat(),
            "symbol": SYMBOL,
            "p0": p0,
            "atr_h1": round(atr, 5),
            "k_up": FORECAST_K_UP,
            "k_down": FORECAST_K_DOWN,
            "horizon_bars": int(horizon),
            "barrier_up": barrier_up,
            "barrier_down": barrier_down,
            "p_up": result.get("p_up"),
            "p_down": result.get("p_down"),
            "p_timeout": result.get("p_timeout"),
            "p_raw": result.get("p_raw"),
            "samples": result.get("samples", []),
            "n_samples": result.get("n_samples", 1),
            "n_valid_samples": result.get("n_valid_samples", 0),
            "probs_valid": bool(result.get("probs_valid")),
            "invalid_reason": result.get("invalid_reason", ""),
            "model": result.get("model", model),
            "used_debate": bool(use_debate),
            "key_reason": result.get("key_reason", ""),
            "inputs_path": inputs_path,
            "input_hash": input_hash,
            "ok": bool(result.get("ok")),
            "error": result.get("error", ""),
            "usage": result.get("usage"),
            "usage_total": _sum_usage(reports["technical"], reports["sentiment"], reports["macro"], reports["debate"], result),
            "momentum_3": momentum,
            "ny_hour": local.hour,
            "weekday": ts_utc.weekday(),
            "news_count": reports.get("news_count", 0),
            "outcome": None,
            "resolved_at_utc": None,
            "bars_to_hit": None,
            "max_favorable_atr": None,
            "max_adverse_atr": None,
        }
        try:
            append_forecast(record)
            summary["written"] += 1
        except Exception as exc:
            LOGGER.exception("could not append forecast record: %s", exc)
            summary["error"] = str(exc)
    summary["ok"] = summary["written"] > 0
    summary["bar_time_utc"] = bar_time.isoformat()
    summary["p0"] = p0
    return summary


# --------------------------------------------------------------------------- #
# Labelling
# --------------------------------------------------------------------------- #
def _parse_iso(value: Any) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def resolve_pending(
    now: datetime | None = None,
    *,
    rates_fetcher: Callable[[str, str, int], pd.DataFrame] = get_rates,
) -> dict[str, int]:
    """Label every unresolved forecast whose horizon has elapsed. Idempotent."""
    now_utc = (now or datetime.now(UTC)).astimezone(UTC)
    rows = read_forecasts()
    counts = {"total": len(rows), "pending": 0, "resolved": 0, "waiting_bars": 0, "errors": 0}

    due: list[int] = []
    oldest: datetime | None = None
    max_horizon = 1
    for index, row in enumerate(rows):
        if row.get("outcome") is not None:
            continue
        ts = _parse_iso(row.get("ts_utc"))
        horizon = int(row.get("horizon_bars") or 0)
        if ts is None or horizon <= 0:
            counts["errors"] += 1
            continue
        if ts + timedelta(hours=horizon) > now_utc:
            continue
        due.append(index)
        oldest = ts if oldest is None or ts < oldest else oldest
        max_horizon = max(max_horizon, horizon)
    counts["pending"] = len(due)
    if not due or oldest is None:
        return counts

    hours_back = int((now_utc - oldest).total_seconds() // 3600) + max_horizon + 10
    count = max(50, min(MAX_RESOLVE_BARS, hours_back))
    try:
        bars = drop_forming_bars(bar_times_to_utc(rates_fetcher(SYMBOL, "H1", count)), now_utc, "H1")
    except Exception as exc:
        LOGGER.warning("resolve_pending: rates unavailable: %s", exc)
        counts["errors"] += len(due)
        return counts
    if bars is None or bars.empty:
        counts["waiting_bars"] += len(due)
        return counts
    bars = bars.sort_values("time").reset_index(drop=True)
    bar_times = pd.to_datetime(bars["time"], utc=True)

    changed = False
    for index in due:
        row = rows[index]
        ts = _parse_iso(row.get("ts_utc"))
        after = bars[bar_times >= pd.Timestamp(ts)]
        bars_after = [{"high": float(r["high"]), "low": float(r["low"])} for _, r in after.iterrows()]
        try:
            label = resolve_outcome(
                bars_after,
                p0=float(row["p0"]),
                atr=float(row.get("atr_h1") or 0.0),
                barrier_up=float(row["barrier_up"]),
                barrier_down=float(row["barrier_down"]),
                horizon_bars=int(row["horizon_bars"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            LOGGER.warning("resolve_pending: bad row %s: %s", row.get("forecast_id"), exc)
            counts["errors"] += 1
            continue
        if label is None:
            counts["waiting_bars"] += 1
            continue
        row.update(label)
        row["resolved_at_utc"] = now_utc.isoformat()
        counts["resolved"] += 1
        changed = True

    if changed:
        write_forecasts(rows)
    return counts
