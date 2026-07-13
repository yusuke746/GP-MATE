from __future__ import annotations

import json
from typing import Any, Final, Literal, TypedDict, cast

from agents.base import analysis_model, get_default_client

SYSTEM_PROMPT = (
    "あなたは経験20年のテクニカルアナリストです。"
    "与えられたデータのみで判断し、必ずJSONで返してください。"
)

TIMEFRAME_TREND_VALUES: Final[tuple[Literal["UP", "DOWN", "RANGE"], ...]] = (
    "UP",
    "DOWN",
    "RANGE",
)

ALIGNMENT_VALUES: Final[tuple[Literal["ALIGNED", "DIVERGENT", "MIXED"], ...]] = (
    "ALIGNED",
    "DIVERGENT",
    "MIXED",
)


FALLBACK_RESPONSE: dict[str, Any] = {
    "trend": "RANGE",
    "signal": "NEUTRAL",
    "key_levels": {},
    "reasoning": "テクニカル分析に失敗したため保守的にNEUTRAL。",
}


class TechnicalAnalysisMeta(TypedDict):
    ok: bool
    model: str
    error: str
    usage: dict[str, int]


class TechnicalAnalysisResult(TypedDict, total=False):
    trend: Literal["UP", "DOWN", "RANGE"]
    signal: Literal["BUY", "SELL", "NEUTRAL"]
    rsi_14: float
    key_levels: dict[str, Any]
    reasoning: str
    d1_trend: Literal["UP", "DOWN", "RANGE"]
    execution_trend: Literal["UP", "DOWN", "RANGE"]
    alignment: Literal["ALIGNED", "DIVERGENT", "MIXED"]
    _meta: TechnicalAnalysisMeta


def _empty_meta(ok: bool, model: str, error: str, usage: dict[str, int] | None = None) -> TechnicalAnalysisMeta:
    return {
        "ok": ok,
        "model": model,
        "error": error,
        "usage": usage or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _normalize_trend(value: Any) -> Literal["UP", "DOWN", "RANGE"]:
    text = str(value or "").upper()
    if text in {"UP", "DOWN", "RANGE"}:
        return cast(Literal["UP", "DOWN", "RANGE"], text)
    return "RANGE"


def _direction_to_signal(trend: Literal["UP", "DOWN", "RANGE"]) -> Literal["BUY", "SELL", "NEUTRAL"]:
    if trend == "UP":
        return "BUY"
    if trend == "DOWN":
        return "SELL"
    return "NEUTRAL"


def _frame_reason(parts: list[str]) -> str:
    filtered = [part for part in parts if part]
    return "、".join(filtered) if filtered else "方向感は限定的"


def _score_frame(frame: dict[str, Any], label: str) -> dict[str, Any]:
    if not isinstance(frame, dict) or not frame:
        return {
            "trend": "RANGE",
            "score": 0.0,
            "reason": f"{label}データが不足しているためRANGE",
            "snapshot": {
                "close": 0.0,
                "rsi_14": 50.0,
                "macd_hist": 0.0,
                "bb_mid": 0.0,
                "bb_upper": 0.0,
                "bb_lower": 0.0,
                "recent_high_20": 0.0,
                "recent_low_20": 0.0,
            },
        }

    close = _as_float(frame.get("close", 0.0), 0.0)
    rsi = _as_float(frame.get("rsi_14", 50.0), 50.0)
    macd_hist = _as_float(frame.get("macd_hist", 0.0), 0.0)
    bb_mid = _as_float(frame.get("bb_mid", 0.0), 0.0)
    bb_upper = _as_float(frame.get("bb_upper", 0.0), 0.0)
    bb_lower = _as_float(frame.get("bb_lower", 0.0), 0.0)
    recent_high = _as_float(frame.get("recent_high_20", 0.0), 0.0)
    recent_low = _as_float(frame.get("recent_low_20", 0.0), 0.0)

    score = 0.0
    reasons: list[str] = []

    if rsi >= 65.0:
        score += 0.8
        reasons.append(f"RSI {rsi:.1f}が強気")
    elif rsi >= 58.0:
        score += 0.4
        reasons.append(f"RSI {rsi:.1f}がやや強気")
    elif rsi <= 35.0:
        score -= 0.8
        reasons.append(f"RSI {rsi:.1f}が弱気")
    elif rsi <= 42.0:
        score -= 0.4
        reasons.append(f"RSI {rsi:.1f}がやや弱気")

    if macd_hist >= 0.10:
        score += 0.6
        reasons.append(f"MACDヒストグラム {macd_hist:.2f} がプラス")
    elif macd_hist <= -0.10:
        score -= 0.6
        reasons.append(f"MACDヒストグラム {macd_hist:.2f} がマイナス")

    if bb_mid > 0.0:
        if close > bb_mid:
            score += 0.2
            reasons.append("終値がBBミッドを上回る")
        elif close < bb_mid:
            score -= 0.2
            reasons.append("終値がBBミッドを下回る")

    if bb_upper > 0.0 and close >= bb_upper:
        score += 0.2
        reasons.append("終値がBB上限に到達")
    elif bb_lower > 0.0 and close <= bb_lower:
        score -= 0.2
        reasons.append("終値がBB下限に到達")

    if recent_high > 0.0 and close >= recent_high * 0.99:
        score += 0.1
        reasons.append("直近高値圏で推移")
    elif recent_low > 0.0 and close <= recent_low * 1.01:
        score -= 0.1
        reasons.append("直近安値圏で推移")

    if score >= 0.8:
        trend: Literal["UP", "DOWN", "RANGE"] = "UP"
    elif score <= -0.8:
        trend = "DOWN"
    else:
        trend = "RANGE"

    return {
        "trend": trend,
        "score": score,
        "reason": _frame_reason(reasons),
        "snapshot": {
            "close": close,
            "rsi_14": rsi,
            "macd_hist": macd_hist,
            "bb_mid": bb_mid,
            "bb_upper": bb_upper,
            "bb_lower": bb_lower,
            "recent_high_20": recent_high,
            "recent_low_20": recent_low,
        },
    }


def _combine_execution_trend(h4_trend: Literal["UP", "DOWN", "RANGE"], h1_trend: Literal["UP", "DOWN", "RANGE"]) -> Literal["UP", "DOWN", "RANGE"]:
    if h4_trend == h1_trend:
        return h4_trend
    if h4_trend == "RANGE":
        return h1_trend
    if h1_trend == "RANGE":
        return h4_trend
    return "RANGE"


def _alignment_from_trends(
    d1_trend: Literal["UP", "DOWN", "RANGE"],
    execution_trend: Literal["UP", "DOWN", "RANGE"],
) -> Literal["ALIGNED", "DIVERGENT", "MIXED"]:
    if d1_trend == "RANGE" or execution_trend == "RANGE":
        return "MIXED"
    if d1_trend == execution_trend:
        return "ALIGNED"
    return "DIVERGENT"


def _build_multitimeframe_baseline(indicator_payload: dict[str, Any]) -> TechnicalAnalysisResult:
    h4_raw = indicator_payload.get("h4", indicator_payload.get("execution", {}))
    h1_raw = indicator_payload.get("h1", indicator_payload.get("execution", {}))
    d1_raw = indicator_payload.get("d1", {})

    h4 = _score_frame(h4_raw if isinstance(h4_raw, dict) else {}, "H4")
    h1 = _score_frame(h1_raw if isinstance(h1_raw, dict) else {}, "H1")
    d1 = _score_frame(d1_raw if isinstance(d1_raw, dict) else {}, "D1")

    execution_trend = _combine_execution_trend(h4["trend"], h1["trend"])
    alignment = _alignment_from_trends(d1["trend"], execution_trend)

    d1_text = f"D1={d1['trend']}"
    exec_text = f"執行足(H4/H1)={execution_trend}"
    if alignment == "ALIGNED":
        alignment_text = "大局と執行が一致しており、強い方向感"
    elif alignment == "DIVERGENT":
        alignment_text = "D1と執行足が逆向きで、短期反発と大局トレンドが対立"
    else:
        alignment_text = "どちらかがレンジで、明確な整合性は弱い"

    reasoning = (
        f"{d1_text}、{exec_text}。{alignment_text}。"
        f"H4の根拠: {h4['reason']}。H1の根拠: {h1['reason']}。D1の根拠: {d1['reason']}。"
    )

    raw_horizontal_levels = indicator_payload.get("horizontal_levels", {})
    horizontal_levels = raw_horizontal_levels if isinstance(raw_horizontal_levels, dict) else {}
    resistances_raw = horizontal_levels.get("resistances", []) if isinstance(horizontal_levels, dict) else []
    supports_raw = horizontal_levels.get("supports", []) if isinstance(horizontal_levels, dict) else []

    key_levels = {
        "d1": {"trend": d1["trend"], "score": d1["score"], "snapshot": d1["snapshot"]},
        "execution": {
            "trend": execution_trend,
            "score": h4["score"] + h1["score"],
            "h4_trend": h4["trend"],
            "h1_trend": h1["trend"],
        },
        "alignment": alignment,
        "frames": {
            "h4": h4["snapshot"],
            "h1": h1["snapshot"],
        },
        "horizontal_levels": {
            "resistances": list(resistances_raw) if isinstance(resistances_raw, list) else [],
            "supports": list(supports_raw) if isinstance(supports_raw, list) else [],
        },
    }

    result: TechnicalAnalysisResult = {
        "trend": execution_trend,
        "signal": _direction_to_signal(execution_trend),
        "rsi_14": _as_float(h1["snapshot"].get("rsi_14", 50.0), 50.0),
        "key_levels": key_levels,
        "reasoning": reasoning,
        "d1_trend": d1["trend"],
        "execution_trend": execution_trend,
        "alignment": alignment,
    }
    return result


def _build_legacy_baseline(indicator_payload: dict[str, Any]) -> TechnicalAnalysisResult:
    signal = str(indicator_payload.get("signal", "NEUTRAL") or "NEUTRAL").upper()
    if signal not in {"BUY", "SELL", "NEUTRAL"}:
        signal = "NEUTRAL"

    trend = str(indicator_payload.get("trend", "RANGE") or "RANGE").upper()
    if trend not in {"UP", "DOWN", "RANGE"}:
        trend = "RANGE"

    raw_horizontal_levels = indicator_payload.get("horizontal_levels", {})
    horizontal_levels = raw_horizontal_levels if isinstance(raw_horizontal_levels, dict) else {}
    resistances_raw = horizontal_levels.get("resistances", []) if isinstance(horizontal_levels, dict) else []
    supports_raw = horizontal_levels.get("supports", []) if isinstance(horizontal_levels, dict) else []

    result: TechnicalAnalysisResult = {
        "trend": cast(Literal["UP", "DOWN", "RANGE"], trend if trend in {"UP", "DOWN", "RANGE"} else "RANGE"),
        "signal": cast(Literal["BUY", "SELL", "NEUTRAL"], signal),
        "rsi_14": _as_float(indicator_payload.get("rsi_14", 50.0), 50.0),
        "key_levels": {
            **(
                dict(indicator_payload.get("key_levels", {}))
                if isinstance(indicator_payload.get("key_levels", {}), dict)
                else {}
            ),
            "horizontal_levels": {
                "resistances": list(resistances_raw) if isinstance(resistances_raw, list) else [],
                "supports": list(supports_raw) if isinstance(supports_raw, list) else [],
            },
        },
        "reasoning": str(indicator_payload.get("reasoning", "テクニカル分析に失敗したため保守的にNEUTRAL。") or "テクニカル分析に失敗したため保守的にNEUTRAL."),
        "d1_trend": "RANGE",
        "execution_trend": cast(Literal["UP", "DOWN", "RANGE"], trend if trend in {"UP", "DOWN", "RANGE"} else "RANGE"),
        "alignment": "MIXED",
    }
    return result


def analyze_technical(indicator_payload: dict[str, Any]) -> dict[str, Any]:
    has_multi_timeframe = any(key in indicator_payload for key in ("h4", "h1", "d1", "execution"))
    baseline = _build_multitimeframe_baseline(indicator_payload) if has_multi_timeframe else _build_legacy_baseline(indicator_payload)

    if has_multi_timeframe:
        user_prompt = (
            "以下の多時間軸インジケータ情報から、D1と執行足(H4/H1)の整合性を説明してください。\n"
            "重要: D1は大局、H4/H1は執行足として扱い、alignmentをALIGNED/DIVERGENT/MIXEDで要約してください。\n"
            "D1が無い/壊れている場合は、執行足のみで安全側に判断してください。\n"
            f"{json.dumps(indicator_payload, ensure_ascii=False)}"
        )
    else:
        user_prompt = (
            "以下のインジケータ情報から、trend/signal/key_levels/reasoningをJSONで返してください。\n"
            f"{json.dumps(indicator_payload, ensure_ascii=False)}"
        )

    result = get_default_client().call_json(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        model=analysis_model(),
        fallback_payload=dict(baseline),
    )

    payload = dict(result.payload)
    merged: TechnicalAnalysisResult = dict(baseline)
    for key in ("reasoning", "key_levels", "trend", "signal"):
        if key in payload:
            merged[key] = payload[key]

    merged["trend"] = baseline["trend"]
    merged["signal"] = baseline["signal"]
    merged["rsi_14"] = baseline["rsi_14"]
    merged["key_levels"] = baseline["key_levels"]
    merged["reasoning"] = str(payload.get("reasoning", baseline["reasoning"]))
    merged["d1_trend"] = baseline["d1_trend"]
    merged["execution_trend"] = baseline["execution_trend"]
    merged["alignment"] = baseline["alignment"]
    merged["_meta"] = {
        "ok": result.ok,
        "model": result.model,
        "error": result.error,
        "usage": {
            "prompt_tokens": result.usage.prompt_tokens,
            "completion_tokens": result.usage.completion_tokens,
            "total_tokens": result.usage.total_tokens,
        },
    }
    return merged
