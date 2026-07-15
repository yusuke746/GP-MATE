from __future__ import annotations

import json
from typing import Any

from agents.base import decision_model, get_default_client
from config import CONFIDENCE_THRESHOLD, SYMBOL, MACRO_BIAS_CARRY_THRESHOLD

SYSTEM_PROMPT = (
    "あなたは最終決定権を持つトレーダーです。"
    "必ず place_trade_order 関数を呼び出して最終判断を返してください。"
    "confidenceが閾値未満なら必ずHOLDにしてください。"
    "ただしHOLDの場合でも、macro/technical/sentimentが方向性を示すなら "
    "directional_bias(BULLISH/BEARISH)とbias_strength、trigger_conditions"
    "(key_levelsに基づく発動価格条件)を必ず設定すること。"
    "特にmacroのmacro_biasとconfidenceは、テクニカルがレンジでも"
    "directional_biasに反映すること。"
)

PLACE_TRADE_ORDER_SCHEMA: dict[str, Any] = {
    "description": "分析結果に基づき売買判断を実行する",
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["BUY", "SELL", "HOLD"]},
            "symbol": {"type": "string"},
            "confidence": {"type": "number", "description": "0-1の確信度"},
            "reasoning": {"type": "string", "description": "判断根拠（日本語）"},
            "risk_level": {"type": "string", "enum": ["LOW", "MID", "HIGH"]},
            "directional_bias": {"type": "string", "enum": ["BULLISH", "BEARISH", "NEUTRAL"]},
            "bias_strength": {"type": "number", "description": "方向性バイアスの強さ0-1"},
            "trigger_conditions": {"type": "array", "items": {"type": "string"}, "description": "バイアス発動の価格条件"},
        },
        "required": ["action", "symbol", "confidence", "reasoning"],
    },
}

FALLBACK_RESPONSE: dict[str, Any] = {
    "action": "HOLD",
    "symbol": SYMBOL,
    "confidence": 0.0,
    "reasoning": "最終判断に失敗したためHOLD。",
    "risk_level": "HIGH",
}


def decide_trade(
    technical_report: dict[str, Any],
    sentiment_report: dict[str, Any],
    debate_report: dict[str, Any],
    macro_report: dict[str, Any] | None = None,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
) -> dict[str, Any]:
    raw_judge_summary = debate_report.get("judge_summary", {})
    judge_summary: dict[str, Any]
    if isinstance(raw_judge_summary, dict):
        judge_summary = dict(raw_judge_summary)
    else:
        judge_summary = {
            "agreements": [],
            "conflicts": [str(raw_judge_summary)] if raw_judge_summary else [],
            "confidence_shift": {"bull": [], "bear": []},
            "stronger_side": "neutral",
        }
    user_payload = {
        "technical": technical_report,
        "sentiment": sentiment_report,
        "macro": macro_report or {},
        "debate": debate_report,
        "judge_summary": judge_summary,
        "constraints": {
            "confidence_threshold": confidence_threshold,
            "symbol": SYMBOL,
        },
    }

    result = get_default_client().call_function(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=json.dumps(user_payload, ensure_ascii=False),
        model=decision_model(),
        function_name="place_trade_order",
        function_schema=PLACE_TRADE_ORDER_SCHEMA,
        fallback_payload=FALLBACK_RESPONSE,
    )

    payload = dict(result.payload)
    action = str(payload.get("action", "HOLD")).upper()
    confidence = float(payload.get("confidence", 0.0) or 0.0)
    confidence = max(0.0, min(1.0, confidence))
    evidence_status = str(sentiment_report.get("evidence_status", "")).upper()
    risk_level = str(payload.get("risk_level") or "HIGH").upper()
    if risk_level not in {"LOW", "MID", "HIGH"}:
        risk_level = "HIGH"

    if action not in {"BUY", "SELL", "HOLD"}:
        action = "HOLD"
    if evidence_status == "INSUFFICIENT":
        action = "HOLD"
        payload["reasoning"] = "ニュース判断材料が不足しているためHOLD。"
    if confidence < confidence_threshold:
        action = "HOLD"

    directional_bias = str(payload.get("directional_bias", "NEUTRAL") or "NEUTRAL").upper()
    if directional_bias not in {"BULLISH", "BEARISH", "NEUTRAL"}:
        directional_bias = "NEUTRAL"
    if directional_bias == "NEUTRAL" and isinstance(macro_report, dict):
        _macro_meta = macro_report.get("_meta", {})
        _macro_ok = bool(_macro_meta.get("ok", False)) if isinstance(_macro_meta, dict) else False
        m_bias = str(macro_report.get("macro_bias", "NEUTRAL") or "NEUTRAL").upper()
        m_conf = float(macro_report.get("confidence", 0.0) or 0.0)
        if _macro_ok and m_bias in {"BULLISH", "BEARISH"} and m_conf >= MACRO_BIAS_CARRY_THRESHOLD:
            directional_bias = m_bias
    payload["directional_bias"] = directional_bias
    payload["bias_strength"] = max(0.0, min(1.0, float(payload.get("bias_strength", 0.0) or 0.0)))
    tc = payload.get("trigger_conditions", [])
    payload["trigger_conditions"] = [str(x) for x in tc] if isinstance(tc, list) else []

    payload["action"] = action
    payload["symbol"] = str(payload.get("symbol") or SYMBOL)
    payload["confidence"] = confidence
    payload["risk_level"] = risk_level

    payload["_meta"] = {
        "ok": result.ok,
        "model": result.model,
        "error": result.error,
        "usage": {
            "prompt_tokens": result.usage.prompt_tokens,
            "completion_tokens": result.usage.completion_tokens,
            "total_tokens": result.usage.total_tokens,
        },
    }
    return payload
