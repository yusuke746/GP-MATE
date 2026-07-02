from __future__ import annotations

import json
from typing import Any

from agents.base import decision_model, get_default_client
from config import CONFIDENCE_THRESHOLD, SYMBOL

SYSTEM_PROMPT = (
    "あなたは最終決定権を持つトレーダーです。"
    "必ず place_trade_order 関数を呼び出して最終判断を返してください。"
    "confidenceが閾値未満なら必ずHOLDにしてください。"
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
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
) -> dict[str, Any]:
    user_payload = {
        "technical": technical_report,
        "sentiment": sentiment_report,
        "debate": debate_report,
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
    evidence_status = str(sentiment_report.get("evidence_status", "")).upper()

    if action not in {"BUY", "SELL", "HOLD"}:
        action = "HOLD"
    if evidence_status == "INSUFFICIENT":
        action = "HOLD"
        payload["reasoning"] = "ニュース判断材料が不足しているためHOLD。"
    if confidence < confidence_threshold:
        action = "HOLD"

    payload["action"] = action
    payload["symbol"] = str(payload.get("symbol") or SYMBOL)
    payload["confidence"] = confidence

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
