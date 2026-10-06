from __future__ import annotations

import json
from typing import Any

from agents.base import decision_model, get_default_client
from config import SYMBOL

SYSTEM_PROMPT = (
    "あなたは保有ポジションを評価するトレーダーです。"
    "保有情報、分析結果、議論結果、マクロ環境を踏まえて、保持(HOLD)か決済(CLOSE)を判断してください。"
    "明確に逆方向の根拠が強い場合のみCLOSEを選び、それ以外はHOLDを優先してください。"
    "逆方向でも確信が弱い場合は慌てて決済せずHOLDにしてください。"
    "同方向または中立ならHOLDにしてください。"
    "macro_context.macro_vs_position が AGAINST の場合は保有方向にマクロの逆風があることを意味する。"
    "macro.key_drivers の根拠が具体的で複数あるならCLOSE寄りの検討材料とし、"
    "単一材料のみならHOLDを維持してください。"
    "確信度の数値は求めない。あなたの判断がそのまま採用される。迷いがあればHOLDを選び、"
    "保有根拠が明確に崩れたと判断したときだけCLOSEを選ぶこと。"
    "必ず evaluate_position_action 関数を呼び出して返答してください。"
)

EVALUATE_POSITION_SCHEMA: dict[str, Any] = {
    "description": "保有ポジションを保持するか決済するか判断する",
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["HOLD", "CLOSE"]},
            "symbol": {"type": "string"},
            "reasoning": {"type": "string", "description": "判断根拠（日本語）"},
            "risk_level": {"type": "string", "enum": ["LOW", "MID", "HIGH"]},
        },
        "required": ["action", "symbol", "reasoning"],
    },
}

FALLBACK_RESPONSE: dict[str, Any] = {
    "action": "HOLD",
    "symbol": SYMBOL,
    "reasoning": "保有評価に失敗したためHOLD。",
    "risk_level": "HIGH",
}


def evaluate_position(
    position_context: dict[str, Any],
    technical_report: dict[str, Any],
    sentiment_report: dict[str, Any],
    debate_report: dict[str, Any],
    macro_report: dict[str, Any] | None = None,
    confidence_threshold: float | None = None,  # deprecated: no longer gates anything; kept for callers
) -> dict[str, Any]:
    """HOLD or CLOSE for an open position. The verdict stands; no confidence gate."""
    _ = confidence_threshold
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

    position_side = str(position_context.get("type", "") or "").upper()
    macro_bias = str((macro_report or {}).get("macro_bias", "NEUTRAL") or "NEUTRAL").upper()
    macro_vs_position = "NEUTRAL"
    if macro_bias == "BULLISH":
        macro_vs_position = "ALIGNED" if position_side == "BUY" else ("AGAINST" if position_side == "SELL" else "NEUTRAL")
    elif macro_bias == "BEARISH":
        macro_vs_position = "ALIGNED" if position_side == "SELL" else ("AGAINST" if position_side == "BUY" else "NEUTRAL")
    macro_context = {
        "macro_bias": macro_bias,
        "position_side": position_side,
        "macro_vs_position": macro_vs_position,
        "macro_reliable": bool((macro_report or {}).get("_meta", {}).get("ok", False)),
    }

    user_payload = {
        "position": position_context,
        "technical": technical_report,
        "sentiment": sentiment_report,
        "macro": macro_report or {},
        "macro_context": macro_context,
        "debate": debate_report,
        "judge_summary": judge_summary,
        "constraints": {
            "symbol": str(position_context.get("symbol") or SYMBOL),
        },
    }

    result = get_default_client().call_function(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=json.dumps(user_payload, ensure_ascii=False),
        model=decision_model(),
        function_name="evaluate_position_action",
        function_schema=EVALUATE_POSITION_SCHEMA,
        fallback_payload=FALLBACK_RESPONSE,
    )

    payload = dict(result.payload)
    action = str(payload.get("action", "HOLD")).upper()
    evidence_status = str(sentiment_report.get("evidence_status", "") or "").upper()
    risk_level = str(payload.get("risk_level") or "HIGH").upper()
    if risk_level not in {"LOW", "MID", "HIGH"}:
        risk_level = "HIGH"

    if action not in {"HOLD", "CLOSE"}:
        action = "HOLD"
    if evidence_status == "INSUFFICIENT":
        action = "HOLD"
        payload["reasoning"] = "ニュース判断材料が不足しているためHOLD。"

    payload["action"] = action
    payload["symbol"] = str(payload.get("symbol") or position_context.get("symbol") or SYMBOL)
    try:
        payload["confidence"] = float(payload["confidence"]) if payload.get("confidence") is not None else None  # logged only
    except (TypeError, ValueError):
        payload["confidence"] = None
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