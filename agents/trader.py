from __future__ import annotations

import json
from typing import Any

from agents.base import decision_model, get_default_client
from config import CONFIDENCE_THRESHOLD, SYMBOL, MACRO_BIAS_CARRY_THRESHOLD

SYSTEM_PROMPT = (
    "あなたは最終決定権を持つトレーダーです。"
    "必ず place_trade_order 関数を呼び出して最終判断を返してください。"
    "confidenceは判断の確からしさ(0-1)を正直に申告すること。"
    "エントリー可否の閾値判定はシステム側で行うため、閾値を意識して数値を調整しないこと。"
    "HOLDの場合でも、macro/technical/sentimentが方向性を示すなら "
    "directional_bias(BULLISH/BEARISH)とbias_strength、trigger_conditions"
    "(key_levelsに基づく発動価格条件)を必ず設定すること。"
    "特にmacroのmacro_biasとconfidenceは、テクニカルがレンジでも"
    "directional_biasに反映すること。"
    "action(BUY/SELL/HOLD)の方向判断はtechnical/macro/sentiment/debateに基づき、"
    "technical_report内のtp_reference_onlyを方向判断に使ってはならない。"
    "tp_reference_onlyはsuggested_tpの算出にのみ使用する。"
    "actionがBUY/SELLの場合、tp_reference_onlyのlevels/round_numbers/prev_day/moving_averagesを参照し、"
    "反発が予想される強レベルの手前にsuggested_tpを数値で設定すること。"
    "複数根拠が重なるほど強いのでconfluence_noteを重視すること。"
    "direction_context.technical.extensionがD1の伸び切り(BBミドルから2ATR超の乖離)を"
    "示す場合、直近の急騰・急落に追随するエントリーは平均回帰による反転リスクが高い。"
    "その局面では押し目/戻りを待つHOLDを優先的に検討し、"
    "それでもエントリーする場合はreasoningで伸び切りリスクを上回る根拠を明示すること。"
    "recent_contextには直近24時間の自分の判断履歴(decisions)と決済結果(recent_closed)が含まれる。"
    "過去判断への盲従は不要だが、数時間前の自分のHOLD判断を覆してエントリーする場合は、"
    "前回から何が新しく変わったのかをreasoningに明示すること。"
    "recent_closedに直近2時間以内のLOSSがあり、同方向へ再エントリーする場合は、"
    "明確な状況変化がない限り見送る(HOLD)こと。"
    "direction_context.technical.adxが強いトレンドを示す場合は、"
    "手前のサポレジで反発しにくいためsuggested_tpを遠めに設定してよい。"
    "ただし最終TPはリスクリワード2Rが上限であり、2Rを超えるsuggested_tpは2Rに丸められる。"
    "したがってsuggested_tpは原則2R以内で、最も反発が強そうなレベルの手前に置くこと。"
    "HOLDの場合や算出根拠が不十分な場合、suggested_tpはnullにすること。"
    "suggested_tp_basisには、その価格にした根拠を簡潔な日本語で記すこと。"
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
            "suggested_tp": {
                "type": ["number", "null"],
                "description": "利確目標価格。反発が予想される強レベルの手前に置く。算出できなければnull",
            },
            "suggested_tp_basis": {
                "type": "string",
                "description": "suggested_tpの根拠(例: キリ番4000と前日安値が重なる4023の手前)",
            },
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


def _safe_float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def _extract_current_price_for_tp_sanity(technical_report: dict[str, Any]) -> float | None:
    try:
        direction_context = technical_report.get("direction_context", {})
        if isinstance(direction_context, dict):
            h1 = direction_context.get("h1", {})
            if isinstance(h1, dict):
                close_1 = _safe_float_or_none(h1.get("close"))
                if close_1 is not None:
                    return close_1

        key_levels = technical_report.get("key_levels", {})
        if isinstance(key_levels, dict):
            frames = key_levels.get("frames", {})
            if isinstance(frames, dict):
                h1_frame = frames.get("h1", {})
                if isinstance(h1_frame, dict):
                    close_2 = _safe_float_or_none(h1_frame.get("close"))
                    if close_2 is not None:
                        return close_2
    except Exception:
        return None
    return None


def decide_trade(
    technical_report: dict[str, Any],
    sentiment_report: dict[str, Any],
    debate_report: dict[str, Any],
    macro_report: dict[str, Any] | None = None,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
    recent_context: dict[str, Any] | None = None,
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
        "recent_context": recent_context or {"decisions": [], "recent_closed": []},
        # The confidence threshold is intentionally NOT exposed to the model:
        # it is enforced in code below, and telling the model the cutoff lets
        # it anchor its self-reported confidence around it.
        "constraints": {
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

    current_price = _extract_current_price_for_tp_sanity(technical_report)
    suggested_tp = _safe_float_or_none(payload.get("suggested_tp"))
    if action == "HOLD":
        suggested_tp = None
    elif suggested_tp is not None and current_price is not None:
        if action == "BUY" and suggested_tp <= current_price:
            suggested_tp = None
        elif action == "SELL" and suggested_tp >= current_price:
            suggested_tp = None

    suggested_tp_basis = str(payload.get("suggested_tp_basis", "") or "")
    payload["suggested_tp"] = suggested_tp
    payload["suggested_tp_basis"] = suggested_tp_basis

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
