from __future__ import annotations

import json
from typing import Any

from agents.base import analysis_model, get_default_client

SYSTEM_PROMPT = (
    "あなたは経験20年のテクニカルアナリストです。"
    "与えられたデータのみで判断し、必ずJSONで返してください。"
)


FALLBACK_RESPONSE: dict[str, Any] = {
    "trend": "RANGE",
    "signal": "NEUTRAL",
    "key_levels": {},
    "reasoning": "テクニカル分析に失敗したため保守的にNEUTRAL。",
}


def analyze_technical(indicator_payload: dict[str, Any]) -> dict[str, Any]:
    user_prompt = (
        "以下のインジケータ情報から、trend/signal/key_levels/reasoningをJSONで返してください。\n"
        f"{json.dumps(indicator_payload, ensure_ascii=False)}"
    )

    result = get_default_client().call_json(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        model=analysis_model(),
        fallback_payload=FALLBACK_RESPONSE,
    )

    payload = dict(result.payload)
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
