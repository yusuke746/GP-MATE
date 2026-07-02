from __future__ import annotations

import json
from typing import Any

from agents.base import analysis_model, get_default_client

SYSTEM_PROMPT = (
    "あなたは金融ニュースのセンチメント専門家です。"
    "ニュース見出しを評価し、score/dominant_news/reasoningをJSONで返してください。"
)

FALLBACK_RESPONSE: dict[str, Any] = {
    "score": 0.0,
    "dominant_news": "N/A",
    "reasoning": "ニュース分析失敗のため中立判定。",
    "news_count": 0,
    "evidence_status": "UNAVAILABLE",
}


def analyze_sentiment(news_items: list[dict[str, Any]]) -> dict[str, Any]:
    if len(news_items) == 0:
        return {
            "score": 0.0,
            "dominant_news": "N/A",
            "reasoning": "ニュースが取得できず判断材料不足。安全側で見送りを推奨。",
            "news_count": 0,
            "evidence_status": "INSUFFICIENT",
            "_meta": {
                "ok": True,
                "model": "none",
                "error": "",
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
            },
        }

    user_prompt = (
        "以下ニュースを強気/中立/弱気で評価し、score(-1~1)を算出してください。\n"
        f"{json.dumps(news_items, ensure_ascii=False)}"
    )

    result = get_default_client().call_json(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        model=analysis_model(),
        fallback_payload=FALLBACK_RESPONSE,
    )

    payload = dict(result.payload)
    payload["news_count"] = len(news_items)
    payload["evidence_status"] = "SUFFICIENT"
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
