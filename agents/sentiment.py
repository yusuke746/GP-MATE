from __future__ import annotations

import json
import logging
from typing import Any

from agents.base import analysis_model, get_default_client

LOGGER = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "あなたはGOLD(XAU/USD)のニュースセンチメント専門家です。"
    "ニュース全体が『金価格にとって』強気か弱気かを評価してください。"
    "注意: 株高・リスクオン材料は株式には強気でも、安全資産の金には中立〜弱気になり得る。"
    "必ず金価格への影響として評価すること。"
    "出力形式: 必ずトップレベルに次の3キーを持つJSONのみを返すこと: "
    "score(-1〜1の総合値。全ニュースを1つに集約した金にとっての方向), "
    "dominant_news(最も影響の大きい見出し), reasoning(日本語の説明)。"
    "ニュースごとの個別評価をevaluationsキーに含めてもよいが、"
    "総合scoreのトップレベル出力を省略してはならない。"
)

FALLBACK_RESPONSE: dict[str, Any] = {
    "score": 0.0,
    "dominant_news": "N/A",
    "reasoning": "ニュース分析失敗のため中立判定。",
    "news_count": 0,
    "evidence_status": "UNAVAILABLE",
}


def _safe_float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def _normalize_sentiment_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Guarantee a top-level aggregate score/dominant_news/reasoning.

    Some models return only per-item `evaluations` despite the format
    instruction; without this, downstream direction logic reads score=0.0
    (always NEUTRAL). Aggregate from the items when the top level is missing.
    """
    score = _safe_float_or_none(payload.get("score"))

    evaluations = payload.get("evaluations")
    items: list[tuple[float, dict[str, Any]]] = []
    if isinstance(evaluations, list):
        for item in evaluations:
            if not isinstance(item, dict):
                continue
            item_score = _safe_float_or_none(item.get("score"))
            if item_score is not None:
                items.append((item_score, item))

    if score is None and items:
        score = sum(value for value, _ in items) / len(items)
        LOGGER.info(
            "sentiment: top-level score missing; aggregated %.3f from %d item evaluations",
            score,
            len(items),
        )

    if score is None:
        score = 0.0

    payload["score"] = max(-1.0, min(1.0, score))

    if not str(payload.get("dominant_news", "") or "").strip():
        if items:
            _, dominant = max(items, key=lambda pair: abs(pair[0]))
            payload["dominant_news"] = str(
                dominant.get("title") or dominant.get("dominant_news") or "N/A"
            )
        else:
            payload["dominant_news"] = "N/A"

    if not str(payload.get("reasoning", "") or "").strip():
        if items:
            parts = [
                str(item.get("reasoning", "") or "").strip()
                for _, item in items
                if str(item.get("reasoning", "") or "").strip()
            ]
            payload["reasoning"] = (
                " / ".join(parts[:4]) if parts else "個別評価の平均から総合スコアを算出。"
            )
        else:
            payload["reasoning"] = "総合スコアの根拠が取得できなかったため中立寄りで扱う。"

    return payload


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
        "以下ニュースが金(XAU/USD)にとって強気/中立/弱気かを評価し、"
        "全体を1つに集約したscore(-1~1)をトップレベルに算出してください。\n"
        f"{json.dumps(news_items, ensure_ascii=False)}"
    )

    result = get_default_client().call_json(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        model=analysis_model(),
        fallback_payload=FALLBACK_RESPONSE,
    )

    payload = _normalize_sentiment_payload(dict(result.payload))
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
