"""News sentiment analyst.

The analyst is asked what is *new* in the headlines and whether that supports
the current price trend continuing or reversing. It is no longer told how to
weight particular headline types; it is told what the headlines are and asked
for its reading. A numeric ``score`` is still emitted for the trade log and
the debate gate, derived from the analyst's stated bias when it does not give
one itself.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Final

from agents.base import analysis_model, get_default_client

LOGGER = logging.getLogger(__name__)

REGIME_VIEW_VALUES: Final[tuple[str, ...]] = ("SUPPORTS_CONTINUATION", "SUPPORTS_REVERSAL", "UNCLEAR")
BIAS_SCORE: Final[dict[str, float]] = {"BULLISH": 0.5, "BEARISH": -0.5, "NEUTRAL": 0.0}

SYSTEM_PROMPT = (
    "あなたはGOLD(XAU/USD)のニュースを読む専門家です。"
    "与えられた見出し(と、あれば本文の抜粋・経済指標の結果)だけを材料に、あなた自身の判断で答えてください。"
    "見出しの種類ごとの採点規則はありません。あなたの読みがそのまま採用されます。"
    "答える問い: (1) この中で『金価格にとって新しい情報』は何か(すでに起きた値動きをなぞるだけの見出しと区別する)。"
    "(2) 新しい情報を総合すると、金にとって強気・弱気・中立のどれか。"
    "(3) それはいまの価格トレンドの継続を支えるか、反転を促すか、どちらとも言えないか。"
    "(4) 最も影響の大きい見出しはどれか。"
    "株式やドルへの影響ではなく、金価格への影響として評価すること。"
    "材料が乏しい、または互いに打ち消し合うときはNEUTRAL/UNCLEARでよく、無理に方向を出さないこと。"
    "出力は次のキーだけを持つJSON: "
    "{gold_bias: 'BULLISH'|'BEARISH'|'NEUTRAL', regime_view: 'SUPPORTS_CONTINUATION'|'SUPPORTS_REVERSAL'|'UNCLEAR', "
    "new_information: string[](新しい情報の要約、無ければ空), price_echo_count: number(値動きをなぞるだけの見出しの本数), "
    "dominant_news: string, reasoning: string(日本語)}"
)

FALLBACK_RESPONSE: dict[str, Any] = {
    "score": 0.0,
    "gold_bias": "NEUTRAL",
    "regime_view": "UNCLEAR",
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
    """Guarantee gold_bias / regime_view / score / dominant_news / reasoning.

    ``score`` comes from the analyst if given, else from ``gold_bias``, else
    from the mean of per-item ``evaluations`` (some models still return those),
    else 0.0. It exists for the log and the debate gate, not as the analysis.
    """
    bias = str(payload.get("gold_bias", "") or "").upper().strip()
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

    if score is None and bias in BIAS_SCORE:
        score = BIAS_SCORE[bias]
    if score is None and items:
        score = sum(value for value, _ in items) / len(items)
        LOGGER.info("sentiment: top-level score missing; aggregated %.3f from %d item evaluations", score, len(items))
    if score is None:
        score = 0.0
    payload["score"] = max(-1.0, min(1.0, score))

    if bias not in BIAS_SCORE:
        bias = "BULLISH" if payload["score"] >= 0.15 else ("BEARISH" if payload["score"] <= -0.15 else "NEUTRAL")
    payload["gold_bias"] = bias

    regime_view = str(payload.get("regime_view", "") or "").upper().strip()
    payload["regime_view"] = regime_view if regime_view in REGIME_VIEW_VALUES else "UNCLEAR"

    new_info = payload.get("new_information")
    payload["new_information"] = [str(x) for x in new_info if str(x).strip()] if isinstance(new_info, list) else []
    try:
        payload["price_echo_count"] = int(payload.get("price_echo_count") or 0)
    except (TypeError, ValueError):
        payload["price_echo_count"] = 0

    if not str(payload.get("dominant_news", "") or "").strip():
        if items:
            _, dominant = max(items, key=lambda pair: abs(pair[0]))
            payload["dominant_news"] = str(dominant.get("title") or dominant.get("dominant_news") or "N/A")
        else:
            payload["dominant_news"] = "N/A"

    if not str(payload.get("reasoning", "") or "").strip():
        if items:
            parts = [str(item.get("reasoning", "") or "").strip() for _, item in items if str(item.get("reasoning", "") or "").strip()]
            payload["reasoning"] = " / ".join(parts[:4]) if parts else "個別評価の平均から総合スコアを算出。"
        else:
            payload["reasoning"] = "総合スコアの根拠が取得できなかったため中立寄りで扱う。"

    return payload


def feed_health(feed_meta: Any) -> str:
    """GOOD (all feeds answered) / DEGRADED (some) / BAD (none) / UNKNOWN (no meta)."""
    if not isinstance(feed_meta, dict):
        return "UNKNOWN"
    try:
        total = int(feed_meta.get("feeds_total", 0) or 0)
        live = int(feed_meta.get("feeds_live", 0) or 0)
    except (TypeError, ValueError):
        return "UNKNOWN"
    if total <= 0:
        return "UNKNOWN"
    if live <= 0:
        return "BAD"
    return "GOOD" if live >= total else "DEGRADED"


def _no_llm_report(*, evidence_status: str, reasoning: str, health: str, news_count: int) -> dict[str, Any]:
    return {
        "score": 0.0,
        "gold_bias": "NEUTRAL",
        "regime_view": "UNCLEAR",
        "new_information": [],
        "price_echo_count": 0,
        "dominant_news": "N/A",
        "reasoning": reasoning,
        "news_count": news_count,
        "evidence_status": evidence_status,
        "feed_health": health,
        "_meta": {
            "ok": True,
            "model": "none",
            "error": "",
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        },
    }


def sentiment_for_feed_state(news_items: list[dict[str, Any]], feed_meta: Any) -> dict[str, Any] | None:
    """Decide from the feed state whether the analyst should run at all.

    BAD feeds (nothing answered): INSUFFICIENT, the trader holds, regardless
    of how many items arrived by other routes.
    Feeds fine but zero items: NO_NEWS, neutral, trading continues.
    Otherwise None: run the analyst.
    """
    health = feed_health(feed_meta)
    if health == "BAD":
        return _no_llm_report(
            evidence_status="INSUFFICIENT",
            reasoning="ニュースフィードが全て取得できず判断材料不足。安全側で見送り。",
            health=health,
            news_count=len(news_items),
        )
    if len(news_items) == 0 and health in {"GOOD", "DEGRADED"}:
        return _no_llm_report(
            evidence_status="NO_NEWS",
            reasoning="フィードは正常だが新規ニュースなし。ニュースは中立として扱う。",
            health=health,
            news_count=0,
        )
    return None


def analyze_sentiment(news_items: list[dict[str, Any]]) -> dict[str, Any]:
    if len(news_items) == 0:
        return {
            "score": 0.0,
            "gold_bias": "NEUTRAL",
            "regime_view": "UNCLEAR",
            "new_information": [],
            "price_echo_count": 0,
            "dominant_news": "N/A",
            "reasoning": "ニュースが取得できず判断材料不足。安全側で見送りを推奨。",
            "news_count": 0,
            "evidence_status": "INSUFFICIENT",
            "_meta": {
                "ok": True,
                "model": "none",
                "error": "",
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            },
        }

    user_prompt = (
        "以下の見出し群について、金にとって新しい情報は何か、総合して強気/弱気/中立のどれか、"
        "いまのトレンドの継続を支えるか反転を促すかを、あなた自身の判断で答えてください。\n"
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
