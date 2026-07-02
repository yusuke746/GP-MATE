from __future__ import annotations

import json
from typing import Any

from agents.base import analysis_model, get_default_client

BULL_SYSTEM_PROMPT = "あなたは強気リサーチャーです。事実ベースで買い根拠をJSONで返してください。"
BEAR_SYSTEM_PROMPT = "あなたは弱気リサーチャーです。リスクと反証をJSONで返してください。"

BULL_FALLBACK: dict[str, Any] = {
    "bull_case": "分析失敗につき十分な買い根拠なし。",
    "conviction": 0.0,
}

BEAR_FALLBACK: dict[str, Any] = {
    "bear_case": "分析失敗につき見送りが妥当。",
    "conviction": 1.0,
}


def run_debate(
    technical_report: dict[str, Any],
    sentiment_report: dict[str, Any],
) -> dict[str, Any]:
    base_context = {
        "technical": technical_report,
        "sentiment": sentiment_report,
    }

    bull_round_1 = get_default_client().call_json(
        system_prompt=BULL_SYSTEM_PROMPT,
        user_prompt=json.dumps(base_context, ensure_ascii=False),
        model=analysis_model(),
        fallback_payload=BULL_FALLBACK,
    )

    bear_round_1 = get_default_client().call_json(
        system_prompt=BEAR_SYSTEM_PROMPT,
        user_prompt=json.dumps(
            {
                **base_context,
                "bull_case": bull_round_1.payload,
            },
            ensure_ascii=False,
        ),
        model=analysis_model(),
        fallback_payload=BEAR_FALLBACK,
    )

    bull_round_2 = get_default_client().call_json(
        system_prompt=BULL_SYSTEM_PROMPT,
        user_prompt=json.dumps(
            {
                **base_context,
                "bull_round_1": bull_round_1.payload,
                "bear_round_1": bear_round_1.payload,
            },
            ensure_ascii=False,
        ),
        model=analysis_model(),
        fallback_payload=BULL_FALLBACK,
    )

    bear_round_2 = get_default_client().call_json(
        system_prompt=BEAR_SYSTEM_PROMPT,
        user_prompt=json.dumps(
            {
                **base_context,
                "bull_round_1": bull_round_1.payload,
                "bear_round_1": bear_round_1.payload,
                "bull_round_2": bull_round_2.payload,
            },
            ensure_ascii=False,
        ),
        model=analysis_model(),
        fallback_payload=BEAR_FALLBACK,
    )

    total_prompt = (
        bull_round_1.usage.prompt_tokens
        + bear_round_1.usage.prompt_tokens
        + bull_round_2.usage.prompt_tokens
        + bear_round_2.usage.prompt_tokens
    )
    total_completion = (
        bull_round_1.usage.completion_tokens
        + bear_round_1.usage.completion_tokens
        + bull_round_2.usage.completion_tokens
        + bear_round_2.usage.completion_tokens
    )

    return {
        "round1": {
            "bull": bull_round_1.payload,
            "bear": bear_round_1.payload,
        },
        "round2": {
            "bull": bull_round_2.payload,
            "bear": bear_round_2.payload,
        },
        "_meta": {
            "ok": all(
                [
                    bull_round_1.ok,
                    bear_round_1.ok,
                    bull_round_2.ok,
                    bear_round_2.ok,
                ]
            ),
            "usage": {
                "prompt_tokens": total_prompt,
                "completion_tokens": total_completion,
                "total_tokens": total_prompt + total_completion,
            },
        },
    }
