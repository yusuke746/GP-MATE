from __future__ import annotations

import json
from typing import Any, Final, Literal, TypedDict

from agents.base import analysis_model, get_default_client
from agents.data.fred_client import MacroData

MACRO_BIAS_VALUES: Final[tuple[Literal["BULLISH", "BEARISH", "NEUTRAL"], ...]] = (
    "BULLISH",
    "BEARISH",
    "NEUTRAL",
)

SYSTEM_PROMPT = (
    "あなたはGOLDのマクロ環境を評価する分析官です。"
    "FREDデータだけを根拠に、macro_bias/confidence/key_drivers/reasoningをJSONで返してください。"
    "最重要の主軸はDTWEXBGSの方向です。DTWEXBGSは広義ドル指数(2006=100基準)であり、"
    "一般的なICE-DXY(90-110)とは水準が異なるため、絶対値ではなく方向(UP/DOWN/FLAT)だけで判断してください。"
    "ドル安(DOWN)は金にポジティブ、ドル高(UP)は金にネガティブです。"
    "FEDFUNDSの方向も重視してください。利下げ方向は金にポジティブ、利上げ/据え置き長期化はネガティブです。"
    "実質金利については、従来の『実質金利上昇→金下落』の単純逆相関だけで判定しないでください。"
    "2025-2026年は財政悪化・通貨信認低下のヘッジ需要により逆相関が崩壊しており、"
    "実質金利が高くても金が上昇しうるため、補助的な文脈情報としてのみ扱ってください。"
    "期待インフレの上昇はインフレヘッジ需要として金にポジティブです。"
    "絶対に例外を投げず、安全側の判断を優先してください。"
)

FALLBACK_REASONING = "FREDまたはLLMの利用に失敗したため、安全側で中立判定。"


class MacroAnalysisMeta(TypedDict):
    ok: bool
    model: str
    usage: dict[str, int]
    error: str


class MacroAnalysisResult(TypedDict):
    macro_bias: Literal["BULLISH", "BEARISH", "NEUTRAL"]
    confidence: float
    key_drivers: list[str]
    reasoning: str
    _meta: MacroAnalysisMeta


def _empty_usage() -> dict[str, int]:
    return {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }


def _clamp_confidence(value: float) -> float:
    return max(0.0, min(1.0, value))


def _normalize_bias(value: Any) -> Literal["BULLISH", "BEARISH", "NEUTRAL"]:
    text = str(value or "").upper()
    if text in {"BULLISH", "BEARISH", "NEUTRAL"}:
        return text  # type: ignore[return-value]
    return "NEUTRAL"


def _direction_text(value: str) -> str:
    return {"UP": "上昇", "DOWN": "下落", "FLAT": "横ばい"}.get(value, "不明")


def _is_positive_direction(value: str) -> bool:
    return value == "DOWN"


def _is_negative_direction(value: str) -> bool:
    return value == "UP"


def _score_macro_environment(fred_data: MacroData) -> tuple[Literal["BULLISH", "BEARISH", "NEUTRAL"], float, list[str], str]:
    dxy = fred_data.get("dxy", {})
    fed_funds = fred_data.get("fed_funds", {})
    breakeven = fred_data.get("breakeven", {})
    real_rate = fred_data.get("real_rate", {})

    dxy_direction = str(dxy.get("direction", "FLAT") or "FLAT").upper()
    fed_direction = str(fed_funds.get("direction", "FLAT") or "FLAT").upper()
    breakeven_direction = str(breakeven.get("direction", "FLAT") or "FLAT").upper()
    real_rate_direction = str(real_rate.get("direction", "FLAT") or "FLAT").upper()

    score = 0.0
    key_drivers: list[str] = []

    if dxy_direction == "DOWN":
        score += 0.60
        key_drivers.append("DTWEXBGSが下落しており、ドル安で金に追い風")
    elif dxy_direction == "UP":
        score -= 0.60
        key_drivers.append("DTWEXBGSが上昇しており、ドル高で金に逆風")
    else:
        key_drivers.append("DTWEXBGSは横ばいで、ドル要因の方向性は限定的")

    if fed_direction == "DOWN":
        score += 0.18
        key_drivers.append("FEDFUNDSが低下方向で、利下げ期待が金に追い風")
    elif fed_direction == "UP":
        score -= 0.18
        key_drivers.append("FEDFUNDSが上昇方向で、引き締め継続が金に逆風")
    else:
        key_drivers.append("FEDFUNDSは横ばいで、政策金利要因は中立寄り")

    if breakeven_direction == "UP":
        score += 0.12
        key_drivers.append("期待インフレ率が上昇しており、インフレヘッジ需要で金に追い風")
    elif breakeven_direction == "DOWN":
        score -= 0.12
        key_drivers.append("期待インフレ率が低下しており、インフレヘッジ需要はやや後退")
    else:
        key_drivers.append("期待インフレ率は横ばいで、ヘッジ需要の変化は限定的")

    if real_rate_direction == "UP":
        key_drivers.append(
            "実質金利は上昇方向だが、2025-2026年は逆相関が崩れているため参考情報に留める"
        )
    elif real_rate_direction == "DOWN":
        key_drivers.append(
            "実質金利は低下方向だが、2025-2026年は補助的な文脈情報としてのみ扱う"
        )
    else:
        key_drivers.append("実質金利は横ばいで、補助情報としての影響は小さい")

    if score >= 0.25:
        macro_bias: Literal["BULLISH", "BEARISH", "NEUTRAL"] = "BULLISH"
    elif score <= -0.25:
        macro_bias = "BEARISH"
    else:
        macro_bias = "NEUTRAL"

    confidence = _clamp_confidence(0.5 + min(0.35, abs(score) * 0.35))

    reasoning = (
        f"主軸のドル要因は{_direction_text(dxy_direction)}で、"
        f"政策金利は{_direction_text(fed_direction)}、期待インフレは{_direction_text(breakeven_direction)}。"
        "実質金利は2025-2026年の構造変化により補助的に扱い、単純な逆相関では判定しない。"
    )
    return macro_bias, confidence, key_drivers, reasoning


def _build_neutral_result(error: str, model: str = "none", ok: bool = False) -> MacroAnalysisResult:
    return {
        "macro_bias": "NEUTRAL",
        "confidence": 0.5,
        "key_drivers": ["FRED取得失敗またはLLM失敗のため安全側で中立"],
        "reasoning": FALLBACK_REASONING,
        "_meta": {
            "ok": ok,
            "model": model,
            "usage": _empty_usage(),
            "error": error,
        },
    }


def _merge_llm_result(
    baseline: MacroAnalysisResult,
    llm_payload: dict[str, Any],
) -> MacroAnalysisResult:
    reasoning = str(llm_payload.get("reasoning") or baseline["reasoning"]).strip()
    key_drivers_raw = llm_payload.get("key_drivers", baseline["key_drivers"])
    key_drivers = [str(item) for item in key_drivers_raw] if isinstance(key_drivers_raw, list) else baseline["key_drivers"]

    return {
        "macro_bias": baseline["macro_bias"],
        "confidence": baseline["confidence"],
        "key_drivers": key_drivers,
        "reasoning": reasoning or baseline["reasoning"],
        "_meta": baseline["_meta"],
    }


def analyze_macro_environment(fred_data: MacroData) -> MacroAnalysisResult:
    meta = fred_data.get("_meta", {}) if isinstance(fred_data, dict) else {}
    if not isinstance(meta, dict) or not bool(meta.get("ok", False)):
        return _build_neutral_result("FRED data unavailable", model=str(meta.get("model", "none") if isinstance(meta, dict) else "none"))

    baseline_bias, baseline_confidence, key_drivers, reasoning = _score_macro_environment(fred_data)
    baseline: MacroAnalysisResult = {
        "macro_bias": baseline_bias,
        "confidence": baseline_confidence,
        "key_drivers": key_drivers,
        "reasoning": reasoning,
        "_meta": {
            "ok": True,
            "model": "rule_based",
            "usage": _empty_usage(),
            "error": "",
        },
    }

    user_prompt = json.dumps(
        {
            "fred_data": fred_data,
            "requirements": {
                "dxy_priority": "DTWEXBGSの方向を最重要視し、絶対値ではなく方向で判定する",
                "real_rate_caveat": "2025-2026年は実質金利と金の逆相関が崩れているため単純弱気に使わない",
                "output_format": {
                    "macro_bias": MACRO_BIAS_VALUES,
                    "confidence": "0.0-1.0",
                    "key_drivers": "list[str]",
                    "reasoning": "string",
                },
            },
        },
        ensure_ascii=False,
    )

    client = get_default_client()
    try:
        result = client.call_json(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            model=analysis_model(),
            fallback_payload={
                "macro_bias": baseline_bias,
                "confidence": baseline_confidence,
                "key_drivers": key_drivers,
                "reasoning": reasoning,
            },
        )
    except Exception as exc:
        return _build_neutral_result(f"LLM call failed: {exc}", model=analysis_model())

    if not bool(result.ok):
        return _build_neutral_result(result.error or "LLM call failed", model=result.model)

    payload = dict(result.payload)
    merged = _merge_llm_result(baseline, payload)
    merged["_meta"] = {
        "ok": True,
        "model": result.model,
        "usage": {
            "prompt_tokens": result.usage.prompt_tokens,
            "completion_tokens": result.usage.completion_tokens,
            "total_tokens": result.usage.total_tokens,
        },
        "error": "",
    }
    merged["macro_bias"] = baseline_bias
    merged["confidence"] = baseline_confidence
    merged["key_drivers"] = key_drivers
    return merged
