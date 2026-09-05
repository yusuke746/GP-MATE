from __future__ import annotations

import json
import logging
from typing import Any, Final, Literal, TypedDict

from agents.base import analysis_model, get_default_client
from agents.data.fred_client import MacroData

LOGGER = logging.getLogger(__name__)

MACRO_BIAS_VALUES: Final[tuple[Literal["BULLISH", "BEARISH", "NEUTRAL"], ...]] = (
    "BULLISH",
    "BEARISH",
    "NEUTRAL",
)

SYSTEM_PROMPT = (
    "あなたはGOLDのマクロ環境を評価する分析官です。"
    "与えられたmacro_data(金利・ドル・期待インフレ・投機ポジション・経済指標の結果)だけを根拠に、"
    "macro_bias/confidence/key_drivers/reasoningをJSONで返してください。"
    "【ドル】主軸はmacro_data.dxyの方向です。sourceがmt5:*なら日次で遅延のないドル指数(ICE-DXY相当)、"
    "fred:DTWEXBGSなら約1週間遅れの広義ドル指数です。いずれも絶対値ではなく方向で判断し、"
    "ドル安(DOWN)は金にポジティブ、ドル高(UP)は金にネガティブです。"
    "【時間軸】各系列にはchange_30d(トレンド)とchange_5d(直近)があります。両者が逆向きなら"
    "『トレンドは継続中だが直近は転換の兆し』として確信度を下げ、reasoningに明記してください。"
    "【金利】us2y(2年債利回り)は市場が織り込む政策金利期待の日次プロキシで、fed_funds(月次)より重視すること。"
    "us2yの低下は利下げ期待=金にポジティブ、上昇はネガティブです。"
    "実質金利(real_rate)は2025-2026年に金との逆相関が崩れているため補助情報に留めてください。"
    "期待インフレ(breakeven)の上昇はインフレヘッジ需要として金にポジティブです。"
    "【ポジション】positioning.cotは投機筋(managed money)の建玉です。crowding=CROWDED_LONGは"
    "買いが過密で反転リスク(利益確定売り)が高い、CROWDED_SHORTは踏み上げ余地がある、と読みます。"
    "positioning.gldはSPDR金ETFの保有量で、増加は実需流入(ポジティブ)、減少は流出(ネガティブ)です。"
    "【指標】recent_releasesは直近の高インパクト米指標で、actual/forecast/surpriseが入ります。"
    "surpriseの符号と大きさを最優先の『新情報』として扱い、first_order_readは一次的な読みに過ぎないので"
    "現在のレジーム(例: 弱いデータで利下げ期待が高まり金が買われる)に照らして解釈してください。"
    "upcoming_eventsは今後24時間の予定で、直前なら方向感を強く出さない理由になります。"
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


DXY_WEIGHT: Final[float] = 0.60
RATES_WEIGHT: Final[float] = 0.18
BREAKEVEN_WEIGHT: Final[float] = 0.12
# A 5-day move against the 30-day trend takes back part of the trend score
# (turning-point warning); a confirming 5-day move adds a little conviction.
SHORT_TERM_CONFLICT_FRACTION: Final[float] = 0.5
SHORT_TERM_CONFIRM_BONUS: Final[float] = 0.08
POSITIONING_WEIGHT: Final[float] = 0.10
GLD_FLOW_WEIGHT: Final[float] = 0.08


def _dir(block: Any, key: str = "direction") -> str:
    if not isinstance(block, dict):
        return "FLAT"
    return str(block.get(key, "FLAT") or "FLAT").upper()


def _trend_with_short_term(
    block: Any,
    weight: float,
    positive_when: str,
    label: str,
    key_drivers: list[str],
) -> float:
    """Score one series: 30d direction carries ``weight``; the 5d direction
    either confirms (small bonus) or conflicts (claws back half)."""
    trend = _dir(block, "direction")
    short = _dir(block, "direction_5d")
    negative_when = "UP" if positive_when == "DOWN" else "DOWN"
    if trend == positive_when:
        score = weight
        note = f"{label}は30日で金に追い風の方向"
    elif trend == negative_when:
        score = -weight
        note = f"{label}は30日で金に逆風の方向"
    else:
        key_drivers.append(f"{label}は30日で横ばい")
        return 0.0

    if short != "FLAT" and short != trend:
        score *= 1.0 - SHORT_TERM_CONFLICT_FRACTION
        note += "だが直近5日は逆行(転換の兆し、確信度を下げる)"
    elif short == trend:
        score += SHORT_TERM_CONFIRM_BONUS if score > 0 else -SHORT_TERM_CONFIRM_BONUS
        note += "で直近5日も同方向(継続を確認)"
    key_drivers.append(note)
    return score


def _score_positioning(positioning: Any, key_drivers: list[str]) -> float:
    if not isinstance(positioning, dict):
        return 0.0
    score = 0.0
    cot = positioning.get("cot")
    if isinstance(cot, dict) and bool(cot.get("_meta", {}).get("ok")):
        crowding = str(cot.get("crowding", "NORMAL"))
        pct = cot.get("net_percentile_window")
        if crowding == "CROWDED_LONG":
            score -= POSITIONING_WEIGHT
            key_drivers.append(f"COT: 投機筋の買いが過密(ネットロング百分位{pct})、利益確定売りの反転リスク")
        elif crowding == "CROWDED_SHORT":
            score += POSITIONING_WEIGHT
            key_drivers.append(f"COT: 投機筋の買いが薄い(百分位{pct})、踏み上げ余地")
        else:
            key_drivers.append(f"COT: 投機筋ポジションは中立圏(百分位{pct})")
    gld = positioning.get("gld")
    if isinstance(gld, dict) and bool(gld.get("_meta", {}).get("ok")):
        direction = str(gld.get("direction_5d", "FLAT"))
        if direction == "UP":
            score += GLD_FLOW_WEIGHT
            key_drivers.append(f"GLD保有量が5日で増加({gld.get('change_5d')}{gld.get('unit', 't')})、ETF資金流入")
        elif direction == "DOWN":
            score -= GLD_FLOW_WEIGHT
            key_drivers.append(f"GLD保有量が5日で減少({gld.get('change_5d')}{gld.get('unit', 't')})、ETF資金流出")
        else:
            key_drivers.append("GLD保有量は5日で横ばい")
    return score


def _describe_releases(macro_data: Any, key_drivers: list[str]) -> None:
    releases = macro_data.get("recent_releases") if isinstance(macro_data, dict) else None
    if isinstance(releases, list):
        for release in releases[:4]:
            if not isinstance(release, dict) or release.get("actual") is None:
                continue
            unit = release.get("unit") or ""
            key_drivers.append(
                f"指標 {release.get('title')}: 結果{release.get('actual')}{unit} "
                f"予想{release.get('forecast')}{unit} (サプライズ{release.get('surprise')})"
            )
    upcoming = macro_data.get("upcoming_events") if isinstance(macro_data, dict) else None
    if isinstance(upcoming, list) and upcoming:
        first = upcoming[0]
        if isinstance(first, dict):
            key_drivers.append(f"予定: {first.get('title')} まで{first.get('hours_ahead')}時間")


def _score_macro_environment(fred_data: MacroData) -> tuple[Literal["BULLISH", "BEARISH", "NEUTRAL"], float, list[str], str]:
    dxy = fred_data.get("dxy", {})
    fed_funds = fred_data.get("fed_funds", {})
    us2y = fred_data.get("us2y", {})
    breakeven = fred_data.get("breakeven", {})
    real_rate = fred_data.get("real_rate", {})

    dxy_direction = _dir(dxy)
    fed_direction = _dir(fed_funds)
    breakeven_direction = _dir(breakeven)
    real_rate_direction = _dir(real_rate)

    score = 0.0
    key_drivers: list[str] = []

    dxy_label = "ドル指数" + ("(MT5日次)" if str(dxy.get("source", "")).startswith("mt5") else "(DTWEXBGS)")
    score += _trend_with_short_term(dxy, DXY_WEIGHT, positive_when="DOWN", label=dxy_label, key_drivers=key_drivers)
    if dxy_direction == "DOWN":
        key_drivers.append("ドル安で金に追い風")
    elif dxy_direction == "UP":
        key_drivers.append("ドル高で金に逆風")

    # Policy-rate expectations: daily 2y yield when present, monthly fed funds otherwise.
    if isinstance(us2y, dict) and us2y.get("value") is not None:
        score += _trend_with_short_term(us2y, RATES_WEIGHT, positive_when="DOWN", label="2年債利回り(利下げ期待)", key_drivers=key_drivers)
        if fed_direction == "DOWN":
            key_drivers.append("FEDFUNDS(月次)も低下方向")
        elif fed_direction == "UP":
            key_drivers.append("FEDFUNDS(月次)は上昇方向")
    elif fed_direction == "DOWN":
        score += RATES_WEIGHT
        key_drivers.append("FEDFUNDSが低下方向で、利下げ期待が金に追い風")
    elif fed_direction == "UP":
        score -= RATES_WEIGHT
        key_drivers.append("FEDFUNDSが上昇方向で、引き締め継続が金に逆風")
    else:
        key_drivers.append("FEDFUNDSは横ばいで、政策金利要因は中立寄り")

    if breakeven_direction == "UP":
        score += BREAKEVEN_WEIGHT
        key_drivers.append("期待インフレ率が上昇しており、インフレヘッジ需要で金に追い風")
    elif breakeven_direction == "DOWN":
        score -= BREAKEVEN_WEIGHT
        key_drivers.append("期待インフレ率が低下しており、インフレヘッジ需要はやや後退")
    else:
        key_drivers.append("期待インフレ率は横ばいで、ヘッジ需要の変化は限定的")

    score += _score_positioning(fred_data.get("positioning"), key_drivers)
    _describe_releases(fred_data, key_drivers)

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
    # The rule-based bias/confidence are authoritative. The LLM narrative is
    # adopted only when its own conclusion agrees with that bias — otherwise the
    # final payload would state e.g. bias=BULLISH with a reasoning text that
    # concludes NEUTRAL, and downstream agents cannot tell which to trust.
    llm_bias = _normalize_bias(llm_payload.get("macro_bias"))
    reasoning = str(llm_payload.get("reasoning") or "").strip()
    if reasoning and llm_bias != baseline["macro_bias"]:
        LOGGER.warning(
            "macro_analyst: LLM bias %s disagrees with rule-based bias %s; keeping rule-based reasoning",
            llm_bias,
            baseline["macro_bias"],
        )
        reasoning = ""

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
            "macro_data": fred_data,
            "rule_based_baseline": {
                "macro_bias": baseline_bias,
                "confidence": baseline_confidence,
                "note": "ルールベースの暫定判定。指標サプライズやポジションの偏りが強ければreasoningで補正理由を述べること",
            },
            "requirements": {
                "dxy_priority": "macro_data.dxyの方向(30d)と直近(5d)を最重要視し、絶対値ではなく方向で判定する",
                "rates_priority": "us2y(2年債)を政策金利期待の主指標とし、fed_fundsは背景情報",
                "releases_priority": "recent_releasesのsurpriseを最新の新情報として最優先で解釈する",
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
