"""Macro analyst: reads the macro inputs and gives its own view.

Earlier versions scored the series with a fixed weight table, told the model
the answer ("rule_based_baseline") and then discarded the model's bias if it
disagreed. That made the "analyst" an if-statement with a narrator. Now the
model is handed the data with provenance notes only (what each series is, how
fresh it is) and asked what it thinks. Nothing in code re-decides the bias.
The only intervention is fail-safe: if the model does not answer, the report
is NEUTRAL and marked as such.
"""

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
REGIME_VIEW_VALUES: Final[tuple[str, ...]] = ("SUPPORTS_CONTINUATION", "SUPPORTS_REVERSAL", "UNCLEAR")
DATA_QUALITY_VALUES: Final[tuple[str, ...]] = ("GOOD", "PARTIAL", "POOR")

SYSTEM_PROMPT = (
    "あなたはGOLD(XAU/USD)のマクロ環境を評価する分析官です。"
    "与えられたmacro_dataだけを材料に、あなた自身の判断で、いまのマクロ環境が金価格にとって"
    "強気(BULLISH)・弱気(BEARISH)・中立(NEUTRAL)のどれかを述べてください。"
    "系列ごとの採点表や『この方向ならこう読む』という固定の規則はありません。あなたの読みがそのまま採用されます。"
    "教科書的な関係(ドル・金利・期待インフレ・ポジション)が現在も成り立っているかどうかも、"
    "与えられた数値から自分で判断してください。成り立っていないと考える根拠があればそう書いてよい。"
    "あわせて、マクロ要因はいまの価格トレンドの継続を支えるか(SUPPORTS_CONTINUATION)、"
    "反転を促すか(SUPPORTS_REVERSAL)、どちらとも言えないか(UNCLEAR)を答え、"
    "その読みが崩れる条件(invalidation: 例『次回CPIが予想を上回る』『ドル指数が◯◯を上抜く』)を1つ書いてください。"
    "材料が乏しい、または互いに打ち消し合うときはNEUTRAL/UNCLEARでよく、無理に方向を出さないこと。"
    "確信度の数値は求めません。強い読みなら、その根拠となる系列名と数値をkey_driversに具体的に列挙してください。"
    "結論に反する材料(counter_evidence)も探して列挙し(無ければ空)、"
    "データ品質 data_quality を GOOD / PARTIAL(欠損・古い値あり) / POOR(判断に足りない) で答え、"
    "判断を保留するなら abstain_reason にその理由を書いて macro_bias=NEUTRAL とすること。"
    "出力は次のキーだけを持つJSON: "
    "{macro_bias: 'BULLISH'|'BEARISH'|'NEUTRAL', regime_view: 'SUPPORTS_CONTINUATION'|'SUPPORTS_REVERSAL'|'UNCLEAR', "
    "key_drivers: string[], counter_evidence: string[], data_quality: 'GOOD'|'PARTIAL'|'POOR', "
    "abstain_reason: string|null, invalidation: string, reasoning: string(日本語)}"
)

# Provenance only: what each field is and how fresh it is. No "this means gold up".
DATA_NOTES: Final[dict[str, str]] = {
    "dxy": "ドル指数。source が mt5:* なら日次で遅延なし(ICE-DXY相当の先物/合成)、fred:DTWEXBGS なら約1週間遅れの広義ドル指数。direction は30日変化の符号、direction_5d は5日変化の符号。",
    "us2y": "米2年債利回り(日次)。change_30d / change_5d は同期間の変化幅(pt)。",
    "us10y": "米10年債利回り(日次)。",
    "fed_funds": "実効FF金利(月次)。",
    "real_rate": "10年TIPS実質利回り(日次)。",
    "breakeven": "10年ブレークイーブン期待インフレ率(日次)。",
    "positioning.cot": "CFTC COT の managed money ネットポジション。net_percentile_window は直近ウィンドウ内の百分位、crowding はその百分位から機械的に付けたラベル(CROWDED_LONG / CROWDED_SHORT / NORMAL)。",
    "positioning.gld": "SPDR Gold Shares の保有量(unit 参照)。change_5d は5日変化。任意項目で、無い場合もある。",
    "recent_releases": "直近48時間の高インパクト米指標。actual / forecast / previous / surprise(actual-forecast)。actual_source が fred:* なら FRED の系列から補完した実績値。",
    "upcoming_events": "今後24時間の高インパクト予定と hours_ahead。",
}

FALLBACK_REASONING = "マクロ分析官が回答しなかったため、安全側で中立扱い。"


class MacroAnalysisMeta(TypedDict):
    ok: bool
    model: str
    usage: dict[str, int]
    error: str


class MacroAnalysisResult(TypedDict, total=False):
    macro_bias: Literal["BULLISH", "BEARISH", "NEUTRAL"]
    regime_view: str
    key_drivers: list[str]
    counter_evidence: list[str]
    data_quality: str
    abstain_reason: str | None
    invalidation: str
    reasoning: str
    source: str
    _meta: MacroAnalysisMeta


def _empty_usage() -> dict[str, int]:
    return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def _normalize_bias(value: Any) -> Literal["BULLISH", "BEARISH", "NEUTRAL"] | None:
    text = str(value or "").upper().strip()
    if text in {"BULLISH", "BEARISH", "NEUTRAL"}:
        return text  # type: ignore[return-value]
    return None


def _normalize_regime_view(value: Any) -> str:
    text = str(value or "").upper().strip()
    return text if text in REGIME_VIEW_VALUES else "UNCLEAR"


def _build_neutral_result(error: str, model: str = "none", ok: bool = False, source: str = "fail_safe") -> MacroAnalysisResult:
    return {
        "macro_bias": "NEUTRAL",
        "regime_view": "UNCLEAR",
        "key_drivers": ["マクロ分析が得られなかったため安全側で中立"],
        "counter_evidence": [],
        "data_quality": "POOR",
        "abstain_reason": error,
        "invalidation": "",
        "reasoning": FALLBACK_REASONING,
        "source": source,
        "_meta": {"ok": ok, "model": model, "usage": _empty_usage(), "error": error},
    }


def _analyst_input(macro_data: MacroData) -> dict[str, Any]:
    """The data as handed to the analyst: a copy without code-made interpretations."""
    data: dict[str, Any] = {}
    for key, value in dict(macro_data).items():
        if key == "_meta":
            continue
        if key == "recent_releases" and isinstance(value, list):
            # first_order_read is a code-made "hawkish/dovish" hint; the analyst
            # is asked to read the surprise, not to be told what it means.
            data[key] = [
                {k: v for k, v in item.items() if k != "first_order_read"} if isinstance(item, dict) else item
                for item in value
            ]
            continue
        data[key] = value
    return data


def analyze_macro_environment(fred_data: MacroData) -> MacroAnalysisResult:
    meta = fred_data.get("_meta", {}) if isinstance(fred_data, dict) else {}
    if not isinstance(meta, dict) or not bool(meta.get("ok", False)):
        return _build_neutral_result(
            "FRED data unavailable",
            model=str(meta.get("model", "none") if isinstance(meta, dict) else "none"),
            source="no_data",
        )

    user_prompt = json.dumps(
        {
            "macro_data": _analyst_input(fred_data),
            "data_notes": DATA_NOTES,
            "question": (
                "この環境は金価格にとって BULLISH / BEARISH / NEUTRAL のどれか。"
                "いまの価格トレンドの継続を支えるか、反転を促すか。読みが崩れる条件は何か。"
            ),
        },
        ensure_ascii=False,
    )

    client = get_default_client()
    try:
        result = client.call_json(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            model=analysis_model(),
            fallback_payload={},
        )
    except Exception as exc:
        return _build_neutral_result(f"LLM call failed: {exc}", model=analysis_model())

    if not bool(result.ok):
        return _build_neutral_result(result.error or "LLM call failed", model=result.model)

    payload = dict(result.payload) if isinstance(result.payload, dict) else {}
    bias = _normalize_bias(payload.get("macro_bias"))
    if bias is None:
        LOGGER.warning("macro_analyst: no valid macro_bias in LLM output; treating as NEUTRAL")
        fallback = _build_neutral_result("macro_bias missing or invalid", model=result.model, ok=True, source="invalid_output")
        fallback["_meta"]["usage"] = {
            "prompt_tokens": result.usage.prompt_tokens,
            "completion_tokens": result.usage.completion_tokens,
            "total_tokens": result.usage.total_tokens,
        }
        return fallback

    key_drivers_raw = payload.get("key_drivers", [])
    key_drivers = [str(item) for item in key_drivers_raw if str(item).strip()] if isinstance(key_drivers_raw, list) else []
    counter_raw = payload.get("counter_evidence", [])
    counter_evidence = [str(item) for item in counter_raw if str(item).strip()] if isinstance(counter_raw, list) else []
    data_quality = str(payload.get("data_quality", "") or "").upper().strip()
    abstain_reason = str(payload.get("abstain_reason") or "").strip() or None
    regime_view = _normalize_regime_view(payload.get("regime_view"))
    if abstain_reason:
        bias, regime_view = "NEUTRAL", "UNCLEAR"
    return {
        "macro_bias": bias,
        "regime_view": regime_view,
        "key_drivers": key_drivers,
        "counter_evidence": counter_evidence,
        "data_quality": data_quality if data_quality in DATA_QUALITY_VALUES else "GOOD",
        "abstain_reason": abstain_reason,
        "invalidation": str(payload.get("invalidation", "") or ""),
        "reasoning": str(payload.get("reasoning", "") or "").strip() or "(reasoning なし)",
        "source": "analyst",
        "_meta": {
            "ok": True,
            "model": result.model,
            "usage": {
                "prompt_tokens": result.usage.prompt_tokens,
                "completion_tokens": result.usage.completion_tokens,
                "total_tokens": result.usage.total_tokens,
            },
            "error": "",
        },
    }
