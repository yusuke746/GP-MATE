from __future__ import annotations

import json
import logging
import re
import time
from operator import add
from typing import Any, Callable, Literal, TypedDict, cast

try:
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_openai import ChatOpenAI
    from langgraph.graph import END, START, StateGraph
    _LANGCHAIN_IMPORT_ERROR = ""
except ModuleNotFoundError as exc:
    HumanMessage = Any  # type: ignore[assignment]
    SystemMessage = Any  # type: ignore[assignment]
    ChatOpenAI = None  # type: ignore[assignment]
    END = "END"  # type: ignore[assignment]
    START = "START"  # type: ignore[assignment]
    StateGraph = None  # type: ignore[assignment]
    _LANGCHAIN_IMPORT_ERROR = str(exc)
from typing_extensions import Annotated

LOGGER = logging.getLogger(__name__)

_BEAR_RAW_LOGGED_ONCE = False
_JUDGE_RAW_LOGGED_ONCE = False

DEBATE_MODEL = "gpt-5.4-mini"
BULL_TEMPERATURE = 0.7
BEAR_TEMPERATURE = 0.3
JUDGE_TEMPERATURE = 0.2

BULL_SYSTEM_PROMPT = (
    "あなたはBullアナリストです。必ず買いの立場を明確に主張し、HOLD寄りに逃げないこと。"
    "あなたは必ず【Bull】の立場です。相手の主張を引用・反論する際、相手の文章を冒頭からそのまま書き写さないこと。"
    "相手の論点を要約して引用し、それに対するBullの立場からの反論を述べること。自分がBullであることを絶対に見失わないこと。"
    "相手(Bear)の主張を名指しで引用し、必ず『相手は○○と言うが、△△を見落としている』形式で反論すること。"
    "マクロと多時間軸も強気材料として解釈し、マクロがBEARISHでも『織り込み済み』や『逆風は過剰評価』として反論してよい。"
    "D1が下でもH4/H1が上向きなら、押し目からの反発や短期優位として主張してよい。"
    "JSONのみで返答し、必ず次のキーを含めること:"
    "{argument: string, confidence: number(0-1), conceded_points: string[]}"
)

BEAR_SYSTEM_PROMPT = (
    "あなたはBearアナリストです。必ず最も弱気な視点を提示する義務がある。"
    "あなたは必ず【Bear】の立場です。相手の主張を引用・反論する際、相手の文章を冒頭からそのまま書き写さないこと。"
    "相手の論点を要約して引用し、それに対するBearの立場からの反論を述べること。自分がBearであることを絶対に見失わないこと。"
    "材料が乏しくても、下落リスクまたは反対材料を最低1つは挙げること。"
    "confidenceは必ず0.3以上で答え、0.0や極端な棄権は禁止。"
    "相手(Bull)の主張の弱点を必ず1つ以上、名指しで突くこと。"
    "必ず『相手は○○と言うが、△△を見落としている』形式で反論すること。"
    "マクロと多時間軸も弱気材料として解釈し、マクロがBULLISHでも実需やヘッジ需要の鈍化として反論してよい。"
    "D1下降トレンド中のH4/H1上昇はダマシや戻り売り候補として主張してよい。"
    "JSONのみで返答し、必ず次のキーを含めること:"
    "{argument: string, confidence: number(0-1), conceded_points: string[]}"
)

BEAR_MAX_ATTEMPTS = 2
JUDGE_MAX_ATTEMPTS = 2
RETRY_BACKOFF_SECONDS = 0.4
STRONGER_SIDE_CONFIDENCE_GAP = 0.05


class UsageStats(TypedDict):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class JudgeSummary(TypedDict):
    agreements: list[str]
    conflicts: list[str]
    confidence_shift: dict[str, list[float]]
    stronger_side: Literal["bull", "bear", "neutral"]


class DebateState(TypedDict):
    technical_report: dict[str, Any]
    sentiment_report: dict[str, Any]
    macro_report: dict[str, Any]
    bull_arguments: Annotated[list[str], add]
    bear_arguments: Annotated[list[str], add]
    bull_conceded_points: Annotated[list[str], add]
    round_count: int
    max_rounds: int
    bull_confidence: float
    bear_confidence: float
    prev_bull_confidence: float
    bull_confidence_history: Annotated[list[float], add]
    bear_confidence_history: Annotated[list[float], add]
    bull_ok_history: Annotated[list[bool], add]
    bear_ok_history: Annotated[list[bool], add]
    judge_summary: JudgeSummary
    bull_ok: bool
    bear_ok: bool
    judge_ok: bool
    bull_error: str
    bear_error: str
    judge_error: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class _RoleResponse(TypedDict):
    argument: str
    confidence: float
    conceded_points: list[str]
    usage: UsageStats
    ok: bool
    error: str


class _JudgeResponse(TypedDict):
    judge_summary: JudgeSummary
    usage: UsageStats
    ok: bool
    error: str


RoleLLMCallable = Callable[[str, DebateState], dict[str, Any]]


class DebateGateDecision(TypedDict):
    should_debate: bool
    reason: str
    technical_direction: Literal["BUY", "SELL", "NEUTRAL"]
    sentiment_direction: Literal["BULLISH", "BEARISH", "NEUTRAL"]
    macro_direction: Literal["BULLISH", "BEARISH", "NEUTRAL"]
    alignment: Literal["ALIGNED", "DIVERGENT", "MIXED"]
    estimated_confidence: float


def _zero_usage() -> UsageStats:
    return {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }


def _default_judge_summary() -> JudgeSummary:
    return {
        "agreements": [],
        "conflicts": ["データ不足のため判断保留"],
        "confidence_shift": {"bull": [], "bear": []},
        "stronger_side": "neutral",
    }


def _technical_direction(technical_report: dict[str, Any]) -> Literal["BUY", "SELL", "NEUTRAL"]:
    signal = str(technical_report.get("signal", "") or "").upper()
    trend = str(technical_report.get("trend", "") or "").upper()
    if signal in {"BUY", "SELL", "NEUTRAL"}:
        return cast(Literal["BUY", "SELL", "NEUTRAL"], signal)
    if "UP" in trend:
        return "BUY"
    if "DOWN" in trend:
        return "SELL"
    return "NEUTRAL"


def _sentiment_direction(sentiment_report: dict[str, Any]) -> Literal["BULLISH", "BEARISH", "NEUTRAL"]:
    score = float(sentiment_report.get("score", 0.0) or 0.0)
    if score >= 0.15:
        return "BULLISH"
    if score <= -0.15:
        return "BEARISH"

    sentiment_label = str(sentiment_report.get("sentiment", "") or "").upper()
    if "BULL" in sentiment_label:
        return "BULLISH"
    if "BEAR" in sentiment_label:
        return "BEARISH"
    return "NEUTRAL"


def _macro_direction(macro_report: dict[str, Any] | None) -> Literal["BULLISH", "BEARISH", "NEUTRAL"]:
    if not isinstance(macro_report, dict):
        return "NEUTRAL"
    bias = str(macro_report.get("macro_bias", "NEUTRAL") or "NEUTRAL").upper()
    if bias in {"BULLISH", "BEARISH", "NEUTRAL"}:
        return cast(Literal["BULLISH", "BEARISH", "NEUTRAL"], bias)
    return "NEUTRAL"


def _alignment_from_technical(technical_report: dict[str, Any]) -> Literal["ALIGNED", "DIVERGENT", "MIXED"]:
    alignment = str(technical_report.get("alignment", "MIXED") or "MIXED").upper()
    if alignment in {"ALIGNED", "DIVERGENT", "MIXED"}:
        return cast(Literal["ALIGNED", "DIVERGENT", "MIXED"], alignment)
    return "MIXED"


def _estimate_uncertainty_confidence(
    technical_report: dict[str, Any],
    sentiment_report: dict[str, Any],
) -> float:
    technical_dir = _technical_direction(technical_report)
    sentiment_dir = _sentiment_direction(sentiment_report)
    trend = str(technical_report.get("trend", "") or "").upper()

    score = 0.55
    if technical_dir == "NEUTRAL":
        score -= 0.08
    else:
        score += 0.06

    if sentiment_dir == "NEUTRAL":
        score -= 0.04
    elif (technical_dir == "BUY" and sentiment_dir == "BULLISH") or (technical_dir == "SELL" and sentiment_dir == "BEARISH"):
        score += 0.08
    elif technical_dir != "NEUTRAL":
        score -= 0.08

    if "STRONG_" in trend:
        score += 0.05
    if trend == "RANGE":
        score -= 0.08

    return max(0.0, min(1.0, score))


def should_execute_debate(
    technical_report: dict[str, Any],
    sentiment_report: dict[str, Any],
    macro_report: dict[str, Any] | None = None,
) -> DebateGateDecision:
    technical_dir = _technical_direction(technical_report)
    sentiment_dir = _sentiment_direction(sentiment_report)
    macro_dir = _macro_direction(macro_report)
    alignment = _alignment_from_technical(technical_report)
    trend = str(technical_report.get("trend", "") or "").upper()
    rsi = float(technical_report.get("rsi_14", 50.0) or 50.0)

    conflict = (technical_dir == "BUY" and sentiment_dir == "BEARISH") or (
        technical_dir == "SELL" and sentiment_dir == "BULLISH"
    )
    macro_conflict = (technical_dir == "BUY" and macro_dir == "BEARISH") or (
        technical_dir == "SELL" and macro_dir == "BULLISH"
    )
    is_strong_trend = trend in {"STRONG_UP", "STRONG_DOWN"}
    is_aligned_strong = (trend == "STRONG_UP" and sentiment_dir == "BULLISH") or (
        trend == "STRONG_DOWN" and sentiment_dir == "BEARISH"
    )
    is_range = trend == "RANGE" or technical_dir == "NEUTRAL"
    is_overheated = rsi > 70.0 or rsi < 30.0
    has_divergence = alignment == "DIVERGENT"

    estimated = _estimate_uncertainty_confidence(technical_report, sentiment_report)

    if has_divergence:
        return {
            "should_debate": True,
            "reason": "議論実行（多時間軸がDIVERGENT）",
            "technical_direction": technical_dir,
            "sentiment_direction": sentiment_dir,
            "macro_direction": macro_dir,
            "alignment": alignment,
            "estimated_confidence": estimated,
        }

    if macro_conflict:
        return {
            "should_debate": True,
            "reason": "議論実行（technicalとmacroが矛盾）",
            "technical_direction": technical_dir,
            "sentiment_direction": sentiment_dir,
            "macro_direction": macro_dir,
            "alignment": alignment,
            "estimated_confidence": estimated,
        }

    if is_range and not conflict and not is_strong_trend:
        return {
            "should_debate": False,
            "reason": "議論スキップ（レンジのため）",
            "technical_direction": technical_dir,
            "sentiment_direction": sentiment_dir,
            "macro_direction": macro_dir,
            "alignment": alignment,
            "estimated_confidence": estimated,
        }

    if conflict:
        return {
            "should_debate": True,
            "reason": "議論実行（technicalとsentimentが矛盾）",
            "technical_direction": technical_dir,
            "sentiment_direction": sentiment_dir,
            "macro_direction": macro_dir,
            "alignment": alignment,
            "estimated_confidence": estimated,
        }

    if is_strong_trend and is_overheated:
        return {
            "should_debate": True,
            "reason": "議論実行（強トレンドかつ過熱）",
            "technical_direction": technical_dir,
            "sentiment_direction": sentiment_dir,
            "macro_direction": macro_dir,
            "alignment": alignment,
            "estimated_confidence": estimated,
        }

    if is_aligned_strong:
        return {
            "should_debate": False,
            "reason": "議論スキップ（明確なトレンドのため）",
            "technical_direction": technical_dir,
            "sentiment_direction": sentiment_dir,
            "macro_direction": macro_dir,
            "alignment": alignment,
            "estimated_confidence": estimated,
        }

    if 0.5 <= estimated <= 0.65:
        return {
            "should_debate": True,
            "reason": "議論実行（不確実局面）",
            "technical_direction": technical_dir,
            "sentiment_direction": sentiment_dir,
            "macro_direction": macro_dir,
            "alignment": alignment,
            "estimated_confidence": estimated,
        }

    return {
        "should_debate": True,
        "reason": "議論実行（通常判定）",
        "technical_direction": technical_dir,
        "sentiment_direction": sentiment_dir,
        "macro_direction": macro_dir,
        "alignment": alignment,
        "estimated_confidence": estimated,
    }


def build_skipped_debate_report(reason: str) -> dict[str, Any]:
    summary = _default_judge_summary()
    summary["conflicts"] = [str(reason)]
    return {
        "bull_arguments": [],
        "bear_arguments": [],
        "bull_conceded_points": [],
        "round_count": 0,
        "bull_confidence": 0.5,
        "bear_confidence": 0.5,
        "prev_bull_confidence": 0.5,
        "bull_confidence_history": [0.5],
        "bear_confidence_history": [0.5],
        "judge_summary": summary,
        "_meta": {
            "ok": True,
            "engine": "langgraph",
            "model": DEBATE_MODEL,
            "debate_executed": False,
            "skip_reason": reason,
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        },
    }


def _is_nonempty_error(value: Any) -> str:
    text = str(value or "").strip()
    return text


def _extract_argument_from_payload(payload: dict[str, Any], fallback: str) -> str:
    candidate_keys = ["argument", "rebuttal", "counter_argument", "thesis"]
    for key in candidate_keys:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    key_points = payload.get("key_points")
    if isinstance(key_points, list) and key_points:
        normalized = [str(x).strip() for x in key_points if str(x).strip()]
        if normalized:
            return " / ".join(normalized)

    return fallback


def _extract_confidence_from_payload(payload: dict[str, Any], default: float) -> float:
    candidate_keys = ["confidence", "conviction", "score"]
    for key in candidate_keys:
        if key in payload:
            return _safe_float(payload.get(key), default)

    risk_view = payload.get("risk_view")
    if isinstance(risk_view, dict):
        for key in candidate_keys:
            if key in risk_view:
                return _safe_float(risk_view.get(key), default)

    return default


def _safe_json_loads(raw_text: str) -> dict[str, Any] | None:
    text = raw_text.strip()
    if not text:
        return None

    candidates: list[str] = [text]

    fenced = re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    for chunk in fenced:
        stripped = chunk.strip()
        if stripped:
            candidates.append(stripped)

    first_obj = text.find("{")
    last_obj = text.rfind("}")
    if first_obj != -1 and last_obj != -1 and first_obj < last_obj:
        candidates.append(text[first_obj : last_obj + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _extract_json_string_list(raw_text: str, key: str) -> list[str]:
    pattern = rf'"{re.escape(key)}"\s*:\s*(\[[\s\S]*?\])'
    match = re.search(pattern, raw_text)
    if match:
        array_text = match.group(1)
        try:
            parsed = json.loads(array_text)
        except Exception:
            parsed = None

        if isinstance(parsed, list):
            values = [str(x).strip() for x in parsed if str(x).strip()]
            return values

        quoted = re.findall(r'"([^"\\]*(?:\\.[^"\\]*)*)"', array_text)
        values = [x.strip() for x in quoted if x.strip()]
        if values:
            return values

    # Lenient fallback for malformed JSON (e.g., missing commas/brackets).
    section_pattern = rf'"{re.escape(key)}"\s*:\s*(.*?)(?=,\s*"(?:agreements|conflicts|stronger_side)"\s*:|\}}\s*$)'
    section_match = re.search(section_pattern, raw_text, flags=re.DOTALL)
    if not section_match:
        return []

    section_text = section_match.group(1)
    quoted = re.findall(r'"([^"\\]*(?:\\.[^"\\]*)*)"', section_text)
    return [x.strip() for x in quoted if x.strip()]


def _extract_json_enum(raw_text: str, key: str, allowed: set[str], default: str) -> str:
    pattern_double = rf'"{re.escape(key)}"\s*:\s*"([^"]+)"'
    pattern_single = rf"'{re.escape(key)}'\s*:\s*'([^']+)'"
    match = re.search(pattern_double, raw_text)
    if not match:
        match = re.search(pattern_single, raw_text)
    if not match:
        return default
    value = str(match.group(1) or "").strip().lower()
    if value in allowed:
        return value
    return default


def _extract_judge_raw_focus(raw_text: str) -> str:
    keys = ["\"agreements\"", "\"conflicts\"", "\"stronger_side\""]
    lowered = raw_text.lower()
    hit_index = -1
    for key in keys:
        idx = lowered.find(key)
        if idx != -1 and (hit_index == -1 or idx < hit_index):
            hit_index = idx

    if hit_index == -1:
        return _slice_log_text(raw_text, 320)

    start = max(0, hit_index - 80)
    end = min(len(raw_text), hit_index + 320)
    return _slice_log_text(raw_text[start:end], 320)


def _slice_log_text(raw_text: str, max_len: int = 500) -> str:
    compact = " ".join(raw_text.split())
    if len(compact) <= max_len:
        return compact
    return compact[: max_len - 3] + "..."


def _build_judge_summary_from_state(state: DebateState, reason: str) -> JudgeSummary:
    summary = _default_judge_summary()

    agreements: list[str] = []
    conceded = state.get("bull_conceded_points", [])
    if isinstance(conceded, list):
        for item in conceded:
            text = str(item or "").strip()
            if text:
                agreements.append(text)

    conflicts: list[str] = []
    bull_arguments = [str(x).strip() for x in state.get("bull_arguments", []) if str(x).strip()]
    bear_arguments = [str(x).strip() for x in state.get("bear_arguments", []) if str(x).strip()]
    pair_count = min(len(bull_arguments), len(bear_arguments), 2)
    for idx in range(pair_count):
        bull_head = bull_arguments[idx][:120]
        bear_head = bear_arguments[idx][:120]
        conflicts.append(f"bull: {bull_head} / bear: {bear_head}")

    if not conflicts and bull_arguments:
        conflicts.append(f"bull主張: {bull_arguments[-1][:180]}")
    if not conflicts and bear_arguments:
        conflicts.append(f"bear主張: {bear_arguments[-1][:180]}")
    if not conflicts:
        conflicts = ["judge失敗のため対立点抽出不可"]

    summary["agreements"] = agreements
    summary["conflicts"] = conflicts
    summary["stronger_side"] = "neutral"
    if reason:
        summary["conflicts"].append(reason)
    return summary


def _recover_judge_summary_from_raw_text(raw_text: str) -> JudgeSummary | None:
    agreements = _extract_json_string_list(raw_text, "agreements")
    conflicts = _extract_json_string_list(raw_text, "conflicts")
    stronger_raw = _extract_json_enum(raw_text, "stronger_side", {"bull", "bear", "neutral"}, "neutral")

    if not agreements and not conflicts and stronger_raw == "neutral":
        return None

    summary = _default_judge_summary()
    summary["agreements"] = agreements
    summary["conflicts"] = conflicts if conflicts else ["judge生レスポンスから部分復旧"]
    summary["stronger_side"] = cast(Literal["bull", "bear", "neutral"], stronger_raw)
    return summary


def _extract_text(response: Any) -> str:
    content = getattr(response, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                text = item.get("text")
                if text is not None:
                    parts.append(str(text))
        return "\n".join(parts).strip()
    return str(content or "")


def _safe_float(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except Exception:
        return default
    return max(0.0, min(1.0, parsed))


def _extract_usage(response: Any) -> UsageStats:
    usage = _zero_usage()

    usage_meta = getattr(response, "usage_metadata", None)
    if isinstance(usage_meta, dict):
        usage["prompt_tokens"] = int(usage_meta.get("input_tokens", usage_meta.get("prompt_tokens", 0)) or 0)
        usage["completion_tokens"] = int(usage_meta.get("output_tokens", usage_meta.get("completion_tokens", 0)) or 0)
        usage["total_tokens"] = int(usage_meta.get("total_tokens", 0) or 0)
        if usage["total_tokens"] == 0:
            usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
        return usage

    response_meta = getattr(response, "response_metadata", None)
    if isinstance(response_meta, dict):
        token_usage = response_meta.get("token_usage", {})
        if isinstance(token_usage, dict):
            usage["prompt_tokens"] = int(token_usage.get("prompt_tokens", 0) or 0)
            usage["completion_tokens"] = int(token_usage.get("completion_tokens", 0) or 0)
            usage["total_tokens"] = int(token_usage.get("total_tokens", 0) or 0)
            if usage["total_tokens"] == 0:
                usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
    return usage


def _usage_from_payload(payload: dict[str, Any]) -> UsageStats:
    raw = payload.get("usage", {})
    if not isinstance(raw, dict):
        return _zero_usage()
    prompt = int(raw.get("prompt_tokens", 0) or 0)
    completion = int(raw.get("completion_tokens", 0) or 0)
    total = int(raw.get("total_tokens", 0) or 0)
    if total == 0:
        total = prompt + completion
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
    }


def _coerce_judge_summary(value: Any) -> JudgeSummary:
    fallback = _default_judge_summary()
    if not isinstance(value, dict):
        if value:
            fallback["conflicts"] = [str(value)]
        return fallback

    agreements_raw = value.get("agreements", [])
    conflicts_raw = value.get("conflicts", [])
    stronger_raw = str(value.get("stronger_side", "neutral") or "neutral").lower()

    agreements = [str(x) for x in agreements_raw] if isinstance(agreements_raw, list) else []
    conflicts = [str(x) for x in conflicts_raw] if isinstance(conflicts_raw, list) else []

    stronger_side: Literal["bull", "bear", "neutral"] = "neutral"
    if stronger_raw in {"bull", "bear", "neutral"}:
        stronger_side = cast(Literal["bull", "bear", "neutral"], stronger_raw)

    return {
        "agreements": agreements,
        "conflicts": conflicts,
        "confidence_shift": {
            "bull": [],
            "bear": [],
        },
        "stronger_side": stronger_side,
    }


def _compute_stronger_side_from_state(state: DebateState) -> Literal["bull", "bear", "neutral"]:
    bull_conf = _latest_valid_role_confidence(
        history=state.get("bull_confidence_history", []),
        ok_history=state.get("bull_ok_history", []),
        default=0.5,
    )
    bear_conf = _latest_valid_role_confidence(
        history=state.get("bear_confidence_history", []),
        ok_history=state.get("bear_ok_history", []),
        default=0.5,
    )

    if bull_conf is None or bear_conf is None:
        return "neutral"

    diff = bull_conf - bear_conf
    if abs(diff) <= STRONGER_SIDE_CONFIDENCE_GAP:
        return "neutral"
    return "bull" if diff > 0 else "bear"


def _build_confidence_shift_from_state(state: DebateState) -> dict[str, list[float]]:
    bull_hist_raw = state.get("bull_confidence_history", [])
    bear_hist_raw = state.get("bear_confidence_history", [])

    bull_hist = [_safe_float(x, 0.5) for x in bull_hist_raw] if isinstance(bull_hist_raw, list) else []
    bear_hist = [_safe_float(x, 0.5) for x in bear_hist_raw] if isinstance(bear_hist_raw, list) else []
    return {
        "bull": bull_hist,
        "bear": bear_hist,
    }


def _latest_valid_role_confidence(
    history: Any,
    ok_history: Any,
    default: float,
) -> float | None:
    if not isinstance(history, list) or not isinstance(ok_history, list):
        return default

    # history format: [initial, round1, round2, ...]
    # ok_history format: [round1_ok, round2_ok, ...]
    if len(history) < 2 or len(ok_history) == 0:
        return default

    usable_rounds = min(len(ok_history), len(history) - 1)
    for idx in range(usable_rounds - 1, -1, -1):
        if bool(ok_history[idx]):
            return _safe_float(history[idx + 1], default)
    return None


def _build_incomplete_marker(state: DebateState) -> str:
    failed_roles: list[str] = []

    bull_ok_history = state.get("bull_ok_history", [])
    bear_ok_history = state.get("bear_ok_history", [])

    bull_fail_count = 0
    if isinstance(bull_ok_history, list):
        bull_fail_count = sum(1 for x in bull_ok_history if not bool(x))
    if bull_fail_count > 0:
        failed_roles.append(f"bull分析が{bull_fail_count}ラウンド失敗")

    bear_fail_count = 0
    if isinstance(bear_ok_history, list):
        bear_fail_count = sum(1 for x in bear_ok_history if not bool(x))
    if bear_fail_count > 0:
        failed_roles.append(f"bear分析が{bear_fail_count}ラウンド失敗")

    if not bool(state.get("judge_ok", True)):
        failed_roles.append("judge失敗")

    if not failed_roles:
        return ""
    return f"議論不完全（{', '.join(failed_roles)}）"


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _normalize_horizontal_entries(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []

    normalized: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        price = _as_float(item.get("price", 0.0), 0.0)
        if price <= 0:
            continue
        normalized.append(
            {
                "price": round(price, 5),
                "score": _as_float(item.get("score", 0.0), 0.0),
                "source": str(item.get("source", "") or ""),
                "timeframe": str(item.get("timeframe", "") or ""),
                "touch_count": int(item.get("touch_count", 0) or 0),
            }
        )
    return normalized


def _build_horizontal_levels_context(technical_report: dict[str, Any]) -> dict[str, Any]:
    key_levels = technical_report.get("key_levels", {}) if isinstance(technical_report, dict) else {}
    if not isinstance(key_levels, dict):
        return {
            "available": False,
            "current_price": "",
            "nearest_resistances": [],
            "nearest_supports": [],
            "summary": "horizontal_levels unavailable",
        }

    frames = key_levels.get("frames", {}) if isinstance(key_levels.get("frames", {}), dict) else {}
    h1 = frames.get("h1", {}) if isinstance(frames.get("h1", {}), dict) else {}
    d1 = key_levels.get("d1", {}) if isinstance(key_levels.get("d1", {}), dict) else {}
    d1_snapshot = d1.get("snapshot", {}) if isinstance(d1.get("snapshot", {}), dict) else {}

    current_price = _as_float(h1.get("close", d1_snapshot.get("close", 0.0)), 0.0)

    horizontal_levels = key_levels.get("horizontal_levels", {})
    if not isinstance(horizontal_levels, dict):
        return {
            "available": False,
            "current_price": round(current_price, 5) if current_price > 0 else "",
            "nearest_resistances": [],
            "nearest_supports": [],
            "summary": "horizontal_levels unavailable",
        }

    resistances = _normalize_horizontal_entries(horizontal_levels.get("resistances", []))
    supports = _normalize_horizontal_entries(horizontal_levels.get("supports", []))

    if current_price > 0:
        resistances = [item for item in resistances if float(item["price"]) > current_price]
        supports = [item for item in supports if float(item["price"]) < current_price]

        resistances.sort(key=lambda item: abs(float(item["price"]) - current_price))
        supports.sort(key=lambda item: abs(float(item["price"]) - current_price))

    nearest_res = resistances[:3]
    nearest_sup = supports[:3]
    available = bool(nearest_res or nearest_sup)

    if available:
        summary = f"current={round(current_price, 5)} res={len(nearest_res)} sup={len(nearest_sup)}"
    else:
        summary = "horizontal_levels empty"

    return {
        "available": available,
        "current_price": round(current_price, 5) if current_price > 0 else "",
        "nearest_resistances": nearest_res,
        "nearest_supports": nearest_sup,
        "summary": summary,
    }


def _invoke_role_llm(role: str, state: DebateState) -> _RoleResponse:
    latest_bear = state["bear_arguments"][-1] if state["bear_arguments"] else "初回ラウンド。"
    latest_bull = state["bull_arguments"][-1] if state["bull_arguments"] else "初回ラウンド。"
    horizontal_levels_context = _build_horizontal_levels_context(state["technical_report"])
    bear_opponent_context = (
        "--- 相手(Bear)の主張ここから ---\n"
        f"{latest_bear}\n"
        "--- 相手(Bear)の主張ここまで ---"
    )
    bull_opponent_context = (
        "--- 相手(Bull)の主張ここから ---\n"
        f"{latest_bull}\n"
        "--- 相手(Bull)の主張ここまで ---"
    )

    if role == "bull":
        system_prompt = BULL_SYSTEM_PROMPT
        user_payload = {
            "self_role": "Bull",
            "technical_report": state["technical_report"],
            "sentiment_report": state["sentiment_report"],
            "macro_report": state.get("macro_report", {}),
            "horizontal_levels_context": horizontal_levels_context,
            "multi_timeframe": {
                "d1_trend": state["technical_report"].get("d1_trend", "RANGE"),
                "execution_trend": state["technical_report"].get("execution_trend", state["technical_report"].get("trend", "RANGE")),
                "alignment": state["technical_report"].get("alignment", "MIXED"),
            },
            "latest_bear_argument": latest_bear,
            "opponent_argument_context": bear_opponent_context,
            "prev_bull_confidence": state["bull_confidence"],
        }
    else:
        system_prompt = BEAR_SYSTEM_PROMPT
        user_payload = {
            "self_role": "Bear",
            "technical_report": state["technical_report"],
            "sentiment_report": state["sentiment_report"],
            "macro_report": state.get("macro_report", {}),
            "horizontal_levels_context": horizontal_levels_context,
            "multi_timeframe": {
                "d1_trend": state["technical_report"].get("d1_trend", "RANGE"),
                "execution_trend": state["technical_report"].get("execution_trend", state["technical_report"].get("trend", "RANGE")),
                "alignment": state["technical_report"].get("alignment", "MIXED"),
            },
            "latest_bull_argument": latest_bull,
            "opponent_argument_context": bull_opponent_context,
            "prev_bear_confidence": state["bear_confidence"],
        }

    temperature = BULL_TEMPERATURE if role == "bull" else BEAR_TEMPERATURE
    fallback: _RoleResponse = {
        "argument": f"{role}分析に失敗したためHOLD寄り。",
        "confidence": 0.0,
        "conceded_points": [],
        "usage": _zero_usage(),
        "ok": False,
        "error": "",
    }

    if ChatOpenAI is None:
        fallback["error"] = f"missing dependency: {_LANGCHAIN_IMPORT_ERROR or 'langchain modules unavailable'}"
        LOGGER.warning("debate_graph %s skipped: %s", role, fallback["error"])
        return fallback

    attempts = BEAR_MAX_ATTEMPTS if role == "bear" else 1
    for attempt in range(attempts):
        try:
            model = ChatOpenAI(model=DEBATE_MODEL, temperature=temperature)
            response = model.invoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=json.dumps(user_payload, ensure_ascii=False)),
                ]
            )
            usage = _extract_usage(response)
            text = _extract_text(response)

            global _BEAR_RAW_LOGGED_ONCE
            if role == "bear" and not _BEAR_RAW_LOGGED_ONCE:
                LOGGER.warning("debate_graph bear raw response(first sample): %s", text)
                _BEAR_RAW_LOGGED_ONCE = True

            if not text.strip():
                fallback["usage"] = usage
                fallback["error"] = "empty output"
                LOGGER.warning(
                    "debate_graph role fallback: role=%s reason=%s raw_full=%s",
                    role,
                    fallback["error"],
                    text,
                )
                if attempt + 1 < attempts:
                    time.sleep(RETRY_BACKOFF_SECONDS * (2**attempt))
                    continue
                return fallback

            payload = _safe_json_loads(text)
            if payload is None:
                fallback["usage"] = usage
                fallback["error"] = "json parse error"
                LOGGER.warning(
                    "debate_graph role fallback: role=%s reason=%s raw_full=%s",
                    role,
                    fallback["error"],
                    text,
                )
                if attempt + 1 < attempts:
                    time.sleep(RETRY_BACKOFF_SECONDS * (2**attempt))
                    continue
                return fallback

            argument = _extract_argument_from_payload(payload, fallback["argument"])
            conceded = payload.get("conceded_points", [])
            conceded_points = [str(x) for x in conceded] if isinstance(conceded, list) else []
            confidence = _extract_confidence_from_payload(payload, default=fallback["confidence"])

            if role == "bear" and confidence < 0.3:
                LOGGER.warning(
                    "debate_graph bear confidence too low for ok response; clamped to 0.3. raw_head=%s",
                    text[:200],
                )
                confidence = 0.3

            if role == "bear" and confidence <= 0.0:
                LOGGER.warning(
                    "debate_graph bear returned non-positive confidence with ok=True; argument_head=%s",
                    argument[:200],
                )
            return {
                "argument": argument,
                "confidence": confidence,
                "conceded_points": conceded_points,
                "usage": usage,
                "ok": True,
                "error": "",
            }
        except Exception as exc:
            LOGGER.warning("debate_graph %s invoke failed(attempt=%s/%s): %s", role, attempt + 1, attempts, exc)
            fallback["error"] = str(exc)
            LOGGER.warning(
                "debate_graph role fallback: role=%s reason=api error raw_full=%s",
                role,
                "",
            )
            if attempt + 1 < attempts:
                time.sleep(RETRY_BACKOFF_SECONDS * (2**attempt))
                continue
            return fallback

    return fallback


def _invoke_judge_llm(state: DebateState) -> _JudgeResponse:
    system_prompt = (
        "あなたはJudgeです。議論全体から合意点・対立点・confidence変遷・論拠の強さを抽出してください。"
        "必ず次のJSON形式で返してください:"
        "{agreements: string[], conflicts: string[], stronger_side:'bull'|'bear'|'neutral'}"
    )
    user_payload = {
        "bull_arguments": state["bull_arguments"],
        "bear_arguments": state["bear_arguments"],
        "bull_confidence": state["bull_confidence"],
        "bear_confidence": state["bear_confidence"],
        "round_count": state["round_count"],
        "bull_conceded_points": state["bull_conceded_points"],
        "bull_confidence_history": state["bull_confidence_history"],
        "bear_confidence_history": state["bear_confidence_history"],
        "macro_report": state.get("macro_report", {}),
        "multi_timeframe": {
            "d1_trend": state["technical_report"].get("d1_trend", "RANGE"),
            "execution_trend": state["technical_report"].get("execution_trend", state["technical_report"].get("trend", "RANGE")),
            "alignment": state["technical_report"].get("alignment", "MIXED"),
        },
    }
    fallback = _default_judge_summary()

    if ChatOpenAI is None:
        summary = _build_judge_summary_from_state(
            state,
            f"missing dependency: {_LANGCHAIN_IMPORT_ERROR or 'langchain modules unavailable'}",
        )
        LOGGER.warning("debate_graph judge skipped: %s", _LANGCHAIN_IMPORT_ERROR or "langchain modules unavailable")
        return {
            "judge_summary": summary,
            "usage": _zero_usage(),
            "ok": False,
            "error": f"missing dependency: {_LANGCHAIN_IMPORT_ERROR or 'langchain modules unavailable'}",
        }

    for attempt in range(JUDGE_MAX_ATTEMPTS):
        try:
            model = ChatOpenAI(model=DEBATE_MODEL, temperature=JUDGE_TEMPERATURE)
            response = model.invoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=json.dumps(user_payload, ensure_ascii=False)),
                ]
            )
            usage = _extract_usage(response)
            text = _extract_text(response)

            global _JUDGE_RAW_LOGGED_ONCE
            if not _JUDGE_RAW_LOGGED_ONCE:
                LOGGER.warning("debate_graph judge raw response(first sample): %s", _slice_log_text(text, 700))
                _JUDGE_RAW_LOGGED_ONCE = True

            if not text.strip():
                if attempt + 1 < JUDGE_MAX_ATTEMPTS:
                    time.sleep(RETRY_BACKOFF_SECONDS * (2**attempt))
                    continue
                summary = _build_judge_summary_from_state(state, "judge empty output")
                return {
                    "judge_summary": summary,
                    "usage": usage,
                    "ok": False,
                    "error": "empty output",
                }

            payload = _safe_json_loads(text)
            if payload is not None:
                summary = _coerce_judge_summary(payload)
                return {
                    "judge_summary": summary,
                    "usage": usage,
                    "ok": True,
                    "error": "",
                }

            recovered = _recover_judge_summary_from_raw_text(text)
            if recovered is not None:
                LOGGER.warning(
                    "debate_graph judge json parse recovered from raw text: raw_head=%s raw_focus=%s",
                    _slice_log_text(text, 260),
                    _extract_judge_raw_focus(text),
                )
                return {
                    "judge_summary": recovered,
                    "usage": usage,
                    "ok": True,
                    "error": "json parse recovered",
                }

            LOGGER.warning(
                "debate_graph judge json parse failed(attempt=%s/%s): raw_head=%s raw_focus=%s",
                attempt + 1,
                JUDGE_MAX_ATTEMPTS,
                _slice_log_text(text, 260),
                _extract_judge_raw_focus(text),
            )
            if attempt + 1 < JUDGE_MAX_ATTEMPTS:
                time.sleep(RETRY_BACKOFF_SECONDS * (2**attempt))
                continue

            summary = _build_judge_summary_from_state(state, "judge json parse error")
            return {
                "judge_summary": summary,
                "usage": usage,
                "ok": False,
                "error": "json parse error",
            }
        except Exception as exc:
            LOGGER.warning("debate_graph judge invoke failed(attempt=%s/%s): %s", attempt + 1, JUDGE_MAX_ATTEMPTS, exc)
            if attempt + 1 < JUDGE_MAX_ATTEMPTS:
                time.sleep(RETRY_BACKOFF_SECONDS * (2**attempt))
                continue
            summary = _build_judge_summary_from_state(state, f"judge invoke failed: {exc}")
            return {
                "judge_summary": summary,
                "usage": _zero_usage(),
                "ok": False,
                "error": str(exc),
            }

    fallback_summary = _build_judge_summary_from_state(state, "judge failed unexpectedly")
    return {
        "judge_summary": fallback_summary,
        "usage": _zero_usage(),
        "ok": False,
        "error": "unexpected judge failure",
    }


def _bull_node(state: DebateState, llm: RoleLLMCallable | None) -> dict[str, Any]:
    role_response = llm("bull", state) if llm is not None else _invoke_role_llm("bull", state)
    prev = float(state["bull_confidence"])
    current = _safe_float(role_response.get("confidence", prev), default=prev)
    conceded = role_response.get("conceded_points", [])
    conceded_points = [str(x) for x in conceded] if isinstance(conceded, list) else []
    usage = _usage_from_payload(role_response)
    return {
        "prev_bull_confidence": prev,
        "bull_confidence": current,
        "bull_confidence_history": [current],
        "bull_ok_history": [bool(role_response.get("ok", True))],
        "bull_arguments": [str(role_response.get("argument", ""))],
        "bull_conceded_points": conceded_points,
        "bull_ok": bool(state.get("bull_ok", True)) and bool(role_response.get("ok", True)),
        "bull_error": _is_nonempty_error(role_response.get("error", "")),
        "prompt_tokens": int(state["prompt_tokens"]) + usage["prompt_tokens"],
        "completion_tokens": int(state["completion_tokens"]) + usage["completion_tokens"],
        "total_tokens": int(state["total_tokens"]) + usage["total_tokens"],
    }


def _bear_node(state: DebateState, llm: RoleLLMCallable | None) -> dict[str, Any]:
    role_response = llm("bear", state) if llm is not None else _invoke_role_llm("bear", state)
    prev = float(state["bear_confidence"])
    current = _safe_float(role_response.get("confidence", prev), default=prev)
    role_ok = bool(role_response.get("ok", True))
    if not role_ok:
        current = prev
    if role_ok and current < 0.3:
        current = 0.3
    usage = _usage_from_payload(role_response)
    return {
        "bear_confidence": current,
        "bear_confidence_history": [current],
        "bear_ok_history": [bool(role_response.get("ok", True))],
        "bear_arguments": [str(role_response.get("argument", ""))],
        "round_count": int(state["round_count"]) + 1,
        "bear_ok": bool(state.get("bear_ok", True)) and role_ok,
        "bear_error": _is_nonempty_error(role_response.get("error", "")),
        "prompt_tokens": int(state["prompt_tokens"]) + usage["prompt_tokens"],
        "completion_tokens": int(state["completion_tokens"]) + usage["completion_tokens"],
        "total_tokens": int(state["total_tokens"]) + usage["total_tokens"],
    }


def _judge_node(state: DebateState, llm: RoleLLMCallable | None) -> dict[str, Any]:
    judge_payload = llm("judge", state) if llm is not None else _invoke_judge_llm(state)
    summary = _coerce_judge_summary(judge_payload.get("judge_summary", judge_payload.get("summary", {})))
    summary["confidence_shift"] = _build_confidence_shift_from_state(state)

    llm_stronger = str(summary.get("stronger_side", "neutral") or "neutral")
    algorithmic_stronger = _compute_stronger_side_from_state(state)
    if llm_stronger in {"bull", "bear", "neutral"} and llm_stronger != algorithmic_stronger:
        LOGGER.warning(
            "Judge stronger_side conflicted with confidence diff; llm=%s algorithmic=%s",
            llm_stronger,
            algorithmic_stronger,
        )
    summary["stronger_side"] = algorithmic_stronger

    technical = state.get("technical_report", {})
    if isinstance(technical, dict):
        trend = str(technical.get("trend", "") or "").upper()
        signal = str(technical.get("signal", "") or "").upper()
        alignment = str(technical.get("alignment", "MIXED") or "MIXED").upper()
        macro_bias = str((state.get("macro_report", {}) or {}).get("macro_bias", "NEUTRAL") or "NEUTRAL").upper()
        contradiction = False
        if ("DOWN" in trend or signal == "SELL") and algorithmic_stronger == "bull":
            contradiction = True
        if ("UP" in trend or signal == "BUY") and algorithmic_stronger == "bear":
            contradiction = True
        if contradiction:
            warn_msg = "テクニカル方向と議論結論が逆転"
            LOGGER.warning(
                "debate_graph contradiction: trend=%s signal=%s stronger_side=%s",
                trend,
                signal,
                algorithmic_stronger,
            )
            conflicts = summary.get("conflicts", [])
            if isinstance(conflicts, list) and warn_msg not in conflicts:
                conflicts.append(warn_msg)
                summary["conflicts"] = [str(x) for x in conflicts]

        macro_conflict_msg = "マクロ方向と技術方向が逆"
        if macro_bias in {"BULLISH", "BEARISH"}:
            technical_dir = _technical_direction(technical)
            if (technical_dir == "BUY" and macro_bias == "BEARISH") or (technical_dir == "SELL" and macro_bias == "BULLISH"):
                conflicts = summary.get("conflicts", [])
                if isinstance(conflicts, list) and macro_conflict_msg not in conflicts:
                    conflicts.append(macro_conflict_msg)
                    summary["conflicts"] = [str(x) for x in conflicts]

        if alignment == "DIVERGENT":
            divergent_msg = "多時間軸がDIVERGENT"
            conflicts = summary.get("conflicts", [])
            if isinstance(conflicts, list) and divergent_msg not in conflicts:
                conflicts.append(divergent_msg)
                summary["conflicts"] = [str(x) for x in conflicts]

    judge_ok_now = bool(judge_payload.get("ok", True))
    temp_state = dict(state)
    temp_state["judge_ok"] = judge_ok_now
    incomplete = _build_incomplete_marker(cast(DebateState, temp_state))
    if incomplete:
        conflicts = summary.get("conflicts", [])
        if isinstance(conflicts, list):
            if incomplete not in conflicts:
                conflicts.append(incomplete)
            summary["conflicts"] = [str(x) for x in conflicts]
        else:
            summary["conflicts"] = [incomplete]

    usage = _usage_from_payload(judge_payload)
    return {
        "judge_summary": summary,
        "judge_ok": judge_ok_now,
        "judge_error": _is_nonempty_error(judge_payload.get("error", "")),
        "prompt_tokens": int(state["prompt_tokens"]) + usage["prompt_tokens"],
        "completion_tokens": int(state["completion_tokens"]) + usage["completion_tokens"],
        "total_tokens": int(state["total_tokens"]) + usage["total_tokens"],
    }


def _should_continue(state: DebateState) -> Literal["bull", "judge"]:
    if int(state["round_count"]) < 2:
        return "bull"
    if int(state["round_count"]) >= int(state["max_rounds"]):
        return "judge"
    delta = abs(float(state["bull_confidence"]) - float(state["prev_bull_confidence"]))
    if delta < 0.03:
        return "judge"
    return "bull"


def _build_graph(llm: RoleLLMCallable | None):
    if StateGraph is None:
        raise RuntimeError(f"langgraph unavailable: {_LANGCHAIN_IMPORT_ERROR or 'missing dependencies'}")

    graph = StateGraph(DebateState)

    graph.add_node("bull", lambda s: _bull_node(s, llm))
    graph.add_node("bear", lambda s: _bear_node(s, llm))
    graph.add_node("judge", lambda s: _judge_node(s, llm))

    graph.add_edge(START, "bull")
    graph.add_edge("bull", "bear")
    graph.add_conditional_edges(
        "bear",
        _should_continue,
        {
            "bull": "bull",
            "judge": "judge",
        },
    )
    graph.add_edge("judge", END)

    return graph.compile()


def run_debate_graph(
    technical_report: dict[str, Any],
    sentiment_report: dict[str, Any],
    macro_report: dict[str, Any] | None = None,
    max_rounds: int = 3,
    llm_override: RoleLLMCallable | None = None,
) -> dict[str, Any]:
    """Run Bull/Bear/Judge debate with LangGraph and return structured report.

    Safe policy: any failure returns HOLD-friendly fallback report.
    """
    initial_state: DebateState = {
        "technical_report": technical_report,
        "sentiment_report": sentiment_report,
        "macro_report": macro_report or {},
        "bull_arguments": [],
        "bear_arguments": [],
        "bull_conceded_points": [],
        "round_count": 0,
        "max_rounds": max(1, int(max_rounds)),
        "bull_confidence": 0.5,
        "bear_confidence": 0.5,
        "prev_bull_confidence": 0.5,
        "bull_confidence_history": [0.5],
        "bear_confidence_history": [0.5],
        "bull_ok_history": [],
        "bear_ok_history": [],
        "judge_summary": _default_judge_summary(),
        "bull_ok": True,
        "bear_ok": True,
        "judge_ok": True,
        "bull_error": "",
        "bear_error": "",
        "judge_error": "",
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }

    try:
        app = _build_graph(llm_override)
        final_state = app.invoke(initial_state)

        summary = _coerce_judge_summary(final_state.get("judge_summary", {}))
        summary["confidence_shift"] = _build_confidence_shift_from_state(cast(DebateState, final_state))
        summary["stronger_side"] = _compute_stronger_side_from_state(cast(DebateState, final_state))

        return {
            "bull_arguments": list(final_state.get("bull_arguments", [])),
            "bear_arguments": list(final_state.get("bear_arguments", [])),
            "bull_conceded_points": list(final_state.get("bull_conceded_points", [])),
            "round_count": int(final_state.get("round_count", 0) or 0),
            "bull_confidence": float(final_state.get("bull_confidence", 0.0) or 0.0),
            "bear_confidence": float(final_state.get("bear_confidence", 0.0) or 0.0),
            "prev_bull_confidence": float(final_state.get("prev_bull_confidence", 0.0) or 0.0),
            "bull_confidence_history": list(final_state.get("bull_confidence_history", [])),
            "bear_confidence_history": list(final_state.get("bear_confidence_history", [])),
            "judge_summary": summary,
            "_meta": {
                "ok": bool(final_state.get("bull_ok", True)) and bool(final_state.get("bear_ok", True)) and bool(final_state.get("judge_ok", True)),
                "engine": "langgraph",
                "model": DEBATE_MODEL,
                "bull_ok": bool(final_state.get("bull_ok", True)),
                "bear_ok": bool(final_state.get("bear_ok", True)),
                "judge_ok": bool(final_state.get("judge_ok", True)),
                "bull_ok_history": [bool(x) for x in final_state.get("bull_ok_history", [])],
                "bear_ok_history": [bool(x) for x in final_state.get("bear_ok_history", [])],
                "bull_error": str(final_state.get("bull_error", "") or ""),
                "bear_error": str(final_state.get("bear_error", "") or ""),
                "judge_error": str(final_state.get("judge_error", "") or ""),
                "usage": {
                    "prompt_tokens": int(final_state.get("prompt_tokens", 0) or 0),
                    "completion_tokens": int(final_state.get("completion_tokens", 0) or 0),
                    "total_tokens": int(final_state.get("total_tokens", 0) or 0),
                },
            },
        }
    except Exception as exc:
        LOGGER.warning("run_debate_graph failed: %s", exc)
        return {
            "bull_arguments": ["Bull分析失敗"],
            "bear_arguments": ["Bear分析失敗"],
            "bull_conceded_points": [],
            "round_count": 0,
            "bull_confidence": 0.0,
            "bear_confidence": 0.0,
            "prev_bull_confidence": 0.0,
            "bull_confidence_history": [0.5],
            "bear_confidence_history": [0.5],
            "bull_ok_history": [],
            "bear_ok_history": [],
            "judge_summary": {
                "agreements": [],
                "conflicts": ["議論エンジン障害のため判断不可", "議論不完全（bull失敗, bear失敗, judge失敗）"],
                "confidence_shift": {"bull": [], "bear": []},
                "stronger_side": "neutral",
            },
            "_meta": {
                "ok": False,
                "engine": "langgraph",
                "model": DEBATE_MODEL,
                "bull_ok": False,
                "bear_ok": False,
                "judge_ok": False,
                "bull_error": str(exc),
                "bear_error": str(exc),
                "judge_error": str(exc),
                "error": str(exc),
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
            },
        }
