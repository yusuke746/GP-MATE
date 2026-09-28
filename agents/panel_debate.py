"""Analyst panel debate: technical, macro and sentiment analysts discuss one
question, "is this a trend-continuation market or a reversal (range) market?",
and a judge writes up where they agree and disagree.

Nobody is assigned a side. Each analyst speaks from their own report, reads the
other two reports and what has been said so far, may agree, disagree or change
their view, and must say what would change it. There is no instruction to
"always argue X", no minimum confidence, no fabricated rebuttal requirement.

Fail-safe policy is limited to *shape*: an analyst who does not answer is
recorded as absent; if the judge does not answer, the regime is taken from a
plain majority of the analysts' stated views and marked as such; if fewer
than two analysts answer, the report is marked not ok and main.py holds.
The report keeps the field names downstream already reads (judge_summary,
regime_summary, stronger_side, _meta.usage).
"""

from __future__ import annotations

import json
import logging
from typing import Any, Final, Literal

from agents.base import get_default_client
from config import MODEL_DEBATE, PANEL_DEBATE_ROUNDS

LOGGER = logging.getLogger(__name__)

AXIS_PANEL: Final[str] = "panel"
PANEL_ROLES: Final[tuple[str, ...]] = ("technical", "macro", "sentiment")
ROLE_LABELS: Final[dict[str, str]] = {
    "technical": "テクニカル分析官",
    "macro": "マクロ分析官",
    "sentiment": "ニュース(センチメント)専門家",
}
REGIME_VIEW_VALUES: Final[tuple[str, ...]] = ("TREND_CONTINUATION", "MEAN_REVERSION", "UNCLEAR")
REGIME_VALUES: Final[tuple[str, ...]] = ("TREND", "RANGE", "TRANSITION")
ENTRY_STYLES: Final[tuple[str, ...]] = ("STOP_BREAKOUT", "LIMIT_PULLBACK", "LIMIT_FADE", "NONE")
CONSENSUS_VALUES: Final[tuple[str, ...]] = ("UNANIMOUS", "MAJORITY", "SPLIT")
# Regime confidence for the log is derived from how many analysts agree, not
# from anyone's self-reported number.
CONSENSUS_CONFIDENCE: Final[dict[str, float]] = {"UNANIMOUS": 0.8, "MAJORITY": 0.6, "SPLIT": 0.4}

PANEL_SYSTEM_PROMPT = (
    "あなたはGOLD(XAU/USD)の分析官パネルの一員として討論に参加します。"
    "あなたの専門は__ROLE__です。論点はただ一つ: いまの相場はトレンド継続局面(押し目買い/戻り売りやブレイク追随が機能する)か、"
    "反転・平均回帰局面(帯の端で逆方向に戻りやすい)か、判断できないか。"
    "自分の専門のレポート(your_report)を出発点に、他の分析官のレポート(other_reports)を読み、"
    "自分の専門から見た客観的な見解を述べてください。transcriptには前の巡までの発言だけが入ります"
    "(第1巡は空で、全員が同じ情報だけを見て独立に再評価します)。"
    "他の分析官に同意してもよいし、反対してもよいし、説得されて見解を変えてもよい"
    "(自分のレポートの見解から変えたらchanged_view=trueにし、どの数値・水準で変えたかをchange_reasonに書く)。"
    "反論は求めません。根拠のない主張はしないこと。根拠が弱ければUNCLEARと答えること。"
    "regime_hintはコードによる機械的な暫定判定で、参考情報にすぎません。従う必要はありません。"
    "出力は次のキーだけを持つJSON: "
    "{regime_view: 'TREND_CONTINUATION'|'MEAN_REVERSION'|'UNCLEAR', direction_if_trend: 'UP'|'DOWN'|'NEUTRAL', "
    "statement: string(日本語、自分の専門からの見解と根拠), responses_to_others: string[](他の分析官の具体的な論点への応答、無ければ空), "
    "key_prices: {continuation_confirms: number|null, reversal_confirms: number|null}, "
    "what_would_change_view: string, changed_view: boolean, change_reason: string}"
)

DEBATE_PROTOCOL_VERSION: Final[str] = "panel-v2-symmetric-round1"
JUDGE_MAX_ATTEMPTS: Final[int] = 2
STATEMENT_KEYS: Final[tuple[str, ...]] = (
    "role", "round", "regime_view", "direction_if_trend", "statement", "responses_to_others",
    "key_prices", "what_would_change_view", "changed_view", "change_reason",
)

PANEL_JUDGE_SYSTEM_PROMPT = (
    "あなたは分析官パネル(テクニカル・マクロ・ニュース)の討論を整理する議長です。"
    "あなた自身の相場観を加えず、3人の発言から次を整理してください: 合意点、対立点、"
    "パネルとしての結論はTREND(継続・順張りが機能)/RANGE(反転・逆張りが機能)/TRANSITION(結論が割れる、または判断できない)のどれか、"
    "TRENDならその方向(direction_if_trend)と、待てる押し目/戻りがあるならLIMIT_PULLBACK、伸び切っておらずブレイクを追えるならSTOP_BREAKOUT、"
    "RANGEならLIMIT_FADE、TRANSITIONならNONE(entry_style)。"
    "発言に出た価格から、継続が確認される価格(continuation_confirms)と反転が確認される価格(reversal_confirms)を拾うこと(無ければnull)。"
    "consensusは3人の見解が一致ならUNANIMOUS、2対1ならMAJORITY、それ以外はSPLIT。"
    "UNCLEARが多い、または見解が割れる場合は無理にTREND/RANGEにせずTRANSITIONとすること。"
    "出力は次のキーだけを持つJSON: "
    "{agreements: string[], conflicts: string[], regime: 'TREND'|'RANGE'|'TRANSITION', "
    "direction_if_trend: 'UP'|'DOWN'|'NEUTRAL', entry_style: 'STOP_BREAKOUT'|'LIMIT_PULLBACK'|'LIMIT_FADE'|'NONE', "
    "key_levels: {continuation_confirms: number|null, reversal_confirms: number|null}, "
    "consensus: 'UNANIMOUS'|'MAJORITY'|'SPLIT', summary: string(日本語)}"
)


def _enum(value: Any, allowed: tuple[str, ...], default: str) -> str:
    text = str(value or "").upper().strip()
    return text if text in allowed else default


def _price(value: Any) -> float | None:
    try:
        return round(float(value), 5) if value is not None else None
    except (TypeError, ValueError):
        return None


def _usage_of(result: Any) -> dict[str, int]:
    usage = getattr(result, "usage", None)
    return {
        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
    }


def _add_usage(total: dict[str, int], part: dict[str, int]) -> None:
    for key in total:
        total[key] += int(part.get(key, 0) or 0)


def _statement_from_payload(role: str, round_index: int, payload: dict[str, Any]) -> dict[str, Any]:
    regime_view = _enum(payload.get("regime_view"), REGIME_VIEW_VALUES, "UNCLEAR")
    direction = _enum(payload.get("direction_if_trend"), ("UP", "DOWN", "NEUTRAL"), "NEUTRAL")
    if regime_view != "TREND_CONTINUATION":
        direction = "NEUTRAL"
    responses = payload.get("responses_to_others")
    key_prices = payload.get("key_prices") if isinstance(payload.get("key_prices"), dict) else {}
    return {
        "role": role,
        "round": round_index,
        "ok": True,
        "regime_view": regime_view,
        "direction_if_trend": direction,
        "statement": str(payload.get("statement", "") or "").strip(),
        "responses_to_others": [str(x) for x in responses if str(x).strip()] if isinstance(responses, list) else [],
        "key_prices": {
            "continuation_confirms": _price(key_prices.get("continuation_confirms")),
            "reversal_confirms": _price(key_prices.get("reversal_confirms")),
        },
        "what_would_change_view": str(payload.get("what_would_change_view", "") or ""),
        "changed_view": bool(payload.get("changed_view", False)),
        "change_reason": str(payload.get("change_reason", "") or ""),
    }


def _absent_statement(role: str, round_index: int, error: str) -> dict[str, Any]:
    return {
        "role": role,
        "round": round_index,
        "ok": False,
        "error": error,
        "regime_view": "UNCLEAR",
        "direction_if_trend": "NEUTRAL",
        "statement": "",
        "responses_to_others": [],
        "key_prices": {"continuation_confirms": None, "reversal_confirms": None},
        "what_would_change_view": "",
        "changed_view": False,
        "change_reason": "",
    }


def _initial_view(role: str, report: dict[str, Any]) -> str:
    """The analyst's stance in its own report, mapped onto the panel vocabulary."""
    raw = str(report.get("regime_view", "") or "").upper()
    return {
        "TREND_CONTINUATION": "TREND_CONTINUATION",
        "SUPPORTS_CONTINUATION": "TREND_CONTINUATION",
        "MEAN_REVERSION": "MEAN_REVERSION",
        "SUPPORTS_REVERSAL": "MEAN_REVERSION",
    }.get(raw, "UNCLEAR")


def _latest_views(transcript: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for statement in transcript:
        if statement.get("ok"):
            latest[str(statement["role"])] = statement
    return latest


def _consensus_of(views: list[str]) -> str:
    if not views:
        return "SPLIT"
    counts: dict[str, int] = {}
    for view in views:
        counts[view] = counts.get(view, 0) + 1
    top = max(counts.values())
    if top == len(views) and len(views) >= 2:
        return "UNANIMOUS"
    if top >= 2 and top > len(views) - top:
        return "MAJORITY"
    return "SPLIT"


def _vote_fallback(transcript: list[dict[str, Any]]) -> dict[str, Any]:
    """Regime from a plain majority of the analysts' latest views (judge absent)."""
    latest = _latest_views(transcript)
    views = [s["regime_view"] for s in latest.values()]
    consensus = _consensus_of(views)
    counts = {v: views.count(v) for v in REGIME_VIEW_VALUES}
    regime = "TRANSITION"
    direction = "NEUTRAL"
    if consensus in {"UNANIMOUS", "MAJORITY"}:
        top = max(counts, key=lambda k: counts[k])
        if top == "TREND_CONTINUATION":
            regime = "TREND"
            directions = [s["direction_if_trend"] for s in latest.values() if s["regime_view"] == "TREND_CONTINUATION"]
            ups, downs = directions.count("UP"), directions.count("DOWN")
            direction = "UP" if ups > downs else ("DOWN" if downs > ups else "NEUTRAL")
            if direction == "NEUTRAL":
                regime = "TRANSITION"
        elif top == "MEAN_REVERSION":
            regime = "RANGE"
    entry_style = {"TREND": "LIMIT_PULLBACK", "RANGE": "LIMIT_FADE"}.get(regime, "NONE")
    return {
        "agreements": [],
        "conflicts": ["議長が回答しなかったため分析官の多数決で代替"],
        "regime": regime,
        "direction_if_trend": direction,
        "entry_style": entry_style,
        "key_levels": {"continuation_confirms": None, "reversal_confirms": None},
        "consensus": consensus,
        "summary": "",
        "source": "vote_fallback",
    }


def _judge_from_payload(payload: dict[str, Any], transcript: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Validated chair verdict, or None when the output cannot be executed on:
    missing/invalid regime or direction enum, TREND without a direction."""
    regime = str(payload.get("regime", "") or "").upper().strip()
    if regime not in REGIME_VALUES:
        return None
    direction_raw = str(payload.get("direction_if_trend", "NEUTRAL") or "NEUTRAL").upper().strip()
    if direction_raw not in ("UP", "DOWN", "NEUTRAL"):
        return None
    direction = direction_raw if regime == "TREND" else "NEUTRAL"
    if regime == "TREND" and direction == "NEUTRAL":
        return None
    entry_style = _enum(payload.get("entry_style"), ENTRY_STYLES, {"TREND": "LIMIT_PULLBACK", "RANGE": "LIMIT_FADE"}.get(regime, "NONE"))
    if regime == "TRANSITION":
        entry_style = "NONE"
    levels = payload.get("key_levels") if isinstance(payload.get("key_levels"), dict) else {}
    views = [s["regime_view"] for s in _latest_views(transcript).values()]
    consensus = _enum(payload.get("consensus"), CONSENSUS_VALUES, _consensus_of(views))
    agreements = payload.get("agreements")
    conflicts = payload.get("conflicts")
    return {
        "agreements": [str(x) for x in agreements] if isinstance(agreements, list) else [],
        "conflicts": [str(x) for x in conflicts] if isinstance(conflicts, list) else [],
        "regime": regime,
        "direction_if_trend": direction,
        "entry_style": entry_style,
        "key_levels": {
            "continuation_confirms": _price(levels.get("continuation_confirms")),
            "reversal_confirms": _price(levels.get("reversal_confirms")),
        },
        "consensus": consensus,
        "summary": str(payload.get("summary", "") or ""),
        "source": "judge",
    }


def _stronger_side(regime: str, direction: str) -> Literal["bull", "bear", "neutral"]:
    if regime == "TREND" and direction == "UP":
        return "bull"
    if regime == "TREND" and direction == "DOWN":
        return "bear"
    return "neutral"


def _vote_distribution(latest: dict[str, dict[str, Any]]) -> dict[str, int]:
    views = [s["regime_view"] for s in latest.values()]
    return {view: views.count(view) for view in REGIME_VIEW_VALUES}


def _ask_chair(llm: Any, judge_payload: dict[str, Any], model: str, transcript: list[dict[str, Any]], usage: dict[str, int]) -> tuple[dict[str, Any] | None, str, int]:
    """Up to JUDGE_MAX_ATTEMPTS chair calls. -> (verdict or None, last error, attempts)."""
    error = ""
    attempts = 0
    for attempt in range(1, JUDGE_MAX_ATTEMPTS + 1):
        attempts = attempt
        payload = dict(judge_payload)
        if attempt > 1:
            payload["retry_note"] = f"前回の回答は無効でした({error})。指定された JSON キーと列挙値だけで答え直してください。"
        try:
            result = llm.call_json(
                system_prompt=PANEL_JUDGE_SYSTEM_PROMPT,
                user_prompt=json.dumps(payload, ensure_ascii=False),
                model=model,
                fallback_payload={},
            )
        except Exception as exc:
            error = str(exc)
            continue
        _add_usage(usage, _usage_of(result))
        raw = result.payload if isinstance(getattr(result, "payload", None), dict) else {}
        if not bool(getattr(result, "ok", False)):
            error = str(getattr(result, "error", "") or "chair call failed")
            continue
        verdict = _judge_from_payload(raw, transcript)
        if verdict is None:
            error = "invalid chair output (regime/direction missing or out of enum)"
            continue
        return verdict, "", attempts
    return None, error, attempts


def run_panel_debate(
    technical_report: dict[str, Any],
    sentiment_report: dict[str, Any],
    macro_report: dict[str, Any] | None = None,
    *,
    rounds: int = PANEL_DEBATE_ROUNDS,
    regime_hint: dict[str, Any] | None = None,
    model: str = MODEL_DEBATE,
    client: Any = None,
) -> dict[str, Any]:
    llm = client or get_default_client()
    reports = {
        "technical": technical_report if isinstance(technical_report, dict) else {},
        "macro": macro_report if isinstance(macro_report, dict) else {},
        "sentiment": sentiment_report if isinstance(sentiment_report, dict) else {},
    }
    hint = dict(regime_hint) if isinstance(regime_hint, dict) else (
        dict(technical_report.get("regime")) if isinstance(technical_report, dict) and isinstance(technical_report.get("regime"), dict) else {}
    )
    initial_views = {role: _initial_view(role, reports[role]) for role in PANEL_ROLES}
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    transcript: list[dict[str, Any]] = []
    errors: list[str] = []
    model_name = model
    total_rounds = max(1, int(rounds))

    for round_index in range(1, total_rounds + 1):
        # Symmetric rounds: everyone sees the three reports plus the statements
        # of PREVIOUS rounds only, never this round's earlier speakers.
        visible = [{k: s[k] for k in STATEMENT_KEYS} for s in transcript if s.get("ok") and int(s["round"]) < round_index]
        for role in PANEL_ROLES:
            user_payload = {
                "round": round_index,
                "your_role": role,
                "your_report": reports[role],
                "your_initial_view": initial_views[role],
                "other_reports": {k: v for k, v in reports.items() if k != role},
                "transcript": visible,
                "regime_hint": hint,
                "question": "いまの相場はトレンド継続局面か、反転・平均回帰局面か、判断できないか。",
            }
            try:
                result = llm.call_json(
                    system_prompt=PANEL_SYSTEM_PROMPT.replace("__ROLE__", ROLE_LABELS[role]),
                    user_prompt=json.dumps(user_payload, ensure_ascii=False),
                    model=model,
                    fallback_payload={},
                )
            except Exception as exc:
                errors.append(f"{role}#{round_index}: {exc}")
                transcript.append(_absent_statement(role, round_index, str(exc)))
                continue
            model_name = str(getattr(result, "model", model) or model)
            _add_usage(usage, _usage_of(result))
            payload = result.payload if isinstance(getattr(result, "payload", None), dict) else {}
            if not bool(getattr(result, "ok", False)) or "regime_view" not in payload:
                error = str(getattr(result, "error", "") or "no regime_view in output")
                errors.append(f"{role}#{round_index}: {error}")
                transcript.append(_absent_statement(role, round_index, error))
                continue
            statement = _statement_from_payload(role, round_index, payload)
            # changed_view is a fact computed against the analyst's own report,
            # not only the flag it set.
            statement["changed_view"] = bool(statement["changed_view"] or statement["regime_view"] != initial_views[role])
            transcript.append(statement)

    latest = _latest_views(transcript)
    enough_analysts = len(latest) >= 2
    judge_status = "FAILED"
    judge_error = ""
    judge_attempts = 0
    verdict: dict[str, Any] | None = None
    if enough_analysts:
        judge_payload = {
            "transcript": [{k: s[k] for k in STATEMENT_KEYS} for s in transcript if s.get("ok")],
            "initial_views": initial_views,
            "absent_analysts": sorted({s["role"] for s in transcript if not s.get("ok")}),
            "regime_hint": hint,
        }
        verdict, judge_error, judge_attempts = _ask_chair(llm, judge_payload, model, transcript, usage)
        if verdict is not None:
            judge_status = "OK" if judge_attempts == 1 else "RECOVERED_RETRY"
    else:
        judge_error = "fewer than two analysts answered"

    # Vote fallback is for the record only: with the chair failed the report is
    # NOT ok, so the trading loop holds; the logger still gets a regime to score.
    if verdict is None:
        verdict = _vote_fallback(transcript)
        verdict["source"] = "vote_fallback_log_only"
        if not enough_analysts:
            verdict["conflicts"] = ["分析官の回答が2名未満のため判定不能"]
            verdict["regime"], verdict["direction_if_trend"], verdict["entry_style"] = "TRANSITION", "NEUTRAL", "NONE"

    distribution = _vote_distribution(latest)
    hint_regime = str(hint.get("regime", "") or "")
    regime_summary = {
        "regime": verdict["regime"],
        # Not a probability: how many of the analysts who answered agree. Never
        # used as an order threshold; kept for the log and later calibration.
        "regime_confidence": None,
        "panel_agreement": CONSENSUS_CONFIDENCE.get(verdict["consensus"], 0.4),
        "panel_votes_available": len(latest),
        "panel_vote_distribution": distribution,
        "panel_consensus_type": verdict["consensus"],
        "direction_if_trend": verdict["direction_if_trend"],
        "entry_style": verdict["entry_style"],
        "key_levels": verdict["key_levels"],
        "consensus": verdict["consensus"],
        "stronger_advocate": "neutral",
        "source": verdict["source"],
    }
    if hint_regime and hint_regime != verdict["regime"]:
        regime_summary["disagrees_with_rule"] = True

    stronger = _stronger_side(verdict["regime"], verdict["direction_if_trend"])
    ok = enough_analysts and judge_status != "FAILED"
    absent = sorted({s["role"] for s in transcript if not s.get("ok")})
    return {
        "axis": AXIS_PANEL,
        "regime_summary": regime_summary,
        "panel_transcript": transcript,
        "panel_views": {
            role: {
                "initial_view": initial_views[role],
                "regime_view": s["regime_view"],
                "direction_if_trend": s["direction_if_trend"],
                "changed_view": s["changed_view"],
                "change_reason": s["change_reason"],
            }
            for role, s in latest.items()
        },
        "bull_arguments": [],
        "bear_arguments": [],
        "bull_conceded_points": [],
        "bear_conceded_points": [],
        "round_count": total_rounds,
        "bull_confidence": 0.0,
        "bear_confidence": 0.0,
        "prev_bull_confidence": 0.0,
        "bull_confidence_history": [],
        "bear_confidence_history": [],
        "judge_summary": {
            "agreements": verdict["agreements"],
            "conflicts": verdict["conflicts"] + ([f"欠席: {', '.join(absent)}"] if absent else []),
            "confidence_shift": {"bull": [], "bear": []},
            "stronger_side": stronger,
            "regime_summary": regime_summary,
            "consensus": verdict["consensus"],
            "summary": verdict.get("summary", ""),
        },
        "_meta": {
            "ok": ok,
            "engine": "panel",
            "model": model_name,
            "judge_ok": judge_status != "FAILED",
            "judge_status": judge_status,
            "judge_attempts": judge_attempts,
            "judge_error": judge_error,
            "analysts_ok": sorted(latest.keys()),
            "errors": errors,
            "usage": usage,
            "debate_protocol_version": DEBATE_PROTOCOL_VERSION,
        },
    }
