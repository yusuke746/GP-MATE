from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from agents.debate_graph import build_skipped_debate_report, run_debate_graph, should_execute_debate
from agents.trader import decide_trade
from config import OPENAI_API_KEY


@dataclass(frozen=True)
class ScenarioCase:
    name: str
    description: str
    technical_report: dict[str, Any]
    sentiment_report: dict[str, Any]
    macro_report: dict[str, Any] | None = None


@dataclass(frozen=True)
class PatternResult:
    debate_report: dict[str, Any]
    trader_report: dict[str, Any]


@dataclass(frozen=True)
class CaseComparison:
    case: ScenarioCase
    with_debate: PatternResult
    without_debate: PatternResult
    action_changed: bool
    confidence_diff: float
    reasoning_changed: bool
    verdict: str
    debate_moved: bool
    debate_executed: bool
    skip_reason: str
    tokens_always_a: int
    tokens_conditional_a: int


@dataclass(frozen=True)
class CaseRepeatMetrics:
    case: ScenarioCase
    repeat: int
    action_change_rate: float
    confidence_diff_mean: float
    confidence_diff_std: float
    stronger_side_mode: str
    stronger_side_mode_rate: float
    stronger_side_distribution: dict[str, int]
    with_debate_buy_rate: float
    with_debate_buy_rate_executed_only: float
    buy_conversion_rate_executed_only: float
    debate_moved_rate: float
    debate_moved_rate_executed_only: float
    debate_execution_rate: float
    skipped_rate: float
    executed_action_change_rate: float
    avg_tokens_a: float
    avg_tokens_always_a: float
    avg_tokens_b: float


def _scenario_cases() -> list[ScenarioCase]:
    return [
        ScenarioCase(
            name="A",
            description="強い上昇トレンド+過熱(意見が割れやすい)",
            technical_report={
                "signal": "BUY",
                "trend": "STRONG_UP",
                "rsi_14": 76.0,
                "macd_hist": 1.8,
                "atr_14": 12.0,
                "note": "trend is bullish but overbought",
                "_meta": {"ok": True, "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}},
            },
            sentiment_report={
                "score": 0.15,
                "sentiment": "MIXED",
                "evidence_status": "SUFFICIENT",
                "headline_summary": "上昇継続期待と過熱警戒が混在",
                "_meta": {"ok": True, "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}},
            },
        ),
        ScenarioCase(
            name="B",
            description="明確な下落トレンド",
            technical_report={
                "signal": "SELL",
                "trend": "STRONG_DOWN",
                "rsi_14": 31.0,
                "macd_hist": -1.5,
                "atr_14": 14.0,
                "note": "persistent lower highs and lower lows",
                "_meta": {"ok": True, "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}},
            },
            sentiment_report={
                "score": -0.45,
                "sentiment": "BEARISH",
                "evidence_status": "SUFFICIENT",
                "headline_summary": "景気減速懸念で安全資産選好",
                "_meta": {"ok": True, "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}},
            },
        ),
        ScenarioCase(
            name="C",
            description="レンジ・方向感なし(最も割れやすい)",
            technical_report={
                "signal": "NEUTRAL",
                "trend": "RANGE",
                "rsi_14": 50.5,
                "macd_hist": 0.03,
                "atr_14": 7.0,
                "note": "no directional edge",
                "_meta": {"ok": True, "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}},
            },
            sentiment_report={
                "score": 0.0,
                "sentiment": "MIXED",
                "evidence_status": "SUFFICIENT",
                "headline_summary": "材料不足で見通し不透明",
                "_meta": {"ok": True, "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}},
            },
        ),
        ScenarioCase(
            name="D",
            description="指標とセンチメントが矛盾(テクニカル強気xニュース弱気)",
            technical_report={
                "signal": "BUY",
                "trend": "UP",
                "rsi_14": 62.0,
                "macd_hist": 0.8,
                "atr_14": 10.0,
                "note": "technical bullish breakout",
                "_meta": {"ok": True, "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}},
            },
            sentiment_report={
                "score": -0.55,
                "sentiment": "STRONGLY_BEARISH",
                "evidence_status": "SUFFICIENT",
                "headline_summary": "地政学リスク拡大で下押し警戒",
                "_meta": {"ok": True, "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}},
            },
        ),
        ScenarioCase(
            name="E",
            description="テクニカル強気×マクロ弱気(テクニカルは買い、でもマクロは逆風)",
            technical_report={
                "signal": "BUY",
                "trend": "UP",
                "d1_trend": "UP",
                "execution_trend": "UP",
                "alignment": "ALIGNED",
                "rsi_14": 61.0,
                "macd_hist": 0.65,
                "atr_14": 9.0,
                "note": "technical bullish while macro is bearish",
                "_meta": {"ok": True, "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}},
            },
            sentiment_report={
                "score": 0.05,
                "sentiment": "MIXED",
                "evidence_status": "SUFFICIENT",
                "headline_summary": "材料は中立寄り",
                "_meta": {"ok": True, "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}},
            },
            macro_report={
                "macro_bias": "BEARISH",
                "confidence": 0.74,
                "key_drivers": [
                    "DTWEXBGSが上昇しており、ドル高で金に逆風",
                    "期待インフレ率が低下しており、インフレヘッジ需要はやや後退",
                ],
                "reasoning": "ドル高と期待インフレ低下が金に逆風のマクロ局面。",
                "_meta": {"ok": True, "model": "rule_based", "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}, "error": ""},
            },
        ),
        ScenarioCase(
            name="F",
            description="多時間軸DIVERGENT(短期上昇×大局下降)",
            technical_report={
                "signal": "BUY",
                "trend": "UP",
                "d1_trend": "DOWN",
                "execution_trend": "UP",
                "alignment": "DIVERGENT",
                "rsi_14": 58.0,
                "macd_hist": 0.35,
                "atr_14": 8.5,
                "note": "short-term bounce against higher-timeframe downtrend",
                "_meta": {"ok": True, "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}},
            },
            sentiment_report={
                "score": 0.1,
                "sentiment": "MIXED",
                "evidence_status": "SUFFICIENT",
                "headline_summary": "方向感は中立寄り",
                "_meta": {"ok": True, "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}},
            },
            macro_report={
                "macro_bias": "NEUTRAL",
                "confidence": 0.5,
                "key_drivers": ["マクロは方向性が弱い"],
                "reasoning": "マクロは中立で、主な対立は多時間軸のDIVERGENT。",
                "_meta": {"ok": True, "model": "rule_based", "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}, "error": ""},
            },
        ),
        ScenarioCase(
            name="G",
            description="複合矛盾(テクニカル強気×マクロ弱気×多時間軸DIVERGENT)",
            technical_report={
                "signal": "BUY",
                "trend": "UP",
                "d1_trend": "DOWN",
                "execution_trend": "UP",
                "alignment": "DIVERGENT",
                "rsi_14": 63.0,
                "macd_hist": 0.55,
                "atr_14": 9.5,
                "note": "bullish execution against bearish macro and divergent higher timeframe",
                "_meta": {"ok": True, "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}},
            },
            sentiment_report={
                "score": 0.0,
                "sentiment": "MIXED",
                "evidence_status": "SUFFICIENT",
                "headline_summary": "材料は中立",
                "_meta": {"ok": True, "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}},
            },
            macro_report={
                "macro_bias": "BEARISH",
                "confidence": 0.77,
                "key_drivers": [
                    "DTWEXBGSが上昇しており、ドル高で金に逆風",
                    "期待インフレ率が低下しており、インフレヘッジ需要はやや後退",
                ],
                "reasoning": "ドル高と期待インフレ低下の逆風に加え、多時間軸はDIVERGENT。",
                "_meta": {"ok": True, "model": "rule_based", "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}, "error": ""},
            },
        ),
    ]


def _neutral_debate_report() -> dict[str, Any]:
    return {
        "bull_arguments": [],
        "bear_arguments": [],
        "bull_conceded_points": [],
        "round_count": 0,
        "bull_confidence": 0.5,
        "bear_confidence": 0.5,
        "prev_bull_confidence": 0.5,
        "judge_summary": {
            "agreements": [],
            "conflicts": [],
            "confidence_shift": {"bull": [], "bear": []},
            "stronger_side": "neutral",
        },
        "_meta": {
            "ok": True,
            "engine": "none",
            "model": "none",
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        },
    }


def _usage(payload: dict[str, Any]) -> dict[str, int]:
    meta = payload.get("_meta", {}) if isinstance(payload, dict) else {}
    usage = meta.get("usage", {}) if isinstance(meta, dict) else {}
    prompt = int(usage.get("prompt_tokens", 0) or 0)
    completion = int(usage.get("completion_tokens", 0) or 0)
    total = int(usage.get("total_tokens", 0) or 0)
    if total == 0:
        total = prompt + completion
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
    }


def _normalize_reasoning(value: Any) -> str:
    return " ".join(str(value or "").split())


def _debate_moved_from_confidence_shift(debate_report: dict[str, Any]) -> bool:
    judge_summary = debate_report.get("judge_summary", {})
    if not isinstance(judge_summary, dict):
        return False

    shift = judge_summary.get("confidence_shift", {})
    if not isinstance(shift, dict):
        return False

    bull_values = shift.get("bull", [])
    bear_values = shift.get("bear", [])
    meta = debate_report.get("_meta", {}) if isinstance(debate_report, dict) else {}
    bull_ok_history = meta.get("bull_ok_history", []) if isinstance(meta, dict) else []
    bear_ok_history = meta.get("bear_ok_history", []) if isinstance(meta, dict) else []

    def _moved(values: Any, ok_history: Any) -> bool:
        if not isinstance(values, list) or len(values) < 2 or not isinstance(ok_history, list):
            return False

        valid: list[float] = []
        valid_rounds = min(len(ok_history), len(values) - 1)
        for idx in range(valid_rounds):
            if not bool(ok_history[idx]):
                continue
            try:
                valid.append(float(values[idx + 1]))
            except Exception:
                continue

        if len(valid) < 2:
            return False
        return abs(valid[-1] - valid[0]) >= 0.03

    return _moved(bull_values, bull_ok_history) or _moved(bear_values, bear_ok_history)


def _did_debate_change_decision(
    action_changed: bool,
    confidence_diff: float,
    confidence_effect_threshold: float,
) -> bool:
    return action_changed or confidence_diff >= confidence_effect_threshold


def _build_verdict(
    debate_executed: bool,
    skip_reason: str,
    decision_changed_by_debate: bool,
) -> str:
    if not debate_executed:
        reason = skip_reason.strip() or "理由不明"
        if reason.startswith("議論スキップ"):
            return reason
        return f"議論スキップ（{reason}）"
    if decision_changed_by_debate:
        return "議論が判断を変えた"
    return "議論したが判断は不変"


def _run_case(case: ScenarioCase, max_rounds: int, confidence_effect_threshold: float) -> CaseComparison:
    always_debate_report = run_debate_graph(
        technical_report=case.technical_report,
        sentiment_report=case.sentiment_report,
        macro_report=case.macro_report,
        max_rounds=max_rounds,
    )
    trader_always_debate = decide_trade(
        technical_report=case.technical_report,
        sentiment_report=case.sentiment_report,
        debate_report=always_debate_report,
    )

    gate = should_execute_debate(case.technical_report, case.sentiment_report, case.macro_report)
    if gate["should_debate"]:
        debate_report = always_debate_report
        trader_with_debate = trader_always_debate
        debate_executed = True
        skip_reason = ""
    else:
        debate_report = build_skipped_debate_report(gate["reason"])
        trader_with_debate = decide_trade(
            technical_report=case.technical_report,
            sentiment_report=case.sentiment_report,
            debate_report=debate_report,
        )
        debate_executed = False
        skip_reason = gate["reason"]

    neutral_debate = _neutral_debate_report()
    trader_without_debate = decide_trade(
        technical_report=case.technical_report,
        sentiment_report=case.sentiment_report,
        debate_report=neutral_debate,
    )

    action_a = str(trader_with_debate.get("action", "HOLD"))
    action_b = str(trader_without_debate.get("action", "HOLD"))
    confidence_a = float(trader_with_debate.get("confidence", 0.0) or 0.0)
    confidence_b = float(trader_without_debate.get("confidence", 0.0) or 0.0)
    confidence_diff = abs(confidence_a - confidence_b)

    reason_a = _normalize_reasoning(trader_with_debate.get("reasoning", ""))
    reason_b = _normalize_reasoning(trader_without_debate.get("reasoning", ""))

    action_changed = action_a != action_b
    reasoning_changed = reason_a != reason_b
    decision_changed_by_debate = _did_debate_change_decision(
        action_changed=action_changed,
        confidence_diff=confidence_diff,
        confidence_effect_threshold=confidence_effect_threshold,
    )
    debate_moved = debate_executed and decision_changed_by_debate
    verdict = _build_verdict(
        debate_executed=debate_executed,
        skip_reason=skip_reason,
        decision_changed_by_debate=decision_changed_by_debate,
    )

    always_debate_tokens = _usage(always_debate_report)["total_tokens"] + _usage(trader_always_debate)["total_tokens"]
    conditional_tokens = _usage(debate_report)["total_tokens"] + _usage(trader_with_debate)["total_tokens"]

    return CaseComparison(
        case=case,
        with_debate=PatternResult(debate_report=debate_report, trader_report=trader_with_debate),
        without_debate=PatternResult(debate_report=neutral_debate, trader_report=trader_without_debate),
        action_changed=action_changed,
        confidence_diff=confidence_diff,
        reasoning_changed=reasoning_changed,
        verdict=verdict,
        debate_moved=debate_moved,
        debate_executed=debate_executed,
        skip_reason=skip_reason,
        tokens_always_a=always_debate_tokens,
        tokens_conditional_a=conditional_tokens,
    )


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return mean, math.sqrt(variance)


def _token_totals_for_result(result: CaseComparison) -> tuple[int, int]:
    usage_debate = _usage(result.with_debate.debate_report)
    usage_a = _usage(result.with_debate.trader_report)
    usage_b = _usage(result.without_debate.trader_report)
    return usage_debate["total_tokens"] + usage_a["total_tokens"], usage_b["total_tokens"]


def _stronger_side_of_result(result: CaseComparison) -> str:
    debate = result.with_debate.debate_report
    judge_summary = debate.get("judge_summary", {}) if isinstance(debate, dict) else {}
    if not isinstance(judge_summary, dict):
        return "neutral"
    return str(judge_summary.get("stronger_side", "neutral") or "neutral")


def _calc_case_repeat_metrics(case: ScenarioCase, results: list[CaseComparison]) -> CaseRepeatMetrics:
    repeat = len(results)

    action_change_count = sum(1 for x in results if x.action_changed)
    confidence_diffs = [float(x.confidence_diff) for x in results]
    conf_mean, conf_std = _mean_std(confidence_diffs)

    stronger_dist: dict[str, int] = {"bull": 0, "bear": 0, "neutral": 0}
    for x in results:
        side = _stronger_side_of_result(x)
        if side not in stronger_dist:
            stronger_dist[side] = 0
        stronger_dist[side] += 1
    stronger_side_mode = max(stronger_dist.items(), key=lambda kv: kv[1])[0]
    stronger_side_mode_rate = (stronger_dist.get(stronger_side_mode, 0) / repeat) if repeat else 0.0

    with_debate_buy_count = sum(
        1 for x in results if str(x.with_debate.trader_report.get("action", "HOLD")).upper() == "BUY"
    )
    debate_moved_count = sum(1 for x in results if x.debate_moved)
    debate_execution_count = sum(1 for x in results if x.debate_executed)
    skipped_count = repeat - debate_execution_count

    executed_results = [x for x in results if x.debate_executed]
    executed_count = len(executed_results)
    executed_action_change_count = sum(1 for x in executed_results if x.action_changed)
    executed_debate_moved_count = sum(1 for x in executed_results if x.debate_moved)
    executed_buy_count = sum(
        1 for x in executed_results if str(x.with_debate.trader_report.get("action", "HOLD")).upper() == "BUY"
    )
    executed_buy_conversion_count = sum(
        1
        for x in executed_results
        if str(x.with_debate.trader_report.get("action", "HOLD")).upper() == "BUY"
        and str(x.without_debate.trader_report.get("action", "HOLD")).upper() != "BUY"
    )

    token_a_values: list[float] = []
    token_always_a_values: list[float] = []
    token_b_values: list[float] = []
    for x in results:
        token_a_values.append(float(x.tokens_conditional_a))
        token_always_a_values.append(float(x.tokens_always_a))
        _, token_b = _token_totals_for_result(x)
        token_b_values.append(float(token_b))

    avg_tokens_a, _ = _mean_std(token_a_values)
    avg_tokens_always_a, _ = _mean_std(token_always_a_values)
    avg_tokens_b, _ = _mean_std(token_b_values)

    return CaseRepeatMetrics(
        case=case,
        repeat=repeat,
        action_change_rate=(action_change_count / repeat) if repeat else 0.0,
        confidence_diff_mean=conf_mean,
        confidence_diff_std=conf_std,
        stronger_side_mode=stronger_side_mode,
        stronger_side_mode_rate=stronger_side_mode_rate,
        stronger_side_distribution=stronger_dist,
        with_debate_buy_rate=(with_debate_buy_count / repeat) if repeat else 0.0,
        with_debate_buy_rate_executed_only=(executed_buy_count / executed_count) if executed_count else 0.0,
        buy_conversion_rate_executed_only=(executed_buy_conversion_count / executed_count) if executed_count else 0.0,
        debate_moved_rate=(debate_moved_count / repeat) if repeat else 0.0,
        debate_moved_rate_executed_only=(executed_debate_moved_count / executed_count) if executed_count else 0.0,
        debate_execution_rate=(debate_execution_count / repeat) if repeat else 0.0,
        skipped_rate=(skipped_count / repeat) if repeat else 0.0,
        executed_action_change_rate=(executed_action_change_count / executed_count) if executed_count else 0.0,
        avg_tokens_a=avg_tokens_a,
        avg_tokens_always_a=avg_tokens_always_a,
        avg_tokens_b=avg_tokens_b,
    )


def _print_case_repeat_metrics(metrics: CaseRepeatMetrics) -> None:
    case = metrics.case
    print(f"\n=== Case {case.name}: {case.description} (repeat={metrics.repeat}) ===")
    print(f"action変化発生率: {metrics.action_change_rate * 100.0:.1f}%")
    print(
        "confidence差: "
        f"mean={metrics.confidence_diff_mean:.3f} "
        f"std={metrics.confidence_diff_std:.3f}"
    )
    print(
        "stronger_side安定性: "
        f"mode={metrics.stronger_side_mode} "
        f"rate={metrics.stronger_side_mode_rate * 100.0:.1f}% "
        f"dist={metrics.stronger_side_distribution}"
    )
    print(f"議論ありBUY率: {metrics.with_debate_buy_rate * 100.0:.1f}%")
    print(f"議論実行ケースBUY率: {metrics.with_debate_buy_rate_executed_only * 100.0:.1f}%")
    print(f"議論実行ケースBUY転換率: {metrics.buy_conversion_rate_executed_only * 100.0:.1f}%")
    print(f"議論が動いた判定率: {metrics.debate_moved_rate * 100.0:.1f}%")
    print(f"議論実行ケースの判定変化率: {metrics.debate_moved_rate_executed_only * 100.0:.1f}%")
    print(f"議論実行率: {metrics.debate_execution_rate * 100.0:.1f}%")
    print(f"スキップ率: {metrics.skipped_rate * 100.0:.1f}%")
    print(f"議論実行ケースaction変化率: {metrics.executed_action_change_rate * 100.0:.1f}%")
    print(
        "平均トークン/ケース: "
        f"A(条件付き)={metrics.avg_tokens_a:.1f}, "
        f"A(常時議論)={metrics.avg_tokens_always_a:.1f}, "
        f"B={metrics.avg_tokens_b:.1f}, "
        f"ratio={(metrics.avg_tokens_a / metrics.avg_tokens_b) if metrics.avg_tokens_b > 0 else 0.0:.2f}x"
    )


def _print_case_result(result: CaseComparison) -> None:
    case = result.case
    trader_a = result.with_debate.trader_report
    trader_b = result.without_debate.trader_report
    debate = result.with_debate.debate_report
    judge_summary = debate.get("judge_summary", {}) if isinstance(debate, dict) else {}
    usage_debate = _usage(debate)
    usage_trader_a = _usage(trader_a)
    usage_trader_b = _usage(trader_b)
    meta_debate = debate.get("_meta", {}) if isinstance(debate, dict) else {}

    print(f"\n=== Case {case.name}: {case.description} ===")
    print(
        "[ゲート] "
        f"debate_executed={result.debate_executed}"
        + (f" reason={result.skip_reason}" if result.skip_reason else "")
    )
    print("[A] 議論あり")
    print(f"  action={trader_a.get('action')} confidence={float(trader_a.get('confidence', 0.0) or 0.0):.3f}")
    print(f"  reasoning={_normalize_reasoning(trader_a.get('reasoning', ''))}")

    print("[B] 議論なし")
    print(f"  action={trader_b.get('action')} confidence={float(trader_b.get('confidence', 0.0) or 0.0):.3f}")
    print(f"  reasoning={_normalize_reasoning(trader_b.get('reasoning', ''))}")

    print("[比較]")
    print(f"  action changed: {result.action_changed}")
    print(f"  confidence diff: {result.confidence_diff:.3f}")
    print(f"  reasoning changed: {result.reasoning_changed}")

    print("[議論の内実]")
    print(f"  round_count={debate.get('round_count', 0)}")
    print(
        "  bull_confidence(prev->last)="
        f"{float(debate.get('prev_bull_confidence', 0.0) or 0.0):.3f}"
        f"->{float(debate.get('bull_confidence', 0.0) or 0.0):.3f}"
    )
    if isinstance(judge_summary, dict):
        conflicts = judge_summary.get("conflicts", [])
        stronger_side = judge_summary.get("stronger_side", "neutral")
        shift = judge_summary.get("confidence_shift", {})
        bull_ok_history = meta_debate.get("bull_ok_history", []) if isinstance(meta_debate, dict) else []
        bear_ok_history = meta_debate.get("bear_ok_history", []) if isinstance(meta_debate, dict) else []
        print(f"  conflicts={conflicts}")
        print(f"  stronger_side={stronger_side}")
        print(f"  confidence_shift={shift}")
        print(f"  bull_ok_history={bull_ok_history}")
        print(f"  bear_ok_history={bear_ok_history}")
    else:
        print("  judge_summary is not structured JSON")

    print("[コスト(トークン)]")
    print(
        f"  A total={usage_debate['total_tokens'] + usage_trader_a['total_tokens']}"
        f" (debate={usage_debate['total_tokens']}, trader={usage_trader_a['total_tokens']})"
    )
    print(f"  B total={usage_trader_b['total_tokens']} (trader only)")
    print(f"[判定] {result.verdict}")


def _write_csv(results_by_case: dict[str, list[CaseComparison]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "repeat_index",
        "case",
        "description",
        "debate_executed",
        "skip_reason",
        "action_a",
        "action_b",
        "action_changed",
        "confidence_a",
        "confidence_b",
        "confidence_diff",
        "reasoning_a",
        "reasoning_b",
        "reasoning_changed",
        "round_count",
        "prev_bull_confidence",
        "bull_confidence",
        "conflicts",
        "stronger_side",
        "confidence_shift",
        "debate_tokens",
        "trader_tokens_a",
        "trader_tokens_b",
        "tokens_always_a",
        "tokens_conditional_a",
        "verdict",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for case_name, case_results in results_by_case.items():
            _ = case_name
            for idx, result in enumerate(case_results, start=1):
                debate = result.with_debate.debate_report
                judge_summary = debate.get("judge_summary", {}) if isinstance(debate, dict) else {}
                usage_debate = _usage(debate)
                usage_a = _usage(result.with_debate.trader_report)
                usage_b = _usage(result.without_debate.trader_report)

                writer.writerow(
                    {
                        "repeat_index": idx,
                        "case": result.case.name,
                        "description": result.case.description,
                        "debate_executed": result.debate_executed,
                        "skip_reason": result.skip_reason,
                        "action_a": result.with_debate.trader_report.get("action", ""),
                        "action_b": result.without_debate.trader_report.get("action", ""),
                        "action_changed": result.action_changed,
                        "confidence_a": float(result.with_debate.trader_report.get("confidence", 0.0) or 0.0),
                        "confidence_b": float(result.without_debate.trader_report.get("confidence", 0.0) or 0.0),
                        "confidence_diff": result.confidence_diff,
                        "reasoning_a": _normalize_reasoning(result.with_debate.trader_report.get("reasoning", "")),
                        "reasoning_b": _normalize_reasoning(result.without_debate.trader_report.get("reasoning", "")),
                        "reasoning_changed": result.reasoning_changed,
                        "round_count": int(debate.get("round_count", 0) or 0),
                        "prev_bull_confidence": float(debate.get("prev_bull_confidence", 0.0) or 0.0),
                        "bull_confidence": float(debate.get("bull_confidence", 0.0) or 0.0),
                        "conflicts": json.dumps(judge_summary.get("conflicts", []), ensure_ascii=False),
                        "stronger_side": judge_summary.get("stronger_side", "neutral"),
                        "confidence_shift": json.dumps(judge_summary.get("confidence_shift", {}), ensure_ascii=False),
                        "debate_tokens": usage_debate["total_tokens"],
                        "trader_tokens_a": usage_a["total_tokens"],
                        "trader_tokens_b": usage_b["total_tokens"],
                        "tokens_always_a": result.tokens_always_a,
                        "tokens_conditional_a": result.tokens_conditional_a,
                        "verdict": result.verdict,
                    }
                )


def _print_summary(results_by_case: dict[str, list[CaseComparison]]) -> None:
    all_results: list[CaseComparison] = []
    for case_results in results_by_case.values():
        all_results.extend(case_results)

    total_runs = len(all_results)
    changed_runs = sum(1 for x in all_results if x.debate_moved)
    avg_conf_diff = sum(x.confidence_diff for x in all_results) / total_runs if total_runs else 0.0
    moved_by_shift = sum(1 for x in all_results if _debate_moved_from_confidence_shift(x.with_debate.debate_report))
    executed_results = [x for x in all_results if x.debate_executed]
    skipped_results = [x for x in all_results if not x.debate_executed]
    executed_count = len(executed_results)
    skipped_count = len(skipped_results)

    executed_action_change_count = sum(1 for x in executed_results if x.action_changed)
    executed_buy_conversion_count = sum(
        1
        for x in executed_results
        if str(x.with_debate.trader_report.get("action", "HOLD")).upper() == "BUY"
        and str(x.without_debate.trader_report.get("action", "HOLD")).upper() != "BUY"
    )
    executed_buy_count = sum(
        1 for x in executed_results if str(x.with_debate.trader_report.get("action", "HOLD")).upper() == "BUY"
    )

    total_tokens_a = 0
    total_tokens_always_a = 0
    total_tokens_b = 0
    debate_execution_count = 0
    for x in all_results:
        total_tokens_a += x.tokens_conditional_a
        total_tokens_always_a += x.tokens_always_a
        _, token_b = _token_totals_for_result(x)
        total_tokens_b += token_b
        if x.debate_executed:
            debate_execution_count += 1

    case_count = len(results_by_case)
    avg_tokens_a_per_case = (total_tokens_a / case_count) if case_count else 0.0
    avg_tokens_b_per_case = (total_tokens_b / case_count) if case_count else 0.0
    token_ratio = (total_tokens_a / total_tokens_b) if total_tokens_b > 0 else 0.0
    cost_reduction = 0.0
    if total_tokens_always_a > 0:
        cost_reduction = (1.0 - (total_tokens_a / total_tokens_always_a)) * 100.0

    print("\n=== Summary ===")
    print(f"判断が変わったラン数(議論実行のみ): {changed_runs}/{executed_count}")
    print(f"confidence変化の平均: {avg_conf_diff:.3f}")
    print(
        "議論が動いた(confidence変遷あり)ラン割合: "
        f"{moved_by_shift}/{total_runs}"
        f" ({(moved_by_shift / total_runs * 100.0) if total_runs else 0.0:.1f}%)"
    )
    print(f"総トークン A(条件付き議論): {total_tokens_a}")
    print(f"総トークン A(常時議論ベースライン): {total_tokens_always_a}")
    print(f"総トークン B(議論なし): {total_tokens_b}")
    print(f"トークン比 A/B: {token_ratio:.2f}x")
    print(
        "議論実行率: "
        f"{debate_execution_count}/{total_runs}"
        f" ({(debate_execution_count / total_runs * 100.0) if total_runs else 0.0:.1f}%)"
    )
    print(
        "スキップ率: "
        f"{skipped_count}/{total_runs}"
        f" ({(skipped_count / total_runs * 100.0) if total_runs else 0.0:.1f}%)"
    )
    print(
        "議論実行ケース action変化率: "
        f"{executed_action_change_count}/{executed_count}"
        f" ({(executed_action_change_count / executed_count * 100.0) if executed_count else 0.0:.1f}%)"
    )
    print(
        "議論実行ケース BUY転換率: "
        f"{executed_buy_conversion_count}/{executed_count}"
        f" ({(executed_buy_conversion_count / executed_count * 100.0) if executed_count else 0.0:.1f}%)"
    )
    print(
        "議論実行ケース BUY率: "
        f"{executed_buy_count}/{executed_count}"
        f" ({(executed_buy_count / executed_count * 100.0) if executed_count else 0.0:.1f}%)"
    )
    print(f"常時議論比コスト削減率: {cost_reduction:.1f}%")
    print(f"1ケースあたり平均トークン A: {avg_tokens_a_per_case:.1f}")
    print(f"1ケースあたり平均トークン B: {avg_tokens_b_per_case:.1f}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="議論あり/なしで最終判断が変わるかをA/B比較する検証スクリプト",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=3,
        help="run_debate_graph の最大ラウンド数",
    )
    parser.add_argument(
        "--confidence-effect-threshold",
        type=float,
        default=0.05,
        help="この閾値以上の confidence 差を『判断差あり』とみなす",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=5,
        help="各ケースの反復回数 (再現性検証)",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="",
        help="結果をCSV保存するパス(任意)",
    )
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()

    if not OPENAI_API_KEY:
        print("OPENAI_API_KEY が未設定です。実API検証のため設定して再実行してください。")
        return 1

    print("=== Verify Debate Effect (Analysis Only / No Orders) ===")
    print("注意: このスクリプトは run_debate_graph + decide_trade のみを実行し、発注は行いません。")

    repeat = max(1, int(args.repeat))
    print(f"repeat={repeat}")

    results_by_case: dict[str, list[CaseComparison]] = {}
    for case in _scenario_cases():
        bucket: list[CaseComparison] = []
        for _ in range(repeat):
            result = _run_case(
                case=case,
                max_rounds=max(1, int(args.max_rounds)),
                confidence_effect_threshold=float(args.confidence_effect_threshold),
            )
            bucket.append(result)
        results_by_case[case.name] = bucket

        # Print one detailed sample and then reproducibility metrics.
        _print_case_result(bucket[-1])
        metrics = _calc_case_repeat_metrics(case, bucket)
        _print_case_repeat_metrics(metrics)

    _print_summary(results_by_case)

    case_d_results = results_by_case.get("D", [])
    if case_d_results:
        case_d_buy = sum(
            1 for x in case_d_results if str(x.with_debate.trader_report.get("action", "HOLD")).upper() == "BUY"
        )
        print(
            f"Case D BUY転換率(議論あり): {case_d_buy}/{len(case_d_results)}"
            f" ({(case_d_buy / len(case_d_results) * 100.0):.1f}%)"
        )

    if args.csv.strip():
        csv_path = Path(args.csv).expanduser()
        if not csv_path.is_absolute():
            csv_path = BASE_DIR / csv_path
        _write_csv(results_by_case=results_by_case, output_path=csv_path)
        print(f"CSV saved: {csv_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
