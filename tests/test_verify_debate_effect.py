from __future__ import annotations

from typing import Any
from unittest.mock import patch

from scripts.verify_debate_effect import (
    CaseComparison,
    CaseRepeatMetrics,
    PatternResult,
    ScenarioCase,
    _build_verdict,
    _calc_case_repeat_metrics,
    _debate_moved_from_confidence_shift,
    _run_case,
    _scenario_cases,
    parse_args,
)


def test_build_verdict_branching_for_skip_and_executed_cases() -> None:
    assert _build_verdict(False, "議論スキップ（レンジのため）", False).startswith("議論スキップ")
    assert _build_verdict(True, "", True) == "議論が判断を変えた"
    assert _build_verdict(True, "", False) == "議論したが判断は不変"


def test_bear_zero_crash_does_not_create_false_positive_when_bull_also_moves() -> None:
    debate_report = {
        "judge_summary": {
            "confidence_shift": {
                "bull": [0.5, 0.61, 0.74],
                "bear": [0.5, 0.86, 0.0, 0.0],
            }
        },
        "_meta": {
            "bull_ok_history": [True, True],
            "bear_ok_history": [True, False, False],
        },
    }

    # bear side crash is excluded; moved result comes from valid bull-side change only.
    assert _debate_moved_from_confidence_shift(debate_report) is True


def test_debate_moved_detected_when_two_valid_points_exist() -> None:
    debate_report = {
        "judge_summary": {
            "confidence_shift": {
                "bull": [0.5, 0.55, 0.63],
                "bear": [0.5, 0.45, 0.42],
            }
        },
        "_meta": {
            "bull_ok_history": [True, True],
            "bear_ok_history": [True, True],
        },
    }

    assert _debate_moved_from_confidence_shift(debate_report) is True


def test_zero_crash_only_report_is_not_counted_as_moved() -> None:
    debate_report = {
        "judge_summary": {
            "confidence_shift": {
                "bull": [0.5, 0.5],
                "bear": [0.5, 0.0, 0.0],
            }
        },
        "_meta": {
            "bull_ok_history": [True],
            "bear_ok_history": [False, False],
        },
    }

    assert _debate_moved_from_confidence_shift(debate_report) is False


def test_parse_args_repeat_option() -> None:
    args = parse_args(["--repeat", "7", "--max-rounds", "4"])
    assert args.repeat == 7
    assert args.max_rounds == 4


def test_repeat_metrics_computation() -> None:
    case = ScenarioCase(
        name="D",
        description="test",
        technical_report={"signal": "BUY", "trend": "UP"},
        sentiment_report={"score": -0.2},
    )

    base_debate = {
        "judge_summary": {
            "conflicts": ["x"],
            "stronger_side": "bull",
            "confidence_shift": {"bull": [0.5, 0.7], "bear": [0.5, 0.6]},
        },
        "_meta": {"bull_ok_history": [True], "bear_ok_history": [True], "usage": {"total_tokens": 100}},
    }
    base_debate_2 = {
        "judge_summary": {
            "conflicts": ["x"],
            "stronger_side": "bear",
            "confidence_shift": {"bull": [0.5, 0.6], "bear": [0.5, 0.7]},
        },
        "_meta": {"bull_ok_history": [True], "bear_ok_history": [True], "usage": {"total_tokens": 120}},
    }

    r1 = CaseComparison(
        case=case,
        with_debate=PatternResult(
            debate_report=base_debate,
            trader_report={"action": "BUY", "confidence": 0.7, "_meta": {"usage": {"total_tokens": 50}}},
        ),
        without_debate=PatternResult(
            debate_report={},
            trader_report={"action": "HOLD", "confidence": 0.5, "_meta": {"usage": {"total_tokens": 20}}},
        ),
        action_changed=True,
        confidence_diff=0.2,
        reasoning_changed=True,
        verdict="議論が判断を変えた",
        debate_moved=True,
        debate_executed=True,
        skip_reason="",
        tokens_always_a=150,
        tokens_conditional_a=150,
    )
    r2 = CaseComparison(
        case=case,
        with_debate=PatternResult(
            debate_report=base_debate_2,
            trader_report={"action": "HOLD", "confidence": 0.6, "_meta": {"usage": {"total_tokens": 60}}},
        ),
        without_debate=PatternResult(
            debate_report={},
            trader_report={"action": "HOLD", "confidence": 0.55, "_meta": {"usage": {"total_tokens": 18}}},
        ),
        action_changed=False,
        confidence_diff=0.05,
        reasoning_changed=True,
        verdict="議論スキップ（明確なトレンドのため）",
        debate_moved=False,
        debate_executed=False,
        skip_reason="議論スキップ（明確なトレンドのため）",
        tokens_always_a=180,
        tokens_conditional_a=60,
    )

    metrics: CaseRepeatMetrics = _calc_case_repeat_metrics(case, [r1, r2])

    assert metrics.repeat == 2
    assert metrics.action_change_rate == 0.5
    assert metrics.confidence_diff_mean > 0.0
    assert metrics.confidence_diff_std >= 0.0
    assert metrics.stronger_side_distribution["bull"] == 1
    assert metrics.stronger_side_distribution["bear"] == 1
    assert metrics.debate_execution_rate == 0.5
    assert metrics.skipped_rate == 0.5
    assert metrics.executed_action_change_rate == 1.0
    assert metrics.with_debate_buy_rate_executed_only == 1.0
    assert metrics.buy_conversion_rate_executed_only == 1.0
    assert metrics.debate_moved_rate_executed_only == 1.0
    assert metrics.avg_tokens_a < metrics.avg_tokens_always_a


def test_scenario_cases_include_new_macro_and_divergent_validation_cases() -> None:
    cases = _scenario_cases()
    names = [case.name for case in cases]

    assert names[:4] == ["A", "B", "C", "D"]
    assert "E" in names
    assert "F" in names
    assert "G" in names

    case_e = next(case for case in cases if case.name == "E")
    case_f = next(case for case in cases if case.name == "F")
    case_g = next(case for case in cases if case.name == "G")

    assert case_e.macro_report is not None and case_e.macro_report["macro_bias"] == "BEARISH"
    assert case_f.technical_report["alignment"] == "DIVERGENT"
    assert case_g.macro_report is not None and case_g.technical_report["alignment"] == "DIVERGENT"


def test_run_case_passes_macro_and_multitimeframe_inputs_into_debate() -> None:
    captured: dict[str, dict[str, Any]] = {}

    case = ScenarioCase(
        name="X",
        description="macro and mtf capture",
        technical_report={
            "signal": "BUY",
            "trend": "UP",
            "d1_trend": "DOWN",
            "execution_trend": "UP",
            "alignment": "DIVERGENT",
        },
        sentiment_report={"score": 0.1},
        macro_report={"macro_bias": "BEARISH", "confidence": 0.7, "_meta": {"ok": True, "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}, "model": "rule_based", "error": ""}},
    )

    def fake_run_debate_graph(
        technical_report: dict[str, Any],
        sentiment_report: dict[str, Any],
        macro_report: dict[str, Any] | None = None,
        max_rounds: int = 3,
        llm_override: Any | None = None,
    ) -> dict[str, Any]:
        _ = sentiment_report, max_rounds, llm_override
        captured["input"] = {
            "macro_report": macro_report or {},
            "alignment": technical_report.get("alignment", ""),
            "d1_trend": technical_report.get("d1_trend", ""),
            "execution_trend": technical_report.get("execution_trend", ""),
        }
        return {
            "bull_arguments": ["x"],
            "bear_arguments": ["y"],
            "bull_conceded_points": [],
            "round_count": 2,
            "bull_confidence": 0.7,
            "bear_confidence": 0.6,
            "prev_bull_confidence": 0.5,
            "bull_confidence_history": [0.5, 0.7],
            "bear_confidence_history": [0.5, 0.6],
            "judge_summary": {"agreements": [], "conflicts": [], "confidence_shift": {"bull": [0.5, 0.7], "bear": [0.5, 0.6]}, "stronger_side": "bull"},
            "_meta": {"ok": True, "engine": "langgraph", "model": "gpt-5.4-mini", "bull_ok": True, "bear_ok": True, "judge_ok": True, "bull_ok_history": [True], "bear_ok_history": [True], "bull_error": "", "bear_error": "", "judge_error": "", "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}},
        }

    def fake_decide_trade(
        technical_report: dict[str, Any],
        sentiment_report: dict[str, Any],
        debate_report: dict[str, Any],
        confidence_threshold: float = 0.6,
    ) -> dict[str, Any]:
        _ = technical_report, sentiment_report, debate_report, confidence_threshold
        return {
            "action": "BUY",
            "confidence": 0.7,
            "reasoning": "safe",
            "risk_level": "LOW",
            "_meta": {"usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}},
        }

    with patch("scripts.verify_debate_effect.run_debate_graph", side_effect=fake_run_debate_graph), patch(
        "scripts.verify_debate_effect.decide_trade", side_effect=fake_decide_trade
    ):
        result = _run_case(case, max_rounds=2, confidence_effect_threshold=0.05)

    assert result.debate_executed is True
    assert captured["input"]["macro_report"]["macro_bias"] == "BEARISH"
    assert captured["input"]["alignment"] == "DIVERGENT"
    assert captured["input"]["d1_trend"] == "DOWN"
    assert captured["input"]["execution_trend"] == "UP"


def test_run_case_handles_missing_macro_report_safely() -> None:
    case = ScenarioCase(
        name="Y",
        description="no macro data",
        technical_report={"signal": "BUY", "trend": "UP", "alignment": "ALIGNED"},
        sentiment_report={"score": 0.1},
        macro_report=None,
    )

    with patch(
        "scripts.verify_debate_effect.run_debate_graph",
        side_effect=lambda technical_report, sentiment_report, macro_report=None, max_rounds=3, llm_override=None: {
            "bull_arguments": ["x"],
            "bear_arguments": ["y"],
            "bull_conceded_points": [],
            "round_count": 2,
            "bull_confidence": 0.7,
            "bear_confidence": 0.6,
            "prev_bull_confidence": 0.5,
            "bull_confidence_history": [0.5, 0.7],
            "bear_confidence_history": [0.5, 0.6],
            "judge_summary": {"agreements": [], "conflicts": [], "confidence_shift": {"bull": [0.5, 0.7], "bear": [0.5, 0.6]}, "stronger_side": "bull"},
            "_meta": {"ok": True, "engine": "langgraph", "model": "gpt-5.4-mini", "bull_ok": True, "bear_ok": True, "judge_ok": True, "bull_ok_history": [True], "bear_ok_history": [True], "bull_error": "", "bear_error": "", "judge_error": "", "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}},
        },
    ), patch(
        "scripts.verify_debate_effect.decide_trade",
        side_effect=lambda technical_report, sentiment_report, debate_report: {
            "action": "BUY",
            "confidence": 0.7,
            "reasoning": "safe",
            "risk_level": "LOW",
            "_meta": {"usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}},
        },
    ):
        result = _run_case(case, max_rounds=2, confidence_effect_threshold=0.05)

    assert result.debate_executed is True
    assert result.with_debate.debate_report["_meta"]["ok"] is True
