from __future__ import annotations

import json
from typing import Any
from unittest.mock import Mock

import agents.debate_graph as debate_graph
from agents import panel_debate
from agents.debate_graph import run_debate_graph


def _result(payload: dict[str, Any] | None, ok: bool = True, error: str = "") -> Mock:
    result = Mock()
    result.ok = ok
    result.payload = payload if payload is not None else {}
    result.model = "gpt-5.6-terra"
    result.error = error
    result.usage = Mock(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    return result


def _view(regime_view: str, direction: str = "NEUTRAL", statement: str = "s", changed: bool = False) -> dict[str, Any]:
    return {
        "regime_view": regime_view,
        "direction_if_trend": direction,
        "statement": statement,
        "responses_to_others": ["macroの利回り上昇は既に価格に反映"],
        "key_prices": {"continuation_confirms": 4390.0, "reversal_confirms": 4340.5},
        "what_would_change_view": "H4で4340.5割れ",
        "changed_view": changed,
    }


def _judge(regime: str, direction: str = "NEUTRAL", entry_style: str | None = None, consensus: str = "MAJORITY") -> dict[str, Any]:
    payload = {
        "agreements": ["4390が節目"],
        "conflicts": ["マクロは反転寄り"],
        "regime": regime,
        "direction_if_trend": direction,
        "key_levels": {"continuation_confirms": 4390.0, "reversal_confirms": 4340.5},
        "consensus": consensus,
        "summary": "テクニカルとニュースは継続、マクロは反転。",
    }
    if entry_style is not None:
        payload["entry_style"] = entry_style
    return payload


class _ScriptedClient:
    """Answers analysts by role (read from the user payload) and the judge last."""

    def __init__(self, answers: dict[str, Any], judge: Any) -> None:
        self.answers = answers
        self.judge = judge
        self.calls: list[dict[str, Any]] = []

    def call_json(self, *, system_prompt: str, user_prompt: str, model: str, fallback_payload: dict[str, Any]) -> Mock:
        payload = json.loads(user_prompt)
        self.calls.append({"system": system_prompt, "payload": payload})
        if "your_role" in payload:
            answer = self.answers.get(payload["your_role"])
            if isinstance(answer, Exception):
                raise answer
            return answer if isinstance(answer, Mock) else _result(answer)
        if isinstance(self.judge, Exception):
            raise self.judge
        return self.judge if isinstance(self.judge, Mock) else _result(self.judge)


def test_panel_majority_trend_up_gives_bull_and_full_transcript() -> None:
    client = _ScriptedClient(
        {
            "technical": _view("TREND_CONTINUATION", "UP", "H4はHIGHER_HIGH継続"),
            "macro": _view("MEAN_REVERSION", statement="2年債が反転"),
            "sentiment": _view("TREND_CONTINUATION", "UP", "新規材料はドル安", changed=True),
        },
        _judge("TREND", "UP", "LIMIT_PULLBACK"),
    )
    technical = {"signal": "BUY", "regime": {"regime": "TREND", "direction": "UP", "entry_style": "STOP_BREAKOUT"}}
    report = panel_debate.run_panel_debate(technical, {"gold_bias": "BULLISH"}, {"macro_bias": "BEARISH"}, client=client, rounds=1)

    assert report["axis"] == "panel"
    assert report["_meta"]["ok"] is True and report["_meta"]["judge_ok"] is True
    assert report["_meta"]["analysts_ok"] == ["macro", "sentiment", "technical"]
    assert report["_meta"]["usage"]["total_tokens"] == 15 * 4
    summary = report["regime_summary"]
    assert summary["regime"] == "TREND" and summary["direction_if_trend"] == "UP"
    assert summary["entry_style"] == "LIMIT_PULLBACK"
    assert summary["consensus"] == "MAJORITY" and summary["regime_confidence"] == 0.6
    assert summary["key_levels"] == {"continuation_confirms": 4390.0, "reversal_confirms": 4340.5}
    assert summary["source"] == "judge" and "disagrees_with_rule" not in summary
    assert report["judge_summary"]["stronger_side"] == "bull"
    assert report["judge_summary"]["conflicts"] == ["マクロは反転寄り"]
    assert report["panel_views"]["sentiment"]["changed_view"] is True
    assert [s["role"] for s in report["panel_transcript"]] == ["technical", "macro", "sentiment"]
    assert report["bull_arguments"] == [] and report["bull_confidence"] == 0.0

    # Each analyst saw its own report, the other two, the prior statements and the hint labelled as such.
    technical_call, macro_call, sentiment_call, judge_call = client.calls
    assert technical_call["payload"]["your_report"] == technical
    assert set(technical_call["payload"]["other_reports"]) == {"macro", "sentiment"}
    assert technical_call["payload"]["transcript"] == []
    assert macro_call["payload"]["transcript"][0]["role"] == "technical"
    assert len(sentiment_call["payload"]["transcript"]) == 2
    assert "マクロ分析官" in macro_call["system"] and "テクニカル分析官" in technical_call["system"]
    assert judge_call["payload"]["absent_analysts"] == []
    assert judge_call["system"] == panel_debate.PANEL_JUDGE_SYSTEM_PROMPT


def test_panel_prompts_assign_no_side_and_allow_unclear() -> None:
    for prompt in (panel_debate.PANEL_SYSTEM_PROMPT, panel_debate.PANEL_JUDGE_SYSTEM_PROMPT):
        for forbidden in ("必ず主張", "逃げないこと", "義務", "0.3以上", "最低1つ"):
            assert forbidden not in prompt
    assert "UNCLEAR" in panel_debate.PANEL_SYSTEM_PROMPT
    assert "見解を変えてもよい" in panel_debate.PANEL_SYSTEM_PROMPT
    assert "従う必要はありません" in panel_debate.PANEL_SYSTEM_PROMPT
    assert "あなた自身の相場観を加えず" in panel_debate.PANEL_JUDGE_SYSTEM_PROMPT


def test_panel_range_verdict_is_neutral_and_flags_rule_disagreement() -> None:
    client = _ScriptedClient(
        {"technical": _view("MEAN_REVERSION"), "macro": _view("MEAN_REVERSION"), "sentiment": _view("UNCLEAR")},
        _judge("range", "UP", consensus="MAJORITY"),  # direction is ignored for RANGE, entry_style defaults
    )
    report = panel_debate.run_panel_debate({"signal": "SELL"}, {}, {}, client=client, regime_hint={"regime": "TREND"})
    summary = report["regime_summary"]
    assert summary["regime"] == "RANGE" and summary["direction_if_trend"] == "NEUTRAL"
    assert summary["entry_style"] == "LIMIT_FADE"
    assert summary["disagrees_with_rule"] is True
    assert report["judge_summary"]["stronger_side"] == "neutral"


def test_panel_falls_back_to_majority_vote_when_judge_fails() -> None:
    client = _ScriptedClient(
        {
            "technical": _view("TREND_CONTINUATION", "DOWN"),
            "macro": _view("TREND_CONTINUATION", "DOWN"),
            "sentiment": _view("UNCLEAR"),
        },
        _result({}, ok=False, error="judge timeout"),
    )
    report = panel_debate.run_panel_debate({}, {}, {}, client=client)
    assert report["_meta"]["ok"] is True and report["_meta"]["judge_ok"] is False
    assert report["_meta"]["judge_error"] == "judge timeout"
    summary = report["regime_summary"]
    assert summary["source"] == "vote_fallback"
    assert summary["regime"] == "TREND" and summary["direction_if_trend"] == "DOWN" and summary["entry_style"] == "LIMIT_PULLBACK"
    assert summary["consensus"] == "MAJORITY"
    assert report["judge_summary"]["stronger_side"] == "bear"
    assert "多数決で代替" in report["judge_summary"]["conflicts"][0]


def test_panel_split_views_become_transition_without_forcing() -> None:
    client = _ScriptedClient(
        {"technical": _view("TREND_CONTINUATION", "UP"), "macro": _view("MEAN_REVERSION"), "sentiment": _view("UNCLEAR")},
        RuntimeError("judge exploded"),
    )
    report = panel_debate.run_panel_debate({}, {}, {}, client=client)
    assert report["regime_summary"]["regime"] == "TRANSITION"
    assert report["regime_summary"]["entry_style"] == "NONE"
    assert report["regime_summary"]["consensus"] == "SPLIT"
    assert report["judge_summary"]["stronger_side"] == "neutral"


def test_panel_records_absent_analyst_and_holds_when_fewer_than_two_answer() -> None:
    client = _ScriptedClient(
        {"technical": _view("TREND_CONTINUATION", "UP"), "macro": RuntimeError("macro down"), "sentiment": _result({}, ok=False, error="bad json")},
        _judge("TREND", "UP"),
    )
    report = panel_debate.run_panel_debate({}, {}, {}, client=client)
    assert report["_meta"]["ok"] is False
    assert report["_meta"]["analysts_ok"] == ["technical"]
    assert report["_meta"]["judge_error"] == "fewer than two analysts answered"
    assert report["regime_summary"]["regime"] == "TRANSITION"
    assert any("欠席: macro, sentiment" in c for c in report["judge_summary"]["conflicts"])
    assert [s["ok"] for s in report["panel_transcript"]] == [True, False, False]
    assert len(client.calls) == 3  # no judge call


def test_two_rounds_let_analysts_reply() -> None:
    client = _ScriptedClient(
        {"technical": _view("TREND_CONTINUATION", "UP"), "macro": _view("MEAN_REVERSION"), "sentiment": _view("UNCLEAR")},
        _judge("TRANSITION"),
    )
    report = panel_debate.run_panel_debate({}, {}, {}, client=client, rounds=2)
    assert len(client.calls) == 7
    assert [s["round"] for s in report["panel_transcript"]] == [1, 1, 1, 2, 2, 2]
    assert len(client.calls[3]["payload"]["transcript"]) == 3  # round-2 technical sees all of round 1
    assert report["round_count"] == 2


def test_run_debate_graph_dispatches_panel_axis(monkeypatch) -> None:
    seen: dict[str, Any] = {}

    def fake_panel(technical, sentiment, macro, *, regime_hint=None, **kwargs):
        seen.update(technical=technical, macro=macro, regime_hint=regime_hint)
        return {"axis": "panel", "judge_summary": {"stronger_side": "neutral"}, "regime_summary": {}, "_meta": {"ok": True}}

    monkeypatch.setattr(panel_debate, "run_panel_debate", fake_panel)
    report = run_debate_graph({"signal": "BUY", "regime": {"regime": "RANGE"}}, {"score": 0.0}, {"macro_bias": "NEUTRAL"}, axis="panel")
    assert report["axis"] == "panel"
    assert seen["regime_hint"] == {"regime": "RANGE"} and seen["macro"] == {"macro_bias": "NEUTRAL"}


def test_run_debate_graph_panel_failure_is_hold_friendly(monkeypatch) -> None:
    monkeypatch.setattr(panel_debate, "run_panel_debate", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    report = run_debate_graph({"signal": "BUY"}, {"score": 0.0}, axis="panel")
    assert report["axis"] == "panel" and report["_meta"]["ok"] is False
    assert report["judge_summary"]["stronger_side"] == "neutral"


def test_gate_sends_directional_macro_with_neutral_technical_to_debate_without_confidence() -> None:
    decision = debate_graph.should_execute_debate(
        {"signal": "NEUTRAL", "trend": "RANGE"},
        {"gold_bias": "NEUTRAL", "score": 0.0},
        {"macro_bias": "BEARISH", "_meta": {"ok": True}},
    )
    assert decision["should_debate"] is True and "マクロが方向性" in decision["reason"]
    # An unreliable macro report does not trigger it; the range skip applies.
    skipped = debate_graph.should_execute_debate(
        {"signal": "NEUTRAL", "trend": "RANGE"},
        {"gold_bias": "NEUTRAL", "score": 0.0},
        {"macro_bias": "BEARISH", "_meta": {"ok": False}},
    )
    assert skipped["should_debate"] is False


def test_sentiment_direction_prefers_stated_bias_over_score() -> None:
    assert debate_graph._sentiment_direction({"gold_bias": "BEARISH", "score": 0.4}) == "BEARISH"
    assert debate_graph._sentiment_direction({"score": 0.4}) == "BULLISH"
    assert debate_graph._sentiment_direction({"gold_bias": "weird", "score": 0.0}) == "NEUTRAL"
