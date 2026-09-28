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
    assert summary["consensus"] == "MAJORITY"
    assert summary["regime_confidence"] is None  # not a probability: see panel_agreement
    assert summary["panel_agreement"] == 0.6 and summary["panel_votes_available"] == 3
    assert summary["panel_vote_distribution"] == {"TREND_CONTINUATION": 2, "MEAN_REVERSION": 1, "UNCLEAR": 0}
    assert summary["panel_consensus_type"] == "MAJORITY"
    assert summary["key_levels"] == {"continuation_confirms": 4390.0, "reversal_confirms": 4340.5}
    assert summary["source"] == "judge" and "disagrees_with_rule" not in summary
    assert report["judge_summary"]["stronger_side"] == "bull"
    assert report["judge_summary"]["conflicts"] == ["マクロは反転寄り"]
    assert report["panel_views"]["sentiment"]["changed_view"] is True
    assert report["panel_views"]["technical"]["initial_view"] == "UNCLEAR"  # report had no regime_view
    assert [s["role"] for s in report["panel_transcript"]] == ["technical", "macro", "sentiment"]
    assert report["_meta"]["judge_status"] == "OK" and report["_meta"]["judge_attempts"] == 1
    assert report["_meta"]["debate_protocol_version"] == panel_debate.DEBATE_PROTOCOL_VERSION
    assert report["bull_arguments"] == [] and report["bull_confidence"] == 0.0

    # Round 1 is symmetric: every analyst sees the three reports and NO statements
    # from this round, so the speaking order cannot anchor anyone.
    technical_call, macro_call, sentiment_call, judge_call = client.calls
    assert technical_call["payload"]["your_report"] == technical
    assert set(technical_call["payload"]["other_reports"]) == {"macro", "sentiment"}
    assert technical_call["payload"]["transcript"] == []
    assert macro_call["payload"]["transcript"] == []
    assert sentiment_call["payload"]["transcript"] == []
    assert macro_call["payload"]["your_initial_view"] == "UNCLEAR"
    assert judge_call["payload"]["initial_views"] == {"technical": "UNCLEAR", "macro": "UNCLEAR", "sentiment": "UNCLEAR"}
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


def test_panel_chair_failure_retries_once_then_is_not_ok_for_trading() -> None:
    client = _ScriptedClient(
        {
            "technical": _view("TREND_CONTINUATION", "DOWN"),
            "macro": _view("TREND_CONTINUATION", "DOWN"),
            "sentiment": _view("UNCLEAR"),
        },
        _result({}, ok=False, error="judge timeout"),
    )
    report = panel_debate.run_panel_debate({}, {}, {}, client=client)
    # One retry with a note, then FAILED: the report is not ok, so main.py holds.
    chair_calls = [c for c in client.calls if "your_role" not in c["payload"]]
    assert len(chair_calls) == 2 and "retry_note" in chair_calls[1]["payload"]
    assert report["_meta"]["ok"] is False and report["_meta"]["judge_ok"] is False
    assert report["_meta"]["judge_status"] == "FAILED" and report["_meta"]["judge_attempts"] == 2
    assert report["_meta"]["judge_error"] == "judge timeout"
    summary = report["regime_summary"]
    assert summary["source"] == "vote_fallback_log_only"
    assert summary["regime"] == "TREND" and summary["direction_if_trend"] == "DOWN" and summary["entry_style"] == "LIMIT_PULLBACK"
    assert summary["consensus"] == "MAJORITY"
    assert report["judge_summary"]["stronger_side"] == "bear"
    assert "多数決で代替" in report["judge_summary"]["conflicts"][0]


def test_panel_chair_recovers_on_retry_and_rejects_trend_without_direction() -> None:
    answers = {"technical": _view("TREND_CONTINUATION", "UP"), "macro": _view("TREND_CONTINUATION", "UP"), "sentiment": _view("UNCLEAR")}
    bad_then_good = [_result({"regime": "TREND", "direction_if_trend": "NEUTRAL"}), _result(_judge("TREND", "UP", "STOP_BREAKOUT"))]

    class _Client(_ScriptedClient):
        def call_json(self, **kwargs):
            payload = json.loads(kwargs["user_prompt"])
            if "your_role" in payload:
                return super().call_json(**kwargs)
            self.calls.append({"system": kwargs["system_prompt"], "payload": payload})
            return bad_then_good.pop(0)

    client = _Client(answers, None)
    report = panel_debate.run_panel_debate({}, {}, {}, client=client)
    assert report["_meta"]["ok"] is True
    assert report["_meta"]["judge_status"] == "RECOVERED_RETRY" and report["_meta"]["judge_attempts"] == 2
    assert report["regime_summary"]["regime"] == "TREND" and report["regime_summary"]["entry_style"] == "STOP_BREAKOUT"
    assert report["regime_summary"]["source"] == "judge"


def test_panel_split_views_become_transition_without_forcing() -> None:
    client = _ScriptedClient(
        {"technical": _view("TREND_CONTINUATION", "UP"), "macro": _view("MEAN_REVERSION"), "sentiment": _view("UNCLEAR")},
        RuntimeError("judge exploded"),
    )
    report = panel_debate.run_panel_debate({}, {}, {}, client=client)
    assert report["regime_summary"]["regime"] == "TRANSITION"
    assert report["regime_summary"]["entry_style"] == "NONE"
    assert report["regime_summary"]["consensus"] == "SPLIT"
    assert report["regime_summary"]["panel_agreement"] == 0.4
    assert report["judge_summary"]["stronger_side"] == "neutral"
    assert report["_meta"]["ok"] is False  # chair exploded twice


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
    # Round 2 is symmetric too: everyone sees exactly the three round-1 statements.
    for call in client.calls[3:6]:
        assert [s["role"] for s in call["payload"]["transcript"]] == ["technical", "macro", "sentiment"]
        assert all(s["round"] == 1 for s in call["payload"]["transcript"])
    assert report["round_count"] == 2
    # changed_view is computed against the analyst's own report, not just the flag.
    report2 = panel_debate.run_panel_debate({"regime_view": "MEAN_REVERSION"}, {}, {}, client=_ScriptedClient(
        {"technical": _view("TREND_CONTINUATION", "UP"), "macro": _view("UNCLEAR"), "sentiment": _view("UNCLEAR")}, _judge("TRANSITION")))
    assert report2["panel_views"]["technical"]["initial_view"] == "MEAN_REVERSION"
    assert report2["panel_views"]["technical"]["changed_view"] is True


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
