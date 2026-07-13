from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import main
import agents.debate_graph as debate_graph
from agents.debate_graph import run_debate_graph
from agents.technical import analyze_technical
from unittest.mock import Mock, patch


def _fake_llm_result() -> Mock:
    result = Mock()
    result.ok = True
    result.payload = {
        "trend": "RANGE",
        "signal": "NEUTRAL",
        "key_levels": {},
        "reasoning": "LLM reasoning",
    }
    result.model = "gpt-5.4-mini"
    result.error = ""
    result.usage = Mock(prompt_tokens=1, completion_tokens=1, total_tokens=2)
    return result


def _patch_client() -> Mock:
    fake_client = Mock()
    fake_client.call_json.return_value = _fake_llm_result()
    return fake_client


def _frame_with_rsi(rsi_14: float) -> dict[str, float]:
    return {
        "close": 100.0,
        "rsi_14": rsi_14,
        "macd_hist": 0.4,
        "bb_upper": 101.0,
        "bb_mid": 98.0,
        "bb_lower": 95.0,
        "atr_14": 2.0,
        "recent_high_20": 100.5,
        "recent_low_20": 96.0,
    }


def test_confidence_changes_with_opposed_arguments() -> None:
    def fake_llm(role: str, state: dict[str, Any]) -> dict[str, Any]:
        if role == "bull":
            if int(state["round_count"]) == 0:
                return {
                    "argument": "Bearの流動性懸念は認識するが、RSI改善が先行。",
                    "confidence": 0.72,
                    "conceded_points": ["短期ボラ上昇"],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
                }
            return {
                "argument": "Bearの出来高指摘を受け、サイズは抑えるが上方向優位。",
                "confidence": 0.61,
                "conceded_points": ["出来高不足"],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
            }
        if role == "bear":
            return {
                "argument": "Bullの買い根拠はあるが、マクロ不確実性が残る。",
                "confidence": 0.78,
                "conceded_points": [],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
            }
        return {
            "judge_summary": {
                "agreements": ["ボラ上昇"],
                "conflicts": ["方向感"],
                "confidence_shift": {"bull": [0.72, 0.61], "bear": [0.78, 0.78]},
                "stronger_side": "bear",
            },
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    report = run_debate_graph(
        technical_report={"signal": "BUY"},
        sentiment_report={"score": 0.1},
        max_rounds=2,
        llm_override=fake_llm,
    )

    assert report["round_count"] == 2
    assert report["bull_confidence"] != 0.5
    assert report["bull_confidence"] != report["prev_bull_confidence"]


def test_early_convergence_when_confidence_delta_small() -> None:
    def fake_llm(role: str, state: dict[str, Any]) -> dict[str, Any]:
        if role == "bull":
            return {
                "argument": "新規材料なし。前提維持。",
                "confidence": 0.51,
                "conceded_points": [],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        if role == "bear":
            return {
                "argument": "反証追加なし。",
                "confidence": 0.52,
                "conceded_points": [],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        return {
            "judge_summary": {
                "agreements": ["材料不足"],
                "conflicts": ["方向感なし"],
                "confidence_shift": {"bull": [0.51, 0.51], "bear": [0.52, 0.52]},
                "stronger_side": "neutral",
            },
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    report = run_debate_graph(
        technical_report={"signal": "NEUTRAL"},
        sentiment_report={"score": 0.0},
        max_rounds=5,
        llm_override=fake_llm,
    )

    assert report["round_count"] == 2


def test_stops_at_max_rounds() -> None:
    def fake_llm(role: str, state: dict[str, Any]) -> dict[str, Any]:
        if role == "bull":
            base = 0.9 if int(state["round_count"]) % 2 == 0 else 0.6
            return {
                "argument": "方向主張を継続。",
                "confidence": base,
                "conceded_points": [],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        if role == "bear":
            return {
                "argument": "反対主張を継続。",
                "confidence": 0.8,
                "conceded_points": [],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        return {
            "judge_summary": {
                "agreements": [],
                "conflicts": ["継続"],
                "confidence_shift": {"bull": [0.9, 0.6, 0.9], "bear": [0.8, 0.8, 0.8]},
                "stronger_side": "neutral",
            },
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    report = run_debate_graph(
        technical_report={"signal": "BUY"},
        sentiment_report={"score": -0.1},
        max_rounds=3,
        llm_override=fake_llm,
    )

    assert report["round_count"] == 3


def test_judge_summary_contains_conflict_points() -> None:
    def fake_llm(role: str, state: dict[str, Any]) -> dict[str, Any]:
        if role in {"bull", "bear"}:
            return {
                "argument": f"{role} argument",
                "confidence": 0.7,
                "conceded_points": [],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        return {
            "judge_summary": {
                "agreements": ["ATR上昇"],
                "conflicts": ["エントリー方向"],
                "confidence_shift": {"bull": [0.7], "bear": [0.7]},
                "stronger_side": "neutral",
            },
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    report = run_debate_graph(
        technical_report={"signal": "BUY"},
        sentiment_report={"score": 0.2},
        max_rounds=2,
        llm_override=fake_llm,
    )

    assert isinstance(report["judge_summary"], dict)
    assert "conflicts" in report["judge_summary"]
    assert "エントリー方向" in report["judge_summary"]["conflicts"]


def test_temperatures_are_diversified_between_bull_and_bear(monkeypatch) -> None:
    temperatures: list[float] = []

    class _FakeResponse:
        def __init__(self, content: str) -> None:
            self.content = content
            self.usage_metadata = {
                "input_tokens": 2,
                "output_tokens": 1,
                "total_tokens": 3,
            }

    class _FakeChatOpenAI:
        def __init__(self, model: str, temperature: float) -> None:
            _ = model
            temperatures.append(temperature)

        def invoke(self, messages: list[Any]) -> _FakeResponse:
            _ = messages
            return _FakeResponse(
                '{"argument":"相手は強気だが過熱を見落とす","confidence":0.6,"conceded_points":[]}'
            )

    monkeypatch.setattr(debate_graph, "ChatOpenAI", _FakeChatOpenAI)

    state: debate_graph.DebateState = {
        "technical_report": {"signal": "BUY"},
        "sentiment_report": {"score": 0.1},
        "bull_arguments": [],
        "bear_arguments": [],
        "bull_conceded_points": [],
        "round_count": 0,
        "max_rounds": 3,
        "bull_confidence": 0.5,
        "bear_confidence": 0.5,
        "prev_bull_confidence": 0.5,
        "bull_confidence_history": [0.5],
        "bear_confidence_history": [0.5],
        "judge_summary": debate_graph._default_judge_summary(),
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

    debate_graph._invoke_role_llm("bull", state)
    debate_graph._invoke_role_llm("bear", state)

    assert debate_graph.BULL_TEMPERATURE in temperatures
    assert debate_graph.BEAR_TEMPERATURE in temperatures


def test_usage_is_accumulated_in_meta() -> None:
    def fake_llm(role: str, state: dict[str, Any]) -> dict[str, Any]:
        if role in {"bull", "bear"}:
            return {
                "argument": f"{role} argument",
                "confidence": 0.7,
                "conceded_points": [],
                "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
            }
        return {
            "judge_summary": {
                "agreements": [],
                "conflicts": ["方向"],
                "confidence_shift": {"bull": [0.7], "bear": [0.7]},
                "stronger_side": "neutral",
            },
            "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
        }

    report = run_debate_graph(
        technical_report={"signal": "BUY"},
        sentiment_report={"score": 0.2},
        max_rounds=2,
        llm_override=fake_llm,
    )

    usage = report["_meta"]["usage"]
    assert usage["prompt_tokens"] > 0
    assert usage["completion_tokens"] > 0
    assert usage["total_tokens"] > 0


def test_role_failure_sets_meta_flags_and_incomplete_marker() -> None:
    def fake_llm(role: str, state: dict[str, Any]) -> dict[str, Any]:
        if role == "bull":
            return {
                "argument": "bull側は分析に失敗したためHOLD寄り。",
                "confidence": 0.0,
                "conceded_points": [],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                "ok": False,
                "error": "json parse error",
            }
        if role == "bear":
            return {
                "argument": "bearは反論を継続",
                "confidence": 0.8,
                "conceded_points": [],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                "ok": True,
                "error": "",
            }
        return {
            "judge_summary": {
                "agreements": [],
                "conflicts": ["方向不一致"],
                "stronger_side": "bear",
            },
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            "ok": True,
            "error": "",
        }

    report = run_debate_graph(
        technical_report={"signal": "BUY"},
        sentiment_report={"score": 0.2},
        max_rounds=2,
        llm_override=fake_llm,
    )

    assert report["_meta"]["bull_ok"] is False
    assert report["_meta"]["bear_ok"] is True
    assert report["_meta"]["ok"] is False
    conflicts = report["judge_summary"]["conflicts"]
    assert any("議論不完全" in x for x in conflicts)


def test_bear_failure_is_recorded_in_meta_and_summary() -> None:
    def fake_llm(role: str, state: dict[str, Any]) -> dict[str, Any]:
        if role == "bull":
            return {
                "argument": "bullは主張を継続",
                "confidence": 0.7,
                "conceded_points": [],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                "ok": True,
                "error": "",
            }
        if role == "bear":
            return {
                "argument": "bear側は分析に失敗したためHOLD寄り。",
                "confidence": 0.0,
                "conceded_points": [],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                "ok": False,
                "error": "api error",
            }
        return {
            "judge_summary": {
                "agreements": [],
                "conflicts": ["方向不一致"],
                "stronger_side": "bear",
            },
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            "ok": True,
            "error": "",
        }

    report = run_debate_graph(
        technical_report={"signal": "SELL", "trend": "STRONG_DOWN"},
        sentiment_report={"score": -0.2},
        max_rounds=2,
        llm_override=fake_llm,
    )

    assert report["_meta"]["bear_ok"] is False
    assert any(not x for x in report["_meta"].get("bear_ok_history", []))
    conflicts = report["judge_summary"]["conflicts"]
    assert any("bear分析が" in x for x in conflicts)


def test_failed_zero_confidence_round_is_excluded_from_stronger_side() -> None:
    def fake_llm(role: str, state: dict[str, Any]) -> dict[str, Any]:
        if role == "bull":
            return {
                "argument": "bull argument",
                "confidence": 0.72,
                "conceded_points": [],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                "ok": True,
                "error": "",
            }
        if role == "bear":
            return {
                "argument": "bear fallback",
                "confidence": 0.0,
                "conceded_points": [],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                "ok": False,
                "error": "json parse error",
            }
        return {
            "judge_summary": {
                "agreements": [],
                "conflicts": ["方向"],
                "stronger_side": "bull",
            },
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            "ok": True,
            "error": "",
        }

    report = run_debate_graph(
        technical_report={"signal": "BUY", "trend": "UP"},
        sentiment_report={"score": 0.1},
        max_rounds=2,
        llm_override=fake_llm,
    )

    assert report["judge_summary"]["stronger_side"] == "neutral"


def test_bear_confidence_has_minimum_floor_when_ok() -> None:
    def fake_llm(role: str, state: dict[str, Any]) -> dict[str, Any]:
        if role == "bull":
            return {
                "argument": "bull argument",
                "confidence": 0.65,
                "conceded_points": [],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                "ok": True,
                "error": "",
            }
        if role == "bear":
            return {
                "argument": "相手は上昇継続と言うが、過熱リスクを見落としている。",
                "confidence": 0.1,
                "conceded_points": [],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                "ok": True,
                "error": "",
            }
        return {
            "judge_summary": {
                "agreements": [],
                "conflicts": ["方向"],
                "stronger_side": "neutral",
            },
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            "ok": True,
            "error": "",
        }

    report = run_debate_graph(
        technical_report={"signal": "BUY", "trend": "UP"},
        sentiment_report={"score": 0.2},
        max_rounds=2,
        llm_override=fake_llm,
    )

    assert report["bear_confidence"] >= 0.3


def test_failed_bear_round_keeps_previous_confidence_not_zero() -> None:
    def fake_llm(role: str, state: dict[str, Any]) -> dict[str, Any]:
        if role == "bull":
            return {
                "argument": "bull argument",
                "confidence": 0.66,
                "conceded_points": [],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                "ok": True,
                "error": "",
            }
        if role == "bear":
            return {
                "argument": "bear fallback",
                "confidence": 0.0,
                "conceded_points": [],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                "ok": False,
                "error": "json parse error",
            }
        return {
            "judge_summary": {
                "agreements": [],
                "conflicts": ["方向"],
                "stronger_side": "neutral",
            },
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            "ok": True,
            "error": "",
        }

    report = run_debate_graph(
        technical_report={"signal": "BUY", "trend": "UP"},
        sentiment_report={"score": 0.1},
        max_rounds=2,
        llm_override=fake_llm,
    )

    bear_hist = report["bear_confidence_history"]
    assert all(float(x) > 0.0 for x in bear_hist)


def test_role_prompts_require_named_weakness_structure() -> None:
    assert "相手は" in debate_graph.BULL_SYSTEM_PROMPT
    assert "見落としている" in debate_graph.BULL_SYSTEM_PROMPT
    assert "相手は" in debate_graph.BEAR_SYSTEM_PROMPT
    assert "見落としている" in debate_graph.BEAR_SYSTEM_PROMPT
    assert "confidenceは必ず0.3以上" in debate_graph.BEAR_SYSTEM_PROMPT


def _base_debate_state_with_levels(horizontal_levels: dict[str, Any]) -> debate_graph.DebateState:
    return {
        "technical_report": {
            "signal": "BUY",
            "trend": "UP",
            "alignment": "ALIGNED",
            "d1_trend": "UP",
            "execution_trend": "UP",
            "key_levels": {
                "d1": {"snapshot": {"close": 2300.0}},
                "frames": {"h1": {"close": 2300.0}},
                "horizontal_levels": horizontal_levels,
            },
        },
        "sentiment_report": {"score": 0.1},
        "macro_report": {"macro_bias": "NEUTRAL"},
        "bull_arguments": [],
        "bear_arguments": [],
        "bull_conceded_points": [],
        "round_count": 0,
        "max_rounds": 3,
        "bull_confidence": 0.5,
        "bear_confidence": 0.5,
        "prev_bull_confidence": 0.5,
        "bull_confidence_history": [0.5],
        "bear_confidence_history": [0.5],
        "bull_ok_history": [],
        "bear_ok_history": [],
        "judge_summary": debate_graph._default_judge_summary(),
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


def test_role_payload_includes_horizontal_levels_for_bull_and_bear(monkeypatch) -> None:
    captured: dict[str, dict[str, Any]] = {}

    class _FakeResponse:
        def __init__(self) -> None:
            self.content = '{"argument":"ok","confidence":0.6,"conceded_points":[]}'
            self.usage_metadata = {
                "input_tokens": 1,
                "output_tokens": 1,
                "total_tokens": 2,
            }

    class _FakeChatOpenAI:
        def __init__(self, model: str, temperature: float) -> None:
            _ = model
            _ = temperature

        def invoke(self, messages: list[Any]) -> _FakeResponse:
            payload = json.loads(str(getattr(messages[1], "content", "{}") or "{}"))
            role = "bull" if "latest_bear_argument" in payload else "bear"
            captured[role] = payload
            return _FakeResponse()

    monkeypatch.setattr(debate_graph, "ChatOpenAI", _FakeChatOpenAI)

    horizontal_levels = {
        "resistances": [
            {"price": 2310.0, "score": 4.0, "source": "cluster", "timeframe": "H4", "touch_count": 3}
        ],
        "supports": [
            {"price": 2290.0, "score": 3.0, "source": "swing", "timeframe": "D1", "touch_count": 2}
        ],
    }
    state = _base_debate_state_with_levels(horizontal_levels)

    debate_graph._invoke_role_llm("bull", state)
    debate_graph._invoke_role_llm("bear", state)

    assert "horizontal_levels_context" in captured["bull"]
    assert "horizontal_levels_context" in captured["bear"]
    assert captured["bull"]["horizontal_levels_context"] == captured["bear"]["horizontal_levels_context"]
    assert captured["bull"]["horizontal_levels_context"]["nearest_resistances"][0]["price"] == 2310.0
    assert captured["bull"]["horizontal_levels_context"]["nearest_supports"][0]["price"] == 2290.0


def test_role_payload_fallback_when_horizontal_levels_empty(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    class _FakeResponse:
        def __init__(self) -> None:
            self.content = '{"argument":"ok","confidence":0.6,"conceded_points":[]}'
            self.usage_metadata = {
                "input_tokens": 1,
                "output_tokens": 1,
                "total_tokens": 2,
            }

    class _FakeChatOpenAI:
        def __init__(self, model: str, temperature: float) -> None:
            _ = model
            _ = temperature

        def invoke(self, messages: list[Any]) -> _FakeResponse:
            payload = json.loads(str(getattr(messages[1], "content", "{}") or "{}"))
            captured.update(payload)
            return _FakeResponse()

    monkeypatch.setattr(debate_graph, "ChatOpenAI", _FakeChatOpenAI)

    state = _base_debate_state_with_levels({"resistances": [], "supports": []})
    _ = debate_graph._invoke_role_llm("bull", state)

    assert "horizontal_levels_context" in captured
    assert captured["horizontal_levels_context"]["available"] is False
    assert captured["horizontal_levels_context"]["nearest_resistances"] == []
    assert captured["horizontal_levels_context"]["nearest_supports"] == []


def test_confidence_shift_uses_state_history_not_judge_generated_values() -> None:
    def fake_llm(role: str, state: dict[str, Any]) -> dict[str, Any]:
        if role == "bull":
            next_value = 0.61 if int(state["round_count"]) == 0 else 0.74
            return {
                "argument": "bull argument",
                "confidence": next_value,
                "conceded_points": [],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        if role == "bear":
            return {
                "argument": "bear argument",
                "confidence": 0.58,
                "conceded_points": [],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        return {
            "judge_summary": {
                "agreements": [],
                "conflicts": ["方向"],
                "confidence_shift": {"bull": [1.0, 1.0], "bear": [1.0, 1.0]},
                "stronger_side": "bear",
            },
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    report = run_debate_graph(
        technical_report={"signal": "BUY"},
        sentiment_report={"score": -0.1},
        max_rounds=2,
        llm_override=fake_llm,
    )

    shift = report["judge_summary"]["confidence_shift"]
    assert shift["bull"] != [1.0, 1.0]
    assert shift["bull"] == report["bull_confidence_history"]
    assert shift["bear"] == report["bear_confidence_history"]


def test_stronger_side_is_bull_when_bull_confidence_is_higher() -> None:
    def fake_llm(role: str, state: dict[str, Any]) -> dict[str, Any]:
        if role == "bull":
            return {
                "argument": "bull argument",
                "confidence": 0.82,
                "conceded_points": [],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        if role == "bear":
            return {
                "argument": "bear argument",
                "confidence": 0.55,
                "conceded_points": [],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        return {
            "judge_summary": {
                "agreements": [],
                "conflicts": ["方向"],
                "stronger_side": "bear",
            },
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    report = run_debate_graph(
        technical_report={"signal": "BUY"},
        sentiment_report={"score": 0.0},
        max_rounds=2,
        llm_override=fake_llm,
    )

    assert report["bull_confidence"] > report["bear_confidence"]
    assert report["judge_summary"]["stronger_side"] == "bull"


def test_stronger_side_becomes_neutral_when_confidence_gap_is_not_clear() -> None:
    def fake_llm(role: str, state: dict[str, Any]) -> dict[str, Any]:
        if role == "bull":
            return {
                "argument": "bull argument",
                "confidence": 0.64,
                "conceded_points": [],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        if role == "bear":
            return {
                "argument": "bear argument",
                "confidence": 0.60,
                "conceded_points": [],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        return {
            "judge_summary": {
                "agreements": [],
                "conflicts": ["方向"],
                "stronger_side": "bull",
            },
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    report = run_debate_graph(
        technical_report={"signal": "BUY"},
        sentiment_report={"score": 0.0},
        max_rounds=2,
        llm_override=fake_llm,
    )

    assert abs((report["bull_confidence"] - report["bear_confidence"]) - 0.04) < 1e-9
    assert report["judge_summary"]["stronger_side"] == "neutral"


def test_run_once_falls_back_to_hold_when_debate_graph_fails(tmp_path: Path, monkeypatch) -> None:
    log_path = tmp_path / "trade_log.csv"
    monkeypatch.setattr(main, "LOG_DIR", tmp_path)
    monkeypatch.setattr(main, "TRADE_LOG_PATH", log_path)

    monkeypatch.setattr(main, "sync_closed_trades", lambda: 0)
    monkeypatch.setattr(main, "is_high_impact_soon", lambda minutes: False)
    monkeypatch.setattr(main, "get_positions", lambda symbol: [])

    def _dummy_df() -> object:
        import pandas as pd

        return pd.DataFrame(
            [
                {
                    "close": 2300.0,
                    "rsi_14": 55.0,
                    "macd": 1.0,
                    "macd_signal": 0.8,
                    "macd_hist": 0.2,
                    "bb_upper": 2310.0,
                    "bb_mid": 2300.0,
                    "bb_lower": 2290.0,
                    "atr_14": 10.0,
                    "recent_high_20": 2320.0,
                    "recent_low_20": 2280.0,
                }
            ]
        )

    monkeypatch.setattr(main, "get_rates", lambda symbol, tf, count: _dummy_df())
    monkeypatch.setattr(main, "add_indicators", lambda df: df)
    monkeypatch.setattr(main, "fetch_news", lambda hours=24: [])
    monkeypatch.setattr(
        main,
        "analyze_technical",
        lambda payload: {
            "signal": "BUY",
            "_meta": {"usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}},
        },
    )
    monkeypatch.setattr(
        main,
        "analyze_sentiment",
        lambda items: {
            "score": 0.1,
            "_meta": {"usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}},
        },
    )
    monkeypatch.setattr(
        main,
        "run_debate_graph",
        lambda t, s, m=None: {
            "judge_summary": {
                "agreements": [],
                "conflicts": ["engine down"],
                "confidence_shift": {"bull": [], "bear": []},
                "stronger_side": "neutral",
            },
            "_meta": {"ok": False, "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}},
        },
    )

    class _Account:
        balance = 500_000

    monkeypatch.setattr(main, "get_account_info", lambda: {"success": True, "data": _Account()})
    monkeypatch.setattr(
        main,
        "build_risk_plan",
        lambda action, entry_price, atr, balance_jpy: {
            "ok": True,
            "action": "BUY",
            "lot": 0.1,
            "sl": 2285.0,
            "tp": 2330.0,
        },
    )
    monkeypatch.setattr(main, "get_spread", lambda symbol: 10.0)
    monkeypatch.setattr(
        main,
        "send_order",
        lambda symbol, action, lot, sl, tp: {"success": True, "retcode": 0},
    )

    result = main.run_once(baseline_spread=10.0)

    assert result["action"] == "HOLD"
    assert "議論エンジン失敗" in result["reasoning"]


def test_gate_executes_debate_on_technical_sentiment_conflict() -> None:
    decision = debate_graph.should_execute_debate(
        technical_report={"signal": "BUY", "trend": "UP", "rsi_14": 62.0},
        sentiment_report={"score": -0.4, "sentiment": "BEARISH"},
    )
    assert decision["should_debate"] is True
    assert "矛盾" in decision["reason"]


def test_gate_executes_debate_on_macro_conflict() -> None:
    decision = debate_graph.should_execute_debate(
        technical_report={"signal": "BUY", "trend": "UP", "rsi_14": 52.0, "alignment": "ALIGNED"},
        sentiment_report={"score": 0.2, "sentiment": "BULLISH"},
        macro_report={"macro_bias": "BEARISH", "confidence": 0.7},
    )
    assert decision["should_debate"] is True
    assert "technicalとmacroが矛盾" in decision["reason"]


def test_gate_executes_debate_on_divergent_alignment() -> None:
    decision = debate_graph.should_execute_debate(
        technical_report={"signal": "BUY", "trend": "UP", "alignment": "DIVERGENT", "rsi_14": 52.0},
        sentiment_report={"score": 0.2, "sentiment": "BULLISH"},
        macro_report={"macro_bias": "NEUTRAL", "confidence": 0.5},
    )
    assert decision["should_debate"] is True
    assert "DIVERGENT" in decision["reason"]


def test_gate_uses_multitimeframe_top_level_rsi_for_overheated_buy() -> None:
    with patch("agents.technical.get_default_client", return_value=_patch_client()):
        technical_report = analyze_technical(
            {
                "d1": _frame_with_rsi(68.0),
                "h4": _frame_with_rsi(66.0),
                "h1": _frame_with_rsi(76.0),
            }
        )

    technical_report["trend"] = "STRONG_UP"
    decision = debate_graph.should_execute_debate(
        technical_report=technical_report,
        sentiment_report={"score": 0.2, "sentiment": "BULLISH"},
    )

    assert technical_report["rsi_14"] == 76.0
    assert decision["should_debate"] is True
    assert "強トレンドかつ過熱" in decision["reason"]


def test_gate_uses_multitimeframe_top_level_rsi_for_overheated_sell() -> None:
    with patch("agents.technical.get_default_client", return_value=_patch_client()):
        technical_report = analyze_technical(
            {
                "d1": _frame_with_rsi(34.0),
                "h4": _frame_with_rsi(32.0),
                "h1": _frame_with_rsi(24.0),
            }
        )

    technical_report["trend"] = "STRONG_DOWN"
    technical_report["signal"] = "SELL"
    decision = debate_graph.should_execute_debate(
        technical_report=technical_report,
        sentiment_report={"score": -0.2, "sentiment": "BEARISH"},
    )

    assert technical_report["rsi_14"] == 24.0
    assert decision["should_debate"] is True
    assert "強トレンドかつ過熱" in decision["reason"]


def test_gate_uses_multitimeframe_top_level_rsi_for_non_overheated_case() -> None:
    with patch("agents.technical.get_default_client", return_value=_patch_client()):
        technical_report = analyze_technical(
            {
                "d1": _frame_with_rsi(63.0),
                "h4": _frame_with_rsi(61.0),
                "h1": _frame_with_rsi(55.0),
            }
        )

    technical_report["trend"] = "STRONG_UP"
    decision = debate_graph.should_execute_debate(
        technical_report=technical_report,
        sentiment_report={"score": 0.2, "sentiment": "BULLISH"},
    )

    assert technical_report["rsi_14"] == 55.0
    assert decision["should_debate"] is False
    assert "明確なトレンドのため" in decision["reason"]


def test_debate_llm_input_includes_macro_and_multitimeframe() -> None:
    captured: dict[str, dict[str, Any]] = {}

    def fake_llm(role: str, state: dict[str, Any]) -> dict[str, Any]:
        captured[role] = {
            "macro_report": state.get("macro_report", {}),
            "alignment": state.get("technical_report", {}).get("alignment", ""),
            "d1_trend": state.get("technical_report", {}).get("d1_trend", ""),
            "execution_trend": state.get("technical_report", {}).get("execution_trend", ""),
        }
        if role == "bull":
            return {
                "argument": "bull argument",
                "confidence": 0.7,
                "conceded_points": [],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        if role == "bear":
            return {
                "argument": "bear argument",
                "confidence": 0.7,
                "conceded_points": [],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        return {
            "judge_summary": {
                "agreements": [],
                "conflicts": [],
                "stronger_side": "neutral",
            },
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    report = run_debate_graph(
        technical_report={"signal": "BUY", "trend": "UP", "alignment": "ALIGNED", "d1_trend": "UP", "execution_trend": "UP"},
        sentiment_report={"score": 0.1},
        macro_report={"macro_bias": "BULLISH", "confidence": 0.6},
        max_rounds=2,
        llm_override=fake_llm,
    )

    assert report["_meta"]["ok"] is True
    assert captured["bull"]["macro_report"]["macro_bias"] == "BULLISH"
    assert captured["bear"]["macro_report"]["macro_bias"] == "BULLISH"
    assert captured["bull"]["alignment"] == "ALIGNED"
    assert captured["bull"]["d1_trend"] == "UP"
    assert captured["bull"]["execution_trend"] == "UP"


def test_debate_runs_safely_when_macro_and_alignment_missing() -> None:
    def fake_llm(role: str, state: dict[str, Any]) -> dict[str, Any]:
        if role == "bull":
            return {
                "argument": "bull argument",
                "confidence": 0.7,
                "conceded_points": [],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        if role == "bear":
            return {
                "argument": "bear argument",
                "confidence": 0.7,
                "conceded_points": [],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        return {
            "judge_summary": {
                "agreements": [],
                "conflicts": [],
                "stronger_side": "neutral",
            },
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    report = run_debate_graph(
        technical_report={"signal": "BUY", "trend": "UP"},
        sentiment_report={"score": 0.1},
        max_rounds=2,
        llm_override=fake_llm,
    )

    assert report["_meta"]["ok"] is True
    assert report["judge_summary"]["stronger_side"] in {"bull", "bear", "neutral"}


def test_gate_skips_debate_on_strong_aligned_trend() -> None:
    decision = debate_graph.should_execute_debate(
        technical_report={"signal": "SELL", "trend": "STRONG_DOWN", "rsi_14": 45.0},
        sentiment_report={"score": -0.6, "sentiment": "BEARISH"},
    )
    assert decision["should_debate"] is False
    assert "明確なトレンド" in decision["reason"]


def test_skipped_debate_report_has_skip_marker() -> None:
    report = debate_graph.build_skipped_debate_report("議論スキップ（レンジのため）")
    assert report["_meta"]["debate_executed"] is False
    assert "議論スキップ" in report["judge_summary"]["conflicts"][0]


def test_judge_malformed_json_recovers_conflicts_and_agreements(monkeypatch) -> None:
    class _FakeResponse:
        def __init__(self, content: str) -> None:
            self.content = content
            self.usage_metadata = {
                "input_tokens": 11,
                "output_tokens": 9,
                "total_tokens": 20,
            }

    class _FakeChatOpenAI:
        def __init__(self, model: str, temperature: float) -> None:
            _ = model, temperature

        def invoke(self, messages: list[Any]) -> _FakeResponse:
            _ = messages
            return _FakeResponse(
                '{"agreements":["ボラ拡大"],"conflicts":["方向感の対立"] "stronger_side":"bear"}'
            )

    monkeypatch.setattr(debate_graph, "ChatOpenAI", _FakeChatOpenAI)
    monkeypatch.setattr(debate_graph.time, "sleep", lambda _: None)

    state: debate_graph.DebateState = {
        "technical_report": {"signal": "BUY", "trend": "UP"},
        "sentiment_report": {"score": 0.1},
        "macro_report": {},
        "bull_arguments": ["bull argument"],
        "bear_arguments": ["bear argument"],
        "bull_conceded_points": ["短期ノイズ"],
        "round_count": 1,
        "max_rounds": 3,
        "bull_confidence": 0.72,
        "bear_confidence": 0.64,
        "prev_bull_confidence": 0.5,
        "bull_confidence_history": [0.5, 0.72],
        "bear_confidence_history": [0.5, 0.64],
        "bull_ok_history": [True],
        "bear_ok_history": [True],
        "judge_summary": debate_graph._default_judge_summary(),
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

    judge = debate_graph._invoke_judge_llm(state)

    assert judge["ok"] is True
    assert "方向感の対立" in judge["judge_summary"]["conflicts"]
    assert "ボラ拡大" in judge["judge_summary"]["agreements"]


def test_judge_total_parse_failure_falls_back_to_state_summary(monkeypatch) -> None:
    class _FakeResponse:
        def __init__(self, content: str) -> None:
            self.content = content
            self.usage_metadata = {
                "input_tokens": 8,
                "output_tokens": 4,
                "total_tokens": 12,
            }

    class _FakeChatOpenAI:
        def __init__(self, model: str, temperature: float) -> None:
            _ = model, temperature

        def invoke(self, messages: list[Any]) -> _FakeResponse:
            _ = messages
            return _FakeResponse("judge says maybe but not json format")

    monkeypatch.setattr(debate_graph, "ChatOpenAI", _FakeChatOpenAI)
    monkeypatch.setattr(debate_graph.time, "sleep", lambda _: None)

    state: debate_graph.DebateState = {
        "technical_report": {"signal": "BUY", "trend": "UP"},
        "sentiment_report": {"score": 0.0},
        "macro_report": {},
        "bull_arguments": ["Bullは上昇継続を主張"],
        "bear_arguments": ["Bearは過熱反落を主張"],
        "bull_conceded_points": ["ATR上昇"],
        "round_count": 1,
        "max_rounds": 3,
        "bull_confidence": 0.71,
        "bear_confidence": 0.67,
        "prev_bull_confidence": 0.5,
        "bull_confidence_history": [0.5, 0.71],
        "bear_confidence_history": [0.5, 0.67],
        "bull_ok_history": [True],
        "bear_ok_history": [True],
        "judge_summary": debate_graph._default_judge_summary(),
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

    judge = debate_graph._invoke_judge_llm(state)

    assert judge["ok"] is False
    assert any("bull:" in item for item in judge["judge_summary"]["conflicts"])
    assert "ATR上昇" in judge["judge_summary"]["agreements"]


def test_judge_valid_json_still_processed_normally(monkeypatch) -> None:
    class _FakeResponse:
        def __init__(self, content: str) -> None:
            self.content = content
            self.usage_metadata = {
                "input_tokens": 6,
                "output_tokens": 6,
                "total_tokens": 12,
            }

    class _FakeChatOpenAI:
        def __init__(self, model: str, temperature: float) -> None:
            _ = model, temperature

        def invoke(self, messages: list[Any]) -> _FakeResponse:
            _ = messages
            return _FakeResponse(
                '{"agreements":["ボラ高は共通認識"],"conflicts":["エントリー方向"],"stronger_side":"neutral"}'
            )

    monkeypatch.setattr(debate_graph, "ChatOpenAI", _FakeChatOpenAI)

    state: debate_graph.DebateState = {
        "technical_report": {"signal": "SELL", "trend": "DOWN"},
        "sentiment_report": {"score": -0.2},
        "macro_report": {},
        "bull_arguments": ["bull argument"],
        "bear_arguments": ["bear argument"],
        "bull_conceded_points": [],
        "round_count": 1,
        "max_rounds": 3,
        "bull_confidence": 0.62,
        "bear_confidence": 0.68,
        "prev_bull_confidence": 0.5,
        "bull_confidence_history": [0.5, 0.62],
        "bear_confidence_history": [0.5, 0.68],
        "bull_ok_history": [True],
        "bear_ok_history": [True],
        "judge_summary": debate_graph._default_judge_summary(),
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

    judge = debate_graph._invoke_judge_llm(state)

    assert judge["ok"] is True
    assert judge["error"] == ""
    assert "エントリー方向" in judge["judge_summary"]["conflicts"]


def test_judge_parse_recovered_sets_ok_and_error_marker(monkeypatch) -> None:
    class _FakeResponse:
        def __init__(self, content: str) -> None:
            self.content = content
            self.usage_metadata = {
                "input_tokens": 7,
                "output_tokens": 5,
                "total_tokens": 12,
            }

    class _FakeChatOpenAI:
        def __init__(self, model: str, temperature: float) -> None:
            _ = model, temperature

        def invoke(self, messages: list[Any]) -> _FakeResponse:
            _ = messages
            # missing comma between agreements and conflicts
            return _FakeResponse('{"agreements":["a"] "conflicts":["b"],"stronger_side":"bear"}')

    monkeypatch.setattr(debate_graph, "ChatOpenAI", _FakeChatOpenAI)
    monkeypatch.setattr(debate_graph.time, "sleep", lambda _: None)

    state: debate_graph.DebateState = {
        "technical_report": {"signal": "BUY", "trend": "UP"},
        "sentiment_report": {"score": 0.0},
        "macro_report": {},
        "bull_arguments": ["bull argument"],
        "bear_arguments": ["bear argument"],
        "bull_conceded_points": [],
        "round_count": 1,
        "max_rounds": 3,
        "bull_confidence": 0.62,
        "bear_confidence": 0.58,
        "prev_bull_confidence": 0.5,
        "bull_confidence_history": [0.5, 0.62],
        "bear_confidence_history": [0.5, 0.58],
        "bull_ok_history": [True],
        "bear_ok_history": [True],
        "judge_summary": debate_graph._default_judge_summary(),
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

    judge = debate_graph._invoke_judge_llm(state)

    assert judge["ok"] is True
    assert judge["error"] == "json parse recovered"
    assert judge["judge_summary"]["conflicts"] == ["b"]


def test_judge_total_failure_keeps_algorithmic_stronger_side_in_graph() -> None:
    def fake_llm(role: str, state: dict[str, Any]) -> dict[str, Any]:
        if role == "bull":
            return {
                "argument": "bull argument",
                "confidence": 0.82,
                "conceded_points": [],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                "ok": True,
                "error": "",
            }
        if role == "bear":
            return {
                "argument": "bear argument",
                "confidence": 0.52,
                "conceded_points": [],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                "ok": True,
                "error": "",
            }
        return {
            "judge_summary": {
                "agreements": [],
                "conflicts": ["judge json parse error"],
                "stronger_side": "neutral",
            },
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            "ok": False,
            "error": "json parse error",
        }

    report = run_debate_graph(
        technical_report={"signal": "BUY", "trend": "UP"},
        sentiment_report={"score": 0.1},
        max_rounds=2,
        llm_override=fake_llm,
    )

    assert report["_meta"]["judge_ok"] is False
    assert report["_meta"]["judge_error"] == "json parse error"
    assert report["judge_summary"]["stronger_side"] == "bull"


def test_judge_parse_failure_logs_raw_focus(monkeypatch, caplog) -> None:
    class _FakeResponse:
        def __init__(self, content: str) -> None:
            self.content = content
            self.usage_metadata = {
                "input_tokens": 5,
                "output_tokens": 4,
                "total_tokens": 9,
            }

    class _FakeChatOpenAI:
        def __init__(self, model: str, temperature: float) -> None:
            _ = model, temperature

        def invoke(self, messages: list[Any]) -> _FakeResponse:
            _ = messages
            return _FakeResponse("noise only, not json")

    monkeypatch.setattr(debate_graph, "ChatOpenAI", _FakeChatOpenAI)
    monkeypatch.setattr(debate_graph.time, "sleep", lambda _: None)

    state: debate_graph.DebateState = {
        "technical_report": {"signal": "BUY", "trend": "UP"},
        "sentiment_report": {"score": 0.0},
        "macro_report": {},
        "bull_arguments": ["bull argument"],
        "bear_arguments": ["bear argument"],
        "bull_conceded_points": [],
        "round_count": 1,
        "max_rounds": 3,
        "bull_confidence": 0.62,
        "bear_confidence": 0.59,
        "prev_bull_confidence": 0.5,
        "bull_confidence_history": [0.5, 0.62],
        "bear_confidence_history": [0.5, 0.59],
        "bull_ok_history": [True],
        "bear_ok_history": [True],
        "judge_summary": debate_graph._default_judge_summary(),
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

    with caplog.at_level("WARNING"):
        judge = debate_graph._invoke_judge_llm(state)

    assert judge["ok"] is False
    assert "json parse error" in judge["error"]
    assert any("raw_focus=" in rec.message for rec in caplog.records)
