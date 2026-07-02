from __future__ import annotations

from unittest.mock import Mock, patch

from agents.base import LLMClient
from agents.debate import run_debate
from agents.sentiment import analyze_sentiment
from agents.technical import analyze_technical
from agents.trader import decide_trade


def test_technical_fallback_shape() -> None:
    result = analyze_technical({"rsi": 50})
    assert "signal" in result
    assert "_meta" in result


def test_sentiment_fallback_shape() -> None:
    result = analyze_sentiment([{"title": "gold steady"}])
    assert "score" in result
    assert "_meta" in result


def test_debate_fallback_shape() -> None:
    result = run_debate({"signal": "NEUTRAL"}, {"score": 0.0})
    assert "round1" in result
    assert "round2" in result
    assert "_meta" in result


def test_trader_enforces_hold_when_low_confidence() -> None:
    result = decide_trade(
        technical_report={"signal": "BUY"},
        sentiment_report={"score": 0.2},
        debate_report={"round1": {}, "round2": {}},
        confidence_threshold=0.6,
    )
    assert result["action"] in {"BUY", "SELL", "HOLD"}
    if result["confidence"] < 0.6:
        assert result["action"] == "HOLD"


def test_trader_forces_hold_when_sentiment_evidence_insufficient() -> None:
    result = decide_trade(
        technical_report={"signal": "BUY"},
        sentiment_report={"score": 0.0, "evidence_status": "INSUFFICIENT"},
        debate_report={"round1": {}, "round2": {}},
        confidence_threshold=0.1,
    )
    assert result["action"] == "HOLD"


def test_trader_clamps_confidence_to_upper_bound() -> None:
    fake_result = Mock()
    fake_result.ok = True
    fake_result.payload = {
        "action": "BUY",
        "symbol": "GOLD#",
        "confidence": 1.5,
        "reasoning": "test",
        "risk_level": "MID",
    }
    fake_result.model = "gpt-5.5"
    fake_result.error = ""
    fake_result.usage = Mock(prompt_tokens=1, completion_tokens=1, total_tokens=2)

    fake_client = Mock()
    fake_client.call_function.return_value = fake_result

    with patch("agents.trader.get_default_client", return_value=fake_client):
        result = decide_trade(
            technical_report={"signal": "BUY"},
            sentiment_report={"score": 0.2},
            debate_report={"round1": {}, "round2": {}},
            confidence_threshold=0.1,
        )

    assert result["confidence"] == 1.0


def test_trader_clamps_confidence_to_lower_bound() -> None:
    fake_result = Mock()
    fake_result.ok = True
    fake_result.payload = {
        "action": "SELL",
        "symbol": "GOLD#",
        "confidence": -0.3,
        "reasoning": "test",
        "risk_level": "LOW",
    }
    fake_result.model = "gpt-5.5"
    fake_result.error = ""
    fake_result.usage = Mock(prompt_tokens=1, completion_tokens=1, total_tokens=2)

    fake_client = Mock()
    fake_client.call_function.return_value = fake_result

    with patch("agents.trader.get_default_client", return_value=fake_client):
        result = decide_trade(
            technical_report={"signal": "SELL"},
            sentiment_report={"score": -0.2},
            debate_report={"round1": {}, "round2": {}},
            confidence_threshold=0.1,
        )

    assert result["confidence"] == 0.0
    assert result["action"] == "HOLD"


def test_trader_sets_risk_level_high_when_missing() -> None:
    fake_result = Mock()
    fake_result.ok = True
    fake_result.payload = {
        "action": "BUY",
        "symbol": "GOLD#",
        "confidence": 0.8,
        "reasoning": "test",
    }
    fake_result.model = "gpt-5.5"
    fake_result.error = ""
    fake_result.usage = Mock(prompt_tokens=1, completion_tokens=1, total_tokens=2)

    fake_client = Mock()
    fake_client.call_function.return_value = fake_result

    with patch("agents.trader.get_default_client", return_value=fake_client):
        result = decide_trade(
            technical_report={"signal": "BUY"},
            sentiment_report={"score": 0.3},
            debate_report={"round1": {}, "round2": {}},
            confidence_threshold=0.1,
        )

    assert result["risk_level"] == "HIGH"


def test_trader_sets_risk_level_high_when_invalid() -> None:
    fake_result = Mock()
    fake_result.ok = True
    fake_result.payload = {
        "action": "BUY",
        "symbol": "GOLD#",
        "confidence": 0.8,
        "reasoning": "test",
        "risk_level": "UNKNOWN",
    }
    fake_result.model = "gpt-5.5"
    fake_result.error = ""
    fake_result.usage = Mock(prompt_tokens=1, completion_tokens=1, total_tokens=2)

    fake_client = Mock()
    fake_client.call_function.return_value = fake_result

    with patch("agents.trader.get_default_client", return_value=fake_client):
        result = decide_trade(
            technical_report={"signal": "BUY"},
            sentiment_report={"score": 0.3},
            debate_report={"round1": {}, "round2": {}},
            confidence_threshold=0.1,
        )

    assert result["risk_level"] == "HIGH"


def test_llm_call_function_returns_not_ok_when_function_call_missing() -> None:
    fallback = {"action": "HOLD"}

    client = LLMClient(api_key="dummy")
    mocked_client = Mock()
    mocked_response = Mock()
    mocked_response.output = [
        type("Message", (), {"type": "message", "name": "", "arguments": ""})()
    ]
    mocked_response.usage = type(
        "Usage", (), {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
    )()
    mocked_client.responses.create.return_value = mocked_response
    client._client = mocked_client

    result = client.call_function(
        system_prompt="sys",
        user_prompt="usr",
        model="gpt-5.5",
        function_name="place_trade_order",
        function_schema={"description": "", "parameters": {}},
        fallback_payload=fallback,
    )

    assert not result.ok
    assert result.error == "function_call not found"
    assert result.payload == fallback


def test_llm_call_function_returns_ok_when_function_call_parsed() -> None:
    fallback = {"action": "HOLD"}

    client = LLMClient(api_key="dummy")
    mocked_client = Mock()
    mocked_response = Mock()
    mocked_response.output = [
        type(
            "FunctionCall",
            (),
            {
                "type": "function_call",
                "name": "place_trade_order",
                "arguments": '{"action": "BUY", "confidence": 0.8}',
            },
        )()
    ]
    mocked_response.usage = type(
        "Usage", (), {"input_tokens": 9, "output_tokens": 4, "total_tokens": 13}
    )()
    mocked_client.responses.create.return_value = mocked_response
    client._client = mocked_client

    result = client.call_function(
        system_prompt="sys",
        user_prompt="usr",
        model="gpt-5.5",
        function_name="place_trade_order",
        function_schema={"description": "", "parameters": {}},
        fallback_payload=fallback,
    )

    assert result.ok
    assert result.error == ""
    assert result.payload["action"] == "BUY"
    assert result.payload["confidence"] == 0.8


def test_llm_call_json_returns_not_ok_on_empty_output() -> None:
    fallback = {"signal": "NEUTRAL"}

    client = LLMClient(api_key="dummy")
    mocked_client = Mock()
    mocked_response = Mock()
    mocked_response.output_text = ""
    mocked_response.usage = type(
        "Usage", (), {"input_tokens": 3, "output_tokens": 0, "total_tokens": 3}
    )()
    mocked_client.responses.create.return_value = mocked_response
    client._client = mocked_client

    result = client.call_json(
        system_prompt="sys",
        user_prompt="usr",
        model="gpt-5.5-mini",
        fallback_payload=fallback,
    )

    assert not result.ok
    assert result.error == "empty output"
    assert result.payload == fallback


def test_llm_call_json_returns_not_ok_on_json_parse_error() -> None:
    fallback = {"signal": "NEUTRAL"}

    client = LLMClient(api_key="dummy")
    mocked_client = Mock()
    mocked_response = Mock()
    mocked_response.output_text = "{invalid json"
    mocked_response.usage = type(
        "Usage", (), {"input_tokens": 3, "output_tokens": 1, "total_tokens": 4}
    )()
    mocked_client.responses.create.return_value = mocked_response
    client._client = mocked_client

    result = client.call_json(
        system_prompt="sys",
        user_prompt="usr",
        model="gpt-5.5-mini",
        fallback_payload=fallback,
    )

    assert not result.ok
    assert result.error == "json parse error"
    assert result.payload == fallback
