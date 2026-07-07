from __future__ import annotations

from unittest.mock import Mock, patch

from agents.evaluate_position import evaluate_position


def _fake_result(action: str, confidence: float, reasoning: str = "test", risk_level: str = "MID") -> Mock:
    result = Mock()
    result.ok = True
    result.payload = {
        "action": action,
        "symbol": "GOLD#",
        "confidence": confidence,
        "reasoning": reasoning,
        "risk_level": risk_level,
    }
    result.model = "gpt-5.5"
    result.error = ""
    result.usage = Mock(prompt_tokens=1, completion_tokens=1, total_tokens=2)
    return result


def test_evaluate_position_closes_on_high_confidence_reverse_signal() -> None:
    fake_client = Mock()
    fake_client.call_function.return_value = _fake_result("CLOSE", 0.84)

    with patch("agents.evaluate_position.get_default_client", return_value=fake_client):
        result = evaluate_position(
            position_context={"symbol": "GOLD#", "type": "BUY", "price_open": 2300.0, "profit": -50.0},
            technical_report={"signal": "SELL", "trend": "DOWN"},
            sentiment_report={"score": -0.2},
            debate_report={"judge_summary": {"stronger_side": "bear"}},
            confidence_threshold=0.7,
        )

    assert result["action"] == "CLOSE"


def test_evaluate_position_holds_on_low_confidence_reverse_signal() -> None:
    fake_client = Mock()
    fake_client.call_function.return_value = _fake_result("CLOSE", 0.45)

    with patch("agents.evaluate_position.get_default_client", return_value=fake_client):
        result = evaluate_position(
            position_context={"symbol": "GOLD#", "type": "BUY", "price_open": 2300.0, "profit": -10.0},
            technical_report={"signal": "SELL", "trend": "DOWN"},
            sentiment_report={"score": -0.1},
            debate_report={"judge_summary": {"stronger_side": "bear"}},
            confidence_threshold=0.7,
        )

    assert result["action"] == "HOLD"


def test_evaluate_position_holds_on_same_direction() -> None:
    fake_client = Mock()
    fake_client.call_function.return_value = _fake_result("HOLD", 0.72)

    with patch("agents.evaluate_position.get_default_client", return_value=fake_client):
        result = evaluate_position(
            position_context={"symbol": "GOLD#", "type": "BUY", "price_open": 2300.0, "profit": 30.0},
            technical_report={"signal": "BUY", "trend": "UP"},
            sentiment_report={"score": 0.2},
            debate_report={"judge_summary": {"stronger_side": "bull"}},
            confidence_threshold=0.7,
        )

    assert result["action"] == "HOLD"


def test_evaluate_position_holds_on_neutral_output() -> None:
    fake_client = Mock()
    fake_client.call_function.return_value = _fake_result("HOLD", 0.65)

    with patch("agents.evaluate_position.get_default_client", return_value=fake_client):
        result = evaluate_position(
            position_context={"symbol": "GOLD#", "type": "SELL", "price_open": 2300.0, "profit": 0.0},
            technical_report={"signal": "NEUTRAL", "trend": "RANGE"},
            sentiment_report={"score": 0.0},
            debate_report={"judge_summary": {"stronger_side": "neutral"}},
            confidence_threshold=0.7,
        )

    assert result["action"] == "HOLD"


def test_evaluate_position_falls_back_to_hold_on_invalid_action() -> None:
    fake_client = Mock()
    fake_client.call_function.return_value = _fake_result("BUY", 0.9)

    with patch("agents.evaluate_position.get_default_client", return_value=fake_client):
        result = evaluate_position(
            position_context={"symbol": "GOLD#", "type": "BUY", "price_open": 2300.0, "profit": 0.0},
            technical_report={"signal": "SELL", "trend": "DOWN"},
            sentiment_report={"score": -0.2},
            debate_report={"judge_summary": {"stronger_side": "bear"}},
            confidence_threshold=0.7,
        )

    assert result["action"] == "HOLD"


def test_evaluate_position_falls_back_to_hold_on_client_failure() -> None:
    fake_result = Mock()
    fake_result.ok = False
    fake_result.payload = {
        "action": "HOLD",
        "symbol": "GOLD#",
        "confidence": 0.0,
        "reasoning": "保有評価に失敗したためHOLD。",
        "risk_level": "HIGH",
    }
    fake_result.model = "gpt-5.5"
    fake_result.error = "function_call not found"
    fake_result.usage = Mock(prompt_tokens=0, completion_tokens=0, total_tokens=0)

    fake_client = Mock()
    fake_client.call_function.return_value = fake_result

    with patch("agents.evaluate_position.get_default_client", return_value=fake_client):
        result = evaluate_position(
            position_context={"symbol": "GOLD#", "type": "BUY", "price_open": 2300.0, "profit": 0.0},
            technical_report={"signal": "SELL", "trend": "DOWN"},
            sentiment_report={"score": -0.2},
            debate_report={"judge_summary": {"stronger_side": "bear"}},
            confidence_threshold=0.7,
        )

    assert result["action"] == "HOLD"


def test_evaluate_position_closes_at_close_threshold_boundary() -> None:
    fake_client = Mock()
    fake_client.call_function.return_value = _fake_result("CLOSE", 0.7)

    with patch("agents.evaluate_position.get_default_client", return_value=fake_client):
        result = evaluate_position(
            position_context={"symbol": "GOLD#", "type": "BUY", "price_open": 2300.0, "profit": -20.0},
            technical_report={"signal": "SELL", "trend": "DOWN"},
            sentiment_report={"score": -0.2},
            debate_report={"judge_summary": {"stronger_side": "bear"}},
            confidence_threshold=0.7,
        )

    assert result["action"] == "CLOSE"


def test_evaluate_position_holds_just_below_close_threshold() -> None:
    fake_client = Mock()
    fake_client.call_function.return_value = _fake_result("CLOSE", 0.69)

    with patch("agents.evaluate_position.get_default_client", return_value=fake_client):
        result = evaluate_position(
            position_context={"symbol": "GOLD#", "type": "BUY", "price_open": 2300.0, "profit": -15.0},
            technical_report={"signal": "SELL", "trend": "DOWN"},
            sentiment_report={"score": -0.1},
            debate_report={"judge_summary": {"stronger_side": "bear"}},
            confidence_threshold=0.7,
        )

    assert result["action"] == "HOLD"


def test_evaluate_position_holds_at_even_confidence_level() -> None:
    fake_client = Mock()
    fake_client.call_function.return_value = _fake_result("CLOSE", 0.6)

    with patch("agents.evaluate_position.get_default_client", return_value=fake_client):
        result = evaluate_position(
            position_context={"symbol": "GOLD#", "type": "BUY", "price_open": 2300.0, "profit": -5.0},
            technical_report={"signal": "SELL", "trend": "DOWN"},
            sentiment_report={"score": -0.05},
            debate_report={"judge_summary": {"stronger_side": "bear"}},
            confidence_threshold=0.7,
        )

    assert result["action"] == "HOLD"