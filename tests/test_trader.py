from __future__ import annotations

from unittest.mock import Mock, patch

from agents.trader import decide_trade


def _fake_result(payload: dict) -> Mock:
    result = Mock()
    result.ok = True
    result.payload = payload
    result.model = "gpt-5.5"
    result.error = ""
    result.usage = Mock(prompt_tokens=1, completion_tokens=1, total_tokens=2)
    return result


def _run_with_payload(
    llm_payload: dict,
    technical_report: dict,
    sentiment_report: dict | None = None,
    macro_report: dict | None = None,
) -> dict:
    fake_client = Mock()
    fake_client.call_function.return_value = _fake_result(llm_payload)

    with patch("agents.trader.get_default_client", return_value=fake_client):
        return decide_trade(
            technical_report=technical_report,
            sentiment_report=sentiment_report or {"score": 0.2},
            debate_report={"judge_summary": {}},
            macro_report=macro_report,
            confidence_threshold=0.1,
        )


def test_suggested_tp_buy_is_kept_when_above_current_price() -> None:
    result = _run_with_payload(
        llm_payload={
            "action": "BUY",
            "symbol": "GOLD#",
            "confidence": 0.8,
            "reasoning": "test",
            "risk_level": "MID",
            "suggested_tp": 4055.0,
            "suggested_tp_basis": "キリ番4050とレジ重なり手前",
        },
        technical_report={
            "direction_context": {"h1": {"close": 4050.0}},
            "tp_reference_only": {
                "levels": {"supports": [], "resistances": []},
                "round_numbers": [4050.0],
                "prev_day": {"high": 4060.0, "low": 4030.0},
                "moving_averages": {"h4_ma20": 4048.0},
            },
        },
    )

    assert isinstance(result["suggested_tp"], float)
    assert result["suggested_tp"] > 4050.0


def test_suggested_tp_sell_is_kept_when_below_current_price() -> None:
    result = _run_with_payload(
        llm_payload={
            "action": "SELL",
            "symbol": "GOLD#",
            "confidence": 0.8,
            "reasoning": "test",
            "risk_level": "MID",
            "suggested_tp": 4040.0,
            "suggested_tp_basis": "サポート手前",
        },
        technical_report={"direction_context": {"h1": {"close": 4050.0}}},
    )

    assert isinstance(result["suggested_tp"], float)
    assert result["suggested_tp"] < 4050.0


def test_suggested_tp_is_forced_none_on_hold() -> None:
    result = _run_with_payload(
        llm_payload={
            "action": "HOLD",
            "symbol": "GOLD#",
            "confidence": 0.9,
            "reasoning": "hold",
            "risk_level": "LOW",
            "suggested_tp": 4100.0,
            "suggested_tp_basis": "dummy",
        },
        technical_report={"direction_context": {"h1": {"close": 4050.0}}},
    )

    assert result["action"] == "HOLD"
    assert result["suggested_tp"] is None


def test_suggested_tp_is_none_when_direction_is_inconsistent() -> None:
    result = _run_with_payload(
        llm_payload={
            "action": "SELL",
            "symbol": "GOLD#",
            "confidence": 0.85,
            "reasoning": "test",
            "risk_level": "MID",
            "suggested_tp": 4060.0,
            "suggested_tp_basis": "不整合ケース",
        },
        technical_report={"direction_context": {"h1": {"close": 4050.0}}},
    )

    assert result["action"] == "SELL"
    assert result["suggested_tp"] is None


def test_suggested_tp_works_without_current_price_by_type_check_only() -> None:
    result = _run_with_payload(
        llm_payload={
            "action": "BUY",
            "symbol": "GOLD#",
            "confidence": 0.8,
            "reasoning": "test",
            "risk_level": "MID",
            "suggested_tp": "4058.5",
            "suggested_tp_basis": "文字列数値",
        },
        technical_report={"signal": "BUY"},
    )

    assert result["action"] == "BUY"
    assert result["suggested_tp"] == 4058.5


def test_decide_trade_completes_when_suggested_tp_is_missing() -> None:
    result = _run_with_payload(
        llm_payload={
            "action": "BUY",
            "symbol": "GOLD#",
            "confidence": 0.8,
            "reasoning": "test",
            "risk_level": "MID",
        },
        technical_report={"direction_context": {"h1": {"close": 4050.0}}},
    )

    assert result["action"] == "BUY"
    assert result["suggested_tp"] is None
    assert result["suggested_tp_basis"] == ""


def test_existing_action_confidence_directional_bias_logic_is_preserved() -> None:
    result = _run_with_payload(
        llm_payload={
            "action": "BUY",
            "symbol": "GOLD#",
            "confidence": 0.8,
            "reasoning": "test",
            "risk_level": "MID",
            "directional_bias": "NEUTRAL",
            "bias_strength": 0.4,
            "trigger_conditions": ["4055上抜け"],
            "suggested_tp": 4060.0,
            "suggested_tp_basis": "レジ手前",
        },
        technical_report={"direction_context": {"h1": {"close": 4050.0}}},
        macro_report={
            "macro_bias": "BULLISH",
            "confidence": 0.9,
            "_meta": {"ok": True},
        },
    )

    assert result["action"] == "BUY"
    assert result["confidence"] == 0.8
    assert result["directional_bias"] == "BULLISH"
    assert result["trigger_conditions"] == ["4055上抜け"]


def test_decide_trade_passes_recent_context_to_model() -> None:
    from unittest.mock import Mock, patch

    from agents.trader import decide_trade

    fake_result = Mock()
    fake_result.ok = True
    fake_result.payload = {
        "action": "HOLD",
        "symbol": "GOLD#",
        "confidence": 0.5,
        "reasoning": "test",
        "risk_level": "MID",
    }
    fake_result.model = "gpt-5.5"
    fake_result.error = ""
    fake_result.usage = Mock(prompt_tokens=1, completion_tokens=1, total_tokens=2)

    fake_client = Mock()
    fake_client.call_function.return_value = fake_result

    recent_context = {
        "decisions": [{"time_utc": "2026-08-06T12:00:00+00:00", "action": "BUY", "confidence": 0.74}],
        "recent_closed": [{"time_utc": "2026-08-06T13:09:17+00:00", "pnl": -8468.0, "result": "LOSS"}],
    }

    with patch("agents.trader.get_default_client", return_value=fake_client):
        decide_trade(
            technical_report={"signal": "BUY"},
            sentiment_report={"score": 0.2},
            debate_report={},
            recent_context=recent_context,
        )

    user_prompt = fake_client.call_function.call_args.kwargs["user_prompt"]
    assert "recent_context" in user_prompt
    assert "-8468" in user_prompt
    assert "LOSS" in user_prompt
