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


def _decide_with_payload(payload: dict) -> dict:
    from unittest.mock import Mock, patch

    from agents.trader import decide_trade

    fake_result = Mock()
    fake_result.ok = True
    fake_result.payload = payload
    fake_result.model = "gpt-5.6-sol"
    fake_result.error = ""
    fake_result.usage = Mock(prompt_tokens=1, completion_tokens=1, total_tokens=2)
    fake_client = Mock()
    fake_client.call_function.return_value = fake_result

    technical_report = {"direction_context": {"h1": {"close": 4404.0}}, "signal": "BUY"}
    with patch("agents.trader.get_default_client", return_value=fake_client):
        return decide_trade(
            technical_report=technical_report,
            sentiment_report={"score": 0.1},
            debate_report={},
        )


def test_pending_orders_validated_on_hold_with_bias() -> None:
    result = _decide_with_payload(
        {
            "action": "HOLD",
            "symbol": "GOLD#",
            "confidence": 0.65,
            "reasoning": "伸び切りのため押し目待ち",
            "risk_level": "MID",
            "directional_bias": "BULLISH",
            "bias_strength": 0.7,
            "pending_orders": [
                {"type": "BUY_LIMIT", "price": 4382.45, "basis": "D1サポート押し目"},
                {"type": "BUY_STOP", "price": 4436.0, "basis": "前日高値ブレイク"},
            ],
        }
    )

    # Valid orders retained, capped at 1 (the first).
    assert len(result["pending_orders"]) == 1
    assert result["pending_orders"][0]["type"] == "BUY_LIMIT"
    assert result["pending_orders"][0]["price"] == 4382.45


def test_pending_orders_dropped_when_wrong_side_or_direction() -> None:
    result = _decide_with_payload(
        {
            "action": "HOLD",
            "symbol": "GOLD#",
            "confidence": 0.65,
            "reasoning": "test",
            "risk_level": "MID",
            "directional_bias": "BULLISH",
            "bias_strength": 0.7,
            "pending_orders": [
                {"type": "SELL_STOP", "price": 4380.0},  # against bias
                {"type": "BUY_LIMIT", "price": 4500.0},  # above current price
                {"type": "BUY_STOP", "price": 4380.0},  # below current price
            ],
        }
    )

    assert result["pending_orders"] == []


def test_pending_orders_cleared_on_market_action_or_weak_bias() -> None:
    on_buy = _decide_with_payload(
        {
            "action": "BUY",
            "symbol": "GOLD#",
            "confidence": 0.9,
            "reasoning": "test",
            "risk_level": "MID",
            "directional_bias": "BULLISH",
            "bias_strength": 0.9,
            "pending_orders": [{"type": "BUY_LIMIT", "price": 4382.45}],
        }
    )
    assert on_buy["pending_orders"] == []

    weak_bias = _decide_with_payload(
        {
            "action": "HOLD",
            "symbol": "GOLD#",
            "confidence": 0.6,
            "reasoning": "test",
            "risk_level": "MID",
            "directional_bias": "BULLISH",
            "bias_strength": 0.4,
            "pending_orders": [{"type": "BUY_LIMIT", "price": 4382.45}],
        }
    )
    assert weak_bias["pending_orders"] == []


def test_suggested_sl_passthrough_and_direction_check() -> None:
    valid = _decide_with_payload(
        {
            "action": "BUY",
            "symbol": "GOLD#",
            "confidence": 0.9,
            "reasoning": "test",
            "risk_level": "MID",
            "suggested_sl": 4380.0,
            "suggested_sl_basis": "4382サポート帯の外側",
        }
    )
    assert valid["suggested_sl"] == 4380.0

    inverted = _decide_with_payload(
        {
            "action": "BUY",
            "symbol": "GOLD#",
            "confidence": 0.9,
            "reasoning": "test",
            "risk_level": "MID",
            "suggested_sl": 4500.0,  # above current price for a BUY
        }
    )
    assert inverted["suggested_sl"] is None

    on_hold = _decide_with_payload(
        {
            "action": "HOLD",
            "symbol": "GOLD#",
            "confidence": 0.6,
            "reasoning": "test",
            "risk_level": "MID",
            "suggested_sl": 4380.0,
        }
    )
    assert on_hold["suggested_sl"] is None


def test_pending_order_sl_direction_check() -> None:
    result = _decide_with_payload(
        {
            "action": "HOLD",
            "symbol": "GOLD#",
            "confidence": 0.65,
            "reasoning": "test",
            "risk_level": "MID",
            "directional_bias": "BULLISH",
            "bias_strength": 0.7,
            "pending_orders": [
                {"type": "BUY_LIMIT", "price": 4382.45, "sl": 4370.0, "basis": "押し目"},
            ],
        }
    )
    assert result["pending_orders"][0]["sl"] == 4370.0

    inverted = _decide_with_payload(
        {
            "action": "HOLD",
            "symbol": "GOLD#",
            "confidence": 0.65,
            "reasoning": "test",
            "risk_level": "MID",
            "directional_bias": "BULLISH",
            "bias_strength": 0.7,
            "pending_orders": [
                {"type": "BUY_LIMIT", "price": 4382.45, "sl": 4390.0, "basis": "押し目"},
            ],
        }
    )
    # Inverted SL is dropped (order kept, ATR fallback applies downstream).
    assert inverted["pending_orders"][0]["sl"] is None
