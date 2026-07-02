from __future__ import annotations

from unittest.mock import Mock, patch

from agents.technical import analyze_technical


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


def _bullish_frame(close: float = 100.0) -> dict[str, float]:
    return {
        "close": close,
        "rsi_14": 68.0,
        "macd_hist": 0.4,
        "bb_upper": 101.0,
        "bb_mid": 98.0,
        "bb_lower": 95.0,
        "atr_14": 2.0,
        "recent_high_20": 100.5,
        "recent_low_20": 96.0,
    }


def _bearish_frame(close: float = 100.0) -> dict[str, float]:
    return {
        "close": close,
        "rsi_14": 32.0,
        "macd_hist": -0.5,
        "bb_upper": 105.0,
        "bb_mid": 102.0,
        "bb_lower": 99.0,
        "atr_14": 2.0,
        "recent_high_20": 104.0,
        "recent_low_20": 99.5,
    }


def _range_frame(close: float = 100.0) -> dict[str, float]:
    return {
        "close": close,
        "rsi_14": 50.0,
        "macd_hist": 0.0,
        "bb_upper": 101.0,
        "bb_mid": 100.0,
        "bb_lower": 99.0,
        "atr_14": 1.5,
        "recent_high_20": 101.0,
        "recent_low_20": 99.0,
    }


def test_multitimeframe_alignment_is_aligned_when_d1_and_execution_match() -> None:
    with patch("agents.technical.get_default_client", return_value=_patch_client()):
        result = analyze_technical({"d1": _bullish_frame(), "h4": _bullish_frame(), "h1": _bullish_frame()})

    assert result["d1_trend"] == "UP"
    assert result["execution_trend"] == "UP"
    assert result["alignment"] == "ALIGNED"
    assert result["trend"] == "UP"
    assert result["signal"] == "BUY"


def test_multitimeframe_alignment_is_divergent_when_d1_and_execution_conflict() -> None:
    with patch("agents.technical.get_default_client", return_value=_patch_client()):
        result = analyze_technical({"d1": _bearish_frame(), "h4": _bullish_frame(), "h1": _bullish_frame()})

    assert result["d1_trend"] == "DOWN"
    assert result["execution_trend"] == "UP"
    assert result["alignment"] == "DIVERGENT"


def test_multitimeframe_alignment_is_mixed_when_one_frame_is_range() -> None:
    with patch("agents.technical.get_default_client", return_value=_patch_client()):
        result = analyze_technical({"d1": _range_frame(), "h4": _bullish_frame(), "h1": _bullish_frame()})

    assert result["d1_trend"] == "RANGE"
    assert result["execution_trend"] == "UP"
    assert result["alignment"] == "MIXED"


def test_multitimeframe_analysis_works_without_d1_data() -> None:
    with patch("agents.technical.get_default_client", return_value=_patch_client()):
        result = analyze_technical({"h4": _bullish_frame(), "h1": _bullish_frame()})

    assert result["d1_trend"] == "RANGE"
    assert result["execution_trend"] == "UP"
    assert result["alignment"] == "MIXED"
    assert result["signal"] == "BUY"
