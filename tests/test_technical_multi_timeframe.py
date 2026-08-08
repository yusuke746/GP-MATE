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
    assert result["rsi_14"] == 68.0


def test_multitimeframe_top_level_rsi_uses_h1_value() -> None:
    h4 = _bullish_frame()
    h1 = _bullish_frame()
    h1["rsi_14"] = 76.0

    with patch("agents.technical.get_default_client", return_value=_patch_client()):
        result = analyze_technical({"d1": _bullish_frame(), "h4": h4, "h1": h1})

    assert result["rsi_14"] == 76.0
    assert result["key_levels"]["frames"]["h1"]["rsi_14"] == 76.0


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


def test_multitimeframe_keeps_horizontal_levels_in_key_levels() -> None:
    payload = {
        "d1": _bullish_frame(),
        "h4": _bullish_frame(),
        "h1": _bullish_frame(),
        "horizontal_levels": {
            "resistances": [
                {
                    "price": 101.5,
                    "score": 4.2,
                    "source": "cluster",
                    "timeframe": "H4",
                    "touch_count": 3,
                }
            ],
            "supports": [
                {
                    "price": 96.5,
                    "score": 3.8,
                    "source": "swing",
                    "timeframe": "D1",
                    "touch_count": 2,
                }
            ],
        },
    }

    with patch("agents.technical.get_default_client", return_value=_patch_client()):
        result = analyze_technical(payload)

    assert "horizontal_levels" in result["key_levels"]
    assert result["key_levels"]["horizontal_levels"]["resistances"][0]["price"] == 101.5
    assert result["key_levels"]["horizontal_levels"]["supports"][0]["price"] == 96.5


def test_score_frame_band_breach_with_extreme_extension_is_not_bullish() -> None:
    from agents.technical import _score_frame

    # Parabolic case modeled on the 2026-08-06 losing trade: D1 close far above
    # the upper band, >2 ATR from the mid.
    frame = {
        "close": 4262.98,
        "rsi_14": 61.35,
        "macd_hist": 29.53,
        "bb_mid": 4075.25,
        "bb_upper": 4216.15,
        "bb_lower": 3934.36,
        "atr_14": 93.68,
        "recent_high_20": 4304.01,
        "recent_low_20": 3959.54,
    }

    scored = _score_frame(frame, "D1")

    assert any("伸び切り警戒" in reason for reason in scored["reason"].split("、"))
    # The +0.2 band bonus must NOT be applied: RSI(+0.4) + MACD(+0.6)
    # + BBミドル上(+0.2) + 高値圏(+0.1) = 1.3 (not 1.5).
    assert abs(scored["score"] - 1.3) < 1e-9


def test_score_frame_normal_band_touch_keeps_momentum_bonus() -> None:
    from agents.technical import _score_frame

    frame = {
        "close": 105.0,
        "rsi_14": 55.0,
        "macd_hist": 0.0,
        "bb_mid": 100.0,
        "bb_upper": 104.9,
        "bb_lower": 95.0,
        "atr_14": 10.0,  # extension = 0.5 ATR -> normal band ride
        "recent_high_20": 0.0,
        "recent_low_20": 0.0,
    }

    scored = _score_frame(frame, "H4")

    assert "終値がBB上限に到達" in scored["reason"]


def test_calc_extension_atr_handles_invalid_inputs() -> None:
    from agents.technical import calc_extension_atr

    assert calc_extension_atr(close=105.0, bb_mid=100.0, atr=10.0) == 0.5
    assert calc_extension_atr(close=105.0, bb_mid=100.0, atr=0.0) is None
    assert calc_extension_atr(close=105.0, bb_mid=0.0, atr=10.0) is None
