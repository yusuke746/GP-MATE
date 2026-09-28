from __future__ import annotations

from unittest.mock import Mock, patch

from agents.technical import analyze_technical


def _fake_llm_result() -> Mock:
    # Legacy-shaped answer (no execution_trend): the analyst did not answer
    # the question, so these tests exercise the rule-based fail-safe path.
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


def _analyst_client(payload: dict) -> Mock:
    result = Mock()
    result.ok = True
    result.payload = payload
    result.model = "gpt-5.6-terra"
    result.error = ""
    result.usage = Mock(prompt_tokens=1, completion_tokens=1, total_tokens=2)
    client = Mock()
    client.call_json.return_value = result
    return client


def test_fallback_path_is_marked_and_keeps_rule_based_read() -> None:
    with patch("agents.technical.get_default_client", return_value=_patch_client()):
        result = analyze_technical({"d1": _bullish_frame(), "h4": _bullish_frame(), "h1": _bullish_frame()})
    assert result["source"] == "rule_based_fallback"
    assert result["reasoning"].startswith("【ルールベース代替】")
    assert result["regime_view"] == "UNCLEAR"
    assert result["rule_based_view"]["execution_trend"] == "UP"


def test_analyst_view_overrides_rule_based_read() -> None:
    import json

    # Indicators are bullish on every frame (the old scorer says UP/ALIGNED/BUY);
    # the analyst reads exhaustion at resistance and says RANGE / MEAN_REVERSION.
    client = _analyst_client(
        {
            "d1_trend": "UP",
            "execution_trend": "RANGE",
            "alignment": "MIXED",
            "regime_view": "MEAN_REVERSION",
            "direction_if_trend": "UP",
            "key_prices": {"supports": [96.5, "95.0"], "resistances": [101.5], "invalidation": "102.2"},
            "evidence": ["H4 LOWER_HIGH", "未充填の弱気FVG 101.0-101.8"],
            "what_would_change_view": "H4終値で101.8上抜け",
            "reasoning": "上位足は上だが執行足は抵抗帯で失速。",
        }
    )
    payload = {
        "direction_context": {
            "d1": _bullish_frame(),
            "h4": _bullish_frame(),
            "h1": _bullish_frame(),
            "structure": {"h4": {"swings": {"highs": [101.0, 100.5], "last_high_pattern": "LOWER_HIGH"}, "fair_value_gaps": []}},
        },
        "tp_reference_only": {"levels": {"supports": [{"price": 96.5}], "resistances": [{"price": 101.5}]}},
    }
    with patch("agents.technical.get_default_client", return_value=client):
        result = analyze_technical(payload)

    assert result["source"] == "analyst"
    assert result["trend"] == "RANGE" and result["signal"] == "NEUTRAL"
    assert result["d1_trend"] == "UP" and result["execution_trend"] == "RANGE" and result["alignment"] == "MIXED"
    assert result["regime_view"] == "MEAN_REVERSION"
    assert result["direction_if_trend"] == "NEUTRAL"  # only meaningful for TREND_CONTINUATION
    analyst_levels = result["key_levels"]["analyst"]
    assert analyst_levels["supports"] == [96.5, 95.0] and analyst_levels["resistances"] == [101.5] and analyst_levels["invalidation"] == 102.2
    assert analyst_levels["support_level_ids"] == [] and analyst_levels["unresolved"] == []  # no catalogue in this payload
    assert result["key_levels"]["horizontal_levels"]["resistances"][0]["price"] == 101.5  # factual part kept
    assert result["evidence"][1].startswith("未充填")
    assert result["what_would_change_view"] == "H4終値で101.8上抜け"
    assert result["reasoning"] == "上位足は上だが執行足は抵抗帯で失速。"
    assert result["rule_based_view"] == {"trend": "UP", "d1_trend": "UP", "execution_trend": "UP", "alignment": "ALIGNED"}
    # The analyst saw the structure facts and the horizontal levels, and no answer key.
    user_prompt = client.call_json.call_args.kwargs["user_prompt"]
    sent = json.loads(user_prompt[user_prompt.index("{"):])
    assert sent["structure"]["h4"]["swings"]["last_high_pattern"] == "LOWER_HIGH"
    assert sent["horizontal_levels"]["supports"][0]["price"] == 96.5
    assert "baseline" not in json.dumps(sent) and "score" not in json.dumps(sent)
    system = client.call_json.call_args.kwargs["system_prompt"]
    assert "必ず" not in system and "UNCLEAR" in system


def test_analyst_call_failure_falls_back_safely() -> None:
    client = _analyst_client({"execution_trend": "UP"})
    client.call_json.return_value.ok = False
    client.call_json.return_value.error = "timeout"
    with patch("agents.technical.get_default_client", return_value=client):
        result = analyze_technical({"d1": _bearish_frame(), "h4": _bearish_frame(), "h1": _bearish_frame()})
    assert result["source"] == "rule_based_fallback"
    assert result["trend"] == "DOWN" and result["_meta"]["ok"] is False and result["_meta"]["error"] == "timeout"


def test_analyst_refers_to_levels_by_id_and_reports_roles_counter_evidence_and_quality() -> None:
    from indicators.price_levels import build_price_levels

    levels = build_price_levels(
        current_price=100.0, atr=2.0,
        horizontal_levels={"supports": [{"price": 96.5, "source": "swing", "timeframe": "D1"}], "resistances": [{"price": 101.5, "source": "cluster", "timeframe": "H4"}]},
        structure={"h4": {"swings": {"highs": [102.2], "lows": []}, "fair_value_gaps": []}},
    )
    client = _analyst_client(
        {
            "d1_trend": "UP", "h4_trend": "UP", "h1_trend": "DOWN", "h1_role": "PULLBACK", "timeframe_relationship": "H4上昇中のH1押し目",
            "execution_trend": "UP", "alignment": "ALIGNED", "regime_view": "TREND_CONTINUATION", "direction_if_trend": "UP",
            "key_levels": {"support_level_ids": ["D1_SWING_SUPPORT_1", "GHOST"], "resistance_level_ids": ["H4_CLUSTER_RESISTANCE_1"], "invalidation_level_id": "H4_SWING_HIGH_1"},
            "evidence": ["HIGHER_LOW"], "counter_evidence": ["H1 RSI 70超"], "data_quality": "partial", "abstain_reason": None,
            "what_would_change_view": "96.5割れ", "reasoning": "押し目。",
        }
    )
    payload = {"direction_context": {"d1": _bullish_frame(), "h4": _bullish_frame(), "h1": _bullish_frame(), "price_levels": levels}}
    with patch("agents.technical.get_default_client", return_value=client):
        result = analyze_technical(payload)

    assert result["h4_trend"] == "UP" and result["h1_trend"] == "DOWN" and result["h1_role"] == "PULLBACK"
    assert result["timeframe_relationship"] == "H4上昇中のH1押し目"
    analyst_levels = result["key_levels"]["analyst"]
    assert analyst_levels["supports"] == [96.5] and analyst_levels["support_level_ids"] == ["D1_SWING_SUPPORT_1"]
    assert analyst_levels["resistances"] == [101.5] and analyst_levels["invalidation"] == 102.2 and analyst_levels["invalidation_level_id"] == "H4_SWING_HIGH_1"
    assert analyst_levels["unresolved"] == ["GHOST"]
    assert result["counter_evidence"] == ["H1 RSI 70超"] and result["data_quality"] == "PARTIAL" and result["abstain_reason"] is None
    user_prompt = client.call_json.call_args.kwargs["user_prompt"]
    assert "price_levels" in user_prompt and "D1_SWING_SUPPORT_1" in user_prompt


def test_analyst_abstain_forces_unclear() -> None:
    client = _analyst_client({"execution_trend": "UP", "regime_view": "TREND_CONTINUATION", "direction_if_trend": "UP", "abstain_reason": "D1が欠損", "data_quality": "POOR"})
    with patch("agents.technical.get_default_client", return_value=client):
        result = analyze_technical({"h4": _bullish_frame(), "h1": _bullish_frame()})
    assert result["regime_view"] == "UNCLEAR" and result["direction_if_trend"] == "NEUTRAL" and result["abstain_reason"] == "D1が欠損"
    assert result["data_quality"] == "POOR"
