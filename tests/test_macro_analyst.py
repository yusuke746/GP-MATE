from __future__ import annotations

from unittest.mock import Mock, patch

from agents.data.fred_client import MacroData
from agents.macro_analyst import analyze_macro_environment


def _base_fred_data() -> MacroData:
    return {
        "dxy": {"value": 120.0, "change_30d": -1.0, "direction": "DOWN"},
        "real_rate": {"value": 2.1, "change_30d": 0.1, "direction": "UP"},
        "us10y": {"value": 4.2, "change_30d": -0.1, "direction": "DOWN"},
        "breakeven": {"value": 2.3, "change_30d": 0.2, "direction": "UP"},
        "fed_funds": {"value": 3.6, "change_30d": -0.1, "direction": "DOWN"},
        "as_of": "2026-07-03",
        "_meta": {"ok": True, "source": "fred", "cached": False, "fetched_at": "2026-07-03", "error": ""},
    }


def test_macro_analyst_returns_neutral_when_fred_failed() -> None:
    result = analyze_macro_environment(
        {
            "dxy": {"value": None, "change_30d": None, "direction": "FLAT"},
            "real_rate": {"value": None, "change_30d": None, "direction": "FLAT"},
            "us10y": {"value": None, "change_30d": None, "direction": "FLAT"},
            "breakeven": {"value": None, "change_30d": None, "direction": "FLAT"},
            "fed_funds": {"value": None, "change_30d": None, "direction": "FLAT"},
            "as_of": "2026-07-03",
            "_meta": {"ok": False, "source": "fred", "cached": False, "fetched_at": "2026-07-03", "error": "network"},
        }
    )

    assert result["macro_bias"] == "NEUTRAL"
    assert result["confidence"] == 0.5
    assert result["_meta"]["ok"] is False


def test_macro_analyst_is_bullish_when_dollar_is_weak(monkeypatch) -> None:
    fred_data = _base_fred_data()
    fred_data["dxy"]["direction"] = "DOWN"
    fred_data["fed_funds"]["direction"] = "DOWN"
    fred_data["breakeven"]["direction"] = "UP"

    fake_result = Mock()
    fake_result.ok = True
    fake_result.payload = {
        "macro_bias": "NEUTRAL",
        "confidence": 0.1,
        "key_drivers": ["LLM"],
        "reasoning": "LLM reasoning",
    }
    fake_result.model = "gpt-5.4-mini"
    fake_result.error = ""
    fake_result.usage = Mock(prompt_tokens=1, completion_tokens=1, total_tokens=2)

    fake_client = Mock()
    fake_client.call_json.return_value = fake_result

    with patch("agents.macro_analyst.get_default_client", return_value=fake_client):
        result = analyze_macro_environment(fred_data)

    assert result["macro_bias"] == "BULLISH"
    assert result["confidence"] > 0.5
    assert any("ドル安" in driver for driver in result["key_drivers"])
    assert result["_meta"]["ok"] is True


def test_macro_analyst_is_bearish_when_dollar_is_strong(monkeypatch) -> None:
    fred_data = _base_fred_data()
    fred_data["dxy"]["direction"] = "UP"
    fred_data["fed_funds"]["direction"] = "UP"
    fred_data["breakeven"]["direction"] = "DOWN"

    fake_result = Mock()
    fake_result.ok = True
    fake_result.payload = {
        "macro_bias": "BULLISH",
        "confidence": 0.99,
        "key_drivers": ["LLM"],
        "reasoning": "LLM reasoning",
    }
    fake_result.model = "gpt-5.4-mini"
    fake_result.error = ""
    fake_result.usage = Mock(prompt_tokens=1, completion_tokens=1, total_tokens=2)

    fake_client = Mock()
    fake_client.call_json.return_value = fake_result

    with patch("agents.macro_analyst.get_default_client", return_value=fake_client):
        result = analyze_macro_environment(fred_data)

    assert result["macro_bias"] == "BEARISH"
    assert result["confidence"] > 0.5
    assert any("ドル高" in driver for driver in result["key_drivers"])


def test_macro_analyst_real_rate_does_not_force_bearish_by_itself() -> None:
    fred_data = _base_fred_data()
    fred_data["dxy"]["direction"] = "DOWN"
    fred_data["fed_funds"]["direction"] = "FLAT"
    fred_data["breakeven"]["direction"] = "UP"
    fred_data["real_rate"]["direction"] = "UP"

    fake_result = Mock()
    fake_result.ok = True
    fake_result.payload = {
        "macro_bias": "NEUTRAL",
        "confidence": 0.1,
        "key_drivers": ["LLM"],
        "reasoning": "LLM reasoning",
    }
    fake_result.model = "gpt-5.4-mini"
    fake_result.error = ""
    fake_result.usage = Mock(prompt_tokens=1, completion_tokens=1, total_tokens=2)

    fake_client = Mock()
    fake_client.call_json.return_value = fake_result

    with patch("agents.macro_analyst.get_default_client", return_value=fake_client):
        result = analyze_macro_environment(fred_data)

    assert result["macro_bias"] == "BULLISH"
    assert any("逆相関が崩れている" in driver for driver in result["key_drivers"])


def test_macro_analyst_llm_failure_returns_neutral_fallback() -> None:
    fred_data = _base_fred_data()

    fake_result = Mock()
    fake_result.ok = False
    fake_result.payload = {"macro_bias": "BULLISH", "confidence": 0.9, "key_drivers": [], "reasoning": "x"}
    fake_result.model = "gpt-5.4-mini"
    fake_result.error = "boom"
    fake_result.usage = Mock(prompt_tokens=0, completion_tokens=0, total_tokens=0)

    fake_client = Mock()
    fake_client.call_json.return_value = fake_result

    with patch("agents.macro_analyst.get_default_client", return_value=fake_client):
        result = analyze_macro_environment(fred_data)

    assert result["macro_bias"] == "NEUTRAL"
    assert result["confidence"] == 0.5
    assert result["_meta"]["ok"] is False
    assert result["_meta"]["error"] == "boom"


# --------------------------------------------------------------------------- #
# LLM confidence merge: downshift only, bounded by MACRO_LLM_CONF_MAX_DOWNSHIFT
# --------------------------------------------------------------------------- #
def _run_with_llm(llm_bias: str, llm_confidence: float, fred_data: MacroData | None = None):
    from agents.macro_analyst import _score_macro_environment

    data = fred_data or _base_fred_data()
    baseline_bias, baseline_conf, _, _ = _score_macro_environment(data)

    fake_result = Mock()
    fake_result.ok = True
    fake_result.payload = {
        "macro_bias": llm_bias,
        "confidence": llm_confidence,
        "key_drivers": ["LLM"],
        "reasoning": "確信度は中程度に抑制する",
    }
    fake_result.model = "gpt-5.6-terra"
    fake_result.error = ""
    fake_result.usage = Mock(prompt_tokens=1, completion_tokens=1, total_tokens=2)
    fake_client = Mock()
    fake_client.call_json.return_value = fake_result
    with patch("agents.macro_analyst.get_default_client", return_value=fake_client):
        result = analyze_macro_environment(data)
    return result, baseline_bias, baseline_conf


def test_llm_confidence_below_baseline_is_adopted_within_band() -> None:
    result, bias, baseline_conf = _run_with_llm("BULLISH", 0.58)
    assert bias == "BULLISH"
    assert baseline_conf > 0.58
    # Adopted as-is when within the 0.15 band, otherwise clamped to baseline - 0.15.
    assert result["confidence"] == round(max(baseline_conf - 0.15, 0.58), 4)
    assert result["confidence"] < baseline_conf
    assert result["_meta"]["confidence_source"] == "llm_downshift"
    assert result["_meta"]["llm_confidence"] == 0.58
    assert result["reasoning"] == "確信度は中程度に抑制する"


def test_llm_confidence_far_below_baseline_is_capped_at_max_downshift() -> None:
    result, _, baseline_conf = _run_with_llm("BULLISH", 0.40)
    assert result["confidence"] == round(baseline_conf - 0.15, 4)
    assert result["_meta"]["confidence_source"] == "llm_downshift"
    assert result["_meta"]["llm_confidence"] == 0.40


def test_llm_confidence_above_baseline_is_ignored() -> None:
    result, _, baseline_conf = _run_with_llm("BULLISH", 0.90)
    assert result["confidence"] == round(baseline_conf, 4)
    assert result["_meta"]["confidence_source"] == "rule_based"
    assert result["_meta"]["llm_confidence"] == 0.90


def test_llm_bias_mismatch_keeps_rule_based_confidence_and_reasoning() -> None:
    result, _, baseline_conf = _run_with_llm("BEARISH", 0.40)
    assert result["confidence"] == round(baseline_conf, 4)
    assert result["_meta"]["confidence_source"] == "rule_based"
    assert result["_meta"]["llm_confidence"] == 0.40
    assert result["reasoning"] != "確信度は中程度に抑制する"


def test_review_case_0689_to_058() -> None:
    # 2026-09-09: rule-based 0.689, LLM 0.58 (2y yield up, PPI ahead) -> 0.58.
    from agents.macro_analyst import _merge_llm_result

    baseline = {
        "macro_bias": "BULLISH",
        "confidence": 0.689,
        "key_drivers": ["rule"],
        "reasoning": "rule reasoning",
        "_meta": {"ok": True, "model": "rule_based", "usage": {}, "error": ""},
    }
    merged = _merge_llm_result(baseline, {"macro_bias": "BULLISH", "confidence": 0.58, "reasoning": "PPI前で抑制"})
    assert merged["confidence"] == 0.58
    assert merged["_meta"]["confidence_source"] == "llm_downshift"
    merged = _merge_llm_result(baseline, {"macro_bias": "BULLISH", "confidence": 0.40, "reasoning": "x"})
    assert merged["confidence"] == 0.539
    merged = _merge_llm_result(baseline, {"macro_bias": "BULLISH", "confidence": 0.90, "reasoning": "x"})
    assert merged["confidence"] == 0.689 and merged["_meta"]["confidence_source"] == "rule_based"
    merged = _merge_llm_result(baseline, {"macro_bias": "BEARISH", "confidence": 0.40, "reasoning": "x"})
    assert merged["confidence"] == 0.689 and merged["_meta"]["confidence_source"] == "rule_based"
    merged = _merge_llm_result(baseline, {"macro_bias": "BULLISH", "confidence": "n/a", "reasoning": "x"})
    assert merged["confidence"] == 0.689 and merged["_meta"]["llm_confidence"] is None
