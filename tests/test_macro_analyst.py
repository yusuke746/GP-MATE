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
