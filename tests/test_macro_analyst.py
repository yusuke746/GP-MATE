from __future__ import annotations

import json
from unittest.mock import Mock, patch

from agents.data.fred_client import MacroData
from agents.macro_analyst import DATA_NOTES, SYSTEM_PROMPT, analyze_macro_environment


def _base_fred_data() -> MacroData:
    return {
        "dxy": {"value": 120.0, "change_30d": -1.0, "direction": "DOWN", "source": "mt5:USDX"},
        "real_rate": {"value": 2.1, "change_30d": 0.1, "direction": "UP"},
        "us10y": {"value": 4.2, "change_30d": -0.1, "direction": "DOWN"},
        "us2y": {"value": 3.6, "change_30d": -0.2, "direction": "DOWN", "change_5d": 0.1, "direction_5d": "UP"},
        "breakeven": {"value": 2.3, "change_30d": 0.2, "direction": "UP"},
        "fed_funds": {"value": 3.6, "change_30d": -0.1, "direction": "DOWN"},
        "as_of": "2026-07-03",
        "_meta": {"ok": True, "source": "fred", "cached": False, "fetched_at": "2026-07-03", "error": ""},
    }


def _client(payload: dict | None, ok: bool = True, error: str = "") -> Mock:
    result = Mock()
    result.ok = ok
    result.payload = payload if payload is not None else {}
    result.model = "gpt-5.6-terra"
    result.error = error
    result.usage = Mock(prompt_tokens=11, completion_tokens=7, total_tokens=18)
    client = Mock()
    client.call_json.return_value = result
    return client


def test_macro_analyst_returns_neutral_when_fred_failed() -> None:
    result = analyze_macro_environment(
        {
            "dxy": {"value": None, "change_30d": None, "direction": "FLAT"},
            "as_of": "2026-07-03",
            "_meta": {"ok": False, "source": "fred", "cached": False, "fetched_at": "2026-07-03", "error": "network"},
        }
    )
    assert result["macro_bias"] == "NEUTRAL"
    assert result["regime_view"] == "UNCLEAR"
    assert result["source"] == "no_data"
    assert result["_meta"]["ok"] is False
    assert "confidence" not in result


def test_macro_analyst_adopts_the_analysts_view_even_against_textbook_reading() -> None:
    # Falling dollar, falling 2y: the old scorer said BULLISH regardless. The
    # analyst reads the 5-day reversal in the 2y and calls it BEARISH; that stands.
    client = _client(
        {
            "macro_bias": "bearish",
            "regime_view": "SUPPORTS_REVERSAL",
            "key_drivers": ["us2y 5日で+0.1pt、30日トレンドと逆行", "DXYはmt5日次で-1.0だが直近横ばい"],
            "invalidation": "us2yが3.5を下抜ける",
            "reasoning": "利下げ期待の巻き戻しが始まっている。",
        }
    )
    with patch("agents.macro_analyst.get_default_client", return_value=client):
        result = analyze_macro_environment(_base_fred_data())

    assert result["macro_bias"] == "BEARISH"
    assert result["regime_view"] == "SUPPORTS_REVERSAL"
    assert result["key_drivers"][0].startswith("us2y")
    assert result["invalidation"] == "us2yが3.5を下抜ける"
    assert result["reasoning"] == "利下げ期待の巻き戻しが始まっている。"
    assert result["source"] == "analyst"
    assert result["_meta"]["ok"] is True and result["_meta"]["usage"]["total_tokens"] == 18
    assert "confidence" not in result


def test_macro_analyst_prompt_has_no_answer_key_and_no_directional_rules() -> None:
    client = _client({"macro_bias": "NEUTRAL", "regime_view": "UNCLEAR", "key_drivers": [], "invalidation": "", "reasoning": "r"})
    with patch("agents.macro_analyst.get_default_client", return_value=client):
        analyze_macro_environment(_base_fred_data())

    sent = json.loads(client.call_json.call_args.kwargs["user_prompt"])
    assert set(sent) == {"macro_data", "data_notes", "question"}
    assert "rule_based_baseline" not in sent and "requirements" not in sent
    assert "_meta" not in sent["macro_data"]
    assert sent["data_notes"] == DATA_NOTES
    # Provenance notes describe what a field is, never what it means for gold.
    for word in ("ポジティブ", "ネガティブ", "追い風", "逆風", "強気", "弱気"):
        assert word not in json.dumps(DATA_NOTES, ensure_ascii=False)
    for phrase in ("ドル安(DOWN)は金にポジティブ", "必ず", "上限とし", "重視すること", "補助情報に留めて"):
        assert phrase not in SYSTEM_PROMPT
    assert "確信度の数値は求めません" in SYSTEM_PROMPT


def test_macro_analyst_llm_failure_returns_neutral_fallback() -> None:
    client = _client({"macro_bias": "BULLISH"}, ok=False, error="boom")
    with patch("agents.macro_analyst.get_default_client", return_value=client):
        result = analyze_macro_environment(_base_fred_data())
    assert result["macro_bias"] == "NEUTRAL"
    assert result["source"] == "fail_safe"
    assert result["_meta"]["ok"] is False and result["_meta"]["error"] == "boom"


def test_macro_analyst_invalid_bias_is_neutral_but_call_counts_as_ok() -> None:
    client = _client({"macro_bias": "SIDEWAYS", "reasoning": "x"})
    with patch("agents.macro_analyst.get_default_client", return_value=client):
        result = analyze_macro_environment(_base_fred_data())
    assert result["macro_bias"] == "NEUTRAL"
    assert result["source"] == "invalid_output"
    assert result["_meta"]["ok"] is True
    assert result["_meta"]["usage"]["total_tokens"] == 18


def test_macro_analyst_exception_is_fail_safe() -> None:
    client = Mock()
    client.call_json.side_effect = RuntimeError("api down")
    with patch("agents.macro_analyst.get_default_client", return_value=client):
        result = analyze_macro_environment(_base_fred_data())
    assert result["macro_bias"] == "NEUTRAL" and "api down" in result["_meta"]["error"]


def test_macro_analyst_records_counter_evidence_quality_and_abstain() -> None:
    client = _client({"macro_bias": "BULLISH", "regime_view": "SUPPORTS_CONTINUATION", "key_drivers": ["dxy -1.0"], "counter_evidence": ["us2y 5日で反発"], "data_quality": "PARTIAL", "abstain_reason": None, "invalidation": "x", "reasoning": "r"})
    with patch("agents.macro_analyst.get_default_client", return_value=client):
        result = analyze_macro_environment(_base_fred_data())
    assert result["counter_evidence"] == ["us2y 5日で反発"] and result["data_quality"] == "PARTIAL" and result["abstain_reason"] is None

    client = _client({"macro_bias": "BULLISH", "regime_view": "SUPPORTS_CONTINUATION", "key_drivers": [], "abstain_reason": "COT が2週間古い", "data_quality": "POOR", "invalidation": "", "reasoning": "r"})
    with patch("agents.macro_analyst.get_default_client", return_value=client):
        result = analyze_macro_environment(_base_fred_data())
    assert result["macro_bias"] == "NEUTRAL" and result["regime_view"] == "UNCLEAR" and result["abstain_reason"] == "COT が2週間古い"
