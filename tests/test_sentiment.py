from __future__ import annotations

from unittest.mock import Mock, patch

from agents.sentiment import _normalize_sentiment_payload, analyze_sentiment


def _fake_client_returning(payload: dict) -> Mock:
    fake_result = Mock()
    fake_result.ok = True
    fake_result.payload = payload
    fake_result.model = "gpt-5.4-mini"
    fake_result.error = ""
    fake_result.usage = Mock(prompt_tokens=1, completion_tokens=1, total_tokens=2)

    fake_client = Mock()
    fake_client.call_json.return_value = fake_result
    return fake_client


def test_sentiment_aggregates_score_from_item_evaluations() -> None:
    # Regression: some models return only per-item evaluations without the
    # top-level aggregate score, which downstream read as 0.0 (always NEUTRAL).
    payload = {
        "evaluations": [
            {"title": "rate cut hopes fade", "score": -0.35, "reasoning": "利下げ観測後退"},
            {"title": "BOJ decision", "score": 0.05, "reasoning": "方向感弱い"},
            {"title": "tariff uncertainty", "score": 0.15, "reasoning": "ヘッジ需要"},
            {"title": "Dow hits record", "score": -0.25, "reasoning": "リスクオン"},
        ]
    }

    with patch("agents.sentiment.get_default_client", return_value=_fake_client_returning(payload)):
        result = analyze_sentiment([{"title": "n1"}, {"title": "n2"}, {"title": "n3"}, {"title": "n4"}])

    assert abs(result["score"] - (-0.35 + 0.05 + 0.15 - 0.25) / 4) < 1e-9
    # Dominant news = the item with the largest absolute score.
    assert result["dominant_news"] == "rate cut hopes fade"
    assert result["reasoning"].strip() != ""


def test_sentiment_keeps_top_level_score_when_present() -> None:
    payload = {
        "score": 0.4,
        "dominant_news": "gold rallies",
        "reasoning": "ドル安",
        "evaluations": [{"title": "x", "score": -0.9, "reasoning": "無視されるべき"}],
    }

    with patch("agents.sentiment.get_default_client", return_value=_fake_client_returning(payload)):
        result = analyze_sentiment([{"title": "n1"}])

    assert result["score"] == 0.4
    assert result["dominant_news"] == "gold rallies"


def test_normalize_sentiment_payload_clamps_score() -> None:
    assert _normalize_sentiment_payload({"score": 1.7})["score"] == 1.0
    assert _normalize_sentiment_payload({"score": -2.0})["score"] == -1.0


def test_normalize_sentiment_payload_defaults_to_neutral_without_data() -> None:
    normalized = _normalize_sentiment_payload({})
    assert normalized["score"] == 0.0
    assert normalized["dominant_news"] == "N/A"
    assert normalized["reasoning"].strip() != ""
