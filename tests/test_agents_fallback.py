from __future__ import annotations

from agents.debate import run_debate
from agents.sentiment import analyze_sentiment
from agents.technical import analyze_technical
from agents.trader import decide_trade


def test_technical_fallback_shape() -> None:
    result = analyze_technical({"rsi": 50})
    assert "signal" in result
    assert "_meta" in result


def test_sentiment_fallback_shape() -> None:
    result = analyze_sentiment([{"title": "gold steady"}])
    assert "score" in result
    assert "_meta" in result


def test_debate_fallback_shape() -> None:
    result = run_debate({"signal": "NEUTRAL"}, {"score": 0.0})
    assert "round1" in result
    assert "round2" in result
    assert "_meta" in result


def test_trader_enforces_hold_when_low_confidence() -> None:
    result = decide_trade(
        technical_report={"signal": "BUY"},
        sentiment_report={"score": 0.2},
        debate_report={"round1": {}, "round2": {}},
        confidence_threshold=0.6,
    )
    assert result["action"] in {"BUY", "SELL", "HOLD"}
    if result["confidence"] < 0.6:
        assert result["action"] == "HOLD"


def test_trader_forces_hold_when_sentiment_evidence_insufficient() -> None:
    result = decide_trade(
        technical_report={"signal": "BUY"},
        sentiment_report={"score": 0.0, "evidence_status": "INSUFFICIENT"},
        debate_report={"round1": {}, "round2": {}},
        confidence_threshold=0.1,
    )
    assert result["action"] == "HOLD"
