from __future__ import annotations

from typing import Any

from indicators.regime import classify_regime


def _frame(adx: float | None, atr: float = 10.0, bb_width_atr: float = 4.0) -> dict[str, Any]:
    frame: dict[str, Any] = {"close": 2300.0, "atr_14": atr, "bb_mid": 2300.0}
    if adx is not None:
        frame["adx_14"] = adx
    frame["bb_upper"] = 2300.0 + bb_width_atr * atr / 2
    frame["bb_lower"] = 2300.0 - bb_width_atr * atr / 2
    return frame


def _report(
    *,
    h4_adx: float | None,
    h1_adx: float | None = None,
    d1_adx: float | None = None,
    alignment: str = "MIXED",
    trend: str = "RANGE",
    d1_trend: str = "RANGE",
    extension: float | None = 0.3,
    h4_bb_width_atr: float = 4.0,
) -> dict[str, Any]:
    return {
        "signal": "NEUTRAL",
        "trend": trend,
        "execution_trend": trend,
        "d1_trend": d1_trend,
        "alignment": alignment,
        "direction_context": {
            "d1": _frame(d1_adx),
            "h4": _frame(h4_adx, bb_width_atr=h4_bb_width_atr),
            "h1": _frame(h1_adx),
            "technical": {"extension": {"d1_close_vs_mid_atr": extension}},
        },
    }


def test_strong_aligned_adx_is_trend_with_breakout_entry() -> None:
    regime = classify_regime(_report(h4_adx=31.0, h1_adx=27.0, d1_adx=28.0, alignment="ALIGNED", trend="UP", d1_trend="UP", extension=0.4))
    assert regime["regime"] == "TREND"
    assert regime["direction"] == "UP"
    assert regime["score"] == 5
    assert regime["confidence"] == 0.8
    assert regime["entry_style"] == "STOP_BREAKOUT"
    assert regime["source"] == "rule_based"
    assert any("H4 ADX" in line for line in regime["evidence"])


def test_stretched_trend_prefers_pullback_limit() -> None:
    regime = classify_regime(_report(h4_adx=31.0, h1_adx=27.0, alignment="ALIGNED", trend="DOWN", d1_trend="DOWN", extension=-1.3))
    assert regime["regime"] == "TREND"
    assert regime["direction"] == "DOWN"
    assert regime["entry_style"] == "LIMIT_PULLBACK"


def test_exhausted_extension_forces_transition_even_when_adx_is_high() -> None:
    regime = classify_regime(_report(h4_adx=35.0, h1_adx=30.0, d1_adx=30.0, alignment="ALIGNED", trend="UP", d1_trend="UP", extension=2.4))
    assert regime["regime"] == "TRANSITION"
    assert regime["direction"] == "NEUTRAL"
    assert regime["trend_direction_raw"] == "UP"
    assert regime["entry_style"] == "NONE"
    assert regime["confidence"] == 0.6
    assert any("伸び切り" in line for line in regime["evidence"])


def test_low_adx_and_squeezed_bands_is_range_with_fade_entry() -> None:
    regime = classify_regime(_report(h4_adx=14.0, h1_adx=12.0, d1_adx=17.0, alignment="MIXED", h4_bb_width_atr=2.0))
    assert regime["regime"] == "RANGE"
    assert regime["score"] == -1
    assert regime["entry_style"] == "LIMIT_FADE"
    assert regime["direction"] == "NEUTRAL"
    assert regime["h4_bb_width_atr"] == 2.0
    assert regime["confidence"] == 0.8


def test_divergent_frames_lower_the_score() -> None:
    aligned = classify_regime(_report(h4_adx=22.0, alignment="ALIGNED", trend="UP", d1_trend="UP"))
    divergent = classify_regime(_report(h4_adx=22.0, alignment="DIVERGENT", trend="UP", d1_trend="DOWN"))
    assert aligned["score"] == 2 and aligned["regime"] == "TRANSITION"
    assert divergent["score"] == 0 and divergent["regime"] == "RANGE"


def test_trend_score_without_direction_is_not_a_trend() -> None:
    regime = classify_regime(_report(h4_adx=31.0, h1_adx=27.0, d1_adx=28.0, alignment="MIXED", trend="RANGE", d1_trend="RANGE"))
    assert regime["score"] == 4
    assert regime["regime"] == "TRANSITION"


def test_missing_inputs_yield_low_confidence_transition_and_never_raise() -> None:
    for payload in (None, {}, {"direction_context": "garbage"}, {"direction_context": {"h4": {"adx_14": "n/a"}}}):
        regime = classify_regime(payload)  # type: ignore[arg-type]
        assert regime["regime"] == "TRANSITION"
        assert regime["confidence"] == 0.3
        assert regime["entry_style"] == "NONE"
        assert regime["h4_adx"] is None
