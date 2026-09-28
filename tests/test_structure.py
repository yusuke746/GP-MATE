from __future__ import annotations

import pandas as pd

from indicators.structure import build_structure_context, detect_fair_value_gaps, recent_swings


def _bars(rows: list[tuple[float, float, float, float]], atr: float = 10.0) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=["open", "high", "low", "close"])
    frame["atr_14"] = atr
    return frame


def test_detect_bullish_fvg_and_partial_fill() -> None:
    # candle1 high 105 -> impulse -> candle3 low 112: bullish gap 105..112 (0.7 ATR)
    rows = [(100, 105, 99, 104), (104, 115, 103, 114), (114, 120, 112, 118), (118, 121, 110, 111)]
    gaps = detect_fair_value_gaps(_bars(rows))
    assert len(gaps) == 1
    gap = gaps[0]
    assert gap["type"] == "BULLISH" and gap["top"] == 112.0 and gap["bottom"] == 105.0
    assert gap["size_atr"] == 0.7 and gap["age_bars"] == 2
    assert gap["filled_pct"] == round((112 - 110) / 7, 2)  # last bar dipped to 110
    assert gap["distance_from_close_atr"] == round(((112 + 105) / 2 - 111) / 10, 2)


def test_filled_gaps_and_small_gaps_are_dropped() -> None:
    filled = [(100, 105, 99, 104), (104, 115, 103, 114), (114, 120, 112, 118), (118, 119, 104, 105)]
    assert detect_fair_value_gaps(_bars(filled)) == []
    tiny = [(100, 105, 99, 104), (104, 108, 103, 107), (107, 109, 106, 108)]  # gap 105..106 = 0.1 ATR
    assert detect_fair_value_gaps(_bars(tiny)) == []
    bearish = [(120, 121, 115, 116), (116, 117, 105, 106), (106, 108, 100, 101)]
    gaps = detect_fair_value_gaps(_bars(bearish))
    assert len(gaps) == 1 and gaps[0]["type"] == "BEARISH" and gaps[0]["top"] == 115.0 and gaps[0]["bottom"] == 108.0


def test_recent_swings_reports_pattern() -> None:
    highs = [10, 12, 11, 10, 11, 14, 12, 11, 12, 13, 12, 11]
    rows = [(h - 1, h, h - 3, h - 1) for h in highs]
    swings = recent_swings(_bars(rows), "H1")
    assert swings["highs"] == [14.0, 13.0]  # index 1 has no two left bars, so it is not a swing
    assert swings["last_high_pattern"] == "LOWER_HIGH"
    assert swings["last_low_pattern"] in {"HIGHER_LOW", "LOWER_LOW", "EQUAL", "UNKNOWN"}


def test_build_structure_context_is_safe_on_empty_or_missing_frames() -> None:
    context = build_structure_context({"d1": None, "h4": pd.DataFrame(), "h1": _bars([(1, 2, 0, 1)] * 3)})
    assert context["d1"] == {} and context["h4"] == {}
    assert context["h1"]["bars"] == 3 and context["h1"]["fair_value_gaps"] == []
