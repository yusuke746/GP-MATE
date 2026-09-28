"""Price-structure facts handed to the technical analyst (no opinions).

The analyst used to see only the latest indicator values per timeframe. A
human technician also looks at *structure*: recent swing highs/lows, whether
the last swings make higher highs or lower lows, and unfilled fair value gaps
(FVG: a three-candle imbalance where candle 1's high and candle 3's low do not
overlap). This module computes those facts from confirmed bars and returns
plain numbers. It never says what they mean; that is the analyst's job.
"""

from __future__ import annotations

from typing import Any, Final

import pandas as pd

from indicators.horizontal_levels import detect_swing_points

# An FVG narrower than this many ATRs is noise, not an imbalance.
FVG_MIN_GAP_ATR: Final[float] = 0.25
# Gaps older than this many bars are usually already traded through or stale.
FVG_MAX_AGE_BARS: Final[int] = 60
FVG_MAX_REPORTED: Final[int] = 3
SWINGS_REPORTED: Final[int] = 3


def _f(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return default if parsed != parsed else parsed


def detect_fair_value_gaps(
    frame: pd.DataFrame,
    *,
    min_gap_atr: float = FVG_MIN_GAP_ATR,
    max_age_bars: int = FVG_MAX_AGE_BARS,
) -> list[dict[str, Any]]:
    """Unfilled FVGs among the last ``max_age_bars`` bars, newest first.

    Bullish gap: low[i+1] > high[i-1] (candle i is the impulse). Bearish gap:
    high[i+1] < low[i-1]. A gap counts as filled once a later bar's range
    covers it completely; partially traded gaps are reported with ``filled_pct``.
    """
    if frame is None or frame.empty or len(frame) < 3:
        return []
    if not {"high", "low", "close"}.issubset(frame.columns):
        return []
    highs = [_f(v) for v in frame["high"].tolist()]
    lows = [_f(v) for v in frame["low"].tolist()]
    atr_series = frame["atr_14"].tolist() if "atr_14" in frame.columns else []
    n = len(frame)
    start = max(1, n - max_age_bars - 1)
    last_close = _f(frame["close"].iloc[-1])
    gaps: list[dict[str, Any]] = []
    for i in range(start, n - 1):
        atr = _f(atr_series[i]) if i < len(atr_series) else 0.0
        if atr <= 0:
            continue
        # bullish: candle i-1 high < candle i+1 low
        top, bottom, kind = lows[i + 1], highs[i - 1], "BULLISH"
        if top - bottom < min_gap_atr * atr:
            top, bottom, kind = lows[i - 1], highs[i + 1], "BEARISH"
            if top - bottom < min_gap_atr * atr:
                continue
        later_high = max(highs[i + 2 :], default=None)
        later_low = min(lows[i + 2 :], default=None)
        if kind == "BULLISH":
            filled = later_low is not None and later_low <= bottom
            traded = 0.0 if later_low is None else max(0.0, min(1.0, (top - later_low) / (top - bottom)))
        else:
            filled = later_high is not None and later_high >= top
            traded = 0.0 if later_high is None else max(0.0, min(1.0, (later_high - bottom) / (top - bottom)))
        if filled:
            continue
        gaps.append(
            {
                "type": kind,
                "top": round(top, 5),
                "bottom": round(bottom, 5),
                "size_atr": round((top - bottom) / atr, 2),
                "age_bars": n - 1 - i,
                "filled_pct": round(traded, 2),
                "distance_from_close_atr": round((((top + bottom) / 2) - last_close) / atr, 2),
            }
        )
    gaps.sort(key=lambda g: g["age_bars"])
    return gaps[:FVG_MAX_REPORTED]


def recent_swings(frame: pd.DataFrame, timeframe: str, count: int = SWINGS_REPORTED) -> dict[str, Any]:
    """Last ``count`` swing highs/lows plus the higher-high / lower-low pattern."""
    points = detect_swing_points(frame, timeframe)
    highs = [p["price"] for p in points if p["kind"] == "high"][-count:]
    lows = [p["price"] for p in points if p["kind"] == "low"][-count:]

    def _pattern(values: list[float], rising_label: str, falling_label: str) -> str:
        if len(values) < 2:
            return "UNKNOWN"
        if values[-1] > values[-2]:
            return rising_label
        if values[-1] < values[-2]:
            return falling_label
        return "EQUAL"

    return {
        "highs": highs,
        "lows": lows,
        "last_high_pattern": _pattern(highs, "HIGHER_HIGH", "LOWER_HIGH"),
        "last_low_pattern": _pattern(lows, "HIGHER_LOW", "LOWER_LOW"),
    }


def build_structure_context(frames: dict[str, pd.DataFrame | None]) -> dict[str, Any]:
    """{timeframe: {swings, fair_value_gaps, bars}} for the frames given (H1/H4/D1).

    Never raises: a frame that cannot be analysed yields an empty entry.
    """
    context: dict[str, Any] = {}
    for timeframe, frame in frames.items():
        key = str(timeframe).lower()
        try:
            if frame is None or frame.empty:
                context[key] = {}
                continue
            context[key] = {
                "bars": int(len(frame)),
                "swings": recent_swings(frame, str(timeframe).upper()),
                "fair_value_gaps": detect_fair_value_gaps(frame),
            }
        except Exception as exc:  # pragma: no cover - defensive
            context[key] = {"error": str(exc)}
    return context
