"""Triple-barrier outcome labelling for forecast-only records.

Given the H1 bars strictly AFTER the forecast bar (chronological), decide
which barrier was touched first within the horizon:

* UP        high >= barrier_up before any low <= barrier_down
* DOWN      low <= barrier_down before any high >= barrier_up
* TIMEOUT   neither within ``horizon_bars`` bars
* AMBIGUOUS both barriers touched inside the same bar (excluded from scoring)

Returns None while fewer than ``horizon_bars`` bars exist and no barrier has
been hit yet (the label is not decidable; retry later). Idempotent.
"""

from __future__ import annotations

from typing import Any

OUTCOMES = ("UP", "DOWN", "TIMEOUT", "AMBIGUOUS")
SCORABLE_OUTCOMES = ("UP", "DOWN", "TIMEOUT")


def resolve_outcome(
    bars_after: list[dict[str, Any]],
    *,
    p0: float,
    atr: float,
    barrier_up: float,
    barrier_down: float,
    horizon_bars: int,
) -> dict[str, Any] | None:
    horizon = max(1, int(horizon_bars))
    window = list(bars_after[:horizon])
    if not window:
        return None

    max_high = float("-inf")
    min_low = float("inf")
    for index, bar in enumerate(window, start=1):
        high = float(bar["high"])
        low = float(bar["low"])
        max_high = max(max_high, high)
        min_low = min(min_low, low)
        hit_up = high >= barrier_up
        hit_down = low <= barrier_down
        if hit_up and hit_down:
            return _result("AMBIGUOUS", index, p0, atr, max_high, min_low)
        if hit_up:
            return _result("UP", index, p0, atr, max_high, min_low)
        if hit_down:
            return _result("DOWN", index, p0, atr, max_high, min_low)

    if len(window) < horizon:
        return None  # not enough bars yet to declare a timeout
    return _result("TIMEOUT", None, p0, atr, max_high, min_low)


def _result(outcome: str, bars_to_hit: int | None, p0: float, atr: float, max_high: float, min_low: float) -> dict[str, Any]:
    safe_atr = atr if atr and atr > 0 else None
    return {
        "outcome": outcome,
        "bars_to_hit": bars_to_hit,
        "max_favorable_atr": round((max_high - p0) / safe_atr, 4) if safe_atr else None,
        "max_adverse_atr": round((p0 - min_low) / safe_atr, 4) if safe_atr else None,
    }
