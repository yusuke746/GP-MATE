from __future__ import annotations

from analysis.forecast_labels import resolve_outcome

P0, ATR, UP, DOWN = 100.0, 2.0, 102.0, 98.0


def _bar(high: float, low: float) -> dict:
    return {"high": high, "low": low}


def test_up_when_upper_barrier_touched_first() -> None:
    bars = [_bar(101.0, 99.0), _bar(102.5, 99.5), _bar(97.0, 96.0)]
    label = resolve_outcome(bars, p0=P0, atr=ATR, barrier_up=UP, barrier_down=DOWN, horizon_bars=6)
    assert label["outcome"] == "UP"
    assert label["bars_to_hit"] == 2
    assert label["max_favorable_atr"] == 1.25
    assert label["max_adverse_atr"] == 0.5


def test_down_when_lower_barrier_touched_first() -> None:
    bars = [_bar(101.5, 99.0), _bar(100.0, 97.9)]
    label = resolve_outcome(bars, p0=P0, atr=ATR, barrier_up=UP, barrier_down=DOWN, horizon_bars=6)
    assert label["outcome"] == "DOWN" and label["bars_to_hit"] == 2


def test_timeout_when_neither_within_horizon() -> None:
    bars = [_bar(101.9, 98.1)] * 6 + [_bar(105.0, 99.0)]  # the 7th bar is beyond the horizon
    label = resolve_outcome(bars, p0=P0, atr=ATR, barrier_up=UP, barrier_down=DOWN, horizon_bars=6)
    assert label["outcome"] == "TIMEOUT" and label["bars_to_hit"] is None
    assert label["max_favorable_atr"] == 0.95 and label["max_adverse_atr"] == 0.95


def test_ambiguous_when_both_barriers_in_same_bar() -> None:
    bars = [_bar(101.0, 99.0), _bar(103.0, 97.0)]
    label = resolve_outcome(bars, p0=P0, atr=ATR, barrier_up=UP, barrier_down=DOWN, horizon_bars=6)
    assert label["outcome"] == "AMBIGUOUS" and label["bars_to_hit"] == 2


def test_unresolved_when_not_enough_bars_and_no_hit() -> None:
    bars = [_bar(101.0, 99.0)] * 3
    assert resolve_outcome(bars, p0=P0, atr=ATR, barrier_up=UP, barrier_down=DOWN, horizon_bars=6) is None
    assert resolve_outcome([], p0=P0, atr=ATR, barrier_up=UP, barrier_down=DOWN, horizon_bars=6) is None


def test_exact_touch_counts_and_is_idempotent() -> None:
    bars = [_bar(102.0, 99.5)]
    first = resolve_outcome(bars, p0=P0, atr=ATR, barrier_up=UP, barrier_down=DOWN, horizon_bars=1)
    second = resolve_outcome(bars, p0=P0, atr=ATR, barrier_up=UP, barrier_down=DOWN, horizon_bars=1)
    assert first == second and first["outcome"] == "UP"
