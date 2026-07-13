from __future__ import annotations

import pandas as pd

from indicators.horizontal_levels import (
    LEVELS_PER_SIDE,
    build_horizontal_levels,
    cluster_swing_levels,
    detect_swing_points,
    filter_levels_by_atr_range,
    generate_round_levels,
)


def _frame_from_high_low(high: list[float], low: list[float]) -> pd.DataFrame:
    close = [(h + l) / 2.0 for h, l in zip(high, low)]
    return pd.DataFrame(
        {
            "open": close,
            "high": high,
            "low": low,
            "close": close,
        }
    )


def test_detect_swing_points_finds_peaks_and_valleys() -> None:
    frame = _frame_from_high_low(
        high=[10.0, 11.0, 15.0, 11.0, 10.0, 12.0, 14.0, 12.0, 10.0],
        low=[9.0, 8.0, 9.0, 8.0, 6.0, 8.0, 9.0, 8.0, 9.0],
    )
    points = detect_swing_points(frame, timeframe="H1", left_bars=1, right_bars=1)

    highs = sorted(float(item["price"]) for item in points if item["kind"] == "high")
    lows = sorted(float(item["price"]) for item in points if item["kind"] == "low")
    assert highs == [14.0, 15.0]
    assert 6.0 in lows


def test_cluster_swing_levels_counts_touch_points() -> None:
    points = [
        {"price": 100.00, "kind": "high", "timeframe": "D1"},
        {"price": 100.10, "kind": "low", "timeframe": "H4"},
        {"price": 99.95, "kind": "high", "timeframe": "H1"},
        {"price": 103.00, "kind": "high", "timeframe": "H1"},
    ]
    clusters = cluster_swing_levels(points, proximity_abs=0.2)

    assert any(item["touch_count"] == 3 for item in clusters)
    assert any(item["touch_count"] == 1 for item in clusters)


def test_generate_round_levels_builds_minor_and_major_steps() -> None:
    levels = generate_round_levels(min_price=2290.0, max_price=2410.0)
    prices = sorted(float(item["price"]) for item in levels)

    assert 2300.0 in prices
    assert 2350.0 in prices
    assert 2400.0 in prices


def test_filter_levels_by_atr_range_excludes_far_levels() -> None:
    levels = [
        {"price": 100.0, "score": 1.0, "source": "round", "timeframe": "ROUND", "touch_count": 1},
        {"price": 107.0, "score": 1.0, "source": "round", "timeframe": "ROUND", "touch_count": 1},
        {"price": 120.0, "score": 1.0, "source": "round", "timeframe": "ROUND", "touch_count": 1},
    ]
    filtered = filter_levels_by_atr_range(levels, current_price=100.0, current_atr=5.0, atr_multiplier=2.0)

    filtered_prices = sorted(float(item["price"]) for item in filtered)
    assert filtered_prices == [100.0, 107.0]


def test_build_horizontal_levels_limits_each_side_to_top_count() -> None:
    empty = pd.DataFrame(columns=["open", "high", "low", "close"])
    levels = build_horizontal_levels(
        d1_frame=empty,
        h4_frame=empty,
        h1_frame=empty,
        current_price=2300.0,
        current_atr=120.0,
    )

    assert 1 <= len(levels["resistances"]) <= LEVELS_PER_SIDE
    assert 1 <= len(levels["supports"]) <= LEVELS_PER_SIDE
    assert all(float(item["price"]) > 2300.0 for item in levels["resistances"])
    assert all(float(item["price"]) < 2300.0 for item in levels["supports"])