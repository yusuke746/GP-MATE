from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Final, Literal, TypedDict

import pandas as pd


SWING_LEFT_BARS: Final[int] = 2
SWING_RIGHT_BARS: Final[int] = 2
CLUSTER_PROXIMITY_ATR_MULTIPLIER: Final[float] = 0.25
ATR_RANGE_MULTIPLIER: Final[float] = 3.0
ROUND_STEP_MINOR: Final[float] = 50.0
ROUND_STEP_MAJOR: Final[float] = 100.0
ROUND_LEVEL_BONUS: Final[float] = 1.2
LEVELS_PER_SIDE: Final[int] = 3

TIMEFRAME_WEIGHTS: Final[dict[str, float]] = {
    "D1": 3.0,
    "H4": 2.0,
    "H1": 1.0,
    "ROUND": 1.0,
}


class SwingPoint(TypedDict):
    price: float
    kind: Literal["high", "low"]
    timeframe: str


class HorizontalLevelEntry(TypedDict):
    price: float
    score: float
    source: Literal["swing", "cluster", "round"]
    timeframe: str
    touch_count: int


class HorizontalLevels(TypedDict):
    resistances: list[HorizontalLevelEntry]
    supports: list[HorizontalLevelEntry]


@dataclass(frozen=True)
class _LevelCandidate:
    price: float
    score: float
    source: Literal["swing", "cluster", "round"]
    timeframe: str
    touch_count: int


def _safe_float(value: object, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def detect_swing_points(
    frame: pd.DataFrame,
    timeframe: str,
    left_bars: int = SWING_LEFT_BARS,
    right_bars: int = SWING_RIGHT_BARS,
) -> list[SwingPoint]:
    if frame.empty or left_bars <= 0 or right_bars <= 0:
        return []
    if "high" not in frame.columns or "low" not in frame.columns:
        return []

    highs = frame["high"].tolist()
    lows = frame["low"].tolist()
    points: list[SwingPoint] = []
    start = left_bars
    end = len(frame) - right_bars

    for idx in range(start, end):
        center_high = _safe_float(highs[idx], 0.0)
        center_low = _safe_float(lows[idx], 0.0)

        left_highs = [_safe_float(v, 0.0) for v in highs[idx - left_bars : idx]]
        right_highs = [_safe_float(v, 0.0) for v in highs[idx + 1 : idx + right_bars + 1]]
        if left_highs and right_highs and center_high > max(left_highs) and center_high > max(right_highs):
            points.append(
                {
                    "price": round(center_high, 5),
                    "kind": "high",
                    "timeframe": timeframe,
                }
            )

        left_lows = [_safe_float(v, 0.0) for v in lows[idx - left_bars : idx]]
        right_lows = [_safe_float(v, 0.0) for v in lows[idx + 1 : idx + right_bars + 1]]
        if left_lows and right_lows and center_low < min(left_lows) and center_low < min(right_lows):
            points.append(
                {
                    "price": round(center_low, 5),
                    "kind": "low",
                    "timeframe": timeframe,
                }
            )

    return points


def cluster_swing_levels(
    swing_points: list[SwingPoint],
    proximity_abs: float,
) -> list[HorizontalLevelEntry]:
    if proximity_abs <= 0 or not swing_points:
        return []

    sorted_points = sorted(swing_points, key=lambda item: float(item["price"]))
    clusters: list[list[SwingPoint]] = []
    current_cluster: list[SwingPoint] = []

    for point in sorted_points:
        if not current_cluster:
            current_cluster = [point]
            continue

        current_avg = sum(float(p["price"]) for p in current_cluster) / len(current_cluster)
        if abs(float(point["price"]) - current_avg) <= proximity_abs:
            current_cluster.append(point)
            continue

        clusters.append(current_cluster)
        current_cluster = [point]

    if current_cluster:
        clusters.append(current_cluster)

    entries: list[HorizontalLevelEntry] = []
    for cluster in clusters:
        prices = [float(p["price"]) for p in cluster]
        avg_price = sum(prices) / len(prices)
        tf_counter = Counter(str(p["timeframe"]) for p in cluster)
        dominant_tf = tf_counter.most_common(1)[0][0] if tf_counter else "H1"
        touch_count = len(cluster)
        weight = TIMEFRAME_WEIGHTS.get(dominant_tf, 1.0)
        score = weight * touch_count
        entries.append(
            {
                "price": round(avg_price, 5),
                "score": round(score, 5),
                "source": "cluster",
                "timeframe": dominant_tf,
                "touch_count": touch_count,
            }
        )

    return entries


def generate_round_levels(
    min_price: float,
    max_price: float,
    step_minor: float = ROUND_STEP_MINOR,
    step_major: float = ROUND_STEP_MAJOR,
) -> list[HorizontalLevelEntry]:
    if min_price <= 0 or max_price <= 0 or max_price < min_price:
        return []
    if step_minor <= 0 or step_major <= 0:
        return []

    start = int(min_price // step_minor)
    end = int(max_price // step_minor)
    entries: list[HorizontalLevelEntry] = []

    for idx in range(start, end + 1):
        price = round(idx * step_minor, 5)
        if price < min_price or price > max_price:
            continue
        is_major = abs((price / step_major) - round(price / step_major)) < 1e-9
        major_boost = 1.2 if is_major else 1.0
        entries.append(
            {
                "price": price,
                "score": round(major_boost * ROUND_LEVEL_BONUS, 5),
                "source": "round",
                "timeframe": "ROUND",
                "touch_count": 1,
            }
        )

    return entries


def filter_levels_by_atr_range(
    levels: list[HorizontalLevelEntry],
    current_price: float,
    current_atr: float,
    atr_multiplier: float = ATR_RANGE_MULTIPLIER,
) -> list[HorizontalLevelEntry]:
    if current_price <= 0 or current_atr <= 0 or atr_multiplier <= 0:
        return []

    radius = current_atr * atr_multiplier
    lower = current_price - radius
    upper = current_price + radius
    return [
        item
        for item in levels
        if lower <= float(item["price"]) <= upper
    ]


def _to_swing_level(point: SwingPoint) -> HorizontalLevelEntry:
    timeframe = str(point["timeframe"])
    weight = TIMEFRAME_WEIGHTS.get(timeframe, 1.0)
    return {
        "price": round(float(point["price"]), 5),
        "score": round(weight, 5),
        "source": "swing",
        "timeframe": timeframe,
        "touch_count": 1,
    }


def _deduplicate_candidates(
    candidates: list[HorizontalLevelEntry],
    proximity_abs: float,
) -> list[HorizontalLevelEntry]:
    if not candidates:
        return []
    if proximity_abs <= 0:
        return candidates

    selected: list[HorizontalLevelEntry] = []
    for candidate in sorted(candidates, key=lambda item: (float(item["score"]), -float(item["touch_count"])), reverse=True):
        if any(abs(float(candidate["price"]) - float(chosen["price"])) <= proximity_abs for chosen in selected):
            continue
        selected.append(candidate)
    return selected


def _top_side_levels(
    filtered: list[HorizontalLevelEntry],
    current_price: float,
    side: Literal["resistance", "support"],
) -> list[HorizontalLevelEntry]:
    if side == "resistance":
        side_levels = [item for item in filtered if float(item["price"]) > current_price]
        side_levels.sort(key=lambda item: (-float(item["score"]), float(item["price"])))
    else:
        side_levels = [item for item in filtered if float(item["price"]) < current_price]
        side_levels.sort(key=lambda item: (-float(item["score"]), -float(item["price"])))

    return side_levels[:LEVELS_PER_SIDE]


def build_horizontal_levels(
    d1_frame: pd.DataFrame,
    h4_frame: pd.DataFrame,
    h1_frame: pd.DataFrame,
    current_price: float,
    current_atr: float,
) -> HorizontalLevels:
    safe_empty: HorizontalLevels = {
        "resistances": [],
        "supports": [],
    }

    if current_price <= 0 or current_atr <= 0:
        return safe_empty

    try:
        swing_points: list[SwingPoint] = []
        swing_points.extend(detect_swing_points(d1_frame, "D1"))
        swing_points.extend(detect_swing_points(h4_frame, "H4"))
        swing_points.extend(detect_swing_points(h1_frame, "H1"))

        proximity_abs = max(current_atr * CLUSTER_PROXIMITY_ATR_MULTIPLIER, 0.01)
        swing_levels = [_to_swing_level(point) for point in swing_points]
        cluster_levels = cluster_swing_levels(swing_points=swing_points, proximity_abs=proximity_abs)

        radius = current_atr * ATR_RANGE_MULTIPLIER
        round_levels = generate_round_levels(
            min_price=max(current_price - radius, 0.01),
            max_price=current_price + radius,
        )

        all_candidates = swing_levels + cluster_levels + round_levels
        filtered = filter_levels_by_atr_range(
            levels=all_candidates,
            current_price=current_price,
            current_atr=current_atr,
            atr_multiplier=ATR_RANGE_MULTIPLIER,
        )
        deduplicated = _deduplicate_candidates(filtered, proximity_abs=proximity_abs * 0.5)

        resistances = _top_side_levels(deduplicated, current_price=current_price, side="resistance")
        supports = _top_side_levels(deduplicated, current_price=current_price, side="support")

        return {
            "resistances": resistances,
            "supports": supports,
        }
    except Exception:
        return safe_empty