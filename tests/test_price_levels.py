from __future__ import annotations

from indicators.price_levels import accepted, build_price_levels, levels_by_id, resolve_level


def _catalogue() -> list[dict]:
    return build_price_levels(
        current_price=4350.0,
        atr=20.0,
        horizontal_levels={
            "supports": [{"price": 4300.0, "score": 5.0, "source": "cluster", "timeframe": "H4", "touch_count": 3}],
            "resistances": [{"price": 4390.0, "score": 4.0, "source": "swing", "timeframe": "D1", "touch_count": 2}],
        },
        structure={
            "h4": {"swings": {"highs": [4380.0, 4395.0], "lows": [4310.0]}, "fair_value_gaps": [{"type": "BULLISH", "top": 4335.0, "bottom": 4325.0, "age_bars": 3, "filled_pct": 0.2}]},
            "h1": {},
        },
        tp_reference={"prev_day": {"high": 4400.0, "low": 4290.0}, "moving_averages": {"h4_ma20": 4340.0, "d1_ma200": None}, "round_numbers": [4300.0, 4350.0, 4400.0, 4700.0]},
    )


def test_build_price_levels_tags_ids_types_distances_and_filters_far_levels() -> None:
    levels = _catalogue()
    by_id = levels_by_id(levels)
    assert by_id["H4_CLUSTER_SUPPORT_1"]["price"] == 4300.0 and by_id["H4_CLUSTER_SUPPORT_1"]["touch_count"] == 3
    assert by_id["D1_SWING_RESISTANCE_1"]["type"] == "SWING_RESISTANCE"
    assert by_id["H4_SWING_HIGH_2"]["price"] == 4395.0 and by_id["H4_SWING_LOW_1"]["side"] == "BELOW"
    assert by_id["H4_FVG_BULLISH_TOP_1"]["price"] == 4335.0 and by_id["H4_FVG_BULLISH_BOTTOM_1"]["filled_pct"] == 0.2
    assert by_id["PREV_DAY_HIGH"]["distance_atr"] == 2.5 and by_id["PREV_DAY_LOW"]["distance_atr"] == -3.0
    assert by_id["H4_MA20"]["type"] == "MOVING_AVERAGE" and "D1_MA200" not in by_id
    assert by_id["ROUND_4350"]["side"] == "AT"
    assert "ROUND_4700" not in by_id  # 17.5 ATR away: beyond MAX_DISTANCE_ATR
    assert [lv["price"] for lv in levels] == sorted(lv["price"] for lv in levels)


def test_resolve_level_by_id_by_nearest_price_and_rejections() -> None:
    levels = _catalogue()
    exact = resolve_level(levels, "H4_SWING_LOW_1", atr=20.0)
    assert exact == {"price": 4310.0, "level_id": "H4_SWING_LOW_1", "anchored": True, "reason": "id"}
    snapped = resolve_level(levels, 4392.5, atr=20.0)  # 2.5 away from 4390 / 4395: within 0.3 ATR (6.0)
    assert snapped["anchored"] and snapped["level_id"] in {"D1_SWING_RESISTANCE_1", "H4_SWING_HIGH_2"} and snapped["reason"] == "nearest"
    far = resolve_level(levels, 4365.0, atr=20.0)  # nearest is 4350 / 4380: 10+ away
    assert far == {"price": 4365.0, "level_id": None, "anchored": False, "reason": "unanchored"}
    assert resolve_level(levels, "NOT_A_LEVEL", atr=20.0)["reason"] == "unknown_id"
    assert resolve_level(levels, None, atr=20.0)["reason"] == "missing"
    assert resolve_level(levels, "4310", atr=20.0)["reason"] == "id" or resolve_level(levels, "4310", atr=20.0)["reason"] == "nearest"
    no_catalogue = resolve_level([], 4365.0, atr=20.0)
    assert no_catalogue["reason"] == "no_catalogue" and accepted(no_catalogue) and not accepted(far)
    assert not accepted(resolve_level(levels, 4365.0, atr=None))  # no ATR -> no tolerance -> unanchored


def test_build_price_levels_is_safe_on_missing_inputs() -> None:
    assert build_price_levels(current_price=0.0, atr=0.0) == []
    levels = build_price_levels(current_price=100.0, atr=1.0, horizontal_levels="junk", structure={"h1": "junk"}, tp_reference={"prev_day": {"high": "x"}})  # type: ignore[arg-type]
    assert levels == []
