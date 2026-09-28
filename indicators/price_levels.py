"""ID-tagged price-level candidates and resolution of LLM references to them.

LLMs asked for free-form prices invent levels that were never in the input,
mix bid/ask/close, or pick one point out of a cluster arbitrarily. Instead,
code builds one catalogue of candidate levels per cycle, each with a stable
``level_id`` (e.g. ``H4_SWING_LOW_1``, ``ROUND_4400``, ``H1_FVG_BULLISH_TOP_1``),
and the analysts, the panel chair and the trader refer to levels by id.

``resolve_level`` maps whatever the model returned (an id, or a raw number as
a fallback) onto the catalogue: an id resolves exactly; a raw number is
accepted only when it sits within ``tolerance_atr`` of a candidate, otherwise
it is reported as unanchored and the caller decides (drop the order, log it).
"""

from __future__ import annotations

from typing import Any, Final

LEVEL_TOLERANCE_ATR: Final[float] = 0.3
MAX_DISTANCE_ATR: Final[float] = 6.0
MAX_LEVELS: Final[int] = 40


def _f(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return None if parsed != parsed or parsed <= 0 else parsed


def _entry(level_id: str, price: float, kind: str, timeframe: str, current_price: float, atr: float, **extra: Any) -> dict[str, Any]:
    distance = round((price - current_price) / atr, 2) if atr > 0 and current_price > 0 else None
    return {
        "level_id": level_id,
        "price": round(price, 5),
        "type": kind,
        "timeframe": timeframe,
        "distance_atr": distance,
        "side": "ABOVE" if current_price and price > current_price else ("BELOW" if current_price and price < current_price else "AT"),
        **extra,
    }


def build_price_levels(
    *,
    current_price: float,
    atr: float,
    horizontal_levels: dict[str, Any] | None = None,
    structure: dict[str, Any] | None = None,
    tp_reference: dict[str, Any] | None = None,
    max_distance_atr: float = MAX_DISTANCE_ATR,
) -> list[dict[str, Any]]:
    """Catalogue of candidate levels within ``max_distance_atr`` of the price, sorted by price."""
    levels: list[dict[str, Any]] = []
    counters: dict[str, int] = {}

    def _next(prefix: str) -> str:
        counters[prefix] = counters.get(prefix, 0) + 1
        return f"{prefix}_{counters[prefix]}"

    hl = horizontal_levels if isinstance(horizontal_levels, dict) else {}
    for side, kind in (("supports", "SUPPORT"), ("resistances", "RESISTANCE")):
        entries = hl.get(side, []) if isinstance(hl.get(side), list) else []
        for item in entries:
            if not isinstance(item, dict):
                continue
            price = _f(item.get("price"))
            if price is None:
                continue
            timeframe = str(item.get("timeframe", "") or "ANY").upper()
            source = str(item.get("source", "") or "level").upper()
            levels.append(
                _entry(
                    _next(f"{timeframe}_{source}_{kind}"),
                    price,
                    f"{source}_{kind}",
                    timeframe,
                    current_price,
                    atr,
                    touch_count=int(item.get("touch_count", 0) or 0),
                    score=item.get("score"),
                )
            )

    st = structure if isinstance(structure, dict) else {}
    for tf_key, block in st.items():
        if not isinstance(block, dict):
            continue
        timeframe = str(tf_key).upper()
        swings = block.get("swings") if isinstance(block.get("swings"), dict) else {}
        for price in swings.get("highs", []) or []:
            value = _f(price)
            if value is not None:
                levels.append(_entry(_next(f"{timeframe}_SWING_HIGH"), value, "SWING_HIGH", timeframe, current_price, atr))
        for price in swings.get("lows", []) or []:
            value = _f(price)
            if value is not None:
                levels.append(_entry(_next(f"{timeframe}_SWING_LOW"), value, "SWING_LOW", timeframe, current_price, atr))
        for gap in block.get("fair_value_gaps", []) or []:
            if not isinstance(gap, dict):
                continue
            kind = str(gap.get("type", "") or "GAP").upper()
            top, bottom = _f(gap.get("top")), _f(gap.get("bottom"))
            index = counters.get(f"{timeframe}_FVG_{kind}", 0) + 1
            counters[f"{timeframe}_FVG_{kind}"] = index
            if top is not None:
                levels.append(_entry(f"{timeframe}_FVG_{kind}_TOP_{index}", top, f"FVG_{kind}_TOP", timeframe, current_price, atr, age_bars=gap.get("age_bars"), filled_pct=gap.get("filled_pct")))
            if bottom is not None:
                levels.append(_entry(f"{timeframe}_FVG_{kind}_BOTTOM_{index}", bottom, f"FVG_{kind}_BOTTOM", timeframe, current_price, atr, age_bars=gap.get("age_bars"), filled_pct=gap.get("filled_pct")))

    ref = tp_reference if isinstance(tp_reference, dict) else {}
    prev_day = ref.get("prev_day") if isinstance(ref.get("prev_day"), dict) else {}
    for key, kind in (("high", "PREV_DAY_HIGH"), ("low", "PREV_DAY_LOW")):
        value = _f(prev_day.get(key))
        if value is not None:
            levels.append(_entry(kind, value, kind, "D1", current_price, atr))
    mas = ref.get("moving_averages") if isinstance(ref.get("moving_averages"), dict) else {}
    for key, value in mas.items():
        price = _f(value)
        if price is not None:
            levels.append(_entry(str(key).upper(), price, "MOVING_AVERAGE", str(key).split("_")[0].upper(), current_price, atr))
    for value in ref.get("round_numbers", []) or []:
        price = _f(value)
        if price is not None:
            label = int(price) if float(price).is_integer() else price
            levels.append(_entry(f"ROUND_{label}", price, "ROUND_NUMBER", "ANY", current_price, atr))

    if atr > 0 and current_price > 0:
        levels = [lv for lv in levels if lv["distance_atr"] is not None and abs(lv["distance_atr"]) <= max_distance_atr]
    levels.sort(key=lambda lv: lv["price"])
    if len(levels) > MAX_LEVELS:
        # Keep the ones closest to the price; far levels matter least for execution.
        levels = sorted(levels, key=lambda lv: abs(lv["distance_atr"] or 0.0))[:MAX_LEVELS]
        levels.sort(key=lambda lv: lv["price"])
    return levels


def levels_by_id(levels: list[dict[str, Any]] | None) -> dict[str, dict[str, Any]]:
    return {str(lv.get("level_id")): lv for lv in (levels or []) if isinstance(lv, dict) and lv.get("level_id")}


def resolve_level(
    levels: list[dict[str, Any]] | None,
    value: Any,
    *,
    atr: float | None = None,
    tolerance_atr: float = LEVEL_TOLERANCE_ATR,
) -> dict[str, Any]:
    """Map an id or a raw price onto the catalogue.

    -> {"price": float|None, "level_id": str|None, "anchored": bool, "reason": str}
    reason: "id" (exact), "nearest" (raw price within tolerance), "unanchored"
    (raw price with no candidate near it), "unknown_id", "missing", or
    "no_catalogue" (raw price, nothing to anchor to: accepted as given but
    ``anchored`` stays False so callers can tell).
    """
    catalogue = levels_by_id(levels)
    if value is None or (isinstance(value, str) and not value.strip()):
        return {"price": None, "level_id": None, "anchored": False, "reason": "missing"}
    if isinstance(value, str) and not _is_number(value):
        entry = catalogue.get(value.strip())
        if entry is None:
            return {"price": None, "level_id": value.strip(), "anchored": False, "reason": "unknown_id"}
        return {"price": float(entry["price"]), "level_id": entry["level_id"], "anchored": True, "reason": "id"}
    price = _f(value)
    if price is None:
        return {"price": None, "level_id": None, "anchored": False, "reason": "missing"}
    if not catalogue:
        return {"price": price, "level_id": None, "anchored": False, "reason": "no_catalogue"}
    nearest = min(catalogue.values(), key=lambda lv: abs(float(lv["price"]) - price))
    gap = abs(float(nearest["price"]) - price)
    limit = tolerance_atr * float(atr) if atr and atr > 0 else None
    if limit is not None and gap <= limit:
        return {"price": float(nearest["price"]), "level_id": nearest["level_id"], "anchored": True, "reason": "nearest"}
    return {"price": price, "level_id": None, "anchored": False, "reason": "unanchored"}


def accepted(resolved: dict[str, Any]) -> bool:
    """True when the reference yielded a usable price (anchored, or raw with no catalogue)."""
    return resolved.get("price") is not None and (bool(resolved.get("anchored")) or resolved.get("reason") == "no_catalogue")


def _is_number(text: str) -> bool:
    try:
        float(text)
    except ValueError:
        return False
    return True
