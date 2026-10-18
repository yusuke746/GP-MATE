from __future__ import annotations

from typing import Any


def should_move_to_breakeven(
    entry: Any,
    initial_sl: Any,
    current_price: Any,
    current_sl: Any,
    side: str,
    buffer: Any,
) -> tuple[bool, float | None]:
    """Return whether to move SL to breakeven and the new SL price.

    Rule:
    - Trigger at +1R in profit (R=abs(entry-initial_sl)).
    - BUY: current_price >= entry + R => new_sl = entry + buffer.
    - SELL: current_price <= entry - R => new_sl = entry - buffer.
    - One-time move only:
      BUY skips when current_sl >= new_sl, SELL skips when current_sl <= new_sl.
    - Never move in an unfavorable direction from current SL.
    """

    normalized_side = str(side or "").strip().upper()
    if normalized_side not in {"BUY", "SELL"}:
        return False, None

    try:
        entry_f = float(entry)
        initial_sl_f = float(initial_sl)
        current_price_f = float(current_price)
        current_sl_f = float(current_sl)
        buffer_f = float(buffer)
    except Exception:
        return False, None

    if entry_f <= 0 or initial_sl_f <= 0 or current_price_f <= 0 or current_sl_f <= 0:
        return False, None
    if buffer_f < 0:
        return False, None

    risk_r = abs(entry_f - initial_sl_f)
    if risk_r <= 0:
        return False, None

    if normalized_side == "BUY":
        trigger_price = entry_f + risk_r
        if current_price_f < trigger_price:
            return False, None

        new_sl = round(entry_f + buffer_f, 5)

        # Already moved (or better) => no second update.
        if current_sl_f >= new_sl:
            return False, None

        # Defensive guard: never move SL to a worse level.
        if new_sl < current_sl_f:
            return False, None

        return True, new_sl

    trigger_price = entry_f - risk_r
    if current_price_f > trigger_price:
        return False, None

    new_sl = round(entry_f - buffer_f, 5)

    # Already moved (or better) => no second update.
    if current_sl_f <= new_sl:
        return False, None

    # Defensive guard: never move SL to a worse level.
    if new_sl > current_sl_f:
        return False, None

    return True, new_sl

MIN_LOT = 0.01
LOT_STEP = 0.01


def partial_close_volume(volume: Any, fraction: Any, min_lot: float = MIN_LOT, step: float = LOT_STEP) -> float:
    """Volume to close at the 1R trigger so the rest can run to the target.

    ``fraction`` of the position, floored to the lot step. 0.0 when the
    feature is off (fraction <= 0), when the position cannot be split (the
    remainder would fall below ``min_lot``), or on invalid input.
    """
    try:
        volume_f = float(volume)
        fraction_f = float(fraction)
    except Exception:
        return 0.0
    if volume_f <= 0 or fraction_f <= 0:
        return 0.0
    fraction_f = min(fraction_f, 1.0)
    steps = int(volume_f * fraction_f / step + 1e-9)
    part = round(steps * step, 2)
    if part < min_lot:
        return 0.0
    remainder = round(volume_f - part, 2)
    if remainder < min_lot:
        # Closing this much would leave a sub-minimum remainder; take one
        # step less if that still leaves a legal split, else do not split.
        part = round(part - step, 2)
        if part < min_lot or round(volume_f - part, 2) < min_lot:
            return 0.0
    return part
