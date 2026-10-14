from __future__ import annotations

from dataclasses import dataclass

from config import (
    ATR_MULTIPLIER_SL,
    CONSECUTIVE_LOSS_LIMIT,
    JPY_USD_RATE_FALLBACK,
    MAX_DAILY_LOSS_PCT,
    MAX_RISK_PERCENT,
    MIN_RISK_REWARD_RATIO,
    RISK_PERCENT,
    RISK_REWARD_RATIO,
    SL_STRUCTURE_BUFFER_USD,
    SPREAD_MULTIPLIER_LIMIT,
)

MIN_LOT = 0.01
CONTRACT_SIZE = 100.0

# Trade horizon declared by the trader. It selects the ATR every geometry
# rule (SL box, 2R TP cap, pending-trigger distance) is measured in, so a
# thesis built on the D1/H4 flow is not executed with an H1-sized stop.
HORIZON_INTRADAY = "INTRADAY"
HORIZON_SWING = "SWING"
TRADE_HORIZONS = (HORIZON_INTRADAY, HORIZON_SWING)
HORIZON_TIMEFRAME = {HORIZON_INTRADAY: "H1", HORIZON_SWING: "H4"}


def normalize_horizon(value: object) -> str:
    """INTRADAY unless the trader explicitly said SWING."""
    text = str(value or "").upper().strip()
    return text if text in TRADE_HORIZONS else HORIZON_INTRADAY


def select_atr(horizon: object, atr_h1: float, atr_h4: float) -> tuple[float, str]:
    """-> (atr, timeframe) the plan is measured in.

    SWING uses the H4 ATR; INTRADAY the H1 ATR. A missing/invalid H4 ATR
    falls back to H1 (reported as "H1" so the log shows what was used).
    """
    try:
        h1 = float(atr_h1 or 0.0)
    except (TypeError, ValueError):
        h1 = 0.0
    try:
        h4 = float(atr_h4 or 0.0)
    except (TypeError, ValueError):
        h4 = 0.0
    if normalize_horizon(horizon) == HORIZON_SWING and h4 > 0:
        return h4, "H4"
    return h1, "H1"


def realised_risk_pct(
    lot: float,
    sl_distance_usd: float,
    balance_jpy: float,
    jpy_usd_rate: float,
    contract_size: float = CONTRACT_SIZE,
) -> float:
    """Fraction of the balance lost if the SL is hit at this lot (0.0 when unknown)."""
    try:
        if balance_jpy <= 0 or jpy_usd_rate <= 0 or lot <= 0 or sl_distance_usd <= 0:
            return 0.0
        loss_jpy = lot * sl_distance_usd * contract_size * jpy_usd_rate
        return round(loss_jpy / balance_jpy, 6)
    except (TypeError, ValueError):
        return 0.0

# Structural SL (suggested_sl) minimum distance in ATR units (noise-stop guard).
# The maximum is the ATR default itself (atr * atr_mult): suggested values may
# only tighten the baseline, never widen it.
SUGGESTED_SL_MIN_ATR = 1.0

# Pending-order placement guards: the trigger level must be a meaningful
# distance from the current price (in ATR units of the trade horizon), but
# still reachable.
PENDING_MIN_DISTANCE_ATR = 0.1
PENDING_MAX_DISTANCE_ATR = 3.0


def execution_geometry(
    current_price: float,
    atr_h1: float,
    atr_h4: float,
    spread_usd: float | None = None,
    atr_mult: float = ATR_MULTIPLIER_SL,
    rr: float = RISK_REWARD_RATIO,
    min_rr: float = MIN_RISK_REWARD_RATIO,
    structure_buffer_usd: float = SL_STRUCTURE_BUFFER_USD,
) -> dict[str, object]:
    """The distance rules build_risk_plan will apply, in USD, per trade horizon.

    Handed to the trader as facts so a plan can be checked against the box
    before it is proposed (a SWING plan with an intraday-sized stop and a
    near target was being rejected for low RR after the fact).
    """
    try:
        spread = float(spread_usd) if spread_usd is not None and float(spread_usd) > 0 else 0.0
    except (TypeError, ValueError):
        spread = 0.0

    def _for(atr: float, timeframe: str) -> dict[str, object]:
        if atr <= 0:
            return {"atr_timeframe": timeframe, "atr": None, "available": False}
        sl_min = round(SUGGESTED_SL_MIN_ATR * atr, 2)
        sl_default = round(atr_mult * atr, 2)

        def _min_tp(sl_distance: float) -> float:
            # (tp - spread) / (sl + spread) >= min_rr
            return round(min_rr * (sl_distance + spread) + spread, 2)

        return {
            "atr_timeframe": timeframe,
            "atr": round(atr, 2),
            "available": True,
            "sl_distance_usd": {
                "min": sl_min,
                "max": sl_default,
                "default_when_suggested_sl_outside_box": sl_default,
                "structure_buffer_added_to_suggested_sl": round(structure_buffer_usd, 2),
            },
            "tp_distance_usd": {
                "max_2r_on_default_sl": round(rr * sl_default, 2),
                "min_for_rr_on_default_sl": _min_tp(sl_default),
                "min_for_rr_on_tightest_sl": _min_tp(sl_min),
            },
            "pending_trigger_distance_usd": {
                "min": round(PENDING_MIN_DISTANCE_ATR * atr, 2),
                "max": round(PENDING_MAX_DISTANCE_ATR * atr, 2),
            },
        }

    return {
        "current_price": round(float(current_price), 2) if current_price else None,
        "spread_usd": round(spread, 2),
        "min_rr_after_spread": min_rr,
        "by_horizon": {
            "INTRADAY": _for(float(atr_h1 or 0.0), "H1"),
            "SWING": _for(float(atr_h4 or 0.0), "H4"),
        },
    }


@dataclass(frozen=True)
class FilterCheckResult:
    ok: bool
    reason: str


def calc_lot(
    balance_jpy: float,
    risk_pct: float,
    sl_distance_usd: float,
    contract_size: float = CONTRACT_SIZE,
    jpy_usd_rate: float = JPY_USD_RATE_FALLBACK,
) -> float:
    """Calculate lot size from account risk and SL distance.

    Returns minimum MIN_LOT (0.01) for valid inputs, so small accounts can
    still trade; note the effective risk then exceeds risk_pct (build_risk_plan
    caps that overshoot at MAX_RISK_PERCENT).
    Returns 0.0 when inputs are invalid to force a no-trade decision.
    """
    if balance_jpy <= 0 or risk_pct <= 0 or sl_distance_usd <= 0 or contract_size <= 0:
        return 0.0

    if jpy_usd_rate <= 0:
        return 0.0

    risk_amount_usd = (balance_jpy / jpy_usd_rate) * risk_pct
    loss_per_lot = sl_distance_usd * contract_size

    if loss_per_lot <= 0:
        return 0.0

    lot = risk_amount_usd / loss_per_lot
    if lot <= 0:
        return 0.0

    return max(MIN_LOT, round(lot, 2))


def calc_sl_tp(
    entry_price: float,
    atr: float,
    action: str,
    atr_mult: float = ATR_MULTIPLIER_SL,
    rr: float = RISK_REWARD_RATIO,
) -> tuple[float, float]:
    """Calculate ATR-based SL/TP for BUY or SELL.

    Raises ValueError for invalid inputs.
    """
    normalized_action = action.upper().strip()
    if normalized_action not in {"BUY", "SELL"}:
        raise ValueError(f"Unsupported action: {action}")

    if entry_price <= 0:
        raise ValueError("entry_price must be positive")
    if atr <= 0:
        raise ValueError("atr must be positive")
    if atr_mult <= 0:
        raise ValueError("atr_mult must be positive")
    if rr <= 0:
        raise ValueError("rr must be positive")

    sl_distance = atr * atr_mult
    tp_distance = sl_distance * rr

    if normalized_action == "BUY":
        sl = entry_price - sl_distance
        tp = entry_price + tp_distance
    else:
        sl = entry_price + sl_distance
        tp = entry_price - tp_distance

    return round(sl, 5), round(tp, 5)


def check_filters(
    confidence: float | None = None,  # deprecated: ignored (the trader's action stands)
    spread: float = -1.0,
    baseline_spread: float = 0.0,
    is_news_soon: bool = False,
    consecutive_losses: int = 0,
    daily_loss_pct: float = 0.0,
    consecutive_loss_limit: int = CONSECUTIVE_LOSS_LIMIT,
    max_daily_loss_pct: float = MAX_DAILY_LOSS_PCT,
    spread_multiplier_limit: float = SPREAD_MULTIPLIER_LIMIT,
) -> FilterCheckResult:
    """Evaluate risk/safety filters (spread, news window, losing streak, daily loss).

    Any failing condition blocks trading. There is no confidence gate: the
    self-reported number never predicted anything, so the decision maker's
    action is taken at face value and only market/risk facts can block it.
    """
    _ = confidence
    if spread < 0:
        return FilterCheckResult(False, "Spread unavailable")

    if baseline_spread <= 0:
        return FilterCheckResult(False, "Baseline spread unavailable")

    if spread > baseline_spread * spread_multiplier_limit:
        return FilterCheckResult(False, "Spread too high")

    if is_news_soon:
        return FilterCheckResult(False, "High impact news window")

    if consecutive_losses >= consecutive_loss_limit:
        return FilterCheckResult(False, "Consecutive loss limit reached")

    if daily_loss_pct >= max_daily_loss_pct:
        return FilterCheckResult(False, "Daily loss limit reached")

    return FilterCheckResult(True, "OK")


def _resolve_structural_sl(
    action: str,
    entry_price: float,
    atr: float,
    atr_mult: float,
    default_sl: float,
    suggested_sl: float | None,
    structure_buffer_usd: float = SL_STRUCTURE_BUFFER_USD,
) -> tuple[float, str]:
    """Adopt the AI's structural SL only when it TIGHTENS the ATR baseline.

    suggested_sl is treated as the structural LEVEL itself (e.g. the support
    cluster price); the stop is placed a fixed buffer BEYOND it so a wick
    through the level does not stop the trade out before the thesis is
    actually invalidated. The buffered distance must land inside
    [SUGGESTED_SL_MIN_ATR x ATR, atr * atr_mult]; anything wider (or
    noise-tight) falls back to the ATR baseline. Mirrors the suggested_tp
    rule: AI suggestions may only shrink the geometry, never expand it.
    """
    try:
        suggested = float(suggested_sl) if suggested_sl is not None else None
    except Exception:
        suggested = None

    if suggested is None or suggested <= 0 or atr <= 0:
        return default_sl, "fallback_atr"

    if action == "BUY" and suggested >= entry_price:
        return default_sl, "fallback_atr"
    if action == "SELL" and suggested <= entry_price:
        return default_sl, "fallback_atr"

    buffer = max(structure_buffer_usd, 0.0)
    buffered = suggested - buffer if action == "BUY" else suggested + buffer

    distance = abs(entry_price - buffered)
    if not (SUGGESTED_SL_MIN_ATR * atr <= distance <= atr_mult * atr):
        return default_sl, "fallback_atr"

    return round(buffered, 5), "suggested"


def build_risk_plan(
    action: str,
    entry_price: float,
    atr: float,
    balance_jpy: float,
    suggested_tp: float | None = None,
    suggested_sl: float | None = None,
    risk_pct: float = RISK_PERCENT,
    atr_mult: float = ATR_MULTIPLIER_SL,
    rr: float = RISK_REWARD_RATIO,
    jpy_usd_rate: float | None = None,
    min_rr: float = MIN_RISK_REWARD_RATIO,
    spread_usd: float = 0.0,
    max_risk_pct: float = MAX_RISK_PERCENT,
) -> dict[str, float | str | bool]:
    """Build lot/SL/TP plan with safe fallback for invalid cases.

    This function never raises and is suitable for production call paths.

    ``atr`` is whatever ATR the caller's trade horizon is measured in (H1 for
    INTRADAY, H4 for SWING); every distance rule below scales with it.

    The lot targets ``risk_pct`` of the balance. Because the broker's 0.01-lot
    floor can make a wide stop risk more than that, the realised risk
    (``effective_risk_pct``) is reported and the plan is rejected with
    ``reason: "risk_cap"`` when it exceeds ``max_risk_pct``; up to the cap
    the overshoot is accepted.

    After SL and TP are final, the reward/risk NET of the current spread is
    checked against ``min_rr``: crossing the spread costs on entry and again
    on exit, so net_rr = (tp_distance - spread) / (sl_distance + spread).
    ``gross_rr`` (no spread) is reported alongside. A plan below ``min_rr``
    returns ``ok: False, reason: "low_rr"`` with the SL/TP left exactly as
    computed (never re-shaped) so the caller can log the geometry the AI
    actually proposed. ``spread_usd`` <= 0 or invalid means "no spread
    information", in which case net equals gross.
    """
    normalized_action = action.upper().strip()
    if normalized_action not in {"BUY", "SELL"}:
        return {
            "ok": False,
            "action": "HOLD",
            "lot": 0.0,
            "sl": 0.0,
            "tp": 0.0,
            "reason": "Invalid action",
        }

    try:
        # The baseline geometry is always ATR-based: SL = ATR x atr_mult and
        # TP = that distance x rr. AI suggestions (suggested_sl/suggested_tp)
        # may only tighten this box, never widen it.
        default_sl, tp_2r = calc_sl_tp(
            entry_price=entry_price,
            atr=atr,
            action=normalized_action,
            atr_mult=atr_mult,
            rr=rr,
        )

        sl, sl_source = _resolve_structural_sl(
            action=normalized_action,
            entry_price=entry_price,
            atr=atr,
            atr_mult=atr_mult,
            default_sl=default_sl,
            suggested_sl=suggested_sl,
        )

        # Default to legacy 2R TP when AI suggestion is unavailable/invalid.
        final_tp = tp_2r
        tp_source = "fallback_2r"

        suggested_tp_value: float | None
        try:
            suggested_tp_value = float(suggested_tp) if suggested_tp is not None else None
        except Exception:
            suggested_tp_value = None

        if suggested_tp_value is not None:
            is_direction_valid = True
            if normalized_action == "BUY" and suggested_tp_value <= entry_price:
                is_direction_valid = False
            elif normalized_action == "SELL" and suggested_tp_value >= entry_price:
                is_direction_valid = False

            if is_direction_valid:
                if normalized_action == "BUY":
                    capped_tp = min(suggested_tp_value, tp_2r)
                else:
                    capped_tp = max(suggested_tp_value, tp_2r)

                final_tp = round(capped_tp, 5)
                tp_source = "suggested" if final_tp == round(suggested_tp_value, 5) else "suggested_capped_2r"

        sl_distance = abs(entry_price - sl)
        tp_distance = abs(final_tp - entry_price)
        try:
            spread_cost = float(spread_usd) if spread_usd is not None and float(spread_usd) > 0 else 0.0
        except (TypeError, ValueError):
            spread_cost = 0.0
        gross_rr = round(tp_distance / sl_distance, 4) if sl_distance > 0 else 0.0
        net_rr = round((tp_distance - spread_cost) / (sl_distance + spread_cost), 4) if sl_distance + spread_cost > 0 else 0.0
        effective_rr = net_rr
        if effective_rr < min_rr:
            # The AI's TP is its reasoning; the SL was widened by the floor.
            # Sending that combination would be a different trade from the
            # one the AI judged, so skip and report instead of moving either.
            return {
                "ok": False,
                "action": "HOLD",
                "lot": 0.0,
                "sl": sl,
                "sl_source": sl_source,
                "tp": final_tp,
                "tp_2r": tp_2r,
                "tp_source": tp_source,
                "effective_rr": effective_rr,
                "gross_rr": gross_rr,
                "spread_cost": round(spread_cost, 5),
                "net_rr": net_rr,
                "rr_rejection_reason": "net_rr_below_min" if gross_rr >= min_rr else "gross_rr_below_min",
                "reason": "low_rr",
            }

        effective_rate = jpy_usd_rate if jpy_usd_rate is not None and jpy_usd_rate > 0 else JPY_USD_RATE_FALLBACK
        lot = calc_lot(
            balance_jpy=balance_jpy,
            risk_pct=risk_pct,
            sl_distance_usd=sl_distance,
            jpy_usd_rate=effective_rate,
        )

        if lot <= 0:
            return {
                "ok": False,
                "action": "HOLD",
                "lot": 0.0,
                "sl": 0.0,
                "tp": 0.0,
                "tp_2r": 0.0,
                "tp_source": "fallback_2r",
                "reason": "Lot calculation failed",
            }

        effective_risk = realised_risk_pct(lot, sl_distance, balance_jpy, effective_rate)
        plan: dict[str, float | str | bool] = {
            "ok": True,
            "action": normalized_action,
            "lot": lot,
            "sl": sl,
            "sl_source": sl_source,
            "tp": final_tp,
            "tp_2r": tp_2r,
            "tp_source": tp_source,
            "effective_rr": effective_rr,
            "gross_rr": gross_rr,
            "spread_cost": round(spread_cost, 5),
            "net_rr": net_rr,
            "rr_rejection_reason": "",
            "effective_risk_pct": effective_risk,
            "reason": "OK",
        }
        if max_risk_pct > 0 and effective_risk > max_risk_pct:
            # The minimum lot already risks more than the cap: the stop is
            # too wide for this balance, so the trade is reported, not sent.
            plan.update({"ok": False, "action": "HOLD", "reason": "risk_cap"})
        return plan
    except Exception as exc:
        return {
            "ok": False,
            "action": "HOLD",
            "lot": 0.0,
            "sl": 0.0,
            "tp": 0.0,
            "tp_2r": 0.0,
            "tp_source": "fallback_2r",
            "reason": str(exc),
        }
