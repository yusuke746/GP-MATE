from __future__ import annotations

from dataclasses import dataclass

from config import (
    ATR_MULTIPLIER_SL,
    CONFIDENCE_THRESHOLD,
    CONSECUTIVE_LOSS_LIMIT,
    JPY_USD_RATE_FALLBACK,
    MAX_DAILY_LOSS_PCT,
    RISK_PERCENT,
    RISK_REWARD_RATIO,
    SPREAD_MULTIPLIER_LIMIT,
)

MIN_LOT = 0.01

# Structural SL (suggested_sl) adoption bounds, in ATR units.
# Too close = noise stop; too far = undersized lot and oversized time-risk.
SUGGESTED_SL_MIN_ATR = 1.0
SUGGESTED_SL_MAX_ATR = 2.5


@dataclass(frozen=True)
class FilterCheckResult:
    ok: bool
    reason: str


def calc_lot(
    balance_jpy: float,
    risk_pct: float,
    sl_distance_usd: float,
    contract_size: float = 100.0,
    jpy_usd_rate: float = JPY_USD_RATE_FALLBACK,
) -> float:
    """Calculate lot size from account risk and SL distance.

    Returns minimum MIN_LOT (0.01) for valid inputs, so small accounts can
    still trade; note the effective risk then exceeds risk_pct.
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
    confidence: float,
    spread: float,
    baseline_spread: float,
    is_news_soon: bool,
    consecutive_losses: int,
    daily_loss_pct: float,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
    consecutive_loss_limit: int = CONSECUTIVE_LOSS_LIMIT,
    max_daily_loss_pct: float = MAX_DAILY_LOSS_PCT,
    spread_multiplier_limit: float = SPREAD_MULTIPLIER_LIMIT,
) -> FilterCheckResult:
    """Evaluate risk/safety filters.

    Any failing condition blocks trading.
    """
    if confidence < confidence_threshold:
        return FilterCheckResult(False, "Low confidence")

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
    default_sl: float,
    suggested_sl: float | None,
) -> tuple[float, str]:
    """Adopt the AI's structural SL when it is on the correct side and its
    distance stays within [SUGGESTED_SL_MIN_ATR, SUGGESTED_SL_MAX_ATR] x ATR;
    otherwise keep the ATR-based default."""
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

    distance = abs(entry_price - suggested)
    if not (SUGGESTED_SL_MIN_ATR * atr <= distance <= SUGGESTED_SL_MAX_ATR * atr):
        return default_sl, "fallback_atr"

    return round(suggested, 5), "suggested"


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
) -> dict[str, float | str | bool]:
    """Build lot/SL/TP plan with safe fallback for invalid cases.

    This function never raises and is suitable for production call paths.
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
        default_sl, _ = calc_sl_tp(
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
            default_sl=default_sl,
            suggested_sl=suggested_sl,
        )

        # R (and hence the 2R cap) follows the actual SL distance, so a
        # structural SL keeps risk/reward geometry consistent.
        risk_distance = abs(entry_price - sl)
        if normalized_action == "BUY":
            tp_2r = round(entry_price + risk_distance * rr, 5)
        else:
            tp_2r = round(entry_price - risk_distance * rr, 5)

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

        return {
            "ok": True,
            "action": normalized_action,
            "lot": lot,
            "sl": sl,
            "sl_source": sl_source,
            "tp": final_tp,
            "tp_2r": tp_2r,
            "tp_source": tp_source,
            "reason": "OK",
        }
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
