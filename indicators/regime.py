"""Deterministic market-regime classifier (TREND / RANGE / TRANSITION).

The debate and the trader kept arguing direction (bull vs bear) while the
decision that actually mattered was *which kind of market this is*: does a
pullback get bought (trend continuation) or does the edge of the band get
faded (range)? This module answers that from indicators alone so the LLM
layers start from a shared, auditable premise instead of re-deriving it.

Inputs come from the technical report's ``direction_context`` (per-timeframe
ADX / Bollinger / ATR snapshots produced by main._extract_latest_features)
plus the multi-timeframe trend labels. Output is a small dict that is safe
to attach to the technical report, hand to the debate as ``regime_hint`` and
to the forecaster / trader as context.
"""

from __future__ import annotations

from typing import Any, Final, Literal

Regime = Literal["TREND", "RANGE", "TRANSITION"]
Direction = Literal["UP", "DOWN", "NEUTRAL"]
EntryStyle = Literal["STOP_BREAKOUT", "LIMIT_PULLBACK", "LIMIT_FADE", "NONE"]

ADX_TREND: Final[float] = 25.0
ADX_RANGE: Final[float] = 20.0
# D1 close more than this many ATRs from the D1 Bollinger mid = exhausted move.
EXTENSION_EXHAUSTION_ATR: Final[float] = 2.0
# Pullback entries only make sense while the trend is not yet stretched.
EXTENSION_PULLBACK_MAX_ATR: Final[float] = 1.0
# H4 Bollinger width (in ATR) below which the market is compressing.
BB_SQUEEZE_ATR: Final[float] = 2.5
TREND_SCORE_MIN: Final[int] = 3
RANGE_SCORE_MAX: Final[int] = 1


def _num(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return None if parsed != parsed else parsed


def _frame(technical_report: dict[str, Any], key: str) -> dict[str, Any]:
    context = technical_report.get("direction_context")
    if isinstance(context, dict):
        frame = context.get(key)
        if isinstance(frame, dict):
            return frame
    return {}


def _extension(technical_report: dict[str, Any]) -> float | None:
    context = technical_report.get("direction_context")
    if not isinstance(context, dict):
        return None
    technical = context.get("technical")
    if not isinstance(technical, dict):
        return None
    extension = technical.get("extension")
    if not isinstance(extension, dict):
        return None
    return _num(extension.get("d1_close_vs_mid_atr"))


def _trend_direction(technical_report: dict[str, Any]) -> Direction:
    execution = str(technical_report.get("execution_trend", technical_report.get("trend", "")) or "").upper()
    d1 = str(technical_report.get("d1_trend", "") or "").upper()
    for label in (execution, d1):
        if "UP" in label:
            return "UP"
        if "DOWN" in label:
            return "DOWN"
    return "NEUTRAL"


def classify_regime(technical_report: dict[str, Any] | None) -> dict[str, Any]:
    """Rule-based regime read. Never raises; unknown inputs yield TRANSITION/low confidence."""
    report = technical_report if isinstance(technical_report, dict) else {}
    h4 = _frame(report, "h4")
    h1 = _frame(report, "h1")
    d1 = _frame(report, "d1")

    h4_adx = _num(h4.get("adx_14"))
    h1_adx = _num(h1.get("adx_14"))
    d1_adx = _num(d1.get("adx_14"))
    extension = _extension(report)
    alignment = str(report.get("alignment", "") or "").upper()
    direction = _trend_direction(report)

    score = 0
    evidence: list[str] = []

    if h4_adx is not None:
        if h4_adx >= ADX_TREND:
            score += 2
            evidence.append(f"H4 ADX {h4_adx:.1f} ≥ {ADX_TREND:.0f}: トレンド強度あり")
        elif h4_adx >= ADX_RANGE:
            score += 1
            evidence.append(f"H4 ADX {h4_adx:.1f}: 中間")
        else:
            evidence.append(f"H4 ADX {h4_adx:.1f} < {ADX_RANGE:.0f}: トレンド弱くレンジ寄り")
    else:
        evidence.append("H4 ADX 不明")

    if d1_adx is not None and d1_adx >= ADX_TREND:
        score += 1
        evidence.append(f"D1 ADX {d1_adx:.1f}: 上位足もトレンド")
    if h1_adx is not None and h1_adx >= ADX_TREND:
        score += 1
        evidence.append(f"H1 ADX {h1_adx:.1f}: 執行足もトレンド")

    if alignment == "ALIGNED" and direction != "NEUTRAL":
        score += 1
        evidence.append(f"多時間軸が{direction}で整合")
    elif alignment == "DIVERGENT":
        score -= 1
        evidence.append("多時間軸がDIVERGENT: レジーム不明瞭")

    bb_width_atr: float | None = None
    upper, lower, atr = _num(h4.get("bb_upper")), _num(h4.get("bb_lower")), _num(h4.get("atr_14"))
    if upper is not None and lower is not None and atr and atr > 0:
        bb_width_atr = (upper - lower) / atr
        if bb_width_atr < BB_SQUEEZE_ATR:
            score -= 1
            evidence.append(f"H4 BB幅 {bb_width_atr:.1f}ATR: 収縮(ブレイク待ちのレンジ)")

    exhausted = extension is not None and abs(extension) >= EXTENSION_EXHAUSTION_ATR
    if exhausted:
        evidence.append(f"D1乖離 {extension:+.2f}ATR: 伸び切り、順張り追随は不利")

    no_adx = h4_adx is None and h1_adx is None and d1_adx is None

    regime: Regime
    if no_adx:
        # No indicator snapshot at all: refuse to call it a range.
        regime = "TRANSITION"
        evidence.append("ADXが全時間軸で不明: レジーム判定不能")
    elif exhausted:
        regime = "TRANSITION"
    elif score >= TREND_SCORE_MIN and direction != "NEUTRAL":
        regime = "TREND"
    elif score <= RANGE_SCORE_MAX:
        regime = "RANGE"
    else:
        regime = "TRANSITION"

    distance = abs(score - 2)
    confidence = round(min(0.9, 0.5 + 0.1 * distance), 2) if regime != "TRANSITION" else round(max(0.3, 0.5 - 0.1 * distance), 2)
    if exhausted:
        confidence = 0.6
    if no_adx:
        confidence = 0.3

    entry_style: EntryStyle
    if regime == "TREND":
        entry_style = "LIMIT_PULLBACK" if (extension is not None and abs(extension) >= EXTENSION_PULLBACK_MAX_ATR) else "STOP_BREAKOUT"
    elif regime == "RANGE":
        entry_style = "LIMIT_FADE"
    else:
        entry_style = "NONE"

    return {
        "regime": regime,
        "direction": direction if regime == "TREND" else "NEUTRAL",
        "trend_direction_raw": direction,
        "confidence": confidence,
        "score": score,
        "entry_style": entry_style,
        "h4_adx": h4_adx,
        "h1_adx": h1_adx,
        "d1_adx": d1_adx,
        "d1_extension_atr": extension,
        "h4_bb_width_atr": round(bb_width_atr, 2) if bb_width_atr is not None else None,
        "evidence": evidence,
        "source": "rule_based",
    }
