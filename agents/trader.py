from __future__ import annotations

import json
from typing import Any

from agents.base import decision_model, get_default_client
from indicators.price_levels import resolve_level
from config import SYMBOL

SYSTEM_PROMPT = (
    "あなたは最終決定権を持つトレーダーです。必ず place_trade_order 関数を呼び出して最終判断を返してください。"
    "確信度の数値は求めない。あなたの決定がそのまま採用される。迷いがあればHOLDを選ぶこと。"
    "HOLDが『確信がない』の表現であり、見送りは正当な判断である。エントリーは1日に数回で十分である。"
    "【材料】technical / macro / sentiment の各分析官レポートと、debate(分析官パネルの討論全文)、judge_summary(議長のまとめ)。"
    "どう読むかはあなたの判断である。"
    "【レジーム】judge_summary.regime_summary が、パネルの結論として現在の相場を TREND / RANGE / TRANSITION、"
    "注文タイプ entry_style(STOP_BREAKOUT / LIMIT_PULLBACK / LIMIT_FADE / NONE)、"
    "継続確認・反転確認の水準(level_id)で示す。パネルが走らなかった場合は technical.regime(コードの暫定判定)が代わりに入る。"
    "technical.tp_reference_only は利確目標の選定専用のデータであり、方向判断の材料ではない。"
    "recent_context は直近24時間の自分の判断履歴(decisions)と決済結果(recent_closed)。盲従は不要だが、"
    "数時間前の自分の HOLD を覆して入る場合や、直近の負けと同方向に再び入る場合は、前回から何が変わったかを reasoning に書くこと。"
    "【システムが受け付けるもの(事実)】"
    "水準は technical.direction_context.price_levels の level_id で指定する。候補に無い水準の指値はシステムが受け付けない。"
    "pending_orders は最大1件が採用され、レジームに応じて受理される: TREND では direction_if_trend 側の注文のみ、"
    "RANGE では帯の端の LIMIT のみ(方向バイアスは不要、STOP は受理されない)、TRANSITION では受理されない。"
    "指値のトリガーは現値から 0.1〜3.0 ATR の範囲にあるものだけが発注され、NY 11:00〜16:45 の判断では指値は発注されない。"
    "SL: 基準は ATR×1.5。suggested_sl は『シナリオ否定点』の構造水準そのもの(ヒゲ抜け用のバッファはシステムが付与する)で、"
    "バッファ後の距離が 1.0〜1.5 ATR に収まるときだけ採用され、それ以外は ATR×1.5 になる。null なら ATR×1.5。"
    "TP: suggested_tp は 2R が上限で、超える分は 2R に丸められる。null なら 2R。"
    "スプレッド控除後のリスクリワードが 1.5 未満の注文は発注されない。その場合、TP を遠ざけるより、より有利な水準の指値に替えるか見送る方が通る。"
    "【書き方】action が BUY/SELL なら suggested_tp_level_id と suggested_sl_level_id を指定し、根拠を suggested_tp_basis / suggested_sl_basis に書く。"
    "HOLD で条件付きの計画があるなら pending_orders に1件(type, entry_level_id, 任意の tp_level_id / sl_level_id, basis)。"
    "ブレイク待ちは BUY_STOP / SELL_STOP、押し目・戻り・帯の端待ちは BUY_LIMIT / SELL_LIMIT。"
    "directional_bias / bias_strength / trigger_conditions は方向の見方があれば書き、無ければ NEUTRAL / 0 / []。"
    "条件が無ければ pending_orders は空配列にする。reasoning は日本語で、何を見て決めたかが分かるように書くこと。"
)

PENDING_ORDER_TYPES = ("BUY_STOP", "BUY_LIMIT", "SELL_STOP", "SELL_LIMIT")
MAX_PENDING_ORDERS = 1
PENDING_MIN_BIAS_STRENGTH = 0.6

PLACE_TRADE_ORDER_SCHEMA: dict[str, Any] = {
    "description": "分析結果に基づき売買判断を実行する",
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["BUY", "SELL", "HOLD"]},
            "symbol": {"type": "string"},
            "reasoning": {"type": "string", "description": "判断根拠（日本語）"},
            "risk_level": {"type": "string", "enum": ["LOW", "MID", "HIGH"]},
            "directional_bias": {"type": "string", "enum": ["BULLISH", "BEARISH", "NEUTRAL"]},
            "bias_strength": {"type": "number", "description": "方向性バイアスの強さ0-1"},
            "trigger_conditions": {"type": "array", "items": {"type": "string"}, "description": "バイアス発動の価格条件"},
            "suggested_tp_level_id": {
                "type": ["string", "null"],
                "description": "利確目標の候補水準 level_id (price_levels から)。無ければnull",
            },
            "suggested_tp": {
                "type": ["number", "null"],
                "description": "利確目標価格(level_id が使えない場合の補助。候補水準に近い値のみ)。算出できなければnull",
            },
            "suggested_tp_basis": {
                "type": "string",
                "description": "suggested_tpの根拠(例: キリ番4000と前日安値が重なる4023の手前)",
            },
            "suggested_sl_level_id": {
                "type": ["string", "null"],
                "description": "シナリオ否定点の候補水準 level_id (price_levels から)。無ければnull",
            },
            "suggested_sl": {
                "type": ["number", "null"],
                "description": "シナリオ否定点となる構造水準そのもの(level_id が使えない場合の補助。マージン不要、バッファはシステム付与)。算出できなければnull",
            },
            "suggested_sl_basis": {
                "type": "string",
                "description": "suggested_slの根拠(例: 4450.70の支持帯割れはシナリオ否定のため4446)",
            },
            "pending_orders": {
                "type": "array",
                "description": "HOLD時の条件付き予約注文(最大1件採用)。条件がなければ空配列",
                "items": {
                    "type": "object",
                    "properties": {
                        "type": {"type": "string", "enum": ["BUY_STOP", "BUY_LIMIT", "SELL_STOP", "SELL_LIMIT"]},
                        "entry_level_id": {"type": ["string", "null"], "description": "発動水準の level_id (price_levels から)"},
                        "price": {"type": ["number", "null"], "description": "発動価格(level_id が使えない場合の補助。候補水準に近い値のみ受理)"},
                        "tp_level_id": {"type": ["string", "null"], "description": "任意の利確目標の level_id"},
                        "sl_level_id": {"type": ["string", "null"], "description": "任意の損切り(シナリオ否定点)の level_id"},
                        "tp": {"type": ["number", "null"], "description": "任意の利確目標(2R上限適用)"},
                        "sl": {"type": ["number", "null"], "description": "任意の損切り=シナリオ否定点の構造水準そのもの(バッファはシステム付与、範囲外はATRベースに自動フォールバック)"},
                        "basis": {"type": "string", "description": "この予約の根拠(日本語)"},
                    },
                    "required": ["type"],
                },
            },
        },
        "required": ["action", "symbol", "reasoning"],
    },
}

FALLBACK_RESPONSE: dict[str, Any] = {
    "action": "HOLD",
    "symbol": SYMBOL,
    "reasoning": "最終判断に失敗したためHOLD。",
    "risk_level": "HIGH",
}


def _safe_float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def _extract_current_price_for_tp_sanity(technical_report: dict[str, Any]) -> float | None:
    try:
        direction_context = technical_report.get("direction_context", {})
        if isinstance(direction_context, dict):
            h1 = direction_context.get("h1", {})
            if isinstance(h1, dict):
                close_1 = _safe_float_or_none(h1.get("close"))
                if close_1 is not None:
                    return close_1

        key_levels = technical_report.get("key_levels", {})
        if isinstance(key_levels, dict):
            frames = key_levels.get("frames", {})
            if isinstance(frames, dict):
                h1_frame = frames.get("h1", {})
                if isinstance(h1_frame, dict):
                    close_2 = _safe_float_or_none(h1_frame.get("close"))
                    if close_2 is not None:
                        return close_2
    except Exception:
        return None
    return None


def _technical_for_trader(technical_report: dict[str, Any], debate_report: Any) -> dict[str, Any]:
    """The technical report as the trader sees it.

    When the panel chair returned a regime verdict, the rule-based regime
    (technical.regime) is withheld: showing both made the trader explain the
    rule-vs-panel disagreement every cycle instead of reading the market. The
    field stays in the report itself (logs, fallback when no debate ran).
    """
    if not isinstance(technical_report, dict):
        return technical_report
    summary = debate_report.get("regime_summary") if isinstance(debate_report, dict) else None
    chair_decided = isinstance(summary, dict) and bool(summary.get("regime")) and str(summary.get("source", "")) == "judge"
    if not chair_decided or "regime" not in technical_report:
        return technical_report
    return {key: value for key, value in technical_report.items() if key != "regime"}


def _levels_and_atr(technical_report: dict[str, Any]) -> tuple[list[dict[str, Any]], float | None]:
    context = technical_report.get("direction_context") if isinstance(technical_report, dict) else None
    if not isinstance(context, dict):
        return [], None
    levels = context.get("price_levels") if isinstance(context.get("price_levels"), list) else []
    h1 = context.get("h1") if isinstance(context.get("h1"), dict) else {}
    return levels, _safe_float_or_none(h1.get("atr_14"))


def _anchor(levels: list[dict[str, Any]], level_id: Any, price: Any, atr: float | None) -> dict[str, Any]:
    """Resolve a level_id (preferred) or a raw price against the catalogue.

    With an empty catalogue raw prices pass through unanchored (legacy
    behaviour); with a catalogue, an unknown id or a far-off price is rejected.
    """
    if level_id not in (None, ""):
        resolved = resolve_level(levels, str(level_id), atr=atr)
        if resolved["anchored"]:
            return resolved
        # Unknown id: fall back to the raw price if one was also given.
        if price in (None, ""):
            return resolved
    resolved = resolve_level(levels, price, atr=atr)
    if resolved["reason"] == "no_catalogue":
        resolved["anchored"] = True  # nothing to anchor to; accept as given (legacy behaviour)
    return resolved


def _panel_regime(debate_report: Any, technical_report: Any) -> dict[str, Any]:
    """The regime the pending-order rules follow: the chair's verdict when the
    panel ran and agreed (MAJORITY/UNANIMOUS), else the rule-based read, else
    unknown. -> {regime, direction, source}"""
    summary = debate_report.get("regime_summary") if isinstance(debate_report, dict) else None
    if isinstance(summary, dict) and summary.get("regime") and str(summary.get("source", "")) == "judge":
        consensus = str(summary.get("panel_consensus_type") or summary.get("consensus") or "").upper()
        if consensus in {"UNANIMOUS", "MAJORITY"}:
            return {"regime": str(summary["regime"]), "direction": str(summary.get("direction_if_trend", "NEUTRAL") or "NEUTRAL"), "source": "judge"}
        return {"regime": "TRANSITION", "direction": "NEUTRAL", "source": "judge_split"}
    rule = technical_report.get("regime") if isinstance(technical_report, dict) else None
    if isinstance(rule, dict) and rule.get("regime"):
        return {"regime": str(rule["regime"]), "direction": str(rule.get("direction", "NEUTRAL") or "NEUTRAL"), "source": "rule_based"}
    return {"regime": "", "direction": "NEUTRAL", "source": "none"}


def _order_allowed_for_regime(order_type: str, regime: dict[str, Any], directional_bias: str, bias_strength: float) -> str:
    """'' when the order type fits the regime, else the trade-log reason it does not.

    TREND: only orders on the trend side (no directional-bias number needed;
    the panel's direction is the evidence). RANGE: LIMIT fades only, either
    side, no bias needed. TRANSITION: nothing. No regime known: legacy rule
    (bias side + bias_strength >= PENDING_MIN_BIAS_STRENGTH).
    """
    kind = str(regime.get("regime", "") or "")
    side = "BUY" if order_type.startswith("BUY") else "SELL"
    if kind == "TRANSITION":
        return "skipped_transition"
    if kind == "TREND":
        direction = str(regime.get("direction", "NEUTRAL") or "NEUTRAL")
        if direction == "NEUTRAL":
            return "skipped_trend_without_direction"
        if (direction == "UP") != (side == "BUY"):
            return "skipped_against_trend"
        return ""
    if kind == "RANGE":
        return "" if order_type.endswith("LIMIT") else "skipped_stop_in_range"
    if directional_bias not in {"BULLISH", "BEARISH"}:
        return "skipped_no_bias"
    if bias_strength < PENDING_MIN_BIAS_STRENGTH:
        return f"skipped_weak_bias:{bias_strength:.2f}"
    if (directional_bias == "BULLISH") != (side == "BUY"):
        return "skipped_against_bias"
    return ""


def _validate_pending_orders(
    raw: Any,
    action: str,
    directional_bias: str,
    bias_strength: float,
    current_price: float | None,
    levels: list[dict[str, Any]] | None = None,
    atr: float | None = None,
    regime: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Validate model-proposed pending orders.

    Only meaningful for HOLD. Which orders are acceptable follows the regime
    (see _order_allowed_for_regime); each must also be on the correct side of
    the current price for its type, and its trigger must be a catalogue level
    (by id, or a price within tolerance of one) when a catalogue is available.
    """
    if action != "HOLD":
        return []
    if not isinstance(raw, list):
        return []
    regime = regime or {}

    valid: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        order_type = str(item.get("type", "") or "").upper().strip()
        if order_type not in PENDING_ORDER_TYPES:
            continue
        if _order_allowed_for_regime(order_type, regime, directional_bias, bias_strength):
            continue
        anchor = _anchor(levels or [], item.get("entry_level_id", item.get("level_id")), item.get("price"), atr)
        if not anchor["anchored"]:
            continue
        price = _safe_float_or_none(anchor["price"])
        if price is None or price <= 0:
            continue
        if current_price is not None:
            if order_type == "BUY_STOP" and price <= current_price:
                continue
            if order_type == "BUY_LIMIT" and price >= current_price:
                continue
            if order_type == "SELL_STOP" and price >= current_price:
                continue
            if order_type == "SELL_LIMIT" and price <= current_price:
                continue

        tp_anchor = _anchor(levels or [], item.get("tp_level_id"), item.get("tp"), atr)
        sl_anchor = _anchor(levels or [], item.get("sl_level_id"), item.get("sl"), atr)
        pending_sl = _safe_float_or_none(sl_anchor["price"]) if sl_anchor["anchored"] else None
        if pending_sl is not None:
            # SL must be on the loss side of the trigger price; drop it (not
            # the whole order) when inverted — ATR fallback applies downstream.
            if order_type.startswith("BUY") and pending_sl >= price:
                pending_sl = None
            elif order_type.startswith("SELL") and pending_sl <= price:
                pending_sl = None

        valid.append(
            {
                "type": order_type,
                "price": round(price, 5),
                "entry_level_id": anchor["level_id"],
                "tp": _safe_float_or_none(tp_anchor["price"]) if tp_anchor["anchored"] else None,
                "tp_level_id": tp_anchor["level_id"] if tp_anchor["anchored"] else None,
                "sl": pending_sl,
                "sl_level_id": sl_anchor["level_id"] if pending_sl is not None else None,
                "basis": str(item.get("basis", "") or ""),
            }
        )
        if len(valid) >= MAX_PENDING_ORDERS:
            break
    return valid


# Self-reported bull/bear confidence rises monotonically on BOTH sides during
# a debate (e.g. bull 0.5->0.77->0.81, bear 0.5->0.71->0.76), so it carries no
# directional information and can even contradict the judge's stronger_side.
# It stays in the debate report / trade log but is not shown to the trader.
DEBATER_CONFIDENCE_KEYS = frozenset(
    {"confidence_shift", "bull_confidence", "bear_confidence", "prev_bull_confidence", "prev_bear_confidence"}
)


def _strip_debater_confidence(payload: Any) -> Any:
    """Deep-copy ``payload`` without debater self-confidence fields."""
    if isinstance(payload, dict):
        return {
            key: _strip_debater_confidence(value)
            for key, value in payload.items()
            if key not in DEBATER_CONFIDENCE_KEYS and not str(key).endswith("_confidence_history")
        }
    if isinstance(payload, list):
        return [_strip_debater_confidence(item) for item in payload]
    return payload


def _describe_pending_proposal(
    raw: Any,
    action: str,
    directional_bias: str,
    bias_strength: float,
    validated: list[dict[str, Any]],
    levels: list[dict[str, Any]] | None = None,
    atr: float | None = None,
    regime: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any] | None]:
    """Explain why a proposed pending order survived or was dropped.

    Returns (status, first_proposal). ``status`` is "" when an order survived
    validation (the placement path then reports placed/skipped_*), otherwise a
    trade-log-ready reason so "the LLM proposed nothing" and "the LLM proposed
    something the validator dropped" are distinguishable in the CSV.
    """
    items = [item for item in raw if isinstance(item, dict)] if isinstance(raw, list) else []
    proposal: dict[str, Any] | None = None
    if items:
        proposal = {
            "type": str(items[0].get("type", "") or "").upper().strip(),
            "price": _safe_float_or_none(items[0].get("price")),
            "entry_level_id": items[0].get("entry_level_id"),
        }
    if validated:
        return "", proposal
    if action != "HOLD":
        return "", proposal
    if not items:
        return "none_proposed", None
    first = items[0]
    first_type = str(first.get("type", "") or "").upper().strip()
    if first_type not in PENDING_ORDER_TYPES:
        return "skipped_invalid_proposal", proposal
    regime_reason = _order_allowed_for_regime(first_type, regime or {}, directional_bias, bias_strength)
    if regime_reason:
        return regime_reason, proposal
    anchor = _anchor(levels or [], first.get("entry_level_id", first.get("level_id")), first.get("price"), atr)
    if not anchor["anchored"]:
        return f"skipped_unanchored_price:{anchor['reason']}", proposal
    return "skipped_invalid_proposal", proposal


def decide_trade(
    technical_report: dict[str, Any],
    sentiment_report: dict[str, Any],
    debate_report: dict[str, Any],
    macro_report: dict[str, Any] | None = None,
    confidence_threshold: float | None = None,  # deprecated: no longer gates anything; kept for callers
    recent_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Final decision. The trader's action stands; there is no confidence gate.

    A self-reported confidence number predicted nothing (executed trades all
    sat at 0.62-0.84, HOLDs at 0.78 median: the model learned the cutoff),
    so it is neither requested nor used. HOLD is how the trader says "not
    sure". Code-side guards (spread, daily loss, losing streak, RR, pending
    distance, time windows) still apply after this function.
    """
    _ = confidence_threshold
    raw_judge_summary = debate_report.get("judge_summary", {})
    judge_summary: dict[str, Any]
    if isinstance(raw_judge_summary, dict):
        judge_summary = dict(raw_judge_summary)
    else:
        judge_summary = {
            "agreements": [],
            "conflicts": [str(raw_judge_summary)] if raw_judge_summary else [],
            "stronger_side": "neutral",
        }
    user_payload = {
        "technical": _technical_for_trader(technical_report, debate_report),
        "sentiment": sentiment_report,
        "macro": macro_report or {},
        "debate": _strip_debater_confidence(debate_report),
        "judge_summary": _strip_debater_confidence(judge_summary),
        "recent_context": recent_context or {"decisions": [], "recent_closed": []},
        "constraints": {
            "symbol": SYMBOL,
        },
    }

    result = get_default_client().call_function(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=json.dumps(user_payload, ensure_ascii=False),
        model=decision_model(),
        function_name="place_trade_order",
        function_schema=PLACE_TRADE_ORDER_SCHEMA,
        fallback_payload=FALLBACK_RESPONSE,
    )

    payload = dict(result.payload)
    action = str(payload.get("action", "HOLD")).upper()
    evidence_status = str(sentiment_report.get("evidence_status", "")).upper()
    risk_level = str(payload.get("risk_level") or "HIGH").upper()
    if risk_level not in {"LOW", "MID", "HIGH"}:
        risk_level = "HIGH"

    if action not in {"BUY", "SELL", "HOLD"}:
        action = "HOLD"
    if evidence_status == "INSUFFICIENT":
        action = "HOLD"
        payload["reasoning"] = "ニュース判断材料が不足しているためHOLD。"

    directional_bias = str(payload.get("directional_bias", "NEUTRAL") or "NEUTRAL").upper()
    if directional_bias not in {"BULLISH", "BEARISH", "NEUTRAL"}:
        directional_bias = "NEUTRAL"
    # The trader's own bias stands; code no longer injects the macro bias when
    # the trader said NEUTRAL.
    payload["directional_bias"] = directional_bias
    payload["bias_strength"] = max(0.0, min(1.0, float(payload.get("bias_strength", 0.0) or 0.0)))
    tc = payload.get("trigger_conditions", [])
    payload["trigger_conditions"] = [str(x) for x in tc] if isinstance(tc, list) else []

    payload["action"] = action
    payload["symbol"] = str(payload.get("symbol") or SYMBOL)
    # Not requested; if the model volunteers one it is logged, never used.
    payload["confidence"] = _safe_float_or_none(payload.get("confidence"))
    payload["risk_level"] = risk_level

    current_price = _extract_current_price_for_tp_sanity(technical_report)
    price_levels, atr_h1 = _levels_and_atr(technical_report)
    # TP / SL: a level_id is preferred; a raw price is kept (the risk manager
    # bounds it) but its anchoring is recorded for the log.
    tp_anchor = _anchor(price_levels, payload.get("suggested_tp_level_id"), payload.get("suggested_tp"), atr_h1)
    sl_anchor = _anchor(price_levels, payload.get("suggested_sl_level_id"), payload.get("suggested_sl"), atr_h1)
    payload["suggested_tp"] = tp_anchor["price"] if tp_anchor["anchored"] else _safe_float_or_none(payload.get("suggested_tp"))
    payload["suggested_sl"] = sl_anchor["price"] if sl_anchor["anchored"] else _safe_float_or_none(payload.get("suggested_sl"))
    payload["suggested_tp_level_id"] = tp_anchor["level_id"] if tp_anchor["anchored"] else None
    payload["suggested_sl_level_id"] = sl_anchor["level_id"] if sl_anchor["anchored"] else None
    payload["tp_anchor_reason"] = tp_anchor["reason"]
    payload["sl_anchor_reason"] = sl_anchor["reason"]
    suggested_tp = _safe_float_or_none(payload.get("suggested_tp"))
    if action == "HOLD":
        suggested_tp = None
    elif suggested_tp is not None and current_price is not None:
        if action == "BUY" and suggested_tp <= current_price:
            suggested_tp = None
        elif action == "SELL" and suggested_tp >= current_price:
            suggested_tp = None

    suggested_tp_basis = str(payload.get("suggested_tp_basis", "") or "")
    payload["suggested_tp"] = suggested_tp
    payload["suggested_tp_basis"] = suggested_tp_basis

    suggested_sl = _safe_float_or_none(payload.get("suggested_sl"))
    if action == "HOLD":
        suggested_sl = None
    elif suggested_sl is not None and current_price is not None:
        # SL must sit on the loss side; inverted values are discarded here and
        # the ATR fallback applies in build_risk_plan.
        if action == "BUY" and suggested_sl >= current_price:
            suggested_sl = None
        elif action == "SELL" and suggested_sl <= current_price:
            suggested_sl = None
    payload["suggested_sl"] = suggested_sl
    payload["suggested_sl_basis"] = str(payload.get("suggested_sl_basis", "") or "")

    raw_pending_orders = payload.get("pending_orders")
    regime = _panel_regime(debate_report, technical_report)
    payload["pending_regime"] = regime
    payload["pending_orders"] = _validate_pending_orders(
        raw=raw_pending_orders,
        action=action,
        directional_bias=directional_bias,
        bias_strength=float(payload.get("bias_strength", 0.0) or 0.0),
        current_price=current_price,
        levels=price_levels,
        atr=atr_h1,
        regime=regime,
    )
    payload["pending_validation"], payload["pending_proposal"] = _describe_pending_proposal(
        raw=raw_pending_orders,
        action=action,
        directional_bias=directional_bias,
        bias_strength=float(payload.get("bias_strength", 0.0) or 0.0),
        validated=payload["pending_orders"],
        levels=price_levels,
        atr=atr_h1,
        regime=regime,
    )

    payload["_meta"] = {
        "ok": result.ok,
        "model": result.model,
        "error": result.error,
        "usage": {
            "prompt_tokens": result.usage.prompt_tokens,
            "completion_tokens": result.usage.completion_tokens,
            "total_tokens": result.usage.total_tokens,
        },
    }
    return payload
