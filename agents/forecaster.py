"""Probability forecaster (forecast-only research path, no trading).

Given the analyst reports and a concrete triple-barrier task (upper barrier,
lower barrier, horizon in H1 bars), the LLM returns calibrated probabilities
p_up / p_down / p_timeout. Nothing here places orders; the output is logged
by scripts/run_forecast_logger.py and scored later against realised bars.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from agents.base import get_default_client
from config import FORECAST_SAMPLES, MODEL_FORECAST

LOGGER = logging.getLogger(__name__)

PROB_KEYS = ("p_up", "p_down", "p_timeout")
SUM_TOLERANCE_LOW = 0.95
SUM_TOLERANCE_HIGH = 1.05

SYSTEM_PROMPT = (
    "あなたは確率予測者です。取引判断ではなく、確率の見積もりだけを行います。"
    "対象はGOLD(XAU/USD)のH1足で、与えられた現在価格P0・H1 ATR・上壁・下壁・期限(本数)に対し、"
    "次のN本のH1足の高値/安値が『上壁に先に到達する』『下壁に先に到達する』『期限内にどちらにも到達しない』"
    "の3つの確率を返してください。"
    "壁への到達は高値が上壁以上、安値が下壁以下になった時点で成立し、先に成立した側が正解になります。"
    "技術・センチメント・マクロの各レポートは参考情報であり、そのまま添付されています。"
    "確率は正直に。過去の同様の局面で実際に起きた頻度として答えること。"
    "自信がなければ1/3ずつに近づけてよい。断定的な言葉で確率を偏らせないこと。"
    "出力は次のキーだけを持つJSONのみ: "
    "{\"p_up\": float, \"p_down\": float, \"p_timeout\": float, \"key_reason\": str}。"
    "3つの確率の合計は1.0にすること。key_reasonは最も重要な根拠を日本語で1〜2文。"
)

FALLBACK_RESPONSE: dict[str, Any] = {
    "p_up": None,
    "p_down": None,
    "p_timeout": None,
    "key_reason": "",
}


def _to_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed != parsed:  # NaN
        return None
    return parsed


def normalize_probabilities(payload: dict[str, Any]) -> tuple[dict[str, float] | None, dict[str, Any], str]:
    """-> (normalized probs or None, raw values, invalid reason).

    Valid when all three are numbers >= 0 and their sum lies within
    [0.95, 1.05]; the normalized version rescales them to sum to exactly 1.
    """
    raw = {key: _to_float(payload.get(key)) for key in PROB_KEYS}
    if any(value is None for value in raw.values()):
        return None, raw, "missing_or_non_numeric"
    values = {key: float(value) for key, value in raw.items() if value is not None}
    if any(value < 0.0 for value in values.values()):
        return None, raw, "negative_probability"
    total = sum(values.values())
    if not (SUM_TOLERANCE_LOW <= total <= SUM_TOLERANCE_HIGH):
        return None, raw, f"sum_out_of_range:{total:.3f}"
    if total <= 0:
        return None, raw, "zero_sum"
    normalized = {key: round(value / total, 6) for key, value in values.items()}
    return normalized, raw, ""


def build_forecast_payload(
    *,
    symbol: str,
    bar_time_utc: str,
    ts_utc: str,
    p0: float,
    atr_h1: float,
    k_up: float,
    k_down: float,
    horizon_bars: int,
    barrier_up: float,
    barrier_down: float,
    technical_report: dict[str, Any],
    sentiment_report: dict[str, Any],
    macro_report: dict[str, Any],
    debate_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The exact user payload handed to the LLM (also archived per forecast)."""
    task = {
        "symbol": symbol,
        "confirmed_bar_start_utc": bar_time_utc,
        "forecast_time_utc": ts_utc,
        "p0_close": round(float(p0), 5),
        "atr_h1": round(float(atr_h1), 5),
        "k_up": k_up,
        "k_down": k_down,
        "barrier_up": round(float(barrier_up), 5),
        "barrier_down": round(float(barrier_down), 5),
        "horizon_bars": int(horizon_bars),
        "question": (
            f"次の{int(horizon_bars)}本のH1足で、高値が{barrier_up:.2f}以上に達する(UP)のと、"
            f"安値が{barrier_down:.2f}以下に達する(DOWN)のと、どちらが先に起きるか。"
            f"どちらも起きなければTIMEOUT。"
        ),
    }
    payload: dict[str, Any] = {
        "task": task,
        "technical": technical_report,
        "sentiment": sentiment_report,
        "macro": macro_report,
    }
    if debate_report is not None:
        payload["debate"] = debate_report
    return payload


def forecast(
    payload: dict[str, Any],
    *,
    model: str = MODEL_FORECAST,
    samples: int = FORECAST_SAMPLES,
    client: Any = None,
) -> dict[str, Any]:
    """Call the LLM ``samples`` times on the same payload and aggregate.

    Returns a dict that never raises:
    ok, model, p_up/p_down/p_timeout (normalized mean over valid samples or
    None), p_raw (first sample's raw values), samples (list of per-call raw
    dicts), probs_valid, invalid_reason, key_reason, error, usage.
    """
    llm = client or get_default_client()
    user_prompt = json.dumps(payload, ensure_ascii=False)
    sample_rows: list[dict[str, Any]] = []
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    errors: list[str] = []
    model_name = model

    for index in range(max(1, int(samples))):
        try:
            result = llm.call_json(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
                model=model,
                fallback_payload=FALLBACK_RESPONSE,
            )
        except Exception as exc:  # fail-safe: record and continue
            errors.append(f"sample{index}: {exc}")
            sample_rows.append({"ok": False, "error": str(exc), "raw": dict(FALLBACK_RESPONSE), "normalized": None})
            continue
        model_name = str(getattr(result, "model", model) or model)
        result_usage = getattr(result, "usage", None)
        if result_usage is not None:
            usage["prompt_tokens"] += int(getattr(result_usage, "prompt_tokens", 0) or 0)
            usage["completion_tokens"] += int(getattr(result_usage, "completion_tokens", 0) or 0)
            usage["total_tokens"] += int(getattr(result_usage, "total_tokens", 0) or 0)
        if not bool(getattr(result, "ok", False)):
            errors.append(f"sample{index}: {getattr(result, 'error', '') or 'llm call failed'}")
            sample_rows.append({"ok": False, "error": str(getattr(result, "error", "")), "raw": dict(FALLBACK_RESPONSE), "normalized": None})
            continue
        raw_payload = dict(result.payload)
        normalized, raw, invalid_reason = normalize_probabilities(raw_payload)
        sample_rows.append(
            {
                "ok": True,
                "error": "",
                "raw": raw,
                "normalized": normalized,
                "invalid_reason": invalid_reason,
                "key_reason": str(raw_payload.get("key_reason", "") or ""),
            }
        )

    valid = [row["normalized"] for row in sample_rows if row.get("normalized")]
    ok_calls = [row for row in sample_rows if row.get("ok")]
    aggregated: dict[str, float | None] = {key: None for key in PROB_KEYS}
    if valid:
        for key in PROB_KEYS:
            aggregated[key] = round(sum(row[key] for row in valid) / len(valid), 6)
    first_ok = ok_calls[0] if ok_calls else None
    invalid_reason = "" if valid else (first_ok.get("invalid_reason", "") if first_ok else "no_successful_call")

    return {
        "ok": bool(ok_calls),
        "model": model_name,
        "p_up": aggregated["p_up"],
        "p_down": aggregated["p_down"],
        "p_timeout": aggregated["p_timeout"],
        "p_raw": first_ok["raw"] if first_ok else dict(FALLBACK_RESPONSE),
        "samples": sample_rows if len(sample_rows) > 1 else [],
        "n_samples": len(sample_rows),
        "n_valid_samples": len(valid),
        "probs_valid": bool(valid),
        "invalid_reason": invalid_reason,
        "key_reason": (first_ok or {}).get("key_reason", ""),
        "error": "; ".join(errors),
        "usage": usage,
    }
