"""Ablation of the forecast-only logger: re-forecast archived inputs with
one report removed and compare calibration per condition.

Every forecast archived its exact LLM payload under logs/forecast_inputs/
<forecast_id>.json, so the same 6-bar triple-barrier task can be re-asked
with, e.g., the sentiment report omitted. Because the outcome label is
already known from the original row, each condition can be scored on the
identical sample and compared pairwise (block bootstrap on the per-row Brier
difference). No trading is involved.
"""

from __future__ import annotations

import copy
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np

from agents.forecaster import forecast as run_forecaster
from analysis.forecast_eval import BLOCK_SIZE, CLASSES, block_bootstrap_ci, brier_per_row, llm_probs, scorable_rows
from analysis.forecast_store import forecasts_path, inputs_dir, read_forecasts
from config import LOG_DIR, MODEL_FORECAST

LOGGER = logging.getLogger(__name__)

ABLATION_FILENAME = "forecast_ablation.jsonl"

# condition -> report keys removed from the archived payload ("task" always stays)
CONDITIONS: dict[str, tuple[str, ...]] = {
    "full_rerun": (),
    "no_sentiment": ("sentiment",),
    "no_macro": ("macro",),
    "no_technical": ("technical",),
    "no_debate": ("debate",),
    "technical_only": ("sentiment", "macro", "debate"),
}
DEFAULT_CONDITIONS = ("no_sentiment", "no_macro", "technical_only", "no_debate")


def condition_applies(payload: dict[str, Any], condition: str) -> bool:
    """False when the condition would remove nothing from this payload (e.g.
    no_debate on a forecast made without a debate): re-forecasting the
    identical input would only spend an API call on noise."""
    keys = CONDITIONS.get(condition, ())
    if not keys:
        return True
    return any(key in payload for key in keys)


def ablation_path() -> Path:
    return Path(LOG_DIR) / ABLATION_FILENAME


def apply_condition(payload: dict[str, Any], condition: str) -> dict[str, Any]:
    """Return a deep copy of the archived LLM payload with the condition's reports removed."""
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition: {condition}")
    modified = copy.deepcopy(payload)
    for key in CONDITIONS[condition]:
        modified.pop(key, None)
    return modified


def load_archived_payload(forecast_id: str, directory: Path | None = None) -> dict[str, Any] | None:
    """The payload the LLM originally saw (looked up by id, not by the stored
    absolute path, so archives copied from another machine still resolve)."""
    path = (directory or inputs_dir()) / f"{forecast_id}.json"
    if not path.exists():
        return None
    try:
        archived = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    payload = archived.get("payload") if isinstance(archived, dict) else None
    return payload if isinstance(payload, dict) and "task" in payload else None


def read_ablation(path: Path | None = None) -> list[dict[str, Any]]:
    target = path or ablation_path()
    if not target.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in target.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def run_ablation(
    conditions: tuple[str, ...] = DEFAULT_CONDITIONS,
    *,
    forecasts: list[dict[str, Any]] | None = None,
    inputs_directory: Path | None = None,
    output_path: Path | None = None,
    model: str = MODEL_FORECAST,
    limit: int | None = None,
    client: Any = None,
    forecaster: Callable[..., dict[str, Any]] = run_forecaster,
) -> dict[str, int]:
    """Re-forecast every scorable original row under each condition.

    Resumable: (forecast_id, condition) pairs already present in the output
    file are skipped, so the script can be stopped and restarted freely.
    """
    rows = forecasts if forecasts is not None else read_forecasts()
    targets = scorable_rows(rows)
    if limit is not None:
        targets = targets[: max(0, int(limit))]
    out_path = output_path or ablation_path()
    done = {(r.get("forecast_id"), r.get("condition")) for r in read_ablation(out_path)}
    counts = {"targets": len(targets), "calls": 0, "skipped_done": 0, "skipped_not_applicable": 0, "missing_inputs": 0, "failed": 0}
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("a", encoding="utf-8") as fp:
        for row in targets:
            forecast_id = str(row.get("forecast_id", ""))
            payload = load_archived_payload(forecast_id, inputs_directory)
            if payload is None:
                counts["missing_inputs"] += 1
                continue
            for condition in conditions:
                if (forecast_id, condition) in done:
                    counts["skipped_done"] += 1
                    continue
                if not condition_applies(payload, condition):
                    counts["skipped_not_applicable"] += 1
                    continue
                try:
                    result = forecaster(apply_condition(payload, condition), model=model, samples=1, client=client)
                except Exception as exc:  # forecaster is fail-safe; belt and braces
                    result = {"ok": False, "error": str(exc), "p_up": None, "p_down": None, "p_timeout": None,
                              "probs_valid": False, "invalid_reason": "exception", "key_reason": "", "model": model,
                              "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}
                counts["calls"] += 1
                if not result.get("ok") or not result.get("probs_valid"):
                    counts["failed"] += 1
                record = {
                    "forecast_id": forecast_id,
                    "condition": condition,
                    "ts_utc": row.get("ts_utc"),
                    "outcome": row.get("outcome"),
                    "p_up": result.get("p_up"),
                    "p_down": result.get("p_down"),
                    "p_timeout": result.get("p_timeout"),
                    "probs_valid": bool(result.get("probs_valid")),
                    "invalid_reason": result.get("invalid_reason", ""),
                    "key_reason": result.get("key_reason", ""),
                    "model": result.get("model", model),
                    "ok": bool(result.get("ok")),
                    "error": result.get("error", ""),
                    "usage": result.get("usage"),
                }
                fp.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
                fp.flush()
                done.add((forecast_id, condition))
    return counts


# --------------------------------------------------------------------------- #
# Comparison
# --------------------------------------------------------------------------- #
def _tilt(rows: list[dict[str, Any]]) -> float:
    return float(np.mean([r["p_up"] - r["p_down"] for r in rows])) if rows else float("nan")


def _direction_accuracy(rows: list[dict[str, Any]]) -> float | None:
    ud = [r for r in rows if r["outcome"] in ("UP", "DOWN")]
    if not ud:
        return None
    return float(np.mean([(r["p_up"] > r["p_down"]) == (r["outcome"] == "UP") for r in ud]))


def compare_conditions(
    forecasts: list[dict[str, Any]],
    ablation: list[dict[str, Any]],
    conditions: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Score the original ('original') and each ablation condition on the
    intersection of forecast_ids they all cover; pairwise bootstrap vs original."""
    original = {r["forecast_id"]: r for r in scorable_rows(forecasts)}
    by_condition: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for r in ablation:
        if r.get("ok") and r.get("probs_valid") and r.get("forecast_id") in original:
            by_condition[str(r["condition"])][str(r["forecast_id"])] = r
    names = list(conditions) if conditions else sorted(by_condition)
    names = [n for n in names if n in by_condition]
    if not names:
        return {"n": 0, "conditions": [], "note": "no ablation rows to compare"}

    common = set(original)
    for name in names:
        common &= set(by_condition[name])
    ids = sorted(common, key=lambda i: str(original[i].get("ts_utc", "")))
    if not ids:
        return {"n": 0, "conditions": names, "note": "no common forecast_ids"}

    outcomes = [str(original[i]["outcome"]) for i in ids]
    base_rates = np.array([outcomes.count(c) / len(outcomes) for c in CLASSES])
    base_brier = brier_per_row(np.tile(base_rates, (len(ids), 1)), outcomes)

    def score(rows: list[dict[str, Any]]) -> tuple[np.ndarray, dict[str, Any]]:
        per_row = brier_per_row(llm_probs(rows), outcomes)
        lo, hi = block_bootstrap_ci(per_row - base_brier)
        return per_row, {
            "n": len(rows),
            "brier": round(float(per_row.mean()), 4),
            "vs_base_rate_ci95": (round(lo, 4), round(hi, 4)),
            "beats_base_rate": hi < 0.0,
            "mean_tilt": round(_tilt(rows), 3),
            "direction_accuracy": None if _direction_accuracy(rows) is None else round(_direction_accuracy(rows), 3),
            "mean_p_timeout": round(float(np.mean([r["p_timeout"] for r in rows])), 3),
        }

    orig_rows = [original[i] for i in ids]
    orig_per_row, orig_stats = score(orig_rows)
    table = [{"condition": "original", **orig_stats, "vs_original_ci95": (0.0, 0.0), "better_than_original": False}]
    for name in names:
        rows = [by_condition[name][i] for i in ids]
        per_row, stats = score(rows)
        lo, hi = block_bootstrap_ci(per_row - orig_per_row)
        table.append({"condition": name, **stats, "vs_original_ci95": (round(lo, 4), round(hi, 4)), "better_than_original": hi < 0.0})
    return {
        "n": len(ids),
        "base_rate_brier": round(float(base_brier.mean()), 4),
        "base_rates": {c: round(float(v), 3) for c, v in zip(CLASSES, base_rates)},
        "actual_timeout_rate": round(outcomes.count("TIMEOUT") / len(outcomes), 3),
        "conditions": table,
        "block_size": BLOCK_SIZE,
    }


def format_comparison(report: dict[str, Any]) -> str:
    lines = ["=== GP-MATE Forecast Ablation ==="]
    if not report.get("n"):
        lines.append(report.get("note", "nothing to compare"))
        return "\n".join(lines)
    lines.append(f"common rows={report['n']}  base_rate Brier={report['base_rate_brier']}  base rates={report['base_rates']}  actual TIMEOUT rate={report['actual_timeout_rate']}")
    lines.append("")
    lines.append(f"{'condition':<16}{'brier':>8}{'  vs base CI95':>20}{'  vs original CI95':>22}{'  tilt':>8}{'  dir_acc':>10}{'  p_timeout':>12}")
    for row in report["conditions"]:
        acc = "-" if row["direction_accuracy"] is None else f"{row['direction_accuracy']:.3f}"
        flag = " *" if row["better_than_original"] else ""
        lines.append(
            f"{row['condition']:<16}{row['brier']:>8.4f}{str(row['vs_base_rate_ci95']):>20}{str(row['vs_original_ci95']):>22}"
            f"{row['mean_tilt']:>+8.3f}{acc:>10}{row['mean_p_timeout']:>12.3f}{flag}"
        )
    lines.append("")
    lines.append("* = 95% block-bootstrap interval of (condition - original) per-row Brier lies below 0.")
    lines.append(f"注意: 隣接サンプルは相関しているためブロック(連続{report['block_size']}件)ブートストラップ。区間がゼロを跨ぐ差は差なしと読む。")
    return "\n".join(lines)
