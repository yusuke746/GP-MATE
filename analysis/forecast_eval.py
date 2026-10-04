"""Scoring for forecast-only records: multi-class Brier, log loss, reliability
tables, breakdowns, three baselines and a block-bootstrap comparison."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from analysis.forecast_labels import SCORABLE_OUTCOMES
from analysis.forecast_store import read_forecasts
from config import MARKET_TZ

CLASSES = ("UP", "DOWN", "TIMEOUT")
PROB_KEYS = ("p_up", "p_down", "p_timeout")
BLOCK_SIZE = 12
BOOTSTRAP_ROUNDS = 1000
EPS = 1e-6

CAVEAT = (
    "注意: 予測期限が重なるため隣接サンプルは相関しており、実効サンプル数は行数より少ない。"
    f"ブートストラップはブロック(連続{BLOCK_SIZE}件)単位で行っている。"
)


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #
def counts_summary(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "total": len(rows),
        "llm_failed": sum(1 for r in rows if not r.get("ok")),
        "invalid_probs": sum(1 for r in rows if r.get("ok") and not r.get("probs_valid")),
        "resolved": sum(1 for r in rows if r.get("outcome") is not None),
        "ambiguous": sum(1 for r in rows if r.get("outcome") == "AMBIGUOUS"),
        "scorable": len(scorable_rows(rows)),
    }


def scorable_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        if not row.get("ok") or not row.get("probs_valid"):
            continue
        if row.get("outcome") not in SCORABLE_OUTCOMES:
            continue
        if any(row.get(key) is None for key in PROB_KEYS):
            continue
        out.append(row)
    out.sort(key=lambda r: str(r.get("ts_utc", "")))
    return out


# --------------------------------------------------------------------------- #
# Scores
# --------------------------------------------------------------------------- #
def _probs(row: dict[str, Any]) -> np.ndarray:
    return np.array([float(row["p_up"]), float(row["p_down"]), float(row["p_timeout"])], dtype=float)


def _onehot(outcome: str) -> np.ndarray:
    return np.array([1.0 if outcome == c else 0.0 for c in CLASSES], dtype=float)


def brier_per_row(probs: np.ndarray, outcomes: list[str]) -> np.ndarray:
    targets = np.stack([_onehot(o) for o in outcomes]) if outcomes else np.zeros((0, 3))
    return ((probs - targets) ** 2).sum(axis=1)


def logloss_per_row(probs: np.ndarray, outcomes: list[str]) -> np.ndarray:
    idx = np.array([CLASSES.index(o) for o in outcomes], dtype=int)
    picked = np.clip(probs[np.arange(len(outcomes)), idx], EPS, 1.0)
    return -np.log(picked)


def llm_probs(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.stack([_probs(r) for r in rows]) if rows else np.zeros((0, 3))


def outcomes_of(rows: list[dict[str, Any]]) -> list[str]:
    return [str(r["outcome"]) for r in rows]


# --------------------------------------------------------------------------- #
# Baselines
# --------------------------------------------------------------------------- #
def baseline_uniform(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.full((len(rows), 3), 1.0 / 3.0)


def base_rates(rows: list[dict[str, Any]]) -> np.ndarray:
    if not rows:
        return np.full(3, 1.0 / 3.0)
    outs = outcomes_of(rows)
    return np.array([outs.count(c) / len(outs) for c in CLASSES], dtype=float)


def baseline_base_rate(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.tile(base_rates(rows), (len(rows), 1))


def baseline_momentum(rows: list[dict[str, Any]]) -> np.ndarray:
    rates = base_rates(rows)
    out = []
    for row in rows:
        momentum = row.get("momentum_3")
        if momentum is None:
            out.append(rates)
        elif float(momentum) > 0:
            out.append(np.array([0.45, 0.25, 0.30]))
        elif float(momentum) < 0:
            out.append(np.array([0.25, 0.45, 0.30]))
        else:
            out.append(rates)
    return np.stack(out) if out else np.zeros((0, 3))


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #
def reliability_table(rows: list[dict[str, Any]], prob_key: str, outcome: str, bins: int = 10) -> list[dict[str, Any]]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    table = []
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        members = [r for r in rows if lo <= float(r[prob_key]) < hi or (i == bins - 1 and float(r[prob_key]) == 1.0)]
        if not members:
            table.append({"bin": f"[{lo:.1f},{hi:.1f})", "n": 0, "mean_pred": None, "actual_rate": None})
            continue
        table.append(
            {
                "bin": f"[{lo:.1f},{hi:.1f})",
                "n": len(members),
                "mean_pred": round(float(np.mean([float(r[prob_key]) for r in members])), 3),
                "actual_rate": round(sum(1 for r in members if r["outcome"] == outcome) / len(members), 3),
            }
        )
    return table


def group_brier(rows: list[dict[str, Any]], key_fn: Callable[[dict[str, Any]], Any]) -> list[dict[str, Any]]:
    groups: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[key_fn(row)].append(row)
    out = []
    for key in sorted(groups, key=lambda k: str(k)):
        members = groups[key]
        scores = brier_per_row(llm_probs(members), outcomes_of(members))
        out.append({"group": key, "n": len(members), "brier": round(float(scores.mean()), 4)})
    return out


def ny_bucket(row: dict[str, Any]) -> str:
    try:
        ts = datetime.fromisoformat(str(row["ts_utc"])).astimezone(MARKET_TZ)
    except (TypeError, ValueError):
        return "?"
    start = (ts.hour // 2) * 2
    return f"NY {start:02d}-{start + 2:02d}"


def weekday_bucket(row: dict[str, Any]) -> str:
    names = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
    try:
        return names[datetime.fromisoformat(str(row["ts_utc"])).weekday()]
    except (TypeError, ValueError):
        return "?"


# --------------------------------------------------------------------------- #
# Block bootstrap
# --------------------------------------------------------------------------- #
def block_bootstrap_ci(
    per_row_values: np.ndarray,
    block_size: int = BLOCK_SIZE,
    rounds: int = BOOTSTRAP_ROUNDS,
    seed: int = 7,
) -> tuple[float, float]:
    """95% interval of the mean under a moving-block bootstrap (rows in time order)."""
    n = len(per_row_values)
    if n == 0:
        return (float("nan"), float("nan"))
    if n <= block_size:
        return (float(per_row_values.mean()), float(per_row_values.mean()))
    rng = np.random.default_rng(seed)
    starts_max = n - block_size
    blocks_needed = int(math.ceil(n / block_size))
    means = np.empty(rounds)
    for r in range(rounds):
        starts = rng.integers(0, starts_max + 1, size=blocks_needed)
        sample = np.concatenate([per_row_values[s : s + block_size] for s in starts])[:n]
        means[r] = sample.mean()
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def anchor_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Scorable rows that carried a reference_base_rates anchor."""
    out = []
    for row in rows:
        ref = row.get("reference_base_rates")
        if isinstance(ref, dict) and all(isinstance(ref.get(k), (int, float)) for k in ("p_up", "p_down", "p_timeout")):
            out.append(row)
    return out


def anchor_probs(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.array([[float(r["reference_base_rates"]["p_up"]), float(r["reference_base_rates"]["p_down"]), float(r["reference_base_rates"]["p_timeout"])] for r in rows])


def anchor_deviation(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """How the LLM moved away from the base-rate anchor it was handed, and whether
    that move pointed the right way.

    Once the forecaster is anchored, "p_up > p_down" mostly echoes the anchor,
    so directional accuracy of the raw probabilities says little. The
    informative quantity is the sign of (p_up - anchor_up) - (p_down -
    anchor_down) against the realised direction, plus how far it moved.
    """
    anchored = anchor_rows(rows)
    if not anchored:
        return None
    llm = llm_probs(anchored)
    ref = anchor_probs(anchored)
    outs = outcomes_of(anchored)
    dev = llm - ref
    tilt = dev[:, 0] - dev[:, 1]  # >0: moved toward UP relative to the anchor
    directional = [(t, o) for t, o in zip(tilt, outs) if o in ("UP", "DOWN") and abs(t) > 1e-9]
    hits = sum(1 for t, o in directional if (t > 0) == (o == "UP"))
    moved_up = int(sum(1 for t in tilt if t > 1e-9))
    moved_down = int(sum(1 for t in tilt if t < -1e-9))
    llm_b = brier_per_row(llm, outs)
    ref_b = brier_per_row(ref, outs)
    lo, hi = block_bootstrap_ci(llm_b - ref_b)
    return {
        "n": len(anchored),
        "brier_llm": round(float(llm_b.mean()), 4),
        "brier_anchor": round(float(ref_b.mean()), 4),
        "llm_minus_anchor_ci95": (round(lo, 4), round(hi, 4)),
        "beats_anchor": hi < 0.0,
        "deviation_direction_n": len(directional),
        "deviation_direction_accuracy": round(hits / len(directional), 3) if directional else None,
        "moved_toward_up": moved_up,
        "moved_toward_down": moved_down,
        "flat": len(anchored) - moved_up - moved_down,
        "abs_tilt_median": round(float(np.median(np.abs(tilt))), 4),
        "abs_tilt_p90": round(float(np.percentile(np.abs(tilt), 90)), 4),
        "mean_deviation": {c: round(float(v), 4) for c, v in zip(CLASSES, dev.mean(axis=0))},
    }


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def evaluate(rows_all: list[dict[str, Any]]) -> dict[str, Any]:
    rows = scorable_rows(rows_all)
    report: dict[str, Any] = {"counts": counts_summary(rows_all), "caveat": CAVEAT}
    if not rows:
        return report
    outs = outcomes_of(rows)
    probs = llm_probs(rows)
    llm_b = brier_per_row(probs, outs)
    report["llm"] = {"brier": round(float(llm_b.mean()), 4), "logloss": round(float(logloss_per_row(probs, outs).mean()), 4), "n": len(rows)}
    base_b = brier_per_row(baseline_base_rate(rows), outs)
    report["baselines"] = {
        "uniform": round(float(brier_per_row(baseline_uniform(rows), outs).mean()), 4),
        "base_rate": round(float(base_b.mean()), 4),
        "momentum": round(float(brier_per_row(baseline_momentum(rows), outs).mean()), 4),
        "base_rates": {c: round(float(v), 3) for c, v in zip(CLASSES, base_rates(rows))},
    }
    diff = llm_b - base_b
    lo, hi = block_bootstrap_ci(diff)
    llm_lo, llm_hi = block_bootstrap_ci(llm_b)
    report["bootstrap"] = {
        "llm_brier_ci95": (round(llm_lo, 4), round(llm_hi, 4)),
        "llm_minus_base_rate_ci95": (round(lo, 4), round(hi, 4)),
        "significantly_better": hi < 0.0,
        "blocks": BLOCK_SIZE,
        "rounds": BOOTSTRAP_ROUNDS,
    }
    report["reliability_p_up"] = reliability_table(rows, "p_up", "UP")
    report["reliability_p_down"] = reliability_table(rows, "p_down", "DOWN")
    report["by_ny_2h"] = group_brier(rows, ny_bucket)
    report["by_weekday"] = group_brier(rows, weekday_bucket)
    report["by_model"] = group_brier(rows, lambda r: str(r.get("model", "")))
    report["by_debate"] = group_brier(rows, lambda r: bool(r.get("used_debate")))
    report["by_debate_axis"] = group_brier(rows, lambda r: str(r.get("debate_axis") or "none") if r.get("used_debate") else "none")
    report["by_regime"] = group_brier(rows, lambda r: str(r.get("regime") or "n/a"))
    report["by_horizon"] = group_brier(rows, lambda r: int(r.get("horizon_bars", 0)))
    report["anchor"] = anchor_deviation(rows)
    return report


def format_report(report: dict[str, Any]) -> str:
    lines = ["=== GP-MATE Forecast Calibration ==="]
    c = report["counts"]
    lines.append(f"rows={c['total']} resolved={c['resolved']} scorable={c['scorable']} ambiguous={c['ambiguous']} llm_failed={c['llm_failed']} invalid_probs={c['invalid_probs']}")
    if "llm" not in report:
        lines.append("(no scorable rows yet)")
        lines.append(report["caveat"])
        return "\n".join(lines)
    llm, base = report["llm"], report["baselines"]
    lines.append(f"\n[Scores] LLM Brier={llm['brier']} logloss={llm['logloss']} (n={llm['n']})")
    lines.append(f"  baselines: uniform={base['uniform']} base_rate={base['base_rate']} momentum={base['momentum']}  base rates={base['base_rates']}")
    b = report["bootstrap"]
    verdict = "YES" if b["significantly_better"] else "no"
    lines.append(f"  LLM Brier 95% CI={b['llm_brier_ci95']}  LLM-base_rate 95% CI={b['llm_minus_base_rate_ci95']}  significantly better than base rate: {verdict}")
    for title, key in (("p_up vs UP rate", "reliability_p_up"), ("p_down vs DOWN rate", "reliability_p_down")):
        lines.append(f"\n[Reliability: {title}]")
        lines.append("  bin        n   mean_pred  actual")
        for row in report[key]:
            if row["n"]:
                lines.append(f"  {row['bin']:<9} {row['n']:>4}   {row['mean_pred']:.3f}     {row['actual_rate']:.3f}")
    for title, key in (("NY 2h bucket", "by_ny_2h"), ("weekday", "by_weekday"), ("model", "by_model"), ("used_debate", "by_debate"), ("debate_axis", "by_debate_axis"), ("regime", "by_regime"), ("horizon", "by_horizon")):
        lines.append(f"\n[Brier by {title}]")
        for row in report.get(key, []):
            lines.append(f"  {str(row['group']):<14} n={row['n']:<5} brier={row['brier']}")
    anchor = report.get("anchor")
    lines.append("\n[Anchor deviation] (rows that carried reference_base_rates)")
    if not anchor:
        lines.append("  none yet")
    else:
        flag = "  <- beats the anchor" if anchor["beats_anchor"] else ""
        lines.append(f"  n={anchor['n']}  Brier LLM={anchor['brier_llm']}  anchor={anchor['brier_anchor']}  LLM-anchor CI95={anchor['llm_minus_anchor_ci95']}{flag}")
        lines.append(
            f"  deviation direction accuracy={anchor['deviation_direction_accuracy']} (n={anchor['deviation_direction_n']})  "
            f"moved toward UP={anchor['moved_toward_up']} DOWN={anchor['moved_toward_down']} flat={anchor['flat']}"
        )
        lines.append(f"  |tilt| median={anchor['abs_tilt_median']} p90={anchor['abs_tilt_p90']}  mean deviation={anchor['mean_deviation']}")
        lines.append("  Anchored forecasts echo the anchor's side; read the deviation direction, not raw p_up vs p_down.")
    lines.append("\n" + report["caveat"])
    return "\n".join(lines)


def write_csv(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(["section", "group", "n", "mean_pred", "actual_rate", "brier"])
        for key in ("reliability_p_up", "reliability_p_down"):
            for row in report.get(key, []):
                writer.writerow([key, row["bin"], row["n"], row["mean_pred"], row["actual_rate"], ""])
        for key in ("by_ny_2h", "by_weekday", "by_model", "by_debate", "by_debate_axis", "by_regime", "by_horizon"):
            for row in report.get(key, []):
                writer.writerow([key, row["group"], row["n"], "", "", row["brier"]])
        anchor = report.get("anchor")
        if anchor:
            writer.writerow(["anchor", "llm", anchor["n"], "", "", anchor["brier_llm"]])
            writer.writerow(["anchor", "anchor", anchor["n"], "", "", anchor["brier_anchor"]])
            writer.writerow(["anchor", "deviation_direction_accuracy", anchor["deviation_direction_n"], "", anchor["deviation_direction_accuracy"], ""])


def evaluate_file(path: Path | None = None) -> dict[str, Any]:
    return evaluate(read_forecasts(path))
