"""Post-hoc calibration of forecast-only probabilities (no LLM calls).

Question: can a fixed transform of the LLM's (p_up, p_down, p_timeout) beat
the base-rate baseline out of sample? Two small families are fitted on the
first part of the record (chronological) and scored on the rest:

* shrink:      q = w * p + (1 - w) * base_rates          (1 parameter)
* shrink+tilt: as above, then move mass ``t`` from p_down to p_up
               (t may be negative), renormalised           (2 parameters)

Base rates used inside the transform are estimated on the fit split only, so
the test split is never seen. If neither transform beats the base rate on the
test split, the LLM's probabilities carry no usable signal at this horizon.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from analysis.forecast_eval import CLASSES, base_rates, block_bootstrap_ci, brier_per_row, llm_probs, outcomes_of, scorable_rows

SHRINK_GRID = np.linspace(0.0, 1.0, 21)
TILT_GRID = np.linspace(-0.25, 0.25, 26)


def _apply(probs: np.ndarray, rates: np.ndarray, w: float, tilt: float = 0.0) -> np.ndarray:
    q = w * probs + (1.0 - w) * rates
    if tilt:
        moved = np.clip(np.minimum(np.abs(tilt), q[:, 1] if tilt > 0 else q[:, 0]), 0.0, None)
        if tilt > 0:
            q[:, 0] += moved
            q[:, 1] -= moved
        else:
            q[:, 0] -= moved
            q[:, 1] += moved
    q = np.clip(q, 1e-6, None)
    return q / q.sum(axis=1, keepdims=True)


def _fit(probs: np.ndarray, outs: list[str], rates: np.ndarray, with_tilt: bool) -> tuple[float, float, float]:
    best = (float("inf"), 1.0, 0.0)
    tilts = TILT_GRID if with_tilt else np.array([0.0])
    for w in SHRINK_GRID:
        for t in tilts:
            score = float(brier_per_row(_apply(probs.copy(), rates, float(w), float(t)), outs).mean())
            if score < best[0]:
                best = (score, float(w), float(t))
    return best


def split_chronologically(rows: list[dict[str, Any]], fit_fraction: float = 0.5) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ordered = scorable_rows(rows)
    cut = int(len(ordered) * fit_fraction)
    return ordered[:cut], ordered[cut:]


def evaluate_calibration(rows: list[dict[str, Any]], fit_fraction: float = 0.5) -> dict[str, Any]:
    fit_rows, test_rows = split_chronologically(rows, fit_fraction)
    if len(fit_rows) < 20 or len(test_rows) < 20:
        return {"n_fit": len(fit_rows), "n_test": len(test_rows), "note": "too few rows for a fit/test split"}

    fit_p, fit_o = llm_probs(fit_rows), outcomes_of(fit_rows)
    test_p, test_o = llm_probs(test_rows), outcomes_of(test_rows)
    fit_rates = base_rates(fit_rows)  # test split never touches its own rates inside the transform

    raw_test = brier_per_row(test_p, test_o)
    base_test = brier_per_row(np.tile(fit_rates, (len(test_rows), 1)), test_o)
    oracle_test = brier_per_row(np.tile(base_rates(test_rows), (len(test_rows), 1)), test_o)

    results: dict[str, Any] = {
        "n_fit": len(fit_rows),
        "n_test": len(test_rows),
        "fit_base_rates": {c: round(float(v), 3) for c, v in zip(CLASSES, fit_rates)},
        "test": {
            "llm_raw": round(float(raw_test.mean()), 4),
            "base_rate_from_fit": round(float(base_test.mean()), 4),
            "base_rate_oracle": round(float(oracle_test.mean()), 4),
        },
        "transforms": [],
    }
    for name, with_tilt in (("shrink", False), ("shrink+tilt", True)):
        fit_score, w, t = _fit(fit_p, fit_o, fit_rates, with_tilt)
        test_scores = brier_per_row(_apply(test_p.copy(), fit_rates, w, t), test_o)
        lo, hi = block_bootstrap_ci(test_scores - base_test)
        results["transforms"].append(
            {
                "name": name,
                "w": round(w, 2),
                "tilt": round(t, 3),
                "fit_brier": round(fit_score, 4),
                "test_brier": round(float(test_scores.mean()), 4),
                "test_vs_base_ci95": (round(lo, 4), round(hi, 4)),
                "beats_base_rate_out_of_sample": hi < 0.0,
            }
        )
    return results


def format_calibration(report: dict[str, Any]) -> str:
    lines = ["=== GP-MATE Forecast Calibration (fit first half, test second half) ==="]
    if "note" in report:
        lines.append(f"n_fit={report['n_fit']} n_test={report['n_test']}: {report['note']}")
        return "\n".join(lines)
    t = report["test"]
    lines.append(f"n_fit={report['n_fit']} n_test={report['n_test']}  fit base rates={report['fit_base_rates']}")
    lines.append(f"test Brier: llm_raw={t['llm_raw']}  base_rate(fit)={t['base_rate_from_fit']}  base_rate(oracle, test-own)={t['base_rate_oracle']}")
    for tr in report["transforms"]:
        flag = "  <- beats base rate out of sample" if tr["beats_base_rate_out_of_sample"] else ""
        lines.append(
            f"  {tr['name']:<12} w={tr['w']:.2f} tilt={tr['tilt']:+.3f}  fit={tr['fit_brier']}  test={tr['test_brier']}  vs base CI95={tr['test_vs_base_ci95']}{flag}"
        )
    lines.append("w=0 means 'ignore the LLM entirely'; a fitted w near 0 says the probabilities carry no usable signal.")
    return "\n".join(lines)
