from __future__ import annotations

import numpy as np

from analysis import forecast_calibration as fc


def _row(i: int, p_up: float, p_down: float, outcome: str) -> dict:
    return {
        "ts_utc": f"2026-09-{1 + i // 24:02d}T{i % 24:02d}:00:00+00:00",
        "ok": True,
        "probs_valid": True,
        "p_up": p_up,
        "p_down": p_down,
        "p_timeout": round(1.0 - p_up - p_down, 6),
        "outcome": outcome,
        "horizon_bars": 6,
    }


def _outcomes(n: int, seed: int = 3) -> list[str]:
    rng = np.random.default_rng(seed)
    return [str(rng.choice(["UP", "DOWN", "TIMEOUT"], p=[0.35, 0.4, 0.25])) for _ in range(n)]


def _sharp_rows(n: int = 200) -> list[dict]:
    rows = []
    for i, outcome in enumerate(_outcomes(n)):
        probs = {"UP": (0.7, 0.15), "DOWN": (0.15, 0.7), "TIMEOUT": (0.15, 0.15)}[outcome]
        rows.append(_row(i, *probs, outcome))
    return rows


def _noise_rows(n: int = 200) -> list[dict]:
    rng = np.random.default_rng(11)
    rows = []
    for i, outcome in enumerate(_outcomes(n)):
        p = rng.dirichlet([2.0, 2.0, 2.0])
        rows.append(_row(i, round(float(p[0]), 6), round(float(p[1]), 6), outcome))
    return rows


def test_split_is_chronological_and_reports_too_few_rows() -> None:
    rows = _sharp_rows(30)
    fit, test = fc.split_chronologically(rows[::-1], 0.5)  # reversed input must still split by time
    assert len(fit) == 15 and len(test) == 15
    assert fit[-1]["ts_utc"] < test[0]["ts_utc"]
    report = fc.evaluate_calibration(rows, 0.5)
    assert report["n_fit"] == 15 and "too few" in report["note"]
    assert "too few" in fc.format_calibration(report)


def test_apply_shrinks_toward_rates_and_moves_mass_between_up_and_down() -> None:
    probs = np.array([[0.7, 0.2, 0.1]])
    rates = np.array([0.3, 0.4, 0.3])
    assert np.allclose(fc._apply(probs.copy(), rates, 1.0), probs)
    assert np.allclose(fc._apply(probs.copy(), rates, 0.0), rates)
    tilted = fc._apply(probs.copy(), rates, 1.0, tilt=0.1)
    assert np.allclose(tilted, [[0.8, 0.1, 0.1]])
    tilted_down = fc._apply(probs.copy(), rates, 1.0, tilt=-0.1)
    assert np.allclose(tilted_down, [[0.6, 0.3, 0.1]])
    # Never moves more mass than the source class holds; rows always renormalise to 1.
    capped = fc._apply(np.array([[0.05, 0.9, 0.05]]), rates, 1.0, tilt=-0.5)
    assert abs(capped.sum() - 1.0) < 1e-9 and capped[0, 0] > 0


def test_sharp_forecaster_keeps_high_weight_and_beats_base_rate() -> None:
    report = fc.evaluate_calibration(_sharp_rows(), 0.5)
    shrink = next(t for t in report["transforms"] if t["name"] == "shrink")
    assert shrink["w"] >= 0.8
    assert shrink["test_brier"] < report["test"]["base_rate_from_fit"]
    assert shrink["beats_base_rate_out_of_sample"] is True
    assert report["test"]["llm_raw"] < report["test"]["base_rate_from_fit"]
    text = fc.format_calibration(report)
    assert "beats base rate out of sample" in text and "w=0 means" in text


def test_noise_forecaster_is_shrunk_to_zero_and_does_not_beat_base_rate() -> None:
    report = fc.evaluate_calibration(_noise_rows(), 0.5)
    shrink = next(t for t in report["transforms"] if t["name"] == "shrink")
    assert shrink["w"] <= 0.2
    assert shrink["beats_base_rate_out_of_sample"] is False
    assert report["test"]["llm_raw"] > report["test"]["base_rate_from_fit"]
    tilt = next(t for t in report["transforms"] if t["name"] == "shrink+tilt")
    assert tilt["beats_base_rate_out_of_sample"] is False
    assert set(report["fit_base_rates"]) == {"UP", "DOWN", "TIMEOUT"}
