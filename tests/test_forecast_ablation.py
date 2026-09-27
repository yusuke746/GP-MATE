from __future__ import annotations

import json
from unittest.mock import Mock

import pytest

from analysis import forecast_ablation as fa


def _payload() -> dict:
    return {
        "task": {"p0_close": 4360.0, "barrier_up": 4380.0, "barrier_down": 4340.0, "horizon_bars": 6},
        "technical": {"signal": "SELL"},
        "sentiment": {"score": -0.6},
        "macro": {"macro_bias": "NEUTRAL"},
    }


def _forecast_row(i: int, outcome: str, p_up: float = 0.35, p_down: float = 0.47) -> dict:
    return {
        "forecast_id": f"id{i:03d}",
        "ts_utc": f"2026-09-{14 + i // 24:02d}T{i % 24:02d}:00:00+00:00",
        "ok": True,
        "probs_valid": True,
        "p_up": p_up,
        "p_down": p_down,
        "p_timeout": round(1 - p_up - p_down, 6),
        "outcome": outcome,
        "horizon_bars": 6,
    }


def test_apply_condition_removes_only_the_named_reports() -> None:
    payload = _payload()
    assert set(fa.apply_condition(payload, "no_sentiment")) == {"task", "technical", "macro"}
    assert set(fa.apply_condition(payload, "no_macro")) == {"task", "technical", "sentiment"}
    assert set(fa.apply_condition(payload, "technical_only")) == {"task", "technical"}
    assert fa.apply_condition(payload, "full_rerun") == payload
    assert set(payload) == {"task", "technical", "sentiment", "macro"}  # original untouched
    with pytest.raises(ValueError):
        fa.apply_condition(payload, "nope")


def test_load_archived_payload_by_id(tmp_path) -> None:
    (tmp_path / "abc.json").write_text(json.dumps({"payload": _payload(), "reports": {}}), encoding="utf-8")
    (tmp_path / "bad.json").write_text("{not json", encoding="utf-8")
    assert fa.load_archived_payload("abc", tmp_path)["task"]["p0_close"] == 4360.0
    assert fa.load_archived_payload("bad", tmp_path) is None
    assert fa.load_archived_payload("missing", tmp_path) is None


def test_run_ablation_calls_per_condition_and_resumes(tmp_path) -> None:
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    forecasts = [_forecast_row(i, "DOWN") for i in range(3)]
    for row in forecasts:
        (inputs / f"{row['forecast_id']}.json").write_text(json.dumps({"payload": _payload()}), encoding="utf-8")
    forecasts.append({**_forecast_row(9, None)})  # unresolved -> not scorable, never called
    out = tmp_path / "ablation.jsonl"
    seen: list[tuple[str, set]] = []

    def fake_forecaster(payload, *, model, samples, client):
        seen.append((model, set(payload)))
        return {"ok": True, "probs_valid": True, "p_up": 0.3, "p_down": 0.5, "p_timeout": 0.2, "key_reason": "x",
                "model": model, "error": "", "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11}}

    counts = fa.run_ablation(("no_sentiment", "no_macro"), forecasts=forecasts, inputs_directory=inputs,
                             output_path=out, model="m", forecaster=fake_forecaster)
    assert counts == {"targets": 3, "calls": 6, "skipped_done": 0, "missing_inputs": 0, "failed": 0}
    assert all(m == "m" for m, _ in seen)
    assert {"task", "technical", "macro"} in [k for _, k in seen] and {"task", "technical", "sentiment"} in [k for _, k in seen]
    rows = fa.read_ablation(out)
    assert len(rows) == 6 and rows[0]["condition"] == "no_sentiment" and rows[0]["outcome"] == "DOWN"

    # Second run: everything already done -> no calls.
    seen.clear()
    again = fa.run_ablation(("no_sentiment", "no_macro"), forecasts=forecasts, inputs_directory=inputs,
                            output_path=out, model="m", forecaster=fake_forecaster)
    assert again["calls"] == 0 and again["skipped_done"] == 6 and seen == []
    assert len(fa.read_ablation(out)) == 6


def test_run_ablation_records_missing_inputs_and_failures(tmp_path) -> None:
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    forecasts = [_forecast_row(0, "UP"), _forecast_row(1, "UP")]
    (inputs / "id000.json").write_text(json.dumps({"payload": _payload()}), encoding="utf-8")
    out = tmp_path / "ablation.jsonl"

    def failing(payload, *, model, samples, client):
        raise RuntimeError("api down")

    counts = fa.run_ablation(("no_macro",), forecasts=forecasts, inputs_directory=inputs, output_path=out,
                             forecaster=failing)
    assert counts["missing_inputs"] == 1 and counts["calls"] == 1 and counts["failed"] == 1
    row = fa.read_ablation(out)[0]
    assert row["ok"] is False and "api down" in row["error"]


def test_compare_conditions_scores_common_rows_and_flags_improvement() -> None:
    # 60 rows, 40 DOWN / 20 UP. Original is mildly wrong-way; no_sentiment is sharp and right.
    forecasts = [_forecast_row(i, "DOWN" if i % 3 else "UP", p_up=0.45, p_down=0.35) for i in range(60)]
    ablation = []
    for row in forecasts:
        good = {"UP": (0.85, 0.1), "DOWN": (0.1, 0.85)}[row["outcome"]]
        ablation.append({"forecast_id": row["forecast_id"], "condition": "no_sentiment", "ok": True, "probs_valid": True,
                         "p_up": good[0], "p_down": good[1], "p_timeout": 0.05, "outcome": row["outcome"]})
        ablation.append({"forecast_id": row["forecast_id"], "condition": "no_macro", "ok": True, "probs_valid": True,
                         "p_up": 0.45, "p_down": 0.35, "p_timeout": 0.2, "outcome": row["outcome"]})
    ablation.append({"forecast_id": "unknown", "condition": "no_macro", "ok": True, "probs_valid": True,
                     "p_up": 0.3, "p_down": 0.3, "p_timeout": 0.4})  # not in originals -> ignored
    ablation.pop(-2)  # drop one no_macro row -> common set shrinks to 59

    report = fa.compare_conditions(forecasts, ablation)
    assert report["n"] == 59
    by = {c["condition"]: c for c in report["conditions"]}
    assert set(by) == {"original", "no_sentiment", "no_macro"}
    assert by["no_sentiment"]["brier"] < by["original"]["brier"]
    assert by["no_sentiment"]["better_than_original"] is True
    assert by["no_sentiment"]["beats_base_rate"] is True
    assert by["no_macro"]["brier"] == by["original"]["brier"]
    assert by["no_macro"]["better_than_original"] is False
    assert by["original"]["mean_tilt"] == pytest.approx(0.1)
    assert by["no_sentiment"]["direction_accuracy"] == 1.0
    text = fa.format_comparison(report)
    assert "no_sentiment" in text and "*" in text and "ブロック" in text


def test_compare_conditions_handles_empty() -> None:
    report = fa.compare_conditions([_forecast_row(0, "UP")], [])
    assert report["n"] == 0
    assert "nothing to compare" in fa.format_comparison(report) or "no ablation" in fa.format_comparison(report)
