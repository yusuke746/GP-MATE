from __future__ import annotations

from unittest.mock import Mock

from agents import forecaster


def _result(payload: dict, ok: bool = True, tokens: int = 100) -> Mock:
    result = Mock()
    result.ok = ok
    result.payload = payload
    result.model = "gpt-5.6-terra"
    result.error = "" if ok else "boom"
    result.usage = Mock(prompt_tokens=tokens, completion_tokens=10, total_tokens=tokens + 10)
    return result


def test_normalize_probabilities_rescales_within_tolerance() -> None:
    probs, raw, reason = forecaster.normalize_probabilities({"p_up": 0.5, "p_down": 0.3, "p_timeout": 0.22})
    assert reason == ""
    assert raw == {"p_up": 0.5, "p_down": 0.3, "p_timeout": 0.22}
    assert abs(sum(probs.values()) - 1.0) < 1e-9
    assert probs["p_up"] == round(0.5 / 1.02, 6)


def test_normalize_probabilities_rejects_out_of_range_sum_and_garbage() -> None:
    probs, _, reason = forecaster.normalize_probabilities({"p_up": 0.7, "p_down": 0.5, "p_timeout": 0.2})
    assert probs is None and reason.startswith("sum_out_of_range")
    probs, _, reason = forecaster.normalize_probabilities({"p_up": "high", "p_down": 0.2, "p_timeout": 0.2})
    assert probs is None and reason == "missing_or_non_numeric"
    probs, _, reason = forecaster.normalize_probabilities({"p_up": -0.1, "p_down": 0.6, "p_timeout": 0.5})
    assert probs is None and reason == "negative_probability"


def test_forecast_single_sample_returns_normalized_and_raw() -> None:
    client = Mock()
    client.call_json.return_value = _result({"p_up": 0.55, "p_down": 0.25, "p_timeout": 0.2, "key_reason": "支持帯"})
    out = forecaster.forecast({"task": {}}, model="m", samples=1, client=client)
    assert out["ok"] is True and out["probs_valid"] is True
    assert out["p_up"] == 0.55 and out["p_down"] == 0.25 and out["p_timeout"] == 0.2
    assert out["p_raw"]["p_up"] == 0.55
    assert out["key_reason"] == "支持帯"
    assert out["samples"] == []  # single sample: no per-sample list
    assert out["usage"]["total_tokens"] == 110
    sent = client.call_json.call_args.kwargs
    assert sent["system_prompt"] == forecaster.SYSTEM_PROMPT
    assert sent["model"] == "m"


def test_forecast_multiple_samples_records_each_and_averages() -> None:
    client = Mock()
    client.call_json.side_effect = [
        _result({"p_up": 0.6, "p_down": 0.2, "p_timeout": 0.2}),
        _result({"p_up": 0.4, "p_down": 0.4, "p_timeout": 0.2}),
        _result({"p_up": 0.9, "p_down": 0.9, "p_timeout": 0.9}),  # invalid sum
    ]
    out = forecaster.forecast({"task": {}}, samples=3, client=client)
    assert out["n_samples"] == 3 and out["n_valid_samples"] == 2
    assert len(out["samples"]) == 3
    assert out["p_up"] == 0.5 and out["p_down"] == 0.3 and out["p_timeout"] == 0.2
    assert out["samples"][2]["invalid_reason"].startswith("sum_out_of_range")
    assert out["usage"]["prompt_tokens"] == 300


def test_forecast_invalid_output_is_flagged_not_raised() -> None:
    client = Mock()
    client.call_json.return_value = _result({"p_up": 0.9, "p_down": 0.9, "p_timeout": 0.0})
    out = forecaster.forecast({"task": {}}, client=client)
    assert out["ok"] is True
    assert out["probs_valid"] is False
    assert out["p_up"] is None
    assert out["invalid_reason"].startswith("sum_out_of_range")


def test_forecast_llm_failure_and_exception_are_fail_safe() -> None:
    client = Mock()
    client.call_json.return_value = _result({}, ok=False)
    out = forecaster.forecast({"task": {}}, client=client)
    assert out["ok"] is False and out["probs_valid"] is False
    assert "boom" in out["error"]

    client.call_json.side_effect = RuntimeError("network")
    out = forecaster.forecast({"task": {}}, client=client)
    assert out["ok"] is False and "network" in out["error"]


def test_build_forecast_payload_states_task_numerically() -> None:
    payload = forecaster.build_forecast_payload(
        symbol="GOLD#", bar_time_utc="2026-09-09T13:00:00+00:00", ts_utc="2026-09-09T14:00:00+00:00",
        p0=4363.5, atr_h1=21.27, k_up=1.0, k_down=1.0, horizon_bars=6,
        barrier_up=4384.77, barrier_down=4342.23,
        technical_report={"signal": "SELL"}, sentiment_report={"score": -0.6}, macro_report={"macro_bias": "NEUTRAL"},
    )
    task = payload["task"]
    assert task["barrier_up"] == 4384.77 and task["barrier_down"] == 4342.23 and task["horizon_bars"] == 6
    assert "6本" in task["question"] and "4384.77" in task["question"]
    assert "debate" not in payload
    assert payload["technical"] == {"signal": "SELL"}
