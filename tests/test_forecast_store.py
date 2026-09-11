from __future__ import annotations

import json

from analysis import forecast_store


def test_append_read_write_roundtrip(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(forecast_store, "LOG_DIR", tmp_path)
    forecast_store.append_forecast({"forecast_id": "a", "outcome": None, "p_up": 0.4})
    forecast_store.append_forecast({"forecast_id": "b", "outcome": None, "p_up": 0.6})
    rows = forecast_store.read_forecasts()
    assert [r["forecast_id"] for r in rows] == ["a", "b"]

    rows[0]["outcome"] = "UP"
    forecast_store.write_forecasts(rows)
    again = forecast_store.read_forecasts()
    assert again[0]["outcome"] == "UP" and again[1]["outcome"] is None
    assert not list(tmp_path.glob(".forecasts-*.tmp"))  # temp file cleaned up


def test_read_skips_corrupt_lines(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(forecast_store, "LOG_DIR", tmp_path)
    path = forecast_store.forecasts_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"forecast_id": "ok"}\nnot json\n\n[1,2]\n', encoding="utf-8")
    rows = forecast_store.read_forecasts()
    assert [r["forecast_id"] for r in rows] == ["ok"]


def test_save_inputs_writes_json_under_forecast_inputs(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(forecast_store, "LOG_DIR", tmp_path)
    path = forecast_store.save_inputs("abc", {"payload": {"task": {"p0": 1.0}}, "reports": {}})
    assert path.endswith("forecast_inputs/abc.json") or path.endswith("forecast_inputs\\abc.json")
    assert json.loads(open(path, encoding="utf-8").read())["payload"]["task"]["p0"] == 1.0
