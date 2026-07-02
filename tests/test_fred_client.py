from __future__ import annotations

from unittest.mock import Mock, patch

from agents.data import fred_client
from agents.data.fred_client import get_macro_data


def _mock_response(json_data: dict[str, object]) -> Mock:
    response = Mock()
    response.raise_for_status = Mock()
    response.json = Mock(return_value=json_data)
    return response


def _fred_series_payload(values: list[tuple[str, str]]) -> dict[str, object]:
    return {
        "observations": [
            {"date": obs_date, "value": obs_value}
            for obs_date, obs_value in values
        ]
    }


def test_get_macro_data_returns_safe_failure_without_api_key(monkeypatch) -> None:
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    fred_client._DAILY_CACHE.clear()

    with patch("agents.data.fred_client.requests.get") as mock_get:
        result = get_macro_data(force_refresh=True)

    assert result["_meta"]["ok"] is False
    assert result["dxy"]["value"] is None
    assert result["_meta"]["error"] == "FRED_API_KEY missing"
    mock_get.assert_not_called()


def test_get_macro_data_returns_structured_data_on_success(monkeypatch) -> None:
    fred_client._DAILY_CACHE.clear()
    monkeypatch.setenv("FRED_API_KEY", "test-key")

    def fake_get(url: str, params: dict[str, object] | None = None, timeout: float | None = None) -> Mock:
        series_id = str((params or {}).get("series_id", ""))
        payloads = {
            "DTWEXBGS": _fred_series_payload(
                [
                    ("2026-06-01", "100.0"),
                    ("2026-07-01", "104.0"),
                ]
            ),
            "DFII10": _fred_series_payload(
                [
                    ("2026-06-01", "1.50"),
                    ("2026-07-01", "1.80"),
                ]
            ),
            "DGS10": _fred_series_payload(
                [
                    ("2026-06-01", "4.00"),
                    ("2026-07-01", "4.20"),
                ]
            ),
            "T10YIE": _fred_series_payload(
                [
                    ("2026-06-01", "2.10"),
                    ("2026-07-01", "2.30"),
                ]
            ),
            "FEDFUNDS": _fred_series_payload(
                [
                    ("2026-05-01", "5.25"),
                    ("2026-06-01", "5.00"),
                ]
            ),
        }
        response = _mock_response(payloads[series_id])
        return response

    with patch("agents.data.fred_client.requests.get", side_effect=fake_get):
        result = get_macro_data(force_refresh=True)

    assert result["_meta"]["ok"] is True
    assert result["dxy"]["value"] == 104.0
    assert result["dxy"]["change_30d"] == 4.0
    assert result["dxy"]["direction"] == "UP"
    assert result["real_rate"]["change_30d"] == 0.30000000000000004
    assert result["us10y"]["direction"] == "UP"
    assert result["breakeven"]["value"] == 2.3
    assert result["fed_funds"]["direction"] == "DOWN"
    assert result["as_of"]


def test_get_macro_data_uses_same_day_cache(monkeypatch) -> None:
    fred_client._DAILY_CACHE.clear()
    monkeypatch.setenv("FRED_API_KEY", "test-key")
    monkeypatch.setattr(fred_client, "_today_key", lambda: "2026-07-03")

    call_count = {"count": 0}

    def fake_get(url: str, params: dict[str, object] | None = None, timeout: float | None = None) -> Mock:
        call_count["count"] += 1
        series_id = str((params or {}).get("series_id", ""))
        return _mock_response(
            _fred_series_payload(
                [
                    ("2026-06-01", "1.0"),
                    ("2026-07-01", "2.0"),
                ]
            )
            if series_id
            else {"observations": []}
        )

    with patch("agents.data.fred_client.requests.get", side_effect=fake_get):
        first = get_macro_data(force_refresh=True)
        second = get_macro_data()

    assert first["_meta"]["ok"] is True
    assert second["_meta"]["cached"] is True
    assert call_count["count"] == 5
