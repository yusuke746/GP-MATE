from __future__ import annotations

from datetime import UTC, datetime

from agents.data.macro_inputs import build_macro_inputs


def _fred() -> dict:
    return {
        "dxy": {"value": 120.0, "change_30d": 1.0, "direction": "UP", "source": "fred:DTWEXBGS"},
        "us2y": {"value": 3.6, "change_30d": -0.2, "direction": "DOWN"},
        "_meta": {"ok": True, "warnings": []},
    }


def test_live_dollar_index_replaces_fred_and_keeps_original() -> None:
    now = datetime(2026, 9, 4, 14, 0, tzinfo=UTC)
    events = [
        {"title": "Non-Farm Employment Change", "currency": "USD", "impact": "high",
         "datetime_utc": datetime(2026, 9, 4, 12, 30, tzinfo=UTC), "forecast": "75K", "previous": "73K", "actual": "142K"},
        {"title": "ISM Services PMI", "currency": "USD", "impact": "high",
         "datetime_utc": datetime(2026, 9, 5, 14, 0, tzinfo=UTC), "forecast": "51.0", "previous": "50.1", "actual": ""},
    ]
    enriched = build_macro_inputs(
        _fred(),
        now=now,
        dollar_index_fetcher=lambda: {"value": 98.2, "change_30d": -1.1, "direction": "DOWN", "change_5d": 0.3,
                                      "direction_5d": "UP", "source": "mt5:USDX", "_meta": {"ok": True, "error": ""}},
        positioning_fetcher=lambda: {"cot": {"crowding": "NORMAL", "_meta": {"ok": True}}, "gld": {"_meta": {"ok": False, "error": "x"}},
                                     "_meta": {"ok": True, "error": ""}},
        calendar_fetcher=lambda: events,
    )
    assert enriched["dxy"]["direction"] == "DOWN"
    assert enriched["dxy"]["source"] == "mt5:USDX"
    assert "_meta" not in enriched["dxy"]
    assert enriched["dxy_fred"]["direction"] == "UP"
    assert enriched["positioning"]["cot"]["crowding"] == "NORMAL"
    assert enriched["recent_releases"][0]["surprise"] == 67.0
    assert enriched["upcoming_events"][0]["title"] == "ISM Services PMI"
    assert enriched["_meta"]["warnings"] == []
    # us2y untouched
    assert enriched["us2y"]["direction"] == "DOWN"


def test_missing_layers_degrade_to_warnings() -> None:
    enriched = build_macro_inputs(
        _fred(),
        dollar_index_fetcher=lambda: {"_meta": {"ok": False, "error": "connect failed"}},
        positioning_fetcher=lambda: (_ for _ in ()).throw(RuntimeError("boom")),
        calendar_fetcher=lambda: None,
    )
    assert enriched["dxy"]["source"] == "fred:DTWEXBGS"
    assert "dxy_fred" not in enriched
    assert "recent_releases" not in enriched
    warnings = enriched["_meta"]["warnings"]
    assert any("live dollar index unavailable" in w for w in warnings)
    assert any("positioning unavailable" in w for w in warnings)
    assert any("calendar unavailable" in w for w in warnings)


def test_empty_macro_data_does_not_raise() -> None:
    enriched = build_macro_inputs(
        {},
        dollar_index_fetcher=lambda: {"_meta": {"ok": False, "error": "x"}},
        positioning_fetcher=lambda: {"_meta": {"ok": False, "error": "y"}},
        calendar_fetcher=lambda: [],
    )
    assert enriched["recent_releases"] == []
    assert enriched["upcoming_events"] == []
    assert enriched["_meta"]["warnings"]
