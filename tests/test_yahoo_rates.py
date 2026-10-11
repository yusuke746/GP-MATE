from __future__ import annotations

from datetime import UTC, datetime, timedelta

from agents.data import yahoo_rates
from agents.data.macro_inputs import build_macro_inputs
from agents.data.yahoo_rates import get_treasury_yields_snapshot, parse_chart_payload, yield_snapshot_from_chart


def _ts(day: datetime) -> int:
    return int(day.replace(tzinfo=UTC).timestamp())


def _payload(symbol: str, closes: list[float | None], last_day: datetime, live: float | None, live_time: datetime | None, prev_close: float | None = None) -> dict:
    days = [last_day - timedelta(days=len(closes) - 1 - i) for i in range(len(closes))]
    meta = {"symbol": symbol, "currency": "USD", "exchangeTimezoneName": "America/New_York"}
    if live is not None:
        meta["regularMarketPrice"] = live
    if live_time is not None:
        meta["regularMarketTime"] = _ts(live_time)
    if prev_close is not None:
        meta["chartPreviousClose"] = prev_close
    return {
        "chart": {
            "result": [{"meta": meta, "timestamp": [_ts(d) for d in days], "indicators": {"quote": [{"close": closes}]}}],
            "error": None,
        }
    }


def test_parse_chart_payload_skips_nulls_and_reads_live_meta() -> None:
    last_day = datetime(2026, 10, 8, 20, 0)
    payload = _payload("^TNX", [5.10, None, 5.20, 5.295, 5.17], last_day, live=5.17, live_time=datetime(2026, 10, 8, 20, 59), prev_close=5.295)
    parsed = parse_chart_payload(payload)
    assert parsed is not None
    assert [v for _, v in parsed["points"]] == [5.10, 5.20, 5.295, 5.17]
    assert parsed["live_price"] == 5.17 and parsed["previous_close"] == 5.295
    assert parsed["live_time"] == datetime(2026, 10, 8, 20, 59, tzinfo=UTC)
    assert parse_chart_payload({"chart": {"result": None, "error": {"code": "Not Found"}}}) is None
    assert parse_chart_payload({"chart": {"result": [{"meta": {}, "timestamp": [], "indicators": {"quote": [{"close": []}]}}]}}) is None
    assert parse_chart_payload(None) is None


def test_snapshot_measures_change_1d_against_the_last_completed_day() -> None:
    # Daily series ends with today's partial bar (10/8); the live quote is the
    # same bar. change_1d must compare against the 10/7 close, not against itself.
    closes = [5.0 + 0.01 * i for i in range(44)] + [5.295, 5.17]  # 46 points, last = today
    last_day = datetime(2026, 10, 8, 14, 0)
    now = datetime(2026, 10, 9, 0, 0, tzinfo=UTC)
    payload = _payload("^TNX", closes, last_day, live=5.17, live_time=datetime(2026, 10, 8, 21, 0))
    snap = yield_snapshot_from_chart(parse_chart_payload(payload), "^TNX", now=now)
    assert snap is not None
    assert snap["value"] == 5.17 and snap["source"] == "yahoo:^TNX" and snap["unit"] == "percent"
    assert abs(snap["change_1d"] - (5.17 - 5.295)) < 1e-9 and snap["direction_1d"] == "DOWN"
    assert snap["as_of"] == "2026-10-08T21:00:00+00:00"
    assert snap["stale_minutes"] == 180.0
    # 5d / 30d are re-based on the live value against completed days
    completed = closes[:-1]
    assert abs(snap["change_5d"] - (5.17 - completed[-5])) < 1e-9
    assert snap["change_30d"] is not None and snap["direction"] in {"UP", "DOWN"}


def test_snapshot_without_live_quote_uses_last_close_and_previous_close() -> None:
    closes = [4.0, 4.1, 4.2]
    payload = _payload("^TYX", closes, datetime(2026, 10, 7, 20, 0), live=None, live_time=None, prev_close=None)
    snap = yield_snapshot_from_chart(parse_chart_payload(payload), "^TYX", now=datetime(2026, 10, 8, 12, 0, tzinfo=UTC))
    assert snap["value"] == 4.2 and snap["stale_minutes"] is None
    assert abs(snap["change_1d"] - 0.1) < 1e-9  # 10/7 close vs 10/6 close (last completed before "now")


def test_get_treasury_yields_snapshot_requires_the_10y_and_lists_missing_symbols() -> None:
    last_day = datetime(2026, 10, 8, 20, 0)

    def _fetch(symbol: str):
        if symbol == "^TNX":
            return _payload(symbol, [5.2, 5.3, 5.17], last_day, live=5.17, live_time=datetime(2026, 10, 8, 21, 0))
        if symbol == "^TYX":
            return _payload(symbol, [5.6, 5.7, 5.55], last_day, live=5.55, live_time=datetime(2026, 10, 8, 21, 0))
        return None  # 2YY=F and ^FVX unreachable

    snap = get_treasury_yields_snapshot(now=datetime(2026, 10, 8, 22, 0, tzinfo=UTC), fetch=_fetch)
    assert snap["_meta"]["ok"] is True and snap["_meta"]["cached"] is False
    assert set(snap) == {"us10y", "us30y", "_meta"}
    assert snap["us10y"]["value"] == 5.17 and snap["us30y"]["source"] == "yahoo:^TYX"
    assert sorted(snap["_meta"]["warnings"]) == ["us2y (2YY=F) unavailable", "us5y (^FVX) unavailable"]

    failed = get_treasury_yields_snapshot(now=datetime(2026, 10, 8, 22, 0, tzinfo=UTC), fetch=lambda symbol: None)
    assert failed["_meta"]["ok"] is False and "us10y" in failed["_meta"]["error"]

    exploding = get_treasury_yields_snapshot(now=datetime(2026, 10, 8, 22, 0, tzinfo=UTC), fetch=lambda symbol: (_ for _ in ()).throw(RuntimeError("boom")))
    assert exploding["_meta"]["ok"] is False


def test_get_treasury_yields_snapshot_caches_live_fetches(monkeypatch) -> None:
    calls: list[str] = []
    last_day = datetime(2026, 10, 8, 20, 0)

    def _fetch(symbol: str):
        calls.append(symbol)
        return _payload(symbol, [5.2, 5.3, 5.17], last_day, live=5.17, live_time=datetime(2026, 10, 8, 21, 0))

    monkeypatch.setattr(yahoo_rates, "fetch_yahoo_chart", _fetch)
    monkeypatch.setattr(yahoo_rates, "_CACHE", {"at": 0.0, "data": None})
    first = get_treasury_yields_snapshot(now=datetime(2026, 10, 8, 22, 0, tzinfo=UTC))
    second = get_treasury_yields_snapshot(now=datetime(2026, 10, 8, 22, 5, tzinfo=UTC))
    assert first["_meta"]["ok"] and len(calls) == len(yahoo_rates.YIELD_SYMBOLS)
    assert second["_meta"]["cached"] is True and len(calls) == len(yahoo_rates.YIELD_SYMBOLS)


def test_build_macro_inputs_replaces_fred_yields_and_keeps_them_for_comparison() -> None:
    fred = {
        "us10y": {"value": 5.295, "change_30d": 0.4, "direction": "UP", "change_5d": 0.1, "direction_5d": "UP", "as_of": "2026-10-07", "source": "fred:DGS10"},
        "us2y": {"value": 4.9, "change_30d": 0.2, "direction": "UP", "source": "fred:DGS2"},
        "_meta": {"ok": True, "warnings": []},
    }
    live = {
        "us10y": {"value": 5.17, "change_1d": -0.125, "direction_1d": "DOWN", "change_5d": -0.05, "direction_5d": "DOWN", "change_30d": 0.3, "direction": "UP", "as_of": "2026-10-08T21:00:00+00:00", "source": "yahoo:^TNX", "stale_minutes": 60.0},
        "us30y": {"value": 5.55, "change_1d": -0.1, "direction_1d": "DOWN", "source": "yahoo:^TYX"},
        "_meta": {"ok": True, "error": "", "warnings": ["us2y (2YY=F) unavailable"]},
    }
    enriched = build_macro_inputs(
        fred,
        dollar_index_fetcher=lambda: {"_meta": {"ok": False, "error": "x"}},
        positioning_fetcher=lambda: {"_meta": {"ok": False, "error": "y"}},
        calendar_fetcher=lambda: [],
        yields_fetcher=lambda: live,
    )
    assert enriched["us10y"]["value"] == 5.17 and enriched["us10y"]["direction_1d"] == "DOWN"
    assert enriched["us10y_fred"]["value"] == 5.295
    assert enriched["us30y"]["source"] == "yahoo:^TYX" and "us30y_fred" not in enriched
    assert enriched["us2y"]["source"] == "fred:DGS2" and "us2y_fred" not in enriched  # not fetched: FRED stays
    assert any("yahoo yields: us2y (2YY=F) unavailable" in w for w in enriched["_meta"]["warnings"])

    degraded = build_macro_inputs(
        fred,
        dollar_index_fetcher=lambda: {"_meta": {"ok": False, "error": "x"}},
        positioning_fetcher=lambda: {"_meta": {"ok": False, "error": "y"}},
        calendar_fetcher=lambda: [],
        yields_fetcher=lambda: {"_meta": {"ok": False, "error": "required yield series unavailable: us10y"}},
    )
    assert degraded["us10y"]["source"] == "fred:DGS10" and "us10y_fred" not in degraded
    assert any("same-day treasury yields unavailable" in w for w in degraded["_meta"]["warnings"])


def test_macro_analyst_data_notes_explain_live_yields_without_directional_words() -> None:
    from agents.macro_analyst import DATA_NOTES

    assert "yahoo:^TNX" in DATA_NOTES["us10y"] and "1日遅れ" in DATA_NOTES["us10y"]
    assert "change_1d" in DATA_NOTES["yields_live"] and "stale_minutes" in DATA_NOTES["yields_live"]
    for key in ("us5y", "us30y"):
        assert key in DATA_NOTES
    for forbidden in ("ポジティブ", "ネガティブ", "追い風", "逆風"):
        assert forbidden not in " ".join(DATA_NOTES.values())
