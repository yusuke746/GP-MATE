from __future__ import annotations

from datetime import UTC, date, datetime

from agents.data import releases


def _event(title: str, when: datetime, forecast: str = "", previous: str = "", actual: str = "") -> dict:
    return {
        "title": title,
        "currency": "USD",
        "impact": "high",
        "datetime_utc": when,
        "forecast": forecast,
        "previous": previous,
        "actual": actual,
    }


def test_parse_calendar_number_units() -> None:
    assert releases.parse_calendar_number("75K") == (75.0, "K")
    assert releases.parse_calendar_number("0.2%") == (0.2, "%")
    assert releases.parse_calendar_number("-1.5%") == (-1.5, "%")
    assert releases.parse_calendar_number("") == (None, "")
    assert releases.parse_calendar_number("n/a") == (None, "")


def test_match_spec_recognises_forexfactory_titles() -> None:
    assert releases.match_spec("Non-Farm Employment Change").key == "nfp"
    assert releases.match_spec("Core CPI m/m").key == "core_cpi_mom"
    assert releases.match_spec("CPI m/m").key == "cpi_mom"
    assert releases.match_spec("Unemployment Claims").key == "jobless_claims"
    assert releases.match_spec("FOMC Statement") is None


def test_nfp_actual_from_payems_diff_and_surprise() -> None:
    nfp_time = datetime(2026, 9, 4, 12, 30, tzinfo=UTC)
    points = [(date(2026, 7, 1), 158_000.0), (date(2026, 8, 1), 158_142.0)]  # +142K for August
    record = releases.build_release_record(
        _event("Non-Farm Employment Change", nfp_time, forecast="75K", previous="73K"),
        {"PAYEMS": points},
    )
    assert record["actual"] == 142.0
    assert record["unit"] == "K"
    assert record["surprise"] == 67.0
    assert record["actual_source"] == "fred:PAYEMS"
    assert record["first_order_read"].startswith("hawkish")


def test_actual_is_withheld_until_fred_updates() -> None:
    # CPI for August is released 2026-09-11; FRED still only has July -> no actual.
    cpi_time = datetime(2026, 9, 11, 12, 30, tzinfo=UTC)
    stale = [(date(2026, 6, 1), 320.0), (date(2026, 7, 1), 320.64)]
    record = releases.build_release_record(_event("CPI m/m", cpi_time, forecast="0.3%"), {"CPIAUCSL": stale})
    assert record["actual"] is None
    assert record["surprise"] is None

    fresh = stale + [(date(2026, 8, 1), 321.28)]
    record = releases.build_release_record(_event("CPI m/m", cpi_time, forecast="0.3%"), {"CPIAUCSL": fresh})
    assert record["actual"] == 0.2
    assert record["surprise"] == -0.1
    assert record["first_order_read"].startswith("dovish")


def test_calendar_actual_takes_precedence_over_fred() -> None:
    when = datetime(2026, 9, 3, 12, 30, tzinfo=UTC)
    record = releases.build_release_record(
        _event("Unemployment Claims", when, forecast="230K", actual="236K"),
        {"ICSA": [(date(2026, 8, 29), 999_000.0)]},
    )
    assert record["actual"] == 236.0
    assert record["actual_source"] == "calendar"
    assert record["surprise"] == 6.0


def test_build_recent_releases_filters_window_and_fetches_once_per_series() -> None:
    now = datetime(2026, 9, 4, 14, 0, tzinfo=UTC)
    events = [
        _event("Non-Farm Employment Change", datetime(2026, 9, 4, 12, 30, tzinfo=UTC), forecast="75K"),
        _event("Unemployment Rate", datetime(2026, 9, 4, 12, 30, tzinfo=UTC), forecast="4.3%"),
        _event("Core CPI m/m", datetime(2026, 8, 12, 12, 30, tzinfo=UTC), forecast="0.3%"),  # too old
        _event("FOMC Statement", datetime(2026, 9, 6, 18, 0, tzinfo=UTC)),  # future
    ]
    calls: list[str] = []

    def fake_fetch(series_id: str):
        calls.append(series_id)
        if series_id == "PAYEMS":
            return [(date(2026, 7, 1), 100.0), (date(2026, 8, 1), 175.0)]
        if series_id == "UNRATE":
            return [(date(2026, 8, 1), 4.4)]
        return []

    records = releases.build_recent_releases(events, now=now, lookback_hours=48, fetch_points=fake_fetch)
    assert [r["title"] for r in records] == ["Non-Farm Employment Change", "Unemployment Rate"]
    assert sorted(calls) == ["PAYEMS", "UNRATE"]
    nfp = records[0]
    assert nfp["actual"] == 75.0 and nfp["surprise"] == 0.0 and nfp["first_order_read"] == ""
    assert records[1]["actual"] == 4.4 and records[1]["surprise"] == 0.1


def test_build_upcoming_events_orders_and_measures_hours() -> None:
    now = datetime(2026, 9, 4, 14, 0, tzinfo=UTC)
    events = [
        _event("ISM Services PMI", datetime(2026, 9, 5, 14, 0, tzinfo=UTC), forecast="51.0"),
        _event("FOMC Statement", datetime(2026, 9, 4, 18, 0, tzinfo=UTC)),
        _event("Retail Sales m/m", datetime(2026, 9, 10, 12, 30, tzinfo=UTC)),  # beyond 24h
    ]
    upcoming = releases.build_upcoming_events(events, now=now, lookahead_hours=24)
    assert [e["title"] for e in upcoming] == ["FOMC Statement", "ISM Services PMI"]
    assert upcoming[0]["hours_ahead"] == 4.0
    assert upcoming[1]["forecast"] == 51.0


def test_releases_as_news_items_render_only_resolved_prints() -> None:
    items = releases.releases_as_news_items(
        [
            {"title": "Non-Farm Employment Change", "actual": 142.0, "forecast": 75.0, "previous": 73.0,
             "surprise": 67.0, "unit": "K", "time_utc": "2026-09-04T12:30:00+00:00"},
            {"title": "FOMC Statement", "actual": None},
        ]
    )
    assert len(items) == 1
    assert items[0]["title"] == "US Non-Farm Employment Change: actual 142.0K vs forecast 75.0K (prev 73.0K) surprise +67.0K"
    assert items[0]["source"] == "economic_calendar+fred"
    assert items[0]["published_at"] == "2026-09-04T12:30:00+00:00"
