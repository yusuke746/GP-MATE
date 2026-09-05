"""Tests for the gold-specific data-source overhaul: 5-day FRED changes,
optional series, calendar event parsing, feed health metadata, synthetic
dollar index, and the revised macro scoring."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from unittest.mock import Mock, patch

from agents.data import fred_client
from agents.macro_analyst import _score_macro_environment
from data import mt5_client, news_client


# --------------------------------------------------------------------------- #
# FRED snapshots
# --------------------------------------------------------------------------- #
def test_snapshot_from_points_daily_series_has_5d_change() -> None:
    start = date(2026, 7, 1)
    points = [(start + timedelta(days=i), 100.0 + i) for i in range(40) if (start + timedelta(days=i)).weekday() < 5]
    snap = fred_client.snapshot_from_points(points)
    assert snap is not None
    assert snap["direction"] == "UP"
    assert snap["change_5d"] == 7.0  # 5 trading days back spans a weekend: +7 calendar days
    assert snap["direction_5d"] == "UP"
    assert snap["as_of"] == points[-1][0].isoformat()


def test_snapshot_from_points_monthly_series_has_no_5d_change() -> None:
    points = [(date(2026, m, 1), 5.0 - 0.1 * m) for m in range(1, 9)]
    snap = fred_client.snapshot_from_points(points)
    assert snap is not None
    assert snap["direction"] == "DOWN"
    assert snap["change_5d"] is None
    assert snap["direction_5d"] == "FLAT"


def _payload(values: list[tuple[str, str]]) -> Mock:
    response = Mock()
    response.raise_for_status = Mock()
    response.json = Mock(return_value={"observations": [{"date": d, "value": v} for d, v in values]})
    return response


def test_optional_series_failure_degrades_with_warning(monkeypatch) -> None:
    fred_client._DAILY_CACHE.clear()
    monkeypatch.setenv("FRED_API_KEY", "k")
    monkeypatch.setattr(fred_client, "REQUEST_MAX_RETRIES", 1)
    monkeypatch.setattr(fred_client.time, "sleep", lambda s: None)

    def fake_get(url, params=None, timeout=None):
        sid = params["series_id"]
        if sid == "DGS2":
            raise RuntimeError("503")
        return _payload([("2026-06-01", "1.0"), ("2026-07-01", "2.0")])

    with patch("agents.data.fred_client.requests.get", side_effect=fake_get):
        data = fred_client.get_macro_data(force_refresh=True)

    assert data["_meta"]["ok"] is True
    assert data["us2y"]["value"] is None
    assert any("us2y" in w for w in data["_meta"]["warnings"])
    assert data["dxy"]["source"] == "fred:DTWEXBGS"


def test_core_series_failure_still_fails(monkeypatch) -> None:
    fred_client._DAILY_CACHE.clear()
    fred_client._LAST_GOOD_CACHE["data"] = None
    monkeypatch.setenv("FRED_API_KEY", "k")
    monkeypatch.setattr(fred_client, "REQUEST_MAX_RETRIES", 1)

    def fake_get(url, params=None, timeout=None):
        if params["series_id"] == "DTWEXBGS":
            raise RuntimeError("503")
        return _payload([("2026-06-01", "1.0"), ("2026-07-01", "2.0")])

    with patch("agents.data.fred_client.requests.get", side_effect=fake_get):
        data = fred_client.get_macro_data(force_refresh=True)
    assert data["_meta"]["ok"] is False


# --------------------------------------------------------------------------- #
# Calendar parsing / feed health
# --------------------------------------------------------------------------- #
CALENDAR_XML = """
<weeklyevents>
  <event><title>Non-Farm Employment Change</title><country>USD</country><date>09-04-2026</date><time>8:30am</time>
         <impact>High</impact><forecast>75K</forecast><previous>73K</previous></event>
  <event><title>Unemployment Rate</title><country>USD</country><date>09-04-2026</date><time>8:30am</time>
         <impact>High</impact><forecast>4.3%</forecast><previous>4.2%</previous></event>
  <event><title>CPI y/y</title><country>EUR</country><date>09-02-2026</date><time>5:00am</time>
         <impact>High</impact><forecast>2.1%</forecast><previous>2.0%</previous></event>
  <event><title>Crude Oil Inventories</title><country>USD</country><date>09-02-2026</date><time>10:30am</time>
         <impact>Low</impact><forecast></forecast><previous>-2.4M</previous></event>
</weeklyevents>
"""


def test_parse_calendar_events_keeps_forecast_and_previous() -> None:
    events = news_client.parse_calendar_events(CALENDAR_XML)
    assert len(events) == 4
    nfp = events[0]
    assert nfp["title"] == "Non-Farm Employment Change"
    assert nfp["currency"] == "USD"
    assert nfp["impact"] == "high"
    assert nfp["forecast"] == "75K" and nfp["previous"] == "73K" and nfp["actual"] == ""
    assert nfp["datetime_utc"] == datetime(2026, 9, 4, 8, 30, tzinfo=UTC)


def test_fetch_calendar_events_filters_currency_and_impact() -> None:
    response = Mock()
    response.raise_for_status = Mock()
    response.text = CALENDAR_XML
    with patch("data.news_client.requests.get", return_value=response):
        events = news_client.fetch_calendar_events()
    assert [e["title"] for e in events] == ["Non-Farm Employment Change", "Unemployment Rate"]

    with patch("data.news_client.requests.get", side_effect=Exception("down")):
        assert news_client.fetch_calendar_events() is None


def test_fetch_news_with_meta_reports_feed_health(monkeypatch) -> None:
    now = datetime.now(UTC).strftime("%a, %d %b %Y %H:%M:%S GMT")
    rss = f"""<rss><channel>
      <item><title>Gold steadies as Fed cut bets firm</title><link>a</link><pubDate>{now}</pubDate></item>
      <item><title>Football results</title><link>c</link><pubDate>{now}</pubDate></item>
    </channel></rss>"""
    monkeypatch.setattr(news_client, "RSS_FEEDS", ("https://ok.example/rss", "https://dead.example/rss"))

    def fake_get(url, params=None, timeout=None, headers=None):
        assert headers and "User-Agent" in headers
        if "dead" in url:
            raise RuntimeError("403")
        response = Mock()
        response.raise_for_status = Mock()
        response.text = rss
        return response

    with patch("data.news_client.requests.get", side_effect=fake_get):
        items, meta = news_client.fetch_news_with_meta(hours=24, max_items=10)

    assert len(items) == 1
    assert meta["feeds_total"] == 2
    assert meta["feeds_live"] == 1
    assert meta["dead_feeds"] == ["https://dead.example/rss"]
    assert meta["raw_items"] == 2
    assert meta["keyword_items"] == 1


# --------------------------------------------------------------------------- #
# Synthetic dollar index
# --------------------------------------------------------------------------- #
def test_synthesize_dollar_index_tracks_dollar_direction() -> None:
    days = [date(2026, 9, 1) + timedelta(days=i) for i in range(3)]
    # EUR falls and JPY weakens -> dollar strengthens -> index rises.
    closes = {
        "EURUSD": [(d, 1.10 - 0.01 * i) for i, d in enumerate(days)],
        "USDJPY": [(d, 150.0 + 1.0 * i) for i, d in enumerate(days)],
        "GBPUSD": [(d, 1.30) for d in days],
    }
    index = mt5_client.synthesize_dollar_index(closes)
    assert [d for d, _ in index] == days
    assert index[0][1] < index[1][1] < index[2][1]


def test_synthesize_dollar_index_uses_common_dates_only() -> None:
    d1, d2 = date(2026, 9, 1), date(2026, 9, 2)
    closes = {"EURUSD": [(d1, 1.1), (d2, 1.1)], "USDJPY": [(d2, 150.0)]}
    index = mt5_client.synthesize_dollar_index(closes)
    assert [d for d, _ in index] == [d2]
    assert mt5_client.synthesize_dollar_index({}) == []


# --------------------------------------------------------------------------- #
# Macro scoring
# --------------------------------------------------------------------------- #
def _macro(**overrides):
    base = {
        "dxy": {"value": 98.0, "change_30d": -1.0, "direction": "DOWN", "change_5d": -0.2, "direction_5d": "DOWN", "source": "mt5:USDX"},
        "us2y": {"value": 3.5, "change_30d": -0.2, "direction": "DOWN", "change_5d": -0.05, "direction_5d": "DOWN"},
        "real_rate": {"value": 1.8, "change_30d": 0.0, "direction": "FLAT"},
        "breakeven": {"value": 2.3, "change_30d": 0.1, "direction": "UP"},
        "fed_funds": {"value": 4.0, "change_30d": 0.0, "direction": "FLAT"},
        "_meta": {"ok": True},
    }
    base.update(overrides)
    return base


def test_confirming_short_term_moves_raise_confidence() -> None:
    bias, conf, drivers, _ = _score_macro_environment(_macro())
    assert bias == "BULLISH"
    assert conf > 0.8
    assert any("同方向" in d for d in drivers)


def test_conflicting_5d_move_claws_back_trend_score() -> None:
    confirming = _score_macro_environment(_macro())
    conflicting = _score_macro_environment(
        _macro(dxy={"value": 98.0, "change_30d": -1.0, "direction": "DOWN", "change_5d": 0.6, "direction_5d": "UP", "source": "mt5:USDX"})
    )
    assert conflicting[1] < confirming[1]
    assert any("転換の兆し" in d for d in conflicting[2])


def test_us2y_takes_over_from_fed_funds_when_present() -> None:
    # 2y yield rising (hawkish) while monthly fed funds is flat -> rates leg negative.
    data = _macro(us2y={"value": 3.9, "change_30d": 0.3, "direction": "UP", "change_5d": 0.1, "direction_5d": "UP"})
    _, _, drivers, _ = _score_macro_environment(data)
    assert any("2年債利回り" in d and "逆風" in d for d in drivers)


def test_crowded_long_positioning_and_gld_outflow_lower_score() -> None:
    plain = _score_macro_environment(_macro())
    crowded = _score_macro_environment(
        _macro(
            positioning={
                "cot": {"crowding": "CROWDED_LONG", "net_percentile_window": 92.0, "_meta": {"ok": True}},
                "gld": {"direction_5d": "DOWN", "change_5d": -4.2, "_meta": {"ok": True}},
                "_meta": {"ok": True},
            }
        )
    )
    assert crowded[1] < plain[1]
    assert any("過密" in d for d in crowded[2])
    assert any("流出" in d for d in crowded[2])


def test_releases_are_described_in_key_drivers() -> None:
    data = _macro(
        recent_releases=[{"title": "Non-Farm Employment Change", "actual": 142.0, "forecast": 75.0, "surprise": 67.0, "unit": "K"}],
        upcoming_events=[{"title": "FOMC Statement", "hours_ahead": 4.0}],
    )
    _, _, drivers, _ = _score_macro_environment(data)
    assert any("Non-Farm" in d and "サプライズ67.0" in d for d in drivers)
    assert any("FOMC Statement" in d for d in drivers)
