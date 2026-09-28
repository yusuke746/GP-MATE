"""Tests for the gold-specific data-source overhaul: 5-day FRED changes,
optional series, calendar event parsing, feed health metadata, synthetic
dollar index, and the data handed to the macro analyst."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from unittest.mock import Mock, patch

from agents.data import fred_client
from agents.macro_analyst import _analyst_input, analyze_macro_environment
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
# Macro analyst input: facts in, code-made interpretations out
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


def test_analyst_input_keeps_series_positioning_and_releases_but_drops_first_order_read() -> None:
    data = _macro(
        positioning={
            "cot": {"crowding": "CROWDED_LONG", "net_percentile_window": 92.0, "_meta": {"ok": True}},
            "gld": {"direction_5d": "DOWN", "change_5d": -4.2, "_meta": {"ok": True}},
            "_meta": {"ok": True},
        },
        recent_releases=[{"title": "Non-Farm Employment Change", "actual": 142.0, "forecast": 75.0, "surprise": 67.0, "unit": "K",
                          "first_order_read": "hawkish_surprise(gold_negative_first_order)"}],
        upcoming_events=[{"title": "FOMC Statement", "hours_ahead": 4.0}],
    )
    handed = _analyst_input(data)
    assert "_meta" not in handed
    assert handed["dxy"]["direction_5d"] == "DOWN" and handed["us2y"]["change_5d"] == -0.05
    assert handed["positioning"]["cot"]["crowding"] == "CROWDED_LONG"
    assert handed["recent_releases"][0]["surprise"] == 67.0
    assert "first_order_read" not in handed["recent_releases"][0]
    assert handed["upcoming_events"][0]["title"] == "FOMC Statement"


def test_macro_analyst_sends_data_notes_without_directional_rules() -> None:
    import json

    fake_result = Mock()
    fake_result.ok = True
    fake_result.payload = {"macro_bias": "BEARISH", "regime_view": "SUPPORTS_REVERSAL", "key_drivers": ["us2y +0.3pt"], "invalidation": "x", "reasoning": "r"}
    fake_result.model = "m"
    fake_result.error = ""
    fake_result.usage = Mock(prompt_tokens=1, completion_tokens=1, total_tokens=2)
    client = Mock()
    client.call_json.return_value = fake_result
    with patch("agents.macro_analyst.get_default_client", return_value=client):
        result = analyze_macro_environment(_macro())
    assert result["macro_bias"] == "BEARISH"  # the analyst's call, even though the dollar is falling
    sent = json.loads(client.call_json.call_args.kwargs["user_prompt"])
    assert "rule_based_baseline" not in sent
    assert "data_notes" in sent and "dxy" in sent["data_notes"]
    system = client.call_json.call_args.kwargs["system_prompt"]
    for forbidden in ("ドル安(DOWN)は金にポジティブ", "必ず", "上限とし"):
        assert forbidden not in system
    for forbidden in ("ポジティブ", "ネガティブ", "追い風", "逆風"):
        assert forbidden not in json.dumps(sent["data_notes"], ensure_ascii=False)
