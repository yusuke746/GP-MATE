from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import Mock, patch

from data.news_client import (
    _parse_calendar_event_datetime,
    fetch_news,
    is_high_impact_soon,
)


def _mock_response(text: str = "", json_data: dict | None = None) -> Mock:
    response = Mock()
    response.text = text
    response.raise_for_status = Mock()
    if json_data is not None:
        response.json = Mock(return_value=json_data)
    else:
        response.json = Mock(return_value={})
    return response


def _calendar_xml(currency: str, impact: str, event_dt: datetime) -> str:
    return f"""
    <weeklyevents>
      <event>
        <title>Test Event</title>
        <currency>{currency}</currency>
        <impact>{impact}</impact>
        <date>{event_dt.strftime('%m-%d-%Y')}</date>
        <time>{event_dt.strftime('%I:%M%p').lower()}</time>
      </event>
    </weeklyevents>
    """


def test_fetch_news_filters_keywords_and_deduplicates() -> None:
    now = datetime.now(UTC).strftime("%a, %d %b %Y %H:%M:%S GMT")
    rss = f"""
    <rss><channel>
      <item><title>Gold rises on weaker dollar</title><link>a</link><pubDate>{now}</pubDate></item>
      <item><title>Gold rises on weaker dollar</title><link>b</link><pubDate>{now}</pubDate></item>
      <item><title>Sports headline</title><link>c</link><pubDate>{now}</pubDate></item>
    </channel></rss>
    """

    with patch("data.news_client.requests.get", return_value=_mock_response(text=rss)):
        items = fetch_news(hours=24, max_items=10)

    assert len(items) == 1
    assert "gold" in items[0]["title"].lower()


def test_is_high_impact_soon_true_when_request_fails() -> None:
    with patch("data.news_client.requests.get", side_effect=Exception("network")):
        assert is_high_impact_soon(minutes=15) is True


def test_parse_calendar_event_datetime_supports_forexfactory_format() -> None:
    parsed = _parse_calendar_event_datetime("07-28-2026", "8:30am")
    assert parsed == datetime(2026, 7, 28, 8, 30, tzinfo=UTC)

    parsed_pm = _parse_calendar_event_datetime("12-01-2026", "02:00pm")
    assert parsed_pm == datetime(2026, 12, 1, 14, 0, tzinfo=UTC)

    assert _parse_calendar_event_datetime("07-28-2026", "All Day") is None
    assert _parse_calendar_event_datetime("", "8:30am") is None


def test_parse_calendar_event_datetime_honors_configured_timezone(monkeypatch) -> None:
    from zoneinfo import ZoneInfo

    import data.news_client as news_client

    monkeypatch.setattr(news_client, "CALENDAR_TZ", ZoneInfo("America/New_York"))

    parsed = _parse_calendar_event_datetime("07-28-2026", "8:30am")
    # 8:30 New York (EDT, UTC-4) == 12:30 UTC
    assert parsed == datetime(2026, 7, 28, 12, 30, tzinfo=UTC)


def test_is_high_impact_soon_detects_nearby_high_impact_usd_event() -> None:
    event_dt = datetime.now(UTC) + timedelta(minutes=5)
    xml = _calendar_xml("USD", "High", event_dt)

    with patch("data.news_client.requests.get", return_value=_mock_response(text=xml)):
        assert is_high_impact_soon(minutes=15) is True


def test_is_high_impact_soon_ignores_far_or_low_impact_events() -> None:
    far_event = datetime.now(UTC) + timedelta(hours=6)
    xml_far = _calendar_xml("USD", "High", far_event)
    with patch("data.news_client.requests.get", return_value=_mock_response(text=xml_far)):
        assert is_high_impact_soon(minutes=15) is False

    near_low = datetime.now(UTC) + timedelta(minutes=5)
    xml_low = _calendar_xml("USD", "Low", near_low)
    with patch("data.news_client.requests.get", return_value=_mock_response(text=xml_low)):
        assert is_high_impact_soon(minutes=15) is False

    near_eur = datetime.now(UTC) + timedelta(minutes=5)
    xml_eur = _calendar_xml("EUR", "High", near_eur)
    with patch("data.news_client.requests.get", return_value=_mock_response(text=xml_eur)):
        assert is_high_impact_soon(minutes=15) is False


def test_fetch_news_returns_empty_when_all_feeds_dead() -> None:
    with patch("data.news_client.requests.get", side_effect=Exception("network")):
        items = fetch_news(hours=24, max_items=10)

    assert items == []


def test_fetch_news_drops_undated_items() -> None:
    # Undated items bypassed the recency window and were observed polluting
    # sentiment input for weeks with the same stale headlines.
    now = datetime.now(UTC).strftime("%a, %d %b %Y %H:%M:%S GMT")
    rss = f"""
    <rss><channel>
      <item><title>Gold climbs as dollar softens</title><link>a</link><pubDate>{now}</pubDate></item>
      <item><title>Gold outlook: Fed minutes ahead</title><link>b</link></item>
    </channel></rss>
    """

    with patch("data.news_client.requests.get", return_value=_mock_response(text=rss)):
        items = fetch_news(hours=24, max_items=10)

    titles = [item["title"] for item in items]
    assert "Gold climbs as dollar softens" in titles
    assert "Gold outlook: Fed minutes ahead" not in titles
