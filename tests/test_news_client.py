from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import Mock, patch

from data.news_client import fetch_news, get_macro_data, is_high_impact_soon


def _mock_response(text: str = "", json_data: dict | None = None) -> Mock:
    response = Mock()
    response.text = text
    response.raise_for_status = Mock()
    if json_data is not None:
        response.json = Mock(return_value=json_data)
    else:
        response.json = Mock(return_value={})
    return response


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


def test_get_macro_data_returns_failure_without_cache_when_fred_unavailable() -> None:
    with patch("data.news_client.requests.get", side_effect=Exception("network")):
        result = get_macro_data()

    assert isinstance(result, dict)
    assert "success" in result


def test_fetch_news_returns_empty_when_all_feeds_dead() -> None:
    with patch("data.news_client.requests.get", side_effect=Exception("network")):
        items = fetch_news(hours=24, max_items=10)

    assert items == []
