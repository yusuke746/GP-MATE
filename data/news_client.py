from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime
from typing import Any
from xml.etree import ElementTree
from zoneinfo import ZoneInfo

import requests

from config import CALENDAR_TIMEZONE_NAME, MAX_NEWS_ITEMS, NEWS_FILTER_MINUTES, RSS_FEEDS

LOGGER = logging.getLogger(__name__)


def _resolve_calendar_tz() -> ZoneInfo:
    try:
        return ZoneInfo(CALENDAR_TIMEZONE_NAME)
    except Exception:
        LOGGER.warning(
            "Invalid CALENDAR_TIMEZONE %r; falling back to UTC", CALENDAR_TIMEZONE_NAME
        )
        return ZoneInfo("UTC")


# Timezone the economic-calendar feed timestamps are expressed in.
# Configurable because feed providers differ; default UTC.
CALENDAR_TZ = _resolve_calendar_tz()

GOLD_KEYWORDS: tuple[str, ...] = (
    "gold",
    "xau",
    "fed",
    "fomc",
    "rate",
    "rates",
    "inflation",
    "cpi",
    "pce",
    "dollar",
    "powell",
    "treasury",
    "yield",
    "geopolitical",
    "bullion",
    "precious metal",
    "safe haven",
    "safe-haven",
    "tariff",
    "payrolls",
    "nonfarm",
    "jobs report",
    "central bank",
    "etf",
)

CALENDAR_XML_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.xml"
REQUEST_TIMEOUT = 10
# Several publishers answer the default python-requests UA with 403; a
# browser-like UA is what their own readers send.
REQUEST_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) GP-MATE/1.0"}
CALENDAR_CURRENCIES = {"USD", "XAU"}


def _safe_get(url: str, params: dict[str, Any] | None = None) -> requests.Response | None:
    try:
        response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT, headers=REQUEST_HEADERS)
        response.raise_for_status()
        return response
    except Exception as exc:
        LOGGER.warning("HTTP request failed: %s (%s)", url, exc)
        return None


def check_rss_feeds_health(feeds: tuple[str, ...] = RSS_FEEDS) -> list[dict[str, Any]]:
    """Check RSS feed liveness and return per-feed status."""
    results: list[dict[str, Any]] = []
    for url in feeds:
        response = _safe_get(url)
        if response is None:
            results.append(
                {
                    "url": url,
                    "ok": False,
                    "status_code": None,
                    "reason": "request_failed",
                }
            )
            continue

        results.append(
            {
                "url": url,
                "ok": True,
                "status_code": int(response.status_code),
                "reason": "ok",
            }
        )
    return results


def _parse_datetime(value: str) -> datetime | None:
    if not value:
        return None
    try:
        dt = parsedate_to_datetime(value)
        if dt.tzinfo is None:
            return dt.replace(tzinfo=UTC)
        return dt.astimezone(UTC)
    except Exception:
        return None


def _contains_keywords(text: str, keywords: tuple[str, ...]) -> bool:
    normalized = text.lower()
    return any(keyword in normalized for keyword in keywords)


def _extract_rss_items(xml_text: str, source: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []

    try:
        root = ElementTree.fromstring(xml_text)
    except Exception:
        return items

    channels = root.findall(".//channel")
    if channels:
        entries = root.findall(".//item")
        for entry in entries:
            title = (entry.findtext("title") or "").strip()
            link = (entry.findtext("link") or "").strip()
            pub_date_raw = (entry.findtext("pubDate") or "").strip()
            pub_date = _parse_datetime(pub_date_raw)
            items.append(
                {
                    "title": title,
                    "link": link,
                    "published_at": pub_date.isoformat() if pub_date else "",
                    "source": source,
                }
            )
        return items

    atom_entries = root.findall(".//{http://www.w3.org/2005/Atom}entry")
    for entry in atom_entries:
        title = (entry.findtext("{http://www.w3.org/2005/Atom}title") or "").strip()
        link_node = entry.find("{http://www.w3.org/2005/Atom}link")
        link = ""
        if link_node is not None:
            link = (link_node.attrib.get("href") or "").strip()
        updated_raw = (entry.findtext("{http://www.w3.org/2005/Atom}updated") or "").strip()
        pub_date = _parse_datetime(updated_raw)
        items.append(
            {
                "title": title,
                "link": link,
                "published_at": pub_date.isoformat() if pub_date else "",
                "source": source,
            }
        )

    return items


def _deduplicate_by_title(news_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    unique_items: list[dict[str, Any]] = []
    for item in news_items:
        key = (item.get("title") or "").strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        unique_items.append(item)
    return unique_items


def fetch_news_with_meta(
    hours: int = 24, max_items: int = MAX_NEWS_ITEMS
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """fetch_news plus feed-health metadata.

    meta = {"feeds_total", "feeds_live", "raw_items", "keyword_items", "dead_feeds"}
    so a starving pipeline (dead feeds vs. nothing matching the keywords) is
    diagnosable from the trade log instead of only from process logs.
    """
    meta: dict[str, Any] = {
        "feeds_total": len(RSS_FEEDS),
        "feeds_live": 0,
        "raw_items": 0,
        "keyword_items": 0,
        "dead_feeds": [],
    }
    if hours <= 0 or max_items <= 0:
        return [], meta

    now_utc = datetime.now(UTC)
    oldest = now_utc - timedelta(hours=hours)

    all_items: list[dict[str, Any]] = []
    skipped_feeds: list[str] = []
    live_feed_count = 0
    for feed_url in RSS_FEEDS:
        response = _safe_get(feed_url)
        if response is None:
            skipped_feeds.append(feed_url)
            continue
        live_feed_count += 1
        all_items.extend(_extract_rss_items(response.text, source=feed_url))

    if skipped_feeds:
        LOGGER.warning("RSS feed skipped this cycle: %s", ", ".join(skipped_feeds))
    meta["feeds_live"] = live_feed_count
    meta["dead_feeds"] = skipped_feeds
    meta["raw_items"] = len(all_items)

    filtered: list[dict[str, Any]] = []
    dropped_undated = 0
    for item in _deduplicate_by_title(all_items):
        title = str(item.get("title") or "")
        if not _contains_keywords(title, GOLD_KEYWORDS):
            continue
        meta["keyword_items"] = int(meta["keyword_items"]) + 1

        published_at_raw = str(item.get("published_at") or "")
        published_at = datetime.fromisoformat(published_at_raw) if published_at_raw else None
        if published_at is not None and published_at.tzinfo is None:
            published_at = published_at.replace(tzinfo=UTC)

        # Undated items cannot be age-checked and were observed persisting in
        # the sentiment input for weeks; treat them as stale and drop them.
        if published_at is None:
            dropped_undated += 1
            continue

        if published_at < oldest:
            continue

        filtered.append(item)

    if dropped_undated:
        LOGGER.info("fetch_news: dropped %d undated item(s) as unverifiable/stale", dropped_undated)

    filtered.sort(key=lambda x: str(x.get("published_at") or ""), reverse=True)
    result = filtered[:max_items]

    if live_feed_count == 0:
        LOGGER.warning("All RSS feeds are unavailable in this cycle. news_count=0")
    elif len(result) == 0:
        LOGGER.warning(
            "No valid news items after filtering. news_count=0 (feeds_live=%d raw=%d keyword=%d)",
            live_feed_count,
            meta["raw_items"],
            meta["keyword_items"],
        )

    return result, meta


def fetch_news(hours: int = 24, max_items: int = MAX_NEWS_ITEMS) -> list[dict[str, Any]]:
    """Fetch and filter gold-related headlines from configured RSS feeds.

    Returns empty list on failure to preserve safe caller behavior.
    """
    items, _ = fetch_news_with_meta(hours=hours, max_items=max_items)
    return items


def _parse_calendar_event_datetime(date_text: str, time_text: str) -> datetime | None:
    date_text = date_text.strip()
    time_text = time_text.strip()
    if not date_text or not time_text:
        return None

    formats = (
        "%m-%d-%Y %I:%M%p",
        "%Y-%m-%d %H:%M",
    )
    candidate = f"{date_text} {time_text}"

    for fmt in formats:
        try:
            dt = datetime.strptime(candidate, fmt)
            return dt.replace(tzinfo=CALENDAR_TZ).astimezone(UTC)
        except Exception:
            continue
    return None


def parse_calendar_events(xml_text: str) -> list[dict[str, Any]]:
    """ForexFactory weekly XML -> event dicts.

    Each dict: title, currency, impact (lowercase), datetime_utc (aware or None),
    forecast, previous, actual (raw strings; actual is usually absent in this feed).
    Raises on malformed XML so callers can decide the fail-safe direction.
    """
    root = ElementTree.fromstring(xml_text)
    events: list[dict[str, Any]] = []
    for event in root.findall(".//event"):
        date_text = (event.findtext("date") or "").strip()
        time_text = (event.findtext("time") or "").strip()
        events.append(
            {
                "title": (event.findtext("title") or "").strip(),
                "currency": (event.findtext("currency") or event.findtext("country") or "").strip().upper(),
                "impact": (event.findtext("impact") or "").strip().lower(),
                "datetime_utc": _parse_calendar_event_datetime(date_text, time_text),
                "forecast": (event.findtext("forecast") or "").strip(),
                "previous": (event.findtext("previous") or "").strip(),
                "actual": (event.findtext("actual") or "").strip(),
            }
        )
    return events


def fetch_calendar_events(
    currencies: set[str] | None = None,
    high_impact_only: bool = True,
) -> list[dict[str, Any]] | None:
    """This week's calendar events for the given currencies (default USD/XAU).

    Returns None when the feed is unreachable or unparsable so callers can
    fail safe; an empty list means the feed answered with nothing relevant.
    """
    response = _safe_get(CALENDAR_XML_URL)
    if response is None:
        return None
    try:
        events = parse_calendar_events(response.text)
    except Exception as exc:
        LOGGER.warning("Calendar parse failed: %s", exc)
        return None
    wanted = currencies or CALENDAR_CURRENCIES
    selected = [e for e in events if e["currency"] in wanted]
    if high_impact_only:
        selected = [e for e in selected if "high" in e["impact"]]
    return selected


def is_high_impact_soon(minutes: int = NEWS_FILTER_MINUTES) -> bool:
    """Return True if high-impact USD/XAU event is near now.

    On API failure or parse uncertainty, returns True (safe side: block new entries).
    """
    if minutes <= 0:
        return False

    events = fetch_calendar_events()
    if events is None:
        return True
    # A week with zero parsable events is far likelier a feed problem than a
    # genuinely empty calendar; keep the legacy fail-safe.
    if not events and _calendar_has_no_events_at_all():
        return True

    now_utc = datetime.now(UTC)
    threshold = timedelta(minutes=minutes)
    for event in events:
        event_dt = event.get("datetime_utc")
        if event_dt is None:
            continue
        if abs(event_dt - now_utc) <= threshold:
            return True
    return False


def _calendar_has_no_events_at_all() -> bool:
    all_events = fetch_calendar_events(currencies={"USD", "XAU", "EUR", "GBP", "JPY", "CAD", "AUD", "NZD", "CHF", "CNY"}, high_impact_only=False)
    return not all_events


