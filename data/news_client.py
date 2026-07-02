from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import requests

from config import FRED_API_KEY, MAX_NEWS_ITEMS, NEWS_FILTER_MINUTES, RSS_FEEDS

LOGGER = logging.getLogger(__name__)

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
)

CALENDAR_XML_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.xml"
CACHE_PATH = Path(__file__).resolve().parent / "macro_cache.json"
REQUEST_TIMEOUT = 10


def _safe_get(url: str, params: dict[str, Any] | None = None) -> requests.Response | None:
    try:
        response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
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


def fetch_news(hours: int = 24, max_items: int = MAX_NEWS_ITEMS) -> list[dict[str, Any]]:
    """Fetch and filter gold-related headlines from configured RSS feeds.

    Returns empty list on failure to preserve safe caller behavior.
    """
    if hours <= 0 or max_items <= 0:
        return []

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

    filtered: list[dict[str, Any]] = []
    for item in _deduplicate_by_title(all_items):
        title = str(item.get("title") or "")
        if not _contains_keywords(title, GOLD_KEYWORDS):
            continue

        published_at_raw = str(item.get("published_at") or "")
        published_at = datetime.fromisoformat(published_at_raw) if published_at_raw else None
        if published_at is not None and published_at.tzinfo is None:
            published_at = published_at.replace(tzinfo=UTC)

        if published_at is not None and published_at < oldest:
            continue

        filtered.append(item)

    filtered.sort(key=lambda x: str(x.get("published_at") or ""), reverse=True)
    result = filtered[:max_items]

    if live_feed_count == 0:
        LOGGER.warning("All RSS feeds are unavailable in this cycle. news_count=0")
    elif len(result) == 0:
        LOGGER.warning("No valid news items after filtering. news_count=0")

    return result


def _parse_calendar_event_datetime(date_text: str, time_text: str) -> datetime | None:
    date_text = date_text.strip()
    time_text = time_text.strip()
    if not date_text or not time_text:
        return None

    formats = (
        "%m-%d-%Y %I:%M%p",
        "%Y-%m-%d %H:%M",
    )
    candidate = f"{date_text} {time_text}".replace(" ", "") if ":" in time_text and "-" in date_text else f"{date_text} {time_text}"

    for fmt in formats:
        try:
            dt = datetime.strptime(candidate, fmt)
            return dt.replace(tzinfo=UTC)
        except Exception:
            continue
    return None


def is_high_impact_soon(minutes: int = NEWS_FILTER_MINUTES) -> bool:
    """Return True if high-impact USD/XAU event is near now.

    On API failure or parse uncertainty, returns True (safe side: block new entries).
    """
    if minutes <= 0:
        return False

    response = _safe_get(CALENDAR_XML_URL)
    if response is None:
        return True

    try:
        root = ElementTree.fromstring(response.text)
        now_utc = datetime.now(UTC)
        threshold = timedelta(minutes=minutes)

        events = root.findall(".//event")
        if not events:
            return True

        for event in events:
            currency = (event.findtext("currency") or "").strip().upper()
            impact = (event.findtext("impact") or "").strip().lower()
            date_text = (event.findtext("date") or "").strip()
            time_text = (event.findtext("time") or "").strip()

            if currency not in {"USD", "XAU"}:
                continue
            if "high" not in impact:
                continue

            event_dt = _parse_calendar_event_datetime(date_text, time_text)
            if event_dt is None:
                continue

            if abs(event_dt - now_utc) <= threshold:
                return True

        return False
    except Exception as exc:
        LOGGER.warning("Calendar parse failed: %s", exc)
        return True


def _load_macro_cache() -> dict[str, Any]:
    if not CACHE_PATH.exists():
        return {}
    try:
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_macro_cache(payload: dict[str, Any]) -> None:
    try:
        CACHE_PATH.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    except Exception as exc:
        LOGGER.warning("Failed to save macro cache: %s", exc)


def _fred_latest_value(series_id: str) -> float | None:
    if not FRED_API_KEY:
        return None

    params = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "sort_order": "desc",
        "limit": 1,
    }
    response = _safe_get("https://api.stlouisfed.org/fred/series/observations", params=params)
    if response is None:
        return None

    try:
        body = response.json()
        observations = body.get("observations", [])
        if not observations:
            return None
        raw_value = observations[0].get("value")
        return float(raw_value)
    except Exception:
        return None


def get_macro_data() -> dict[str, Any]:
    """Fetch key macro data from FRED with daily cache.

    Returns cached data if fetched today. Returns success=False on hard failure.
    """
    today = datetime.now(UTC).date().isoformat()
    cache = _load_macro_cache()

    if cache.get("date") == today and isinstance(cache.get("data"), dict):
        return {"success": True, "data": cache["data"], "source": "cache"}

    fed_funds = _fred_latest_value("FEDFUNDS")
    cpi = _fred_latest_value("CPIAUCSL")
    us10y = _fred_latest_value("DGS10")

    if fed_funds is None and cpi is None and us10y is None:
        if isinstance(cache.get("data"), dict):
            return {"success": True, "data": cache["data"], "source": "stale_cache"}
        return {"success": False, "reason": "FRED fetch failed"}

    data = {
        "fed_funds": fed_funds,
        "cpi": cpi,
        "us10y": us10y,
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }
    _save_macro_cache({"date": today, "data": data})

    return {"success": True, "data": data, "source": "fred"}
