"""Economic-release surprises for the macro and sentiment analysts.

The ForexFactory calendar we already download for the pre-news trading halt
carries the consensus forecast and previous value of every high-impact USD
release, but not the actual print. FRED publishes the actual within minutes
of the release for the series that move gold most, so we join the two:

    surprise = actual (FRED, converted to the calendar's unit) - forecast (FF)

Only releases whose title matches the curated map below get an actual; the
rest are still passed through with forecast/previous so the analysts know the
event happened.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any, Callable, Final

from agents.data.fred_client import fetch_fred_observations

LOGGER = logging.getLogger(__name__)

Points = list[tuple[date, float]]


@dataclass(frozen=True)
class ReleaseSpec:
    key: str
    series_id: str
    title_patterns: tuple[str, ...]
    # Turn the observation series into the headline number in calendar units.
    transform: Callable[[Points], float | None]
    unit: str
    max_period_age_days: int  # how far back the matching observation may be dated
    # Simple first-order read for gold; the analysts weigh the regime themselves.
    hawkish_when_above_forecast: bool


def _latest_level(points: Points) -> float | None:
    return points[-1][1] if points else None


def _latest_diff_thousands(points: Points) -> float | None:
    if len(points) < 2:
        return None
    return points[-1][1] - points[-2][1]


def _latest_mom_pct(points: Points) -> float | None:
    if len(points) < 2 or points[-2][1] == 0:
        return None
    return round(100.0 * (points[-1][1] / points[-2][1] - 1.0), 2)


def _latest_level_thousands(points: Points) -> float | None:
    return points[-1][1] / 1000.0 if points else None


RELEASE_SPECS: Final[tuple[ReleaseSpec, ...]] = (
    ReleaseSpec("nfp", "PAYEMS", (r"non-?farm employment", r"non-?farm.*payroll", r"\bnfp\b"), _latest_diff_thousands, "K", 45, True),
    ReleaseSpec("unemployment", "UNRATE", (r"unemployment rate",), _latest_level, "%", 45, False),
    ReleaseSpec("avg_hourly_earnings", "CES0500000003", (r"average hourly earnings",), _latest_mom_pct, "%", 45, True),
    ReleaseSpec("cpi_mom", "CPIAUCSL", (r"^cpi m/m", r"consumer price index m/m"), _latest_mom_pct, "%", 45, True),
    ReleaseSpec("core_cpi_mom", "CPILFESL", (r"^core cpi m/m",), _latest_mom_pct, "%", 45, True),
    ReleaseSpec("core_pce_mom", "PCEPILFE", (r"core pce price index m/m",), _latest_mom_pct, "%", 45, True),
    ReleaseSpec("ppi_mom", "PPIFIS", (r"^ppi m/m",), _latest_mom_pct, "%", 45, True),
    ReleaseSpec("retail_sales_mom", "RSAFS", (r"^retail sales m/m",), _latest_mom_pct, "%", 45, True),
    ReleaseSpec("jobless_claims", "ICSA", (r"unemployment claims", r"initial jobless claims"), _latest_level_thousands, "K", 10, False),
    ReleaseSpec("gdp_qoq", "A191RL1Q225SBEA", (r"gdp q/q",), _latest_level, "%", 120, True),
)

_NUMBER_RE = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*([KMB%]?)\s*$", re.IGNORECASE)


def parse_calendar_number(text: Any) -> tuple[float | None, str]:
    """'75K' -> (75.0, 'K'); '0.2%' -> (0.2, '%'); '' -> (None, '')."""
    match = _NUMBER_RE.match(str(text or ""))
    if not match:
        return None, ""
    return float(match.group(1)), match.group(2).upper()


def match_spec(title: str) -> ReleaseSpec | None:
    lowered = title.strip().lower()
    for spec in RELEASE_SPECS:
        for pattern in spec.title_patterns:
            if re.search(pattern, lowered):
                return spec
    return None


def actual_from_points(spec: ReleaseSpec, points: Points, event_time: datetime) -> float | None:
    """The headline number only if the observation covering this release exists.

    Monthly series are dated on the 1st of the reference month and published
    the following month, so an observation dated within ``max_period_age_days``
    before the event means the print is out; older means FRED has not updated
    yet and we must not report the previous month's value as 'actual'.
    """
    if not points:
        return None
    latest_date = points[-1][0]
    if (event_time.date() - latest_date).days > spec.max_period_age_days:
        return None
    if latest_date > event_time.date():
        return None
    value = spec.transform(points)
    return None if value is None else round(float(value), 2)


def build_release_record(
    event: dict[str, Any],
    points_by_series: dict[str, Points],
) -> dict[str, Any]:
    """Attach actual/surprise to one calendar event dict (from fetch_calendar_events)."""
    title = str(event.get("title") or "")
    forecast, unit = parse_calendar_number(event.get("forecast"))
    previous, _ = parse_calendar_number(event.get("previous"))
    actual_text = event.get("actual")
    actual, _ = parse_calendar_number(actual_text)
    spec = match_spec(title)
    source = "calendar" if actual is not None else ""

    if actual is None and spec is not None:
        event_time = event.get("datetime_utc")
        if isinstance(event_time, datetime):
            actual = actual_from_points(spec, points_by_series.get(spec.series_id, []), event_time)
            if actual is not None:
                source = f"fred:{spec.series_id}"
                unit = unit or spec.unit

    surprise = round(actual - forecast, 2) if actual is not None and forecast is not None else None
    read = ""
    if surprise is not None and spec is not None and abs(surprise) > 1e-9:
        hawkish = (surprise > 0) == spec.hawkish_when_above_forecast
        read = "hawkish_surprise(gold_negative_first_order)" if hawkish else "dovish_surprise(gold_positive_first_order)"

    when = event.get("datetime_utc")
    return {
        "title": title,
        "time_utc": when.isoformat() if isinstance(when, datetime) else "",
        "impact": str(event.get("impact") or ""),
        "forecast": forecast,
        "previous": previous,
        "actual": actual,
        "unit": unit,
        "surprise": surprise,
        "actual_source": source,
        "first_order_read": read,
    }


def build_recent_releases(
    events: list[dict[str, Any]],
    now: datetime | None = None,
    lookback_hours: int = 48,
    fetch_points: Callable[[str], Points] | None = None,
) -> list[dict[str, Any]]:
    """High-impact USD events in the lookback window, with actuals where known."""
    now = now or datetime.now(UTC)
    oldest = now - timedelta(hours=lookback_hours)
    recent = [
        e for e in events
        if isinstance(e.get("datetime_utc"), datetime) and oldest <= e["datetime_utc"] <= now
    ]
    if not recent:
        return []

    fetcher = fetch_points or (lambda series_id: fetch_fred_observations(series_id, limit=6))
    points_by_series: dict[str, Points] = {}
    for event in recent:
        spec = match_spec(str(event.get("title") or ""))
        if spec is None or spec.series_id in points_by_series:
            continue
        if parse_calendar_number(event.get("actual"))[0] is not None:
            continue  # calendar already carries the print
        try:
            points_by_series[spec.series_id] = fetcher(spec.series_id)
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.warning("release actual fetch failed (%s): %s", spec.series_id, exc)
            points_by_series[spec.series_id] = []

    records = [build_release_record(e, points_by_series) for e in recent]
    records.sort(key=lambda r: r["time_utc"], reverse=True)
    return records


def build_upcoming_events(
    events: list[dict[str, Any]],
    now: datetime | None = None,
    lookahead_hours: int = 24,
) -> list[dict[str, Any]]:
    now = now or datetime.now(UTC)
    horizon = now + timedelta(hours=lookahead_hours)
    upcoming = [
        e for e in events
        if isinstance(e.get("datetime_utc"), datetime) and now < e["datetime_utc"] <= horizon
    ]
    out = []
    for e in sorted(upcoming, key=lambda x: x["datetime_utc"]):
        forecast, unit = parse_calendar_number(e.get("forecast"))
        previous, _ = parse_calendar_number(e.get("previous"))
        out.append(
            {
                "title": str(e.get("title") or ""),
                "time_utc": e["datetime_utc"].isoformat(),
                "hours_ahead": round((e["datetime_utc"] - now).total_seconds() / 3600, 1),
                "impact": str(e.get("impact") or ""),
                "forecast": forecast,
                "previous": previous,
                "unit": unit,
            }
        )
    return out


def releases_as_news_items(releases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Render release records as headline-shaped items for the sentiment agent.

    Structured prints are the highest-signal 'news' a gold trader gets, and
    they also keep the sentiment input non-empty on release days when RSS
    feeds are thin.
    """
    items: list[dict[str, Any]] = []
    for r in releases:
        if r.get("actual") is None:
            continue
        unit = r.get("unit") or ""
        parts = [f"US {r['title']}: actual {r['actual']}{unit}"]
        if r.get("forecast") is not None:
            parts.append(f"vs forecast {r['forecast']}{unit}")
        if r.get("previous") is not None:
            parts.append(f"(prev {r['previous']}{unit})")
        if r.get("surprise") is not None:
            parts.append(f"surprise {r['surprise']:+}{unit}")
        items.append(
            {
                "title": " ".join(parts),
                "link": "",
                "published_at": r.get("time_utc", ""),
                "source": "economic_calendar+fred",
            }
        )
    return items
