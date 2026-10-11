"""Same-day US Treasury yields from Yahoo Finance (no API key).

Why: the FRED yield series (DGS10 etc.) are daily closes published the next
business day. On 2026-10-08 the system judged at 20:00 NY still "seeing" the
10/7 close near a 24-year high, seven hours after a strong 30-year auction had
already pushed yields down across the curve and gold through two stops. Yahoo's
chart endpoint quotes the CBOE yield indices (^TNX, ^TYX, ^FVX) and the 2-year
yield future (2YY=F) with a delay of minutes, free and keyless.

The snapshot keeps the FRED shape (value / change_30d / direction / change_5d /
direction_5d / as_of / source) so the macro analyst's data_notes still apply,
and adds change_1d (vs the previous daily close) and stale_minutes (age of the
live quote, large over the weekend). Everything is fail-safe: a symbol that
cannot be fetched is simply absent and listed in _meta.warnings.
"""

from __future__ import annotations

import copy
import logging
import time
from datetime import UTC, date, datetime
from typing import Any, Callable, Final

import requests

from agents.data.fred_client import snapshot_from_points

LOGGER = logging.getLogger(__name__)

YAHOO_CHART_URL: Final[str] = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
REQUEST_HEADERS: Final[dict[str, str]] = {"User-Agent": "Mozilla/5.0 (GP-MATE macro inputs)"}
REQUEST_TIMEOUT_SECONDS: Final[float] = 15.0
CHART_RANGE: Final[str] = "3mo"
CHART_INTERVAL: Final[str] = "1d"
CACHE_TTL_SECONDS: Final[int] = 600

# Macro key -> Yahoo symbol. ^TNX/^TYX/^FVX are quoted in percent (4.25 = 4.25%).
# The 2-year has no live Yahoo index: the yield future 2YY=F was found 18 days
# stale and 0.3pt off FRED on 2026-10-11, so the 2-year stays on FRED DGS2.
YIELD_SYMBOLS: Final[dict[str, str]] = {
    "us5y": "^FVX",
    "us10y": "^TNX",
    "us30y": "^TYX",
}
# A quote older than this (a long weekend plus a holiday) is a dead symbol,
# not a fresh read; it is dropped so the FRED close is used instead.
MAX_STALE_MINUTES: Final[float] = 5 * 24 * 60.0
# Without the 10-year the snapshot is not worth replacing FRED with.
REQUIRED_KEYS: Final[frozenset[str]] = frozenset({"us10y"})

_CACHE: dict[str, Any] = {"at": 0.0, "data": None}


def fetch_yahoo_chart(symbol: str, range_: str = CHART_RANGE, interval: str = CHART_INTERVAL) -> dict[str, Any] | None:
    """One chart request; the raw JSON payload or None on any failure (never raises)."""
    try:
        response = requests.get(
            YAHOO_CHART_URL.format(symbol=requests.utils.quote(symbol, safe="")),
            params={"range": range_, "interval": interval, "includePrePost": "false"},
            headers=REQUEST_HEADERS,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        payload = response.json()
        return payload if isinstance(payload, dict) else None
    except Exception as exc:
        LOGGER.warning("yahoo chart request failed (%s): %s", symbol, exc)
        return None


def _to_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed == parsed else None


def parse_chart_payload(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    """Yahoo v8 chart JSON -> {points: [(date, close)], live_price, live_time, previous_close, symbol}.

    Returns None when the payload carries an error or no usable closes. Null
    closes (holidays, partial bars) are skipped.
    """
    if not isinstance(payload, dict):
        return None
    chart = payload.get("chart")
    if not isinstance(chart, dict):
        return None
    if chart.get("error"):
        return None
    results = chart.get("result")
    if not isinstance(results, list) or not results or not isinstance(results[0], dict):
        return None
    result = results[0]
    meta = result.get("meta") if isinstance(result.get("meta"), dict) else {}
    timestamps = result.get("timestamp") if isinstance(result.get("timestamp"), list) else []
    quotes = result.get("indicators", {}).get("quote") if isinstance(result.get("indicators"), dict) else None
    closes = quotes[0].get("close") if isinstance(quotes, list) and quotes and isinstance(quotes[0], dict) else None
    closes = closes if isinstance(closes, list) else []

    points: list[tuple[date, float]] = []
    for ts, close in zip(timestamps, closes):
        value = _to_float(close)
        ts_value = _to_float(ts)
        if value is None or ts_value is None:
            continue
        points.append((datetime.fromtimestamp(ts_value, tz=UTC).date(), value))
    # One point per day (the last close of that day wins), sorted.
    by_day: dict[date, float] = {}
    for day, value in points:
        by_day[day] = value
    points = sorted(by_day.items())

    live_price = _to_float(meta.get("regularMarketPrice"))
    live_ts = _to_float(meta.get("regularMarketTime"))
    live_time = datetime.fromtimestamp(live_ts, tz=UTC) if live_ts is not None else None
    previous_close = _to_float(meta.get("chartPreviousClose"))
    if previous_close is None:
        previous_close = _to_float(meta.get("previousClose"))
    if not points and live_price is None:
        return None
    return {
        "symbol": str(meta.get("symbol") or ""),
        "points": points,
        "live_price": live_price,
        "live_time": live_time,
        "previous_close": previous_close,
    }


def yield_snapshot_from_chart(parsed: dict[str, Any], symbol: str, now: datetime | None = None) -> dict[str, Any] | None:
    """FRED-shaped snapshot with the live quote overlaid on the daily history."""
    now = now or datetime.now(UTC)
    points: list[tuple[date, float]] = list(parsed.get("points") or [])
    live_price: float | None = parsed.get("live_price")
    live_time: datetime | None = parsed.get("live_time")

    snapshot = snapshot_from_points(points)
    if snapshot is None:
        if live_price is None:
            return None
        snapshot = {"value": live_price, "change_30d": None, "direction": "FLAT", "change_5d": None, "direction_5d": "FLAT", "as_of": ""}
    result: dict[str, Any] = dict(snapshot)

    # The daily series may already end with today's partial bar; the live quote
    # supersedes it, and change_1d is measured against the last COMPLETED day.
    if live_price is not None:
        reference_day = live_time.date() if live_time is not None else now.date()
    else:
        # No live quote: the value IS the last daily point, so "previous" is
        # the day before it.
        reference_day = points[-1][0] if points else now.date()
    completed = [p for p in points if p[0] < reference_day]
    previous_close: float | None = completed[-1][1] if completed else parsed.get("previous_close")

    if live_price is not None:
        result["value"] = live_price
        result["as_of"] = live_time.isoformat() if live_time is not None else now.isoformat()
        result["stale_minutes"] = round((now - live_time).total_seconds() / 60.0, 1) if live_time is not None else None
        # Re-base the window changes on the live value so they are not a bar behind.
        if completed:
            target = reference_day.toordinal() - 30
            base_30d = next((v for d, v in reversed(completed) if d.toordinal() <= target), None)
            if base_30d is not None:
                result["change_30d"] = round(live_price - base_30d, 4)
            if len(completed) >= 5:
                result["change_5d"] = round(live_price - completed[-5][1], 4)
    else:
        result["stale_minutes"] = None
    value = _to_float(result.get("value"))
    result["change_1d"] = round(value - previous_close, 4) if value is not None and previous_close is not None else None
    for key in ("change_30d", "change_5d", "change_1d"):
        change = result.get(key)
        result[f"direction{'' if key == 'change_30d' else '_' + key.split('_')[1]}"] = (
            "FLAT" if change is None or change == 0 else ("UP" if change > 0 else "DOWN")
        )
    result["source"] = f"yahoo:{symbol}"
    result["unit"] = "percent"
    return result


def get_treasury_yields_snapshot(
    now: datetime | None = None,
    fetch: Callable[[str], dict[str, Any] | None] | None = None,
    symbols: dict[str, str] | None = None,
) -> dict[str, Any]:
    """{us2y, us5y, us10y, us30y, _meta}. ok when the 10-year was fetched.

    Cached for CACHE_TTL_SECONDS between calls (the scheduler and the forecast
    logger both ask within minutes); an injected ``fetch`` bypasses the cache.
    """
    now = now or datetime.now(UTC)
    wanted = symbols or YIELD_SYMBOLS
    use_cache = fetch is None and symbols is None
    if use_cache and _CACHE["data"] is not None and time.monotonic() - float(_CACHE["at"]) < CACHE_TTL_SECONDS:
        cached = copy.deepcopy(_CACHE["data"])
        cached["_meta"]["cached"] = True
        return cached

    fetcher = fetch or fetch_yahoo_chart
    out: dict[str, Any] = {}
    warnings: list[str] = []
    for key, symbol in wanted.items():
        try:
            parsed = parse_chart_payload(fetcher(symbol))
            snapshot = yield_snapshot_from_chart(parsed, symbol, now=now) if parsed else None
        except Exception as exc:
            LOGGER.warning("yahoo yields: %s (%s) failed safely: %s", key, symbol, exc)
            snapshot = None
        if snapshot is None:
            warnings.append(f"{key} ({symbol}) unavailable")
            continue
        stale = snapshot.get("stale_minutes")
        if stale is not None and float(stale) > MAX_STALE_MINUTES:
            warnings.append(f"{key} ({symbol}) stale: quote is {float(stale) / 1440.0:.1f} days old")
            continue
        out[key] = snapshot

    ok = all(key in out for key in REQUIRED_KEYS)
    out["_meta"] = {
        "ok": ok,
        "source": "yahoo",
        "cached": False,
        "fetched_at": now.isoformat(),
        "error": "" if ok else "required yield series unavailable: " + ", ".join(sorted(REQUIRED_KEYS - set(out))),
        "warnings": warnings,
    }
    if use_cache and ok:
        _CACHE["at"] = time.monotonic()
        _CACHE["data"] = copy.deepcopy(out)
    return out
