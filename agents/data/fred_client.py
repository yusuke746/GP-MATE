from __future__ import annotations

import copy
import logging
import os
import time
from datetime import date, datetime, timedelta
from threading import Lock
from typing import Any, Final, Literal, TypedDict, cast

import requests

LOGGER = logging.getLogger(__name__)

FRED_BASE_URL: Final[str] = "https://api.stlouisfed.org/fred/series/observations"
ALPHA_VANTAGE_BASE_URL: Final[str] = "https://www.alphavantage.co/query"
REQUEST_TIMEOUT_SECONDS: Final[float] = 10.0
REQUEST_MAX_RETRIES: Final[int] = 2
STALE_CACHE_MAX_AGE_DAYS: Final[int] = 5
CACHE_LOCK = Lock()

SERIES_MAP: Final[dict[str, str]] = {
    "dxy": "DTWEXBGS",
    "real_rate": "DFII10",
    "us10y": "DGS10",
    "breakeven": "T10YIE",
    "fed_funds": "FEDFUNDS",
}


class MacroSeriesSnapshot(TypedDict):
    value: float | None
    change_30d: float | None
    direction: Literal["UP", "DOWN", "FLAT"]


class MacroMeta(TypedDict):
    ok: bool
    source: str
    cached: bool
    fetched_at: str
    error: str


class MacroData(TypedDict):
    dxy: MacroSeriesSnapshot
    real_rate: MacroSeriesSnapshot
    us10y: MacroSeriesSnapshot
    breakeven: MacroSeriesSnapshot
    fed_funds: MacroSeriesSnapshot
    as_of: str
    _meta: MacroMeta


class AlphaVantageFxResult(TypedDict):
    from_symbol: str
    to_symbol: str
    rate: float | None
    as_of: str
    _meta: MacroMeta


_DAILY_CACHE: dict[str, MacroData] = {}
_LAST_GOOD_CACHE: dict[str, Any] = {"data": None, "fetched_date": None}


def _today_key(now: datetime | None = None) -> str:
    current = now or datetime.now()
    return current.date().isoformat()


def _empty_snapshot() -> MacroSeriesSnapshot:
    return {
        "value": None,
        "change_30d": None,
        "direction": "FLAT",
    }


def _empty_macro_data(as_of: str, source: str, error: str, cached: bool = False) -> MacroData:
    return {
        "dxy": _empty_snapshot(),
        "real_rate": _empty_snapshot(),
        "us10y": _empty_snapshot(),
        "breakeven": _empty_snapshot(),
        "fed_funds": _empty_snapshot(),
        "as_of": as_of,
        "_meta": {
            "ok": False,
            "source": source,
            "cached": cached,
            "fetched_at": as_of,
            "error": error,
        },
    }


def _empty_alpha_vantage_data(from_symbol: str, to_symbol: str, error: str, cached: bool = False) -> AlphaVantageFxResult:
    as_of = _today_key()
    return {
        "from_symbol": from_symbol,
        "to_symbol": to_symbol,
        "rate": None,
        "as_of": as_of,
        "_meta": {
            "ok": False,
            "source": "alpha_vantage",
            "cached": cached,
            "fetched_at": as_of,
            "error": error,
        },
    }


def _parse_float(value: Any) -> float | None:
    text = str(value or "").strip()
    if not text or text == ".":
        return None
    try:
        return float(text)
    except Exception:
        return None


def _safe_get_json(url: str, params: dict[str, Any]) -> dict[str, Any] | None:
    last_error = ""
    for attempt in range(REQUEST_MAX_RETRIES):
        try:
            response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
            response.raise_for_status()
            payload = response.json()
            if isinstance(payload, dict):
                return payload
            last_error = "non-dict json"
        except Exception as exc:
            last_error = str(exc)
            LOGGER.warning("FRED request failed (attempt=%s/%s): %s", attempt + 1, REQUEST_MAX_RETRIES, exc)
        if attempt + 1 < REQUEST_MAX_RETRIES:
            time.sleep(0.2 * (attempt + 1))
    LOGGER.warning("FRED request exhausted retries: %s", last_error)
    return None


def _direction_from_change(change_30d: float | None) -> Literal["UP", "DOWN", "FLAT"]:
    if change_30d is None:
        return "FLAT"
    if change_30d > 0:
        return "UP"
    if change_30d < 0:
        return "DOWN"
    return "FLAT"


def _series_to_snapshot(observations: list[dict[str, Any]]) -> MacroSeriesSnapshot | None:
    parsed: list[tuple[date, float]] = []
    for row in observations:
        obs_date_raw = str(row.get("date") or "").strip()
        value = _parse_float(row.get("value"))
        if not obs_date_raw or value is None:
            continue
        try:
            obs_date = date.fromisoformat(obs_date_raw)
        except Exception:
            continue
        parsed.append((obs_date, value))

    if not parsed:
        return None

    parsed.sort(key=lambda item: item[0])
    latest_date, latest_value = parsed[-1]
    target_date = latest_date - timedelta(days=30)

    target_value: float | None = None
    for obs_date, value in parsed:
        if obs_date <= target_date:
            target_value = value

    change_30d: float | None
    if target_value is None:
        change_30d = None
    else:
        change_30d = latest_value - target_value

    return {
        "value": latest_value,
        "change_30d": change_30d,
        "direction": _direction_from_change(change_30d),
    }


def _fetch_fred_series(series_id: str, api_key: str) -> MacroSeriesSnapshot | None:
    payload = _safe_get_json(
        FRED_BASE_URL,
        {
            "series_id": series_id,
            "api_key": api_key,
            "file_type": "json",
            "sort_order": "desc",
            "limit": 200,
        },
    )
    if payload is None:
        return None

    observations = payload.get("observations", [])
    if not isinstance(observations, list):
        return None

    return _series_to_snapshot(cast(list[dict[str, Any]], observations))


def _build_macro_data(api_key: str) -> MacroData:
    as_of = _today_key()
    result = _empty_macro_data(as_of=as_of, source="fred", error="")

    if not api_key:
        result["_meta"]["error"] = "FRED_API_KEY missing"
        return result

    fetched: dict[str, MacroSeriesSnapshot] = {}
    for key, series_id in SERIES_MAP.items():
        snapshot = _fetch_fred_series(series_id=series_id, api_key=api_key)
        if snapshot is None:
            result["_meta"]["error"] = f"failed to fetch series_id={series_id}"
            return result
        fetched[key] = snapshot

    result.update(fetched)
    result["_meta"]["ok"] = True
    result["_meta"]["error"] = ""
    return result


def get_macro_data(force_refresh: bool = False, api_key: str | None = None) -> MacroData:
    resolved_key = (api_key or os.getenv("FRED_API_KEY", "")).strip()
    today = _today_key()

    with CACHE_LOCK:
        if not force_refresh and today in _DAILY_CACHE:
            cached = copy.deepcopy(_DAILY_CACHE[today])
            cached["_meta"]["cached"] = True
            return cached

    result = _build_macro_data(resolved_key)

    with CACHE_LOCK:
        if result["_meta"]["ok"]:
            # Successful fetches refresh both daily cache and last-good snapshot.
            _DAILY_CACHE[today] = copy.deepcopy(result)
            _LAST_GOOD_CACHE["data"] = copy.deepcopy(result)
            _LAST_GOOD_CACHE["fetched_date"] = date.fromisoformat(today)
            return result

        # On failure, return a stale last-good snapshot if it is still within TTL.
        last_good = _LAST_GOOD_CACHE.get("data")
        last_date = _LAST_GOOD_CACHE.get("fetched_date")
        if last_good is not None and isinstance(last_date, date):
            age_days = (date.fromisoformat(today) - last_date).days
            if 0 <= age_days <= STALE_CACHE_MAX_AGE_DAYS:
                stale = copy.deepcopy(last_good)
                stale["_meta"]["cached"] = True
                stale["_meta"]["ok"] = True
                stale["_meta"]["error"] = (
                    f"stale fallback age={age_days}d "
                    f"(fetch failed: {result['_meta']['error']})"
                )
                LOGGER.warning(
                    "get_macro_data: using stale last-good cache (age=%sd) due to fetch failure",
                    age_days,
                )
                return stale
            LOGGER.warning(
                "get_macro_data: last-good cache too old (age=%sd > %sd); returning neutral failure",
                age_days,
                STALE_CACHE_MAX_AGE_DAYS,
            )

        # If no valid last-good exists, return the neutral failure result without caching it.
        return result


def get_alpha_vantage_fx_rate(
    from_symbol: str = "USD",
    to_symbol: str = "JPY",
    api_key: str | None = None,
) -> AlphaVantageFxResult:
    resolved_key = (api_key or os.getenv("ALPHA_VANTAGE_API_KEY", "")).strip()
    if not resolved_key:
        return _empty_alpha_vantage_data(from_symbol, to_symbol, "ALPHA_VANTAGE_API_KEY missing")

    payload = _safe_get_json(
        ALPHA_VANTAGE_BASE_URL,
        {
            "function": "CURRENCY_EXCHANGE_RATE",
            "from_currency": from_symbol,
            "to_currency": to_symbol,
            "apikey": resolved_key,
        },
    )
    if payload is None:
        return _empty_alpha_vantage_data(from_symbol, to_symbol, "request failed")

    data = payload.get("Realtime Currency Exchange Rate", {})
    if not isinstance(data, dict):
        return _empty_alpha_vantage_data(from_symbol, to_symbol, "unexpected response format")

    rate_text = str(data.get("5. Exchange Rate", "") or "").strip()
    rate = _parse_float(rate_text)
    if rate is None:
        return _empty_alpha_vantage_data(from_symbol, to_symbol, "missing exchange rate")

    as_of = _today_key()
    return {
        "from_symbol": from_symbol,
        "to_symbol": to_symbol,
        "rate": rate,
        "as_of": as_of,
        "_meta": {
            "ok": True,
            "source": "alpha_vantage",
            "cached": False,
            "fetched_at": as_of,
            "error": "",
        },
    }
