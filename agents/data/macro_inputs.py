"""Assemble everything the macro analyst sees into one enriched MacroData.

Layers, each optional and fail-safe:

1. FRED series (get_macro_data): real rate, 10y/2y yields, breakeven, fed funds,
   and the week-lagged broad dollar index as a fallback.
2. Live dollar index from MT5 (get_dollar_index_snapshot) replacing the FRED
   dollar read when available; the FRED one is kept under ``dxy_fred``.
3. Gold positioning (CFTC managed money, SPDR GLD holdings).
4. Economic calendar: high-impact USD releases of the last 48h with actual
   prints and surprises, and the events scheduled in the next 24h.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any, Callable

from agents.data.fred_client import MacroData
from agents.data.positioning import get_positioning
from agents.data.releases import build_recent_releases, build_upcoming_events
from config import EVENTS_LOOKAHEAD_HOURS, RELEASES_LOOKBACK_HOURS
from data.mt5_client import get_dollar_index_snapshot
from data.news_client import fetch_calendar_events

LOGGER = logging.getLogger(__name__)


def _safe(label: str, fn: Callable[[], Any], default: Any) -> Any:
    try:
        return fn()
    except Exception as exc:
        LOGGER.warning("macro_inputs: %s failed safely: %s", label, exc)
        return default


def build_macro_inputs(
    macro_data: MacroData | dict[str, Any],
    now: datetime | None = None,
    dollar_index_fetcher: Callable[[], dict[str, Any]] = get_dollar_index_snapshot,
    positioning_fetcher: Callable[[], dict[str, Any]] = get_positioning,
    calendar_fetcher: Callable[[], list[dict[str, Any]] | None] = fetch_calendar_events,
) -> dict[str, Any]:
    """Return a copy of ``macro_data`` with live dollar index, positioning and
    calendar context attached. Never raises; missing layers are simply absent
    and noted in ``_meta.warnings``."""
    now = now or datetime.now(UTC)
    enriched: dict[str, Any] = dict(macro_data or {})
    meta = dict(enriched.get("_meta") or {})
    warnings = list(meta.get("warnings") or [])

    # 2. live dollar index ---------------------------------------------------
    live_dxy = _safe("dollar index", dollar_index_fetcher, {"_meta": {"ok": False, "error": "exception"}})
    if isinstance(live_dxy, dict) and bool(live_dxy.get("_meta", {}).get("ok")):
        if "dxy" in enriched:
            enriched["dxy_fred"] = enriched["dxy"]
        snapshot = {k: v for k, v in live_dxy.items() if k != "_meta"}
        enriched["dxy"] = snapshot
    else:
        err = live_dxy.get("_meta", {}).get("error", "") if isinstance(live_dxy, dict) else "unknown"
        warnings.append(f"live dollar index unavailable ({err}); using FRED DTWEXBGS (lags ~1 week)")

    # 3. positioning ---------------------------------------------------------
    positioning = _safe("positioning", positioning_fetcher, {"_meta": {"ok": False, "error": "exception"}})
    if isinstance(positioning, dict):
        enriched["positioning"] = positioning
        if not bool(positioning.get("_meta", {}).get("ok")):
            warnings.append(f"positioning unavailable ({positioning.get('_meta', {}).get('error', '')})")

    # 4. calendar ------------------------------------------------------------
    events = _safe("calendar", calendar_fetcher, None)
    if isinstance(events, list):
        enriched["recent_releases"] = _safe(
            "recent releases",
            lambda: build_recent_releases(events, now=now, lookback_hours=RELEASES_LOOKBACK_HOURS),
            [],
        )
        enriched["upcoming_events"] = _safe(
            "upcoming events",
            lambda: build_upcoming_events(events, now=now, lookahead_hours=EVENTS_LOOKAHEAD_HOURS),
            [],
        )
    else:
        warnings.append("economic calendar unavailable")

    meta["warnings"] = warnings
    enriched["_meta"] = meta
    return enriched
