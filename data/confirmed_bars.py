"""Confirmed (closed) bars only, shared by the trading loop and the forecast logger.

MT5's ``copy_rates_from_pos(..., 0, count)`` returns the forming bar as the
last row. At an 08:00 NY judgment that row is a bar that opened seconds ago,
so any indicator computed on it is noise. Every consumer must therefore:

1. fetch a few more bars than needed,
2. convert bar times to true UTC (MT5 stamps server wall-clock as UTC),
3. drop bars whose close time is after ``now``,
4. keep the last ``count`` closed bars,
5. only then compute indicators, swings, gaps and levels.

``get_confirmed_rates`` does 1-4 and reports what it did so the trade log can
show which bar a decision actually used.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Callable, Final

import pandas as pd

from data import mt5_client
from data.mt5_client import get_rates

TIMEFRAME_SECONDS: Final[dict[str, int]] = {
    "M1": 60,
    "M5": 300,
    "M15": 900,
    "M30": 1800,
    "H1": 3600,
    "H4": 14400,
    "D1": 86400,
}
# Extra bars requested so that dropping the forming bar still leaves ``count``.
FETCH_MARGIN: Final[int] = 2


def timeframe_seconds(timeframe: str) -> int:
    return TIMEFRAME_SECONDS.get(str(timeframe).upper(), 3600)


def bar_times_to_utc(frame: pd.DataFrame) -> pd.DataFrame:
    """MT5 bar times are server wall-clock stamped as UTC; reinterpret them.

    Uses the same rule as closed-deal timestamps (MT5_SERVER_TIMEZONE). With
    no timezone configured the frame is returned unchanged (legacy: as UTC).
    """
    if frame is None or frame.empty or "time" not in frame.columns:
        return frame
    if mt5_client.SERVER_TZ is None:
        return frame
    converted = frame.copy()
    converted["time"] = pd.to_datetime(
        [mt5_client._deal_epoch_to_utc(int(pd.Timestamp(t).timestamp())) for t in converted["time"]],
        utc=True,
    )
    return converted


def drop_forming_bars(frame: pd.DataFrame, now_utc: datetime, timeframe: str) -> pd.DataFrame:
    """Keep only bars whose close time (start + timeframe) is <= now."""
    if frame is None or frame.empty or "time" not in frame.columns:
        return frame
    close_times = pd.to_datetime(frame["time"], utc=True) + pd.Timedelta(seconds=timeframe_seconds(timeframe))
    return frame[close_times <= pd.Timestamp(now_utc)].reset_index(drop=True)


def get_confirmed_rates(
    symbol: str,
    timeframe: str,
    count: int,
    now_utc: datetime | None = None,
    fetch: Callable[[str, str, int], pd.DataFrame] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Last ``count`` closed bars of ``symbol``/``timeframe`` plus a meta dict.

    meta: requested, fetched, closed_bar_count, dropped_open_bar,
    last_closed_bar_time (ISO, UTC), bar_age_seconds (now - last close),
    timeframe. A frame without a ``time`` column (test doubles) is passed
    through with dropped_open_bar=None because nothing can be verified.
    """
    now = now_utc or datetime.now(UTC)
    fetcher = fetch or get_rates
    raw = fetcher(symbol, timeframe, count + FETCH_MARGIN)
    meta: dict[str, Any] = {
        "timeframe": str(timeframe).upper(),
        "requested": int(count),
        "fetched": 0 if raw is None else int(len(raw)),
        "closed_bar_count": 0,
        "dropped_open_bar": None,
        "last_closed_bar_time": "",
        "bar_age_seconds": None,
    }
    if raw is None or raw.empty:
        return pd.DataFrame(), meta
    if "time" not in raw.columns:
        frame = raw.tail(count).reset_index(drop=True)
        meta["closed_bar_count"] = int(len(frame))
        return frame, meta

    converted = bar_times_to_utc(raw)
    confirmed = drop_forming_bars(converted, now, timeframe)
    meta["dropped_open_bar"] = bool(len(confirmed) < len(converted))
    frame = confirmed.tail(count).reset_index(drop=True)
    meta["closed_bar_count"] = int(len(frame))
    if not frame.empty:
        last_open = pd.Timestamp(frame["time"].iloc[-1])
        if last_open.tzinfo is None:
            last_open = last_open.tz_localize("UTC")
        last_close = last_open + pd.Timedelta(seconds=timeframe_seconds(timeframe))
        meta["last_closed_bar_time"] = last_close.isoformat()
        meta["bar_age_seconds"] = int((pd.Timestamp(now) - last_close).total_seconds())
    return frame, meta
