from __future__ import annotations

from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

from data import confirmed_bars as cb
from data import mt5_client


def _frame(end_open: datetime, bars: int, hours: int = 1) -> pd.DataFrame:
    times = [end_open - timedelta(hours=hours * (bars - 1 - i)) for i in range(bars)]
    frame = pd.DataFrame({"time": pd.to_datetime(times, utc=True), "close": [100.0 + i for i in range(bars)]})
    return frame


def test_get_confirmed_rates_drops_forming_bar_and_reports_meta(monkeypatch) -> None:
    monkeypatch.setattr(mt5_client, "SERVER_TZ", None)
    now = datetime(2026, 9, 28, 12, 0, 30, tzinfo=UTC)  # 30s into the 12:00 H1 bar
    calls: list[int] = []

    def fetch(symbol, tf, count):
        calls.append(count)
        return _frame(datetime(2026, 9, 28, 12, 0, tzinfo=UTC), count)  # last row = forming 12:00 bar

    frame, meta = cb.get_confirmed_rates("GOLD#", "H1", 5, now, fetch=fetch)
    assert calls == [5 + cb.FETCH_MARGIN]
    assert len(frame) == 5 and meta["closed_bar_count"] == 5 and meta["requested"] == 5
    assert pd.Timestamp(frame["time"].iloc[-1]) == pd.Timestamp("2026-09-28T11:00:00Z")
    assert meta["dropped_open_bar"] is True
    assert meta["last_closed_bar_time"] == "2026-09-28T12:00:00+00:00"
    assert meta["bar_age_seconds"] == 30
    assert meta["timeframe"] == "H1"


def test_get_confirmed_rates_keeps_all_when_fetch_returned_only_closed_bars(monkeypatch) -> None:
    monkeypatch.setattr(mt5_client, "SERVER_TZ", None)
    now = datetime(2026, 9, 28, 12, 0, 30, tzinfo=UTC)
    frame, meta = cb.get_confirmed_rates("GOLD#", "H4", 3, now, fetch=lambda s, tf, c: _frame(datetime(2026, 9, 28, 4, 0, tzinfo=UTC), c, hours=4))
    assert meta["dropped_open_bar"] is False and len(frame) == 3
    assert meta["last_closed_bar_time"] == "2026-09-28T08:00:00+00:00"
    assert meta["bar_age_seconds"] == 4 * 3600 + 30


def test_get_confirmed_rates_reinterprets_server_wall_clock(monkeypatch) -> None:
    monkeypatch.setattr(mt5_client, "SERVER_TZ", ZoneInfo("Europe/Athens"))  # UTC+3 in September
    now = datetime(2026, 9, 28, 12, 0, 30, tzinfo=UTC)
    # Server stamps 15:00 local as "15:00Z"; that is the forming 12:00Z bar.
    frame, meta = cb.get_confirmed_rates("GOLD#", "H1", 4, now, fetch=lambda s, tf, c: _frame(datetime(2026, 9, 28, 15, 0, tzinfo=UTC), c))
    assert meta["dropped_open_bar"] is True
    assert pd.Timestamp(frame["time"].iloc[-1]) == pd.Timestamp("2026-09-28T11:00:00Z")


def test_get_confirmed_rates_handles_empty_and_timeless_frames() -> None:
    frame, meta = cb.get_confirmed_rates("GOLD#", "H1", 5, datetime.now(UTC), fetch=lambda s, tf, c: pd.DataFrame())
    assert frame.empty and meta["closed_bar_count"] == 0 and meta["dropped_open_bar"] is None
    timeless = pd.DataFrame({"close": [1.0] * 8})
    frame, meta = cb.get_confirmed_rates("GOLD#", "H1", 5, datetime.now(UTC), fetch=lambda s, tf, c: timeless)
    assert len(frame) == 5 and meta["dropped_open_bar"] is None and meta["last_closed_bar_time"] == ""


def test_drop_forming_bars_and_timeframe_seconds() -> None:
    now = datetime(2026, 9, 28, 12, 0, tzinfo=UTC)
    frame = _frame(datetime(2026, 9, 28, 12, 0, tzinfo=UTC), 3)
    assert len(cb.drop_forming_bars(frame, now, "H1")) == 2  # the 11:00 bar closed exactly at 12:00 and counts
    assert cb.timeframe_seconds("d1") == 86400 and cb.timeframe_seconds("unknown") == 3600
