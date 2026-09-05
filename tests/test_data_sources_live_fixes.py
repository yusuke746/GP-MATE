"""Regressions from the first production run of scripts/check_data_sources.py."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from types import SimpleNamespace

from agents.data import positioning, releases
from data import mt5_client, news_client


def test_parse_datetime_accepts_zone_less_timestamps() -> None:
    # investing.com RSS: every item was dropped as "undated" before.
    assert news_client._parse_datetime("2026-09-04 22:51:00") == datetime(2026, 9, 4, 22, 51, tzinfo=UTC)
    assert news_client._parse_datetime("2026-09-04T22:51:00Z") == datetime(2026, 9, 4, 22, 51, tzinfo=UTC)
    assert news_client._parse_datetime("Fri, 04 Sep 2026 22:51:00 GMT") == datetime(2026, 9, 4, 22, 51, tzinfo=UTC)
    assert news_client._parse_datetime("not a date") is None
    assert news_client._parse_datetime("") is None


def test_parse_gld_csv_handles_cr_line_endings() -> None:
    rows = ["SPDR Gold Trust", "Date,Close,Tonnes in the Trust,NAV", "01-Aug-2026,300,950.0,300"]
    rows += [f"{d:02d}-Aug-2026,300,{950 + d}.0,300" for d in range(2, 12)]
    text = "\r".join(rows) + "\r"
    result = positioning.parse_gld_csv(text)
    assert result["_meta"]["ok"] is True
    assert result["as_of"] == "2026-08-11"
    assert result["tonnes"] == 961.0
    assert result["change_5d"] == 5.0


def test_fred_actual_is_rounded_to_calendar_precision() -> None:
    when = datetime(2026, 9, 4, 12, 30, tzinfo=UTC)
    # 0.27% m/m from FRED vs a 0.3% consensus is a match, not a -0.03 miss.
    points = [(date(2026, 7, 1), 100.0), (date(2026, 8, 1), 100.27)]
    record = releases.build_release_record(
        {"title": "Average Hourly Earnings m/m", "datetime_utc": when, "forecast": "0.3%", "previous": "0.3%", "actual": ""},
        {"CES0500000003": points},
    )
    assert record["actual"] == 0.3
    assert record["surprise"] == 0.0
    assert record["first_order_read"] == ""

    nfp = releases.build_release_record(
        {"title": "Non-Farm Employment Change", "datetime_utc": when, "forecast": "55K", "previous": "73K", "actual": ""},
        {"PAYEMS": [(date(2026, 7, 1), 158_000.0), (date(2026, 8, 1), 158_162.4)]},
    )
    assert nfp["actual"] == 162.0
    assert nfp["surprise"] == 107.0


def test_find_symbol_variants_matches_broker_suffixes() -> None:
    names = ["GOLD#", "EURUSD#", "EURUSDmicro", "USDJPY#", "GBPUSD.r", "USDX", "BTCUSD#"]
    assert mt5_client.find_symbol_variants(names, "EURUSD") == ["EURUSD#", "EURUSDmicro"]
    assert mt5_client.find_symbol_variants(names, "USDX") == ["USDX"]
    assert mt5_client.find_symbol_variants(names, "USDSEK") == []




def test_dollar_index_snapshot_discovers_suffixed_pairs(monkeypatch) -> None:
    start = datetime(2026, 7, 1, tzinfo=UTC)
    days = [start + timedelta(days=i) for i in range(40)]
    epochs = [int(d.timestamp()) for d in days]

    class FakeMt5:
        TIMEFRAME_D1 = 16408

        def symbols_get(self):
            return [SimpleNamespace(name=n) for n in ("GOLD#", "EURUSD#", "USDJPY#", "GBPUSD#")]

        def symbol_info(self, name):
            return SimpleNamespace(visible=True) if name in {"GOLD#", "EURUSD#", "USDJPY#", "GBPUSD#"} else None

        def symbol_select(self, name, flag):
            return True

        def copy_rates_from_pos(self, symbol, tf, start, count):
            base = {"EURUSD#": 1.10, "USDJPY#": 150.0, "GBPUSD#": 1.30}[symbol]
            drift = {"EURUSD#": -0.001, "USDJPY#": 0.2, "GBPUSD#": 0.0}[symbol]
            return [{"time": t, "close": base + drift * i} for i, t in enumerate(epochs)]

    monkeypatch.setattr(mt5_client, "mt5", FakeMt5())
    monkeypatch.setattr(mt5_client, "connect", lambda: True)
    monkeypatch.setattr(mt5_client, "disconnect", lambda: None)
    monkeypatch.setattr(mt5_client, "SERVER_TZ", None)

    snap = mt5_client.get_dollar_index_snapshot(candidates=("USDX",))
    assert snap["_meta"]["ok"] is True
    assert snap["source"] == "mt5:synthetic(EURUSD#,USDJPY#,GBPUSD#)"
    assert snap["direction"] == "UP"  # EUR down + JPY weaker -> dollar up
    assert snap["direction_5d"] == "UP"


def test_dollar_index_snapshot_reports_symbol_sample_when_nothing_matches(monkeypatch) -> None:
    class FakeMt5:
        TIMEFRAME_D1 = 16408

        def symbols_get(self):
            return [SimpleNamespace(name="GOLD#"), SimpleNamespace(name="BTCUSD#")]

        def symbol_info(self, name):
            return None

    monkeypatch.setattr(mt5_client, "mt5", FakeMt5())
    monkeypatch.setattr(mt5_client, "connect", lambda: True)
    monkeypatch.setattr(mt5_client, "disconnect", lambda: None)

    snap = mt5_client.get_dollar_index_snapshot(candidates=("USDX",))
    assert snap["_meta"]["ok"] is False
    assert "GOLD#" in snap["_meta"]["error"]
