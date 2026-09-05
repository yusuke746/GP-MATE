from __future__ import annotations

import importlib

import config


def test_blank_env_values_fall_back_to_defaults(monkeypatch) -> None:
    # .env.example ships every key as "KEY="; a blank must behave like unset.
    monkeypatch.setenv("COT_DATASET_URL", "")
    monkeypatch.setenv("COT_MARKET_NAME", "   ")
    monkeypatch.setenv("GLD_HOLDINGS_URL", "")
    monkeypatch.setenv("DXY_SYMBOL_CANDIDATES", "")
    monkeypatch.setenv("RSS_FEEDS", "")
    monkeypatch.setenv("MODEL_ANALYSIS", "")
    monkeypatch.setenv("RELEASES_LOOKBACK_HOURS", "")
    try:
        reloaded = importlib.reload(config)
        assert reloaded.COT_DATASET_URL.startswith("https://publicreporting.cftc.gov/")
        assert reloaded.COT_MARKET_NAME == "GOLD - COMMODITY EXCHANGE INC."
        assert reloaded.GLD_HOLDINGS_URL.startswith("https://www.spdrgoldshares.com/")
        assert reloaded.DXY_SYMBOL_CANDIDATES == ("USDX", "USDX#", "DXY", "USDOLLAR", "DX")
        assert reloaded.RSS_FEEDS == reloaded.DEFAULT_RSS_FEEDS
        assert reloaded.MODEL_ANALYSIS == "gpt-5.6-terra"
        assert reloaded.RELEASES_LOOKBACK_HOURS == 48
    finally:
        monkeypatch.undo()
        importlib.reload(config)


def test_explicit_env_values_still_win(monkeypatch) -> None:
    monkeypatch.setenv("COT_MARKET_NAME", "GOLD - CUSTOM")
    monkeypatch.setenv("DXY_SYMBOL_CANDIDATES", "USDX.r, DX1!")
    monkeypatch.setenv("RSS_FEEDS", "https://a.example/rss,https://b.example/rss")
    try:
        reloaded = importlib.reload(config)
        assert reloaded.COT_MARKET_NAME == "GOLD - CUSTOM"
        assert reloaded.DXY_SYMBOL_CANDIDATES == ("USDX.r", "DX1!")
        assert reloaded.RSS_FEEDS == ("https://a.example/rss", "https://b.example/rss")
    finally:
        monkeypatch.undo()
        importlib.reload(config)
