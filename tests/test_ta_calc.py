from __future__ import annotations

import pandas as pd

from indicators.ta_calc import add_indicators


def _sample_ohlc(rows: int = 120) -> pd.DataFrame:
    base = 2300.0
    close = [base + i * 0.4 for i in range(rows)]
    open_ = [c - 0.2 for c in close]
    high = [c + 1.0 for c in close]
    low = [c - 1.0 for c in close]
    volume = [1000 + i for i in range(rows)]

    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "tick_volume": volume,
        }
    )


def test_add_indicators_returns_empty_when_required_columns_missing() -> None:
    df = pd.DataFrame({"close": [1, 2, 3]})
    out = add_indicators(df)
    assert out.empty


def test_add_indicators_adds_all_expected_columns() -> None:
    df = _sample_ohlc(120)
    out = add_indicators(df)

    expected_columns = {
        "rsi_14",
        "macd",
        "macd_signal",
        "macd_hist",
        "bb_upper",
        "bb_mid",
        "bb_lower",
        "atr_14",
        "recent_high_20",
        "recent_low_20",
    }
    assert expected_columns.issubset(set(out.columns))


def test_rsi_is_within_range() -> None:
    df = _sample_ohlc(120)
    out = add_indicators(df)

    rsi = out["rsi_14"].dropna()
    assert not rsi.empty
    assert (rsi >= 0).all()
    assert (rsi <= 100).all()


def test_recent_levels_are_consistent() -> None:
    df = _sample_ohlc(120)
    out = add_indicators(df)

    tail = out.iloc[-1]
    expected_high = out["high"].iloc[-20:].max()
    expected_low = out["low"].iloc[-20:].min()

    assert tail["recent_high_20"] == expected_high
    assert tail["recent_low_20"] == expected_low


def test_atr_is_non_negative() -> None:
    df = _sample_ohlc(120)
    out = add_indicators(df)

    atr = out["atr_14"].dropna()
    assert not atr.empty
    assert (atr >= 0).all()


def test_macd_hist_equals_macd_minus_signal() -> None:
    df = _sample_ohlc(120)
    out = add_indicators(df)

    diff = (out["macd"] - out["macd_signal"] - out["macd_hist"]).abs().dropna()
    assert (diff < 1e-9).all()
