"""Candidate entry signals for the event-study harness.

Contract for every signal function:

* input  : a DataFrame that already went through ``indicators.ta_calc.add_indicators``
* output : ``pd.Series`` of +1 (long) / -1 (short) / 0 (nothing), same index
* timing : the value at bar ``t`` may only use bars ``<= t`` (the bar is
  closed). The harness enters at the OPEN of bar ``t+1``.
  ``tests/test_event_study.py`` enforces this with a truncation test.

Signals fire on the first bar a condition becomes true (not on every bar it
stays true) so one market move is not counted as many events.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

Signal = Callable[[pd.DataFrame], pd.Series]


def _first_true(condition: pd.Series) -> pd.Series:
    cond = condition.fillna(False).astype(bool)
    return cond & ~cond.shift(1, fill_value=False)


def _combine(long_cond: pd.Series, short_cond: pd.Series) -> pd.Series:
    out = pd.Series(0, index=long_cond.index, dtype="int8")
    out[long_cond & ~short_cond] = 1
    out[short_cond & ~long_cond] = -1
    return out


def bb_reversion(df: pd.DataFrame) -> pd.Series:
    """Fade a close outside the 20/2.0 Bollinger band (mean reversion to the mid)."""
    long_cond = _first_true(df["close"] < df["bb_lower"])
    short_cond = _first_true(df["close"] > df["bb_upper"])
    return _combine(long_cond, short_cond)


def ma_deviation(df: pd.DataFrame, period: int = 50, k_atr: float = 2.0) -> pd.Series:
    """Fade a close stretched more than ``k_atr`` ATR away from the SMA.

    A significantly NEGATIVE result means the opposite (momentum) has the edge.
    """
    sma = df["close"].rolling(period, min_periods=period).mean()
    stretch = (df["close"] - sma) / df["atr_14"].replace(0.0, np.nan)
    long_cond = _first_true(stretch < -k_atr)
    short_cond = _first_true(stretch > k_atr)
    return _combine(long_cond, short_cond)


def donchian_breakout(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    """Follow a close beyond the prior ``lookback``-bar high/low (type-2 logic)."""
    prior_high = df["high"].shift(1).rolling(lookback, min_periods=lookback).max()
    prior_low = df["low"].shift(1).rolling(lookback, min_periods=lookback).min()
    long_cond = _first_true(df["close"] > prior_high)
    short_cond = _first_true(df["close"] < prior_low)
    return _combine(long_cond, short_cond)


def liquidity_sweep(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    """Wick through the prior ``lookback``-bar extreme but CLOSE back inside.

    Same trigger level as ``donchian_breakout`` with the opposite reading, so
    the two results should be read side by side.
    """
    prior_high = df["high"].shift(1).rolling(lookback, min_periods=lookback).max()
    prior_low = df["low"].shift(1).rolling(lookback, min_periods=lookback).min()
    long_cond = (df["low"] < prior_low) & (df["close"] > prior_low)
    short_cond = (df["high"] > prior_high) & (df["close"] < prior_high)
    return _combine(long_cond.fillna(False), short_cond.fillna(False))


def fvg_retest(df: pd.DataFrame, min_gap_atr: float = 0.25, max_age: int = 24) -> pd.Series:
    """First return into a fair value gap that holds on a closing basis.

    Bullish FVG at bar t: low[t] > high[t-2]  -> zone [high[t-2], low[t]].
    Long when a later bar (within ``max_age``) trades into the zone and closes
    at or above its bottom. A zone is consumed by its first touch either way.
    Bearish is the mirror image.
    """
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    atr = df["atr_14"].to_numpy(dtype=float)
    n = len(df)
    sig = np.zeros(n, dtype="int8")
    conflict = np.zeros(n, dtype=bool)
    zones: list[tuple[int, float, float, int]] = []  # side, bottom, top, born

    for t in range(2, n):
        survivors: list[tuple[int, float, float, int]] = []
        for side, bottom, top, born in zones:
            if t - born > max_age:
                continue
            if side == 1 and low[t] <= top:
                if close[t] >= bottom:
                    conflict[t] |= sig[t] == -1
                    sig[t] = 1
                continue
            if side == -1 and high[t] >= bottom:
                if close[t] <= top:
                    conflict[t] |= sig[t] == 1
                    sig[t] = -1
                continue
            survivors.append((side, bottom, top, born))
        zones = survivors

        a = atr[t]
        if not np.isfinite(a) or a <= 0:
            continue
        if low[t] - high[t - 2] >= min_gap_atr * a:
            zones.append((1, high[t - 2], low[t], t))
        if low[t - 2] - high[t] >= min_gap_atr * a:
            zones.append((-1, high[t], low[t - 2], t))

    sig[conflict] = 0
    return pd.Series(sig, index=df.index, dtype="int8")


SIGNALS: dict[str, Signal] = {
    "bb_reversion": bb_reversion,
    "ma_deviation": ma_deviation,
    "donchian_breakout": donchian_breakout,
    "liquidity_sweep": liquidity_sweep,
    "fvg_retest": fvg_retest,
}
