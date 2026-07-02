from __future__ import annotations

from typing import Final

import numpy as np

import pandas as pd

REQUIRED_COLUMNS: Final[tuple[str, ...]] = ("open", "high", "low", "close")


def _validate_input(df: pd.DataFrame) -> bool:
    if df.empty:
        return False
    return all(column in df.columns for column in REQUIRED_COLUMNS)


def _calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.where(avg_loss != 0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def _calc_macd(close: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = close.ewm(span=12, adjust=False).mean()
    ema_slow = close.ewm(span=26, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    return macd, signal, hist


def _calc_bbands(close: pd.Series, period: int = 20, std_mult: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series]:
    basis = close.rolling(window=period, min_periods=period).mean()
    std = close.rolling(window=period, min_periods=period).std(ddof=0)
    upper = basis + std_mult * std
    lower = basis - std_mult * std
    return upper, basis, lower


def _calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]

    prev_close = close.shift(1)
    tr_1 = high - low
    tr_2 = (high - prev_close).abs()
    tr_3 = (low - prev_close).abs()

    true_range = pd.concat([tr_1, tr_2, tr_3], axis=1).max(axis=1)
    atr = true_range.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    return atr


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add RSI/MACD/BB/ATR/recent highs-lows to a price DataFrame.

    Returns an empty DataFrame when required columns are missing.
    """
    if not _validate_input(df):
        return pd.DataFrame()

    out = df.copy()

    out["rsi_14"] = _calc_rsi(out["close"], period=14)

    macd, macd_signal, macd_hist = _calc_macd(out["close"])
    out["macd"] = macd
    out["macd_signal"] = macd_signal
    out["macd_hist"] = macd_hist

    bb_upper, bb_mid, bb_lower = _calc_bbands(out["close"], period=20, std_mult=2.0)
    out["bb_upper"] = bb_upper
    out["bb_mid"] = bb_mid
    out["bb_lower"] = bb_lower

    out["atr_14"] = _calc_atr(out, period=14)

    out["recent_high_20"] = out["high"].rolling(window=20, min_periods=20).max()
    out["recent_low_20"] = out["low"].rolling(window=20, min_periods=20).min()

    return out
