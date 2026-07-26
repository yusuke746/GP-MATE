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


def _calc_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    tr_1 = high - low
    tr_2 = (high - df["close"].shift(1)).abs()
    tr_3 = (low - df["close"].shift(1)).abs()
    true_range = pd.concat([tr_1, tr_2, tr_3], axis=1).max(axis=1)

    atr = true_range.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / atr.replace(0.0, np.nan))
    minus_di = 100 * (minus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / atr.replace(0.0, np.nan))

    di_sum = (plus_di + minus_di).replace(0.0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / di_sum
    adx = dx.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    return adx.fillna(0.0)


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
    out["adx_14"] = _calc_adx(out, period=14)

    out["recent_high_20"] = out["high"].rolling(window=20, min_periods=20).max()
    out["recent_low_20"] = out["low"].rolling(window=20, min_periods=20).min()

    return out
