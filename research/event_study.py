"""Event-study harness: does a bar-close signal carry information, and does
it survive costs?

Deliberately independent of ``config`` / MT5 so it runs on any machine from
the CSVs written by ``scripts/export_ohlcv.py``.

Conventions
-----------
* Signal decided at the close of bar t, entry at the OPEN of bar t+1.
* Returns are in units of ATR(14) at bar t, so symbols are comparable.
* ``gross``  : mean signed forward return (no costs).
* ``excess`` : gross minus the same calendar year's unconditional drift in
  the traded direction. This is the "is there information?" number; it stops
  a bull market from making every long signal look good.
* ``net``    : gross minus one spread. This is the "is it tradable?" number.
* t-stats use week-clustered standard errors because events bunch together
  and forward windows overlap.
* The holdout period is never evaluated unless explicitly unsealed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Iterable

import numpy as np
import pandas as pd

from indicators.ta_calc import add_indicators
from research.signals import Signal

ADX_SPLIT = 20.0


# --------------------------------------------------------------------------- #
# Loading and data quality
# --------------------------------------------------------------------------- #
def load_ohlcv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True)
    df = df.drop_duplicates("time_utc").sort_values("time_utc").set_index("time_utc")
    if "spread" not in df.columns:
        df["spread"] = 0
    return df


def infer_point(df: pd.DataFrame) -> float:
    sample = df["close"].dropna().to_numpy(dtype=float)[-5000:]
    for digits in range(0, 7):
        scaled = sample * (10**digits)
        if np.allclose(scaled, np.round(scaled), atol=1e-6):
            return 10.0 ** (-digits)
    return 1e-5


def data_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Per-year bar count and spread sanity. Old broker history is often thin
    or carries spread=0; both silently flatter a backtest."""
    by_year = df.groupby(df.index.year)
    out = pd.DataFrame(
        {
            "bars": by_year["close"].size(),
            "spread_zero_pct": by_year["spread"].apply(lambda s: round(float((s <= 0).mean() * 100), 1)),
            "spread_median": by_year["spread"].median(),
        }
    )
    full_years = out["bars"].iloc[1:-1] if len(out) > 2 else out["bars"]
    typical = float(full_years.median()) if len(full_years) else float(out["bars"].median())
    out["thin"] = out["bars"] < 0.8 * typical
    return out


# --------------------------------------------------------------------------- #
# Preparation: forward returns, costs, triple barrier
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Prepared:
    frame: pd.DataFrame
    horizons: tuple[int, ...]
    tb_k: float
    tb_horizon: int
    point: float
    cost_note: str


def _cost_in_price(df: pd.DataFrame, point: float, cost_points: float | None) -> tuple[pd.Series, str]:
    if cost_points is not None:
        return pd.Series(cost_points * point, index=df.index), f"fixed {cost_points:g} points"
    spread = df["spread"].astype(float).where(df["spread"] > 0)
    positive = spread.dropna()
    if positive.empty:
        raise ValueError("CSV has no positive spread values; pass --cost-points")
    # Bars with spread=0 are missing data, not free trading: fill from the
    # recorded spreads, and never assume cheaper than the recent median.
    recent_median = float(positive.iloc[-5000:].median())
    filled = spread.fillna(recent_median).clip(lower=recent_median * 0.5)
    share = float((df["spread"] <= 0).mean() * 100)
    return filled * point, f"CSV spread (recent median {recent_median:g} pts; {share:.0f}% of bars were 0 and filled)"


def triple_barrier(df: pd.DataFrame, atr: pd.Series, k: float, horizon: int) -> tuple[pd.Series, pd.Series]:
    """Outcome for a trade entered at open[t+1] with barriers +-k*ATR[t].

    Returns (outcome, timeout_return_atr). outcome: +1 up first, -1 down
    first, 0 timeout, NaN when ambiguous (both inside one bar) or undecidable.
    """
    entry = df["open"].shift(-1)
    up = entry + k * atr
    down = entry - k * atr
    outcome = pd.Series(np.nan, index=df.index)
    undecided = entry.notna() & atr.notna() & (atr > 0)
    ambiguous = pd.Series(False, index=df.index)
    for j in range(1, horizon + 1):
        hit_up = (df["high"].shift(-j) >= up) & undecided
        hit_down = (df["low"].shift(-j) <= down) & undecided
        both = hit_up & hit_down
        outcome[hit_up & ~both] = 1.0
        outcome[hit_down & ~both] = -1.0
        ambiguous |= both
        undecided &= ~(hit_up | hit_down)
    final_close = df["close"].shift(-horizon)
    timeout = undecided & final_close.notna()
    outcome[timeout] = 0.0
    timeout_ret = ((final_close - entry) / atr).where(timeout)
    return outcome, timeout_ret


def prepare(
    raw: pd.DataFrame,
    horizons: Iterable[int] = (1, 3, 6, 12),
    tb_k: float = 1.0,
    tb_horizon: int = 6,
    point: float | None = None,
    cost_points: float | None = None,
) -> Prepared:
    df = add_indicators(raw)
    if df.empty:
        raise ValueError("input is missing OHLC columns")
    point_value = point if point else infer_point(df)
    atr = df["atr_14"].where(df["atr_14"] > 0)
    entry = df["open"].shift(-1)
    horizons_t = tuple(sorted({int(h) for h in horizons if int(h) > 0}))
    for h in horizons_t:
        df[f"fwd_{h}"] = (df["close"].shift(-h) - entry) / atr
    cost_price, cost_note = _cost_in_price(df, point_value, cost_points)
    df["cost_atr"] = cost_price / atr
    df["tb"], df["tb_timeout_ret"] = triple_barrier(df, atr, tb_k, tb_horizon)
    df["week"] = df.index.strftime("%G-%V")
    df["year"] = df.index.year
    return Prepared(df, horizons_t, tb_k, tb_horizon, point_value, cost_note)


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def cluster_t(values: np.ndarray, clusters: np.ndarray) -> float:
    """t-stat of the mean with cluster-robust SE (clusters = calendar weeks)."""
    n = len(values)
    if n < 2:
        return float("nan")
    resid = values - values.mean()
    sums = pd.Series(resid).groupby(clusters).sum().to_numpy()
    groups = len(sums)
    if groups < 2:
        return float("nan")
    se = math.sqrt(groups / (groups - 1) * float((sums**2).sum())) / n
    return float(values.mean() / se) if se > 0 else float("nan")


def bonferroni_t(tests: int, alpha: float = 0.05) -> float:
    return NormalDist().inv_cdf(1 - alpha / 2 / max(1, tests))


def _segments(events: pd.DataFrame) -> Iterable[tuple[str, str, pd.DataFrame]]:
    for side_name, side_mask in (
        ("all", events["side"] != 0),
        ("long", events["side"] == 1),
        ("short", events["side"] == -1),
    ):
        for regime_name, regime_mask in (
            ("all", pd.Series(True, index=events.index)),
            (f"adx<{ADX_SPLIT:g}", events["adx_14"] < ADX_SPLIT),
            (f"adx>={ADX_SPLIT:g}", events["adx_14"] >= ADX_SPLIT),
        ):
            yield side_name, regime_name, events[side_mask & regime_mask]


def evaluate(
    prepared: Prepared,
    signal: Signal,
    name: str,
    dev_end: str = "2022-12-31",
    unseal_holdout: bool = False,
    min_events: int = 30,
) -> pd.DataFrame:
    df = prepared.frame
    side = signal(df).reindex(df.index).fillna(0).astype(int)
    cutoff = pd.Timestamp(dev_end, tz="UTC") + pd.Timedelta(days=1)
    periods = [("dev", df.index < cutoff)]
    if unseal_holdout:
        periods.append(("holdout", df.index >= cutoff))

    rows: list[dict[str, object]] = []
    for period_name, period_mask in periods:
        universe = df[period_mask]
        if universe.empty:
            continue
        decided = universe["tb"].isin([1.0, -1.0])
        p_up = float((universe.loc[decided, "tb"] == 1.0).mean()) if decided.any() else float("nan")
        events = universe[side[period_mask] != 0].copy()
        events["side"] = side[period_mask][side[period_mask] != 0]

        for side_name, regime_name, seg in _segments(events):
            for h in prepared.horizons:
                col = f"fwd_{h}"
                valid = seg[seg[col].notna() & seg["cost_atr"].notna()]
                if len(valid) < min_events:
                    continue
                drift = universe.groupby("year")[col].mean()
                signed = (valid["side"] * valid[col]).to_numpy()
                excess = (valid["side"] * (valid[col] - valid["year"].map(drift))).to_numpy()
                net = signed - valid["cost_atr"].to_numpy()
                row: dict[str, object] = {
                    "signal": name,
                    "period": period_name,
                    "side": side_name,
                    "regime": regime_name,
                    "h": h,
                    "n": len(valid),
                    "gross": round(float(signed.mean()), 4),
                    "excess": round(float(excess.mean()), 4),
                    "t_excess": round(cluster_t(excess, valid["week"].to_numpy()), 2),
                    "net": round(float(net.mean()), 4),
                    "t_net": round(cluster_t(net, valid["week"].to_numpy()), 2),
                }
                if h == prepared.horizons[-1] or h == prepared.tb_horizon:
                    row.update(_triple_barrier_stats(valid, prepared.tb_k, p_up))
                rows.append(row)
    return pd.DataFrame(rows)


def _triple_barrier_stats(events: pd.DataFrame, k: float, p_up: float) -> dict[str, object]:
    tb = events[events["tb"].notna()]
    if tb.empty:
        return {}
    hit = tb["tb"] * tb["side"]  # +1 win, -1 loss, 0 timeout
    decided = hit != 0
    pnl = np.where(hit > 0, k, np.where(hit < 0, -k, (tb["side"] * tb["tb_timeout_ret"]).fillna(0.0)))
    pnl = pnl - tb["cost_atr"].to_numpy()
    base = np.where(tb["side"] == 1, p_up, 1 - p_up)
    return {
        "tb_win": round(float((hit[decided] > 0).mean()), 3) if decided.any() else float("nan"),
        "tb_base": round(float(base[decided.to_numpy()].mean()), 3) if decided.any() else float("nan"),
        "tb_timeout": round(float((~decided).mean()), 3),
        "tb_net": round(float(pnl.mean()), 4),
    }
