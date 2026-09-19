from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from indicators.ta_calc import add_indicators
from research.event_study import cluster_t, evaluate, prepare, triple_barrier
from research.signals import SIGNALS


def _random_walk(n: int = 6000, seed: int = 7, drift: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    steps = rng.normal(drift, 1.0, n)
    close = 2000 + np.cumsum(steps)
    open_ = np.concatenate([[2000.0], close[:-1]])
    high = np.maximum(open_, close) + np.abs(rng.normal(0, 0.6, n))
    low = np.minimum(open_, close) - np.abs(rng.normal(0, 0.6, n))
    index = pd.date_range("2015-01-05", periods=n, freq="h", tz="UTC")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "tick_volume": 100, "spread": 30},
        index=index,
    ).round(2)


@pytest.mark.parametrize("name", list(SIGNALS))
def test_signals_do_not_look_ahead(name: str) -> None:
    full = add_indicators(_random_walk(1500))
    cut = 1100
    on_full = SIGNALS[name](full).iloc[:cut]
    on_prefix = SIGNALS[name](add_indicators(_random_walk(1500).iloc[:cut]))
    assert on_full.to_numpy().tolist() == on_prefix.to_numpy().tolist()
    assert set(np.unique(on_full)).issubset({-1, 0, 1})


def test_triple_barrier_hand_built() -> None:
    index = pd.date_range("2020-01-01", periods=5, freq="h", tz="UTC")
    df = pd.DataFrame(
        {
            "open": [100, 100, 100, 100, 100],
            "high": [100, 100.5, 102.5, 100, 100],
            "low": [100, 99.5, 99.5, 100, 100],
            "close": [100, 100, 102, 100, 100],
        },
        index=index,
        dtype=float,
    )
    atr = pd.Series(2.0, index=index)
    outcome, _ = triple_barrier(df, atr, k=1.0, horizon=3)
    # Entry at open[1]=100, barriers 102/98: bar 1 touches neither, bar 2 hits 102.
    assert outcome.iloc[0] == 1.0


def test_harness_finds_planted_edge_and_ignores_noise() -> None:
    raw = _random_walk()
    prepared = prepare(raw, horizons=(1, 3), tb_horizon=3, cost_points=0)
    frame = prepared.frame

    def cheating(df: pd.DataFrame) -> pd.Series:  # knows the next bar: must light up
        nxt = np.sign(df["close"].shift(-1) - df["open"].shift(-1)).fillna(0)
        return nxt.where(np.arange(len(df)) % 7 == 0, 0).astype(int)

    def coin(df: pd.DataFrame) -> pd.Series:
        rng = np.random.default_rng(1)
        return pd.Series(rng.choice([-1, 0, 0, 0, 1], len(df)), index=df.index)

    planted = evaluate(prepared, cheating, "cheat", dev_end="2030-01-01")
    noise = evaluate(prepared, coin, "coin", dev_end="2030-01-01")
    top = planted[(planted["side"] == "all") & (planted["regime"] == "all") & (planted["h"] == 1)].iloc[0]
    flat = noise[(noise["side"] == "all") & (noise["regime"] == "all") & (noise["h"] == 1)].iloc[0]
    assert top["t_excess"] > 10
    assert abs(flat["t_excess"]) < 3.5
    assert len(frame) == len(raw)


def test_drift_is_not_credited_to_a_long_only_signal() -> None:
    prepared = prepare(_random_walk(drift=0.15), horizons=(6,), cost_points=0)

    def always_long(df: pd.DataFrame) -> pd.Series:
        return pd.Series(np.where(np.arange(len(df)) % 5 == 0, 1, 0), index=df.index)

    row = evaluate(prepared, always_long, "long", dev_end="2030-01-01").query("side=='all' and regime=='all'").iloc[0]
    assert row["gross"] > 0.1          # bull market makes it look profitable...
    assert abs(row["t_excess"]) < 3    # ...but there is no information in it


def test_holdout_stays_sealed_by_default() -> None:
    prepared = prepare(_random_walk(), horizons=(1,), cost_points=0)
    table = evaluate(prepared, SIGNALS["bb_reversion"], "bb", dev_end="2015-04-30")
    assert set(table["period"]) == {"dev"}


def test_cluster_t_is_more_conservative_than_naive() -> None:
    rng = np.random.default_rng(0)
    shocks = np.repeat(rng.normal(0.2, 1.0, 60), 20)  # 20 identical events per week
    weeks = np.repeat(np.arange(60), 20)
    naive = shocks.mean() / (shocks.std(ddof=1) / np.sqrt(len(shocks)))
    assert abs(cluster_t(shocks, weeks)) < abs(naive) / 3
