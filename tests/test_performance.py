from __future__ import annotations

from pathlib import Path

import pandas as pd

from analysis.performance import calc_metrics, daily_summary, load_trades


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


def test_metrics_pf_winrate_maxdd() -> None:
    df = pd.DataFrame(
        {
            "timestamp_utc": pd.to_datetime(
                [
                    "2026-07-01T00:00:00Z",
                    "2026-07-01T01:00:00Z",
                    "2026-07-01T02:00:00Z",
                    "2026-07-01T03:00:00Z",
                ],
                utc=True,
            ),
            "action": ["BUY", "SELL", "BUY", "SELL"],
            "pnl": [100.0, -50.0, -50.0, 200.0],
        }
    )
    df.attrs["hold_count"] = 2

    metrics = calc_metrics(df)

    assert metrics["total_trades"] == 4
    assert metrics["win_rate"] == 50.0
    assert metrics["profit_factor"] == 3.0
    assert round(metrics["max_drawdown_pct"], 2) == 100.0
    assert metrics["hold_count"] == 2


def test_zero_rows_safe() -> None:
    df = pd.DataFrame(columns=["timestamp_utc", "action", "pnl"])
    df.attrs["hold_count"] = 0

    metrics = calc_metrics(df)
    summary = daily_summary(df)

    assert metrics["total_trades"] == 0
    assert metrics["profit_factor"] is None
    assert summary.empty


def test_profit_factor_infinite_when_no_loss() -> None:
    df = pd.DataFrame(
        {
            "timestamp_utc": pd.to_datetime(["2026-07-01T00:00:00Z", "2026-07-01T01:00:00Z"], utc=True),
            "action": ["BUY", "SELL"],
            "pnl": [100.0, 50.0],
        }
    )
    metrics = calc_metrics(df)
    assert metrics["profit_factor"] == float("inf")


def test_load_trades_filters_settled_and_counts_hold(tmp_path: Path) -> None:
    csv_path = tmp_path / "trade_log.csv"
    _write_csv(
        csv_path,
        [
            {
                "timestamp_utc": "2026-07-01T00:00:00Z",
                "action": "HOLD",
                "pnl": "",
            },
            {
                "timestamp_utc": "2026-07-01T01:00:00Z",
                "action": "BUY",
                "pnl": 100,
            },
            {
                "timestamp_utc": "2026-07-01T02:00:00Z",
                "action": "SELL",
                "pnl": -40,
            },
            {
                "timestamp_utc": "2026-07-01T03:00:00Z",
                "action": "BUY",
                "pnl": "",
            },
        ],
    )

    trades = load_trades(str(csv_path))

    assert len(trades) == 2
    assert trades.attrs.get("hold_count") == 1
    assert set(trades["action"].tolist()) == {"BUY", "SELL"}
