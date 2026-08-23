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


from analysis.performance import (
    confidence_band_summary,
    link_decisions_to_outcomes,
    load_raw_log,
    threshold_blocked_summary,
)


def _write_raw_log(path: Path, rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def test_link_decisions_by_position_id(tmp_path: Path) -> None:
    csv_path = tmp_path / "trade_log.csv"
    _write_raw_log(
        csv_path,
        [
            {
                "timestamp_utc": "2026-07-01T09:00:00Z",
                "action": "BUY",
                "confidence": 0.72,
                "reasoning": "entry",
                "order_success": True,
                "position_id": "111",
                "pnl": "",
            },
            {
                "timestamp_utc": "2026-07-01T15:00:00Z",
                "action": "SELL",
                "confidence": "",
                "reasoning": "closed_trade_sync",
                "order_success": True,
                "position_id": "111",
                "pnl": 120.0,
            },
        ],
    )

    linked = link_decisions_to_outcomes(load_raw_log(str(csv_path)))

    assert len(linked) == 1
    assert linked.iloc[0]["matched_by"] == "position_id"
    assert linked.iloc[0]["realized_pnl"] == 120.0
    assert linked.iloc[0]["confidence"] == 0.72


def test_link_decisions_sequential_fallback(tmp_path: Path) -> None:
    csv_path = tmp_path / "trade_log.csv"
    _write_raw_log(
        csv_path,
        [
            {
                "timestamp_utc": "2026-07-01T09:00:00Z",
                "action": "BUY",
                "confidence": 0.65,
                "reasoning": "entry",
                "order_success": True,
                "position_id": "",
                "pnl": "",
            },
            {
                "timestamp_utc": "2026-07-01T12:00:00Z",
                "action": "SELL",
                "confidence": "",
                "reasoning": "closed_trade_sync",
                "order_success": True,
                "position_id": "",
                "pnl": -80.0,
            },
            {
                "timestamp_utc": "2026-07-02T09:00:00Z",
                "action": "SELL",
                "confidence": 0.8,
                "reasoning": "entry",
                "order_success": True,
                "position_id": "",
                "pnl": "",
            },
        ],
    )

    linked = link_decisions_to_outcomes(load_raw_log(str(csv_path)))

    assert len(linked) == 2
    first = linked.iloc[0]
    second = linked.iloc[1]
    assert first["matched_by"] == "sequence"
    assert first["realized_pnl"] == -80.0
    # Second decision is still open: no realized PnL.
    assert second["matched_by"] == ""
    assert pd.isna(second["realized_pnl"])


def test_link_ignores_failed_orders_and_hold_rows(tmp_path: Path) -> None:
    csv_path = tmp_path / "trade_log.csv"
    _write_raw_log(
        csv_path,
        [
            {
                "timestamp_utc": "2026-07-01T09:00:00Z",
                "action": "HOLD",
                "confidence": 0.5,
                "reasoning": "range",
                "order_success": False,
                "position_id": "",
                "pnl": "",
            },
            {
                "timestamp_utc": "2026-07-01T10:00:00Z",
                "action": "BUY",
                "confidence": 0.75,
                "reasoning": "entry",
                "order_success": False,
                "position_id": "",
                "pnl": "",
            },
        ],
    )

    linked = link_decisions_to_outcomes(load_raw_log(str(csv_path)))

    assert linked.empty


def test_confidence_band_summary_groups_results() -> None:
    linked = pd.DataFrame(
        {
            "timestamp_utc": pd.to_datetime(
                ["2026-07-01T09:00:00Z", "2026-07-02T09:00:00Z", "2026-07-03T09:00:00Z"],
                utc=True,
            ),
            "action": ["BUY", "SELL", "BUY"],
            "confidence": [0.62, 0.72, 0.73],
            "realized_pnl": [100.0, -50.0, 80.0],
            "matched_by": ["position_id", "position_id", "sequence"],
        }
    )

    bands = confidence_band_summary(linked)

    band_060 = bands[bands["band"] == "[0.60-0.65)"].iloc[0]
    band_070 = bands[bands["band"] == "[0.70-0.75)"].iloc[0]
    assert band_060["settled"] == 1
    assert band_060["win_rate"] == 100.0
    assert band_070["settled"] == 2
    assert band_070["win_rate"] == 50.0
    assert abs(band_070["profit_factor"] - (80.0 / 50.0)) < 1e-9


def test_threshold_blocked_summary_counts_biased_holds(tmp_path: Path) -> None:
    csv_path = tmp_path / "trade_log.csv"
    _write_raw_log(
        csv_path,
        [
            {
                "timestamp_utc": "2026-07-01T09:00:00Z",
                "action": "HOLD",
                "confidence": 0.62,
                "reasoning": "threshold hold",
                "directional_bias": "BEARISH",
                "order_success": False,
                "pnl": "",
            },
            {
                "timestamp_utc": "2026-07-01T10:00:00Z",
                "action": "HOLD",
                "confidence": 0.62,
                "reasoning": "range",
                "directional_bias": "NEUTRAL",
                "order_success": False,
                "pnl": "",
            },
            {
                "timestamp_utc": "2026-07-01T11:00:00Z",
                "action": "HOLD",
                "confidence": "",
                "reasoning": "breakeven_monitor",
                "directional_bias": "",
                "order_success": False,
                "pnl": 30.0,
            },
        ],
    )

    blocked = threshold_blocked_summary(load_raw_log(str(csv_path)))

    band_060 = blocked[blocked["band"] == "[0.60-0.65)"].iloc[0]
    assert band_060["holds"] == 2
    assert band_060["with_bias"] == 1
    assert blocked["holds"].sum() == 2


def test_slot_summary_groups_by_ny_judgment_slot() -> None:
    from analysis.performance import slot_summary

    linked = pd.DataFrame(
        {
            # 12:00 UTC = 08:00 NY (EDT), 14:30 UTC = 10:30 NY.
            "timestamp_utc": pd.to_datetime(
                ["2026-08-06T12:00:00Z", "2026-08-13T12:00:00Z", "2026-08-06T14:30:00Z"],
                utc=True,
            ),
            "action": ["BUY", "BUY", "BUY"],
            "confidence": [0.74, 0.7, 0.74],
            "realized_pnl": [-8468.0, 5000.0, -10773.0],
            "matched_by": ["position_id"] * 3,
        }
    )

    slots = slot_summary(linked)

    row_0800 = slots[slots["slot_ny"] == "08:00"].iloc[0]
    row_1030 = slots[slots["slot_ny"] == "10:30"].iloc[0]
    assert row_0800["decisions"] == 2
    assert row_0800["wins"] == 1
    assert row_0800["total_pnl"] == -3468.0
    assert row_1030["decisions"] == 1
    assert row_1030["win_rate"] == 0.0
