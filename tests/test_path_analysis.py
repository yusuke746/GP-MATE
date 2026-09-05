from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
import pytest

from analysis import path_analysis as pa


def _bars(start: str, ohlc: list[tuple[float, float, float, float]], step_min: int = 5) -> pd.DataFrame:
    t0 = pd.Timestamp(start, tz="UTC")
    rows = []
    for i, (o, h, l, c) in enumerate(ohlc):
        rows.append({"time": t0 + pd.Timedelta(minutes=step_min * i), "open": o, "high": h, "low": l, "close": c})
    return pd.DataFrame(rows)


def _buy_trade(entry: float | None = 100.0, order_type: str = "BUY") -> pa.Trade:
    return pa.Trade(
        order_type=order_type,
        direction="BUY",
        decision_time=pd.Timestamp("2026-08-18T12:00:00Z"),
        sl=97.0,
        tp=106.0,
        exit_time=pd.Timestamp("2026-08-18T12:10:00Z"),
        exit_price=100.0,
        pnl_jpy=-6.0,
        lot=0.02,
        confidence=0.8,
        entry_price=entry,
    )


# Tuesday 12:00 UTC; bars: run to +1.4R, fall back to entry, then run to TP.
BARS = _bars(
    "2026-08-18T12:00:00Z",
    [
        (100.0, 101.0, 99.5, 100.8),
        (100.8, 104.2, 100.5, 103.9),
        (103.9, 104.0, 99.9, 100.2),
        (100.2, 106.5, 100.0, 106.2),
    ],
)


def test_excursions_are_exact_from_bars() -> None:
    trade = _buy_trade()
    assert pa.resolve_entry(trade, BARS)
    mfe, mae = pa.excursions(trade, BARS)
    assert mfe == pytest.approx(4.2)
    assert mae == pytest.approx(0.5)


def test_breakeven_rule_exits_at_entry_while_no_be_reaches_tp() -> None:
    trade = _buy_trade()
    assert pa.resolve_entry(trade, BARS)
    path = pa.path_bars(trade, BARS)

    with_be = pa.simulate(trade, path, pa.Scenario("rule", breakeven_at_r=1.0))
    assert with_be.exit_kind == "BE"
    assert abs(with_be.r - (pa.BREAKEVEN_OFFSET_USD / 3.0)) < 1e-9

    without_be = pa.simulate(trade, path, pa.Scenario("none", breakeven_at_r=None))
    assert without_be.exit_kind == "TP"
    assert without_be.r == 2.0


def test_partial_take_profit_banks_half_then_breakevens() -> None:
    trade = _buy_trade()
    assert pa.resolve_entry(trade, BARS)
    path = pa.path_bars(trade, BARS)
    result = pa.simulate(trade, path, pa.Scenario("half", partial_at_r=1.0, breakeven_at_r=1.0))
    assert result.exit_kind == "BE"
    assert result.banked_r == 0.5
    assert abs(result.r - (0.5 + 0.5 * pa.BREAKEVEN_OFFSET_USD / 3.0)) < 1e-9


def test_tighter_stop_is_hit_when_wider_stop_survives() -> None:
    trade = _buy_trade()
    assert pa.resolve_entry(trade, BARS)
    path = pa.path_bars(trade, BARS)
    # Bar 3 low is 99.9: a stop tightened to 99.95 is hit, the real 97 stop is not.
    tight = pa.simulate(trade, path, pa.Scenario("tight", breakeven_at_r=None, sl_shift_usd=-2.95))
    assert tight.exit_kind == "SL"
    wide = pa.simulate(trade, path, pa.Scenario("wide", breakeven_at_r=None, sl_shift_usd=+1.0))
    assert wide.exit_kind == "TP"


def test_bar_touching_both_levels_scores_as_stop() -> None:
    trade = _buy_trade()
    bars = _bars("2026-08-18T12:00:00Z", [(100.0, 107.0, 96.0, 101.0)])
    assert pa.resolve_entry(trade, bars)
    result = pa.simulate(trade, bars, pa.Scenario("x", breakeven_at_r=None))
    assert result.exit_kind == "SL"
    assert result.r == -1.0


def test_pending_fill_detected_from_bars() -> None:
    trade = pa.Trade(
        order_type="SELL_STOP",
        direction="SELL",
        decision_time=pd.Timestamp("2026-08-18T12:00:00Z"),
        sl=101.0,
        tp=94.0,
        exit_time=pd.Timestamp("2026-08-18T12:15:00Z"),
        exit_price=94.0,
        pnl_jpy=1000.0,
        lot=0.02,
        confidence=0.8,
        entry_price=98.0,
    )
    bars = _bars(
        "2026-08-18T12:00:00Z",
        [(100.0, 100.5, 99.0, 99.2), (99.2, 99.8, 97.9, 98.1), (98.1, 98.3, 93.8, 94.1)],
    )
    assert pa.resolve_entry(trade, bars)
    assert trade.entry_time == pd.Timestamp("2026-08-18T12:05:00Z")
    result = pa.simulate(trade, pa.path_bars(trade, bars), pa.Scenario("x", breakeven_at_r=None))
    assert result.exit_kind == "TP"
    assert abs(result.r - (4.0 / 3.0)) < 1e-9


def test_unfilled_pending_is_skipped() -> None:
    trade = _buy_trade(entry=95.0, order_type="BUY_LIMIT")
    assert not pa.resolve_entry(trade, BARS)


def test_weekend_flat_cutoff_closes_open_scenario() -> None:
    trade = _buy_trade()
    # Friday 2026-08-21 20:00 UTC = 16:00 NY; the third bar is 16:30 NY (flat cutoff).
    bars = _bars(
        "2026-08-21T20:00:00Z",
        [(100.0, 101.0, 99.5, 100.5), (100.5, 101.5, 100.0, 101.0), (101.0, 102.0, 100.5, 101.5)],
        step_min=15,
    )
    trade.decision_time = pd.Timestamp("2026-08-21T20:00:00Z")
    trade.exit_time = pd.Timestamp("2026-08-21T20:30:00Z")
    assert pa.resolve_entry(trade, bars)
    result = pa.simulate(trade, pa.path_bars(trade, bars), pa.Scenario("x", breakeven_at_r=None))
    assert result.exit_kind == "weekend_flat"
    assert result.r == 0.5


def test_load_trades_links_by_position_id_and_sequence(tmp_path: Path) -> None:
    columns = ["timestamp_utc", "position_id", "action", "entry_price", "exit_price", "pnl", "confidence",
               "reasoning", "lot", "sl", "tp", "order_success"]
    rows = [
        # market BUY without position_id -> sequential match
        ["2026-08-18T12:00:00+00:00", "", "BUY", "", "", "", "0.7", "買い", "0.02", "97", "106", "True"],
        ["2026-08-18T14:00:00+00:00", "", "SELL", "0", "106.0", "1200", "", "closed_trade_sync", "0.02", "", "", "True"],
        # pending with position_id -> exact match
        ["2026-08-19T12:00:00+00:00", "555", "BUY_LIMIT", "98.0", "", "", "0.8", "pending_order: x", "0.02", "95", "104", "True"],
        ["2026-08-19T12:00:00+00:00", "", "HOLD", "", "", "", "0.8", "様子見", "0", "0", "0", "True"],
        ["2026-08-19T16:00:00+00:00", "555", "SELL", "0", "95.0", "-600", "", "closed_trade_sync", "0.02", "", "", "True"],
    ]
    log_path = tmp_path / "trade_log.csv"
    with log_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(columns)
        writer.writerows(rows)

    trades = pa.load_trades(log_path)
    assert [t.order_type for t in trades] == ["BUY", "BUY_LIMIT"]
    assert trades[0].entry_price is None
    assert trades[0].exit_price == 106.0
    assert trades[1].entry_price == 98.0
    assert trades[1].pnl_jpy == -600.0


def test_analyze_builds_table_with_scenarios() -> None:
    trade = _buy_trade()
    table = pa.analyze([trade], BARS)
    assert len(table) == 1
    row = table.iloc[0]
    assert row["kind"] == "market"
    assert row["mfe_R"] == 1.4
    assert row["rule_BE@1R_exit"] == "BE"
    assert row["no_BE_exit"] == "TP"
    summary = pa.scenario_summary(table)
    assert set(summary["scenario"]) >= {"actual", "rule_BE@1R", "no_BE"}
