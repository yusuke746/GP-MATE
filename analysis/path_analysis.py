"""Bar-level path analysis of GP-MATE trades.

Reconstructs each closed trade's price path from exported OHLCV bars (see
scripts/export_ohlcv.py) and answers what the 15-minute excursion sampler
cannot: exact MFE/MAE, how each trade would have ended under alternative exit
rules, and how sensitive the stop was to the structural buffer.

Usage:
    python analysis/path_analysis.py logs/trade_log.csv logs/ohlcv_GOLD#_M5.csv

Simulations are spread/slippage-free and resolve intrabar ambiguity
pessimistically (a bar touching both SL and TP counts as SL). Scenarios that
outlive the real trade keep walking the bars until TP/SL, the weekend-flat
cutoff (Friday FRIDAY_FLAT_TIME_NY) or the horizon (default 48h).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import FRIDAY_FLAT_TIME_NY  # noqa: E402

MARKET_TZ = ZoneInfo("America/New_York")
DECISION_ACTIONS = ("BUY", "SELL", "BUY_LIMIT", "BUY_STOP", "SELL_LIMIT", "SELL_STOP")
BREAKEVEN_OFFSET_USD = 0.1
DEFAULT_HORIZON_HOURS = 48.0
NY_OPEN_WINDOW = ((8, 0), (9, 45))


@dataclass
class Trade:
    order_type: str
    direction: str
    decision_time: pd.Timestamp
    sl: float
    tp: float
    exit_time: pd.Timestamp
    exit_price: float
    pnl_jpy: float
    lot: float
    confidence: float
    entry_price: float | None = None
    entry_time: pd.Timestamp | None = None

    @property
    def kind(self) -> str:
        return "market" if self.order_type in {"BUY", "SELL"} else "pending"

    @property
    def risk_usd(self) -> float:
        return abs(float(self.entry_price or 0.0) - self.sl)


@dataclass
class SimResult:
    r: float
    exit_kind: str
    exit_time: pd.Timestamp | None
    banked_r: float = 0.0


@dataclass(frozen=True)
class Scenario:
    name: str
    breakeven_at_r: float | None = 1.0
    partial_at_r: float | None = None
    sl_shift_usd: float = 0.0
    time_stop_hours: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)


DEFAULT_SCENARIOS: tuple[Scenario, ...] = (
    Scenario("rule_BE@1R"),
    Scenario("no_BE", breakeven_at_r=None),
    Scenario("BE@1.5R", breakeven_at_r=1.5),
    Scenario("half@1R+BE", partial_at_r=1.0),
    Scenario("SL_buffer-2", sl_shift_usd=-2.0),
    Scenario("SL_buffer+2", sl_shift_usd=2.0),
    Scenario("time_stop_8h", time_stop_hours=8.0),
)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_bars(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    time_col = "time_utc" if "time_utc" in df.columns else "time"
    out = pd.DataFrame(
        {
            "time": pd.to_datetime(df[time_col], utc=True, errors="coerce", format="ISO8601"),
            "open": pd.to_numeric(df["open"], errors="coerce"),
            "high": pd.to_numeric(df["high"], errors="coerce"),
            "low": pd.to_numeric(df["low"], errors="coerce"),
            "close": pd.to_numeric(df["close"], errors="coerce"),
        }
    ).dropna()
    return out.sort_values("time").reset_index(drop=True)


def _num(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return default if pd.isna(parsed) else parsed


def load_trades(trade_log_path: str | Path) -> list[Trade]:
    """Pair executed entry decisions with closed_trade_sync rows.

    Same matching rules as analysis.performance.link_decisions_to_outcomes:
    exact position_id join first, then latest-earlier-unmatched sequential
    fallback (valid because MAX_POSITIONS=1 keeps trades sequential).
    """
    df = pd.read_csv(trade_log_path, dtype=str, keep_default_na=False)
    df["_ts"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce", format="ISO8601")
    action = df["action"].str.upper().str.strip()
    reasoning = df["reasoning"].str.strip()
    success = df["order_success"].str.lower() == "true"

    decisions = df[action.isin(DECISION_ACTIONS) & success & (reasoning != "closed_trade_sync")].sort_values("_ts")
    outcomes = df[(reasoning == "closed_trade_sync") & (pd.to_numeric(df["pnl"], errors="coerce").notna())].sort_values("_ts")

    matched_outcome: dict[Any, Any] = {}
    used: set[Any] = set()
    for idx, row in decisions.iterrows():
        pid = str(row.get("position_id", "")).strip()
        if not pid:
            continue
        hits = outcomes[(outcomes["position_id"].str.strip() == pid) & (~outcomes.index.isin(used))]
        if hits.empty:
            continue
        matched_outcome[idx] = hits.index[0]
        used.add(hits.index[0])
    for out_idx, out_row in outcomes.iterrows():
        if out_idx in used:
            continue
        candidates = decisions[(~decisions.index.isin(matched_outcome)) & (decisions["_ts"] <= out_row["_ts"])]
        if candidates.empty:
            continue
        matched_outcome[candidates.index[-1]] = out_idx
        used.add(out_idx)

    trades: list[Trade] = []
    for dec_idx, out_idx in matched_outcome.items():
        dec = decisions.loc[dec_idx]
        out = outcomes.loc[out_idx]
        order_type = str(dec["action"]).upper().strip()
        entry_price = _num(dec.get("entry_price"))
        trades.append(
            Trade(
                order_type=order_type,
                direction="BUY" if order_type.startswith("BUY") else "SELL",
                decision_time=dec["_ts"],
                sl=_num(dec.get("sl")),
                tp=_num(dec.get("tp")),
                exit_time=out["_ts"],
                exit_price=_num(out.get("exit_price")),
                pnl_jpy=_num(out.get("pnl")),
                lot=_num(dec.get("lot")),
                confidence=_num(dec.get("confidence")),
                entry_price=entry_price if entry_price > 0 else None,
            )
        )
    trades.sort(key=lambda t: t.decision_time)
    return trades


# --------------------------------------------------------------------------- #
# Path reconstruction
# --------------------------------------------------------------------------- #
def resolve_entry(trade: Trade, bars: pd.DataFrame) -> bool:
    """Fill in entry_time/entry_price from the bars. Returns False if the
    pending order never filled before the recorded exit."""
    window = bars[(bars["time"] >= trade.decision_time) & (bars["time"] <= trade.exit_time)]
    if window.empty:
        return False
    if trade.kind == "market":
        trade.entry_time = window.iloc[0]["time"]
        if trade.entry_price is None:
            trade.entry_price = float(window.iloc[0]["open"])
        return True

    price = float(trade.entry_price or 0.0)
    if price <= 0:
        return False
    if trade.order_type in {"BUY_LIMIT", "SELL_STOP"}:
        hit = window[window["low"] <= price]
    else:
        hit = window[window["high"] >= price]
    if hit.empty:
        return False
    trade.entry_time = hit.iloc[0]["time"]
    return True


def path_bars(trade: Trade, bars: pd.DataFrame, horizon_hours: float = DEFAULT_HORIZON_HOURS) -> pd.DataFrame:
    assert trade.entry_time is not None
    end = trade.entry_time + timedelta(hours=horizon_hours)
    return bars[(bars["time"] >= trade.entry_time) & (bars["time"] <= end)].reset_index(drop=True)


def excursions(trade: Trade, bars: pd.DataFrame) -> tuple[float, float]:
    """Exact (MFE, MAE) in USD between entry and the recorded exit."""
    assert trade.entry_time is not None and trade.entry_price is not None
    window = bars[(bars["time"] >= trade.entry_time) & (bars["time"] <= trade.exit_time)]
    if window.empty:
        return 0.0, 0.0
    if trade.direction == "BUY":
        return float(window["high"].max() - trade.entry_price), float(trade.entry_price - window["low"].min())
    return float(trade.entry_price - window["low"].min()), float(window["high"].max() - trade.entry_price)


def _is_past_weekend_flat(ts: pd.Timestamp) -> bool:
    local = ts.tz_convert(MARKET_TZ)
    hour, minute = FRIDAY_FLAT_TIME_NY
    return local.weekday() == 4 and (local.hour, local.minute) >= (hour, minute)


def simulate(trade: Trade, bars: pd.DataFrame, scenario: Scenario) -> SimResult:
    """Walk the bars under the scenario's exit rules and return the result in R."""
    assert trade.entry_time is not None and trade.entry_price is not None
    entry = float(trade.entry_price)
    sign = 1.0 if trade.direction == "BUY" else -1.0
    risk = trade.risk_usd
    if risk <= 0 or bars.empty:
        return SimResult(r=0.0, exit_kind="invalid", exit_time=None)

    sl = trade.sl - sign * scenario.sl_shift_usd
    tp = trade.tp
    size = 1.0
    banked = 0.0
    be_done = False
    partial_done = False
    time_limit = trade.entry_time + timedelta(hours=scenario.time_stop_hours) if scenario.time_stop_hours else None

    def favorable(price: float) -> float:
        return sign * (price - entry)

    def finish(price: float, kind: str, ts: pd.Timestamp) -> SimResult:
        return SimResult(r=banked + size * favorable(price) / risk, exit_kind=kind, exit_time=ts, banked_r=banked)

    for _, bar in bars.iterrows():
        ts = bar["time"]
        high, low, close = float(bar["high"]), float(bar["low"]), float(bar["close"])
        best = high if sign > 0 else low
        worst = low if sign > 0 else high

        if _is_past_weekend_flat(ts):
            return finish(close, "weekend_flat", ts)
        if time_limit is not None and ts >= time_limit:
            return finish(close, "time_stop", ts)

        # Stop first: a bar spanning both levels is scored as a loss.
        if sign * (worst - sl) <= 0:
            kind = "BE" if be_done and abs(sl - entry) <= BREAKEVEN_OFFSET_USD + 1e-9 else "SL"
            return finish(sl, kind, ts)
        if sign * (best - tp) >= 0:
            return finish(tp, "TP", ts)

        # Management rules take effect from the next bar (the live monitor
        # also acts after the move, never inside the bar).
        if scenario.partial_at_r is not None and not partial_done and favorable(best) >= scenario.partial_at_r * risk:
            banked += 0.5 * scenario.partial_at_r
            size = 0.5
            partial_done = True
        if scenario.breakeven_at_r is not None and not be_done and favorable(best) >= scenario.breakeven_at_r * risk:
            sl = entry + sign * BREAKEVEN_OFFSET_USD
            be_done = True

    last = bars.iloc[-1]
    return finish(float(last["close"]), "horizon", last["time"])


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _slot_label(ts: pd.Timestamp) -> str:
    local = ts.tz_convert(MARKET_TZ)
    return f"{local.hour:02d}:{'30' if local.minute >= 30 else '00'}"


def _in_ny_open_window(ts: pd.Timestamp) -> bool:
    local = ts.tz_convert(MARKET_TZ)
    (h0, m0), (h1, m1) = NY_OPEN_WINDOW
    return (h0, m0) <= (local.hour, local.minute) <= (h1, m1)


def analyze(trades: list[Trade], bars: pd.DataFrame, scenarios: tuple[Scenario, ...] = DEFAULT_SCENARIOS,
            horizon_hours: float = DEFAULT_HORIZON_HOURS) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for trade in trades:
        if not resolve_entry(trade, bars):
            continue
        assert trade.entry_time is not None and trade.entry_price is not None
        risk = trade.risk_usd
        if risk <= 0:
            continue
        mfe, mae = excursions(trade, bars)
        sign = 1.0 if trade.direction == "BUY" else -1.0
        actual_r = sign * (trade.exit_price - trade.entry_price) / risk if trade.exit_price > 0 else float("nan")
        row: dict[str, Any] = {
            "decision_ny": trade.decision_time.tz_convert(MARKET_TZ).strftime("%m-%d %H:%M"),
            "slot": _slot_label(trade.decision_time),
            "kind": trade.kind,
            "dir": trade.direction,
            "fill_delay_h": round((trade.entry_time - trade.decision_time).total_seconds() / 3600, 1),
            "hold_h": round((trade.exit_time - trade.entry_time).total_seconds() / 3600, 1),
            "risk_usd": round(risk, 2),
            "plan_rr": round(abs(trade.tp - trade.entry_price) / risk, 2),
            "actual_R": round(actual_r, 2),
            "mfe_R": round(mfe / risk, 2),
            "mae_R": round(mae / risk, 2),
            "exit_in_ny_open": _in_ny_open_window(trade.exit_time),
            "pnl_jpy": trade.pnl_jpy,
        }
        path = path_bars(trade, bars, horizon_hours)
        for scenario in scenarios:
            result = simulate(trade, path, scenario)
            row[f"{scenario.name}_R"] = round(result.r, 2)
            row[f"{scenario.name}_exit"] = result.exit_kind
        rows.append(row)
    return pd.DataFrame(rows)


def scenario_summary(table: pd.DataFrame, scenarios: tuple[Scenario, ...] = DEFAULT_SCENARIOS) -> pd.DataFrame:
    if table.empty:
        return pd.DataFrame()
    rows = [{"scenario": "actual", "sum_R": table["actual_R"].sum(), "avg_R": table["actual_R"].mean(),
             "win_rate": (table["actual_R"] > 0).mean(), "exits": "-"}]
    for scenario in scenarios:
        r = table[f"{scenario.name}_R"]
        exits = table[f"{scenario.name}_exit"].value_counts().to_dict()
        rows.append({"scenario": scenario.name, "sum_R": r.sum(), "avg_R": r.mean(), "win_rate": (r > 0).mean(),
                     "exits": " ".join(f"{k}:{v}" for k, v in sorted(exits.items()))})
    out = pd.DataFrame(rows)
    out["sum_R"] = out["sum_R"].round(2)
    out["avg_R"] = out["avg_R"].round(3)
    out["win_rate"] = (out["win_rate"] * 100).round(1)
    return out


def print_report(trade_log_path: str, bars_path: str, horizon_hours: float = DEFAULT_HORIZON_HOURS) -> None:
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 60)
    bars = load_bars(bars_path)
    trades = load_trades(trade_log_path)
    print("=== GP-MATE Path Analysis ===")
    print(f"bars: {len(bars)} ({bars['time'].min()} .. {bars['time'].max()})" if not bars.empty else "bars: 0")
    print(f"linked trades: {len(trades)}")
    table = analyze(trades, bars, horizon_hours=horizon_hours)
    if table.empty:
        print("no trades could be reconstructed from the bars (check the date range).")
        return
    print(f"reconstructed: {len(table)}\n")

    base_cols = ["decision_ny", "slot", "kind", "dir", "fill_delay_h", "hold_h", "risk_usd", "plan_rr",
                 "actual_R", "mfe_R", "mae_R", "exit_in_ny_open", "pnl_jpy"]
    print("[Per-trade path]")
    print(table[base_cols].to_string(index=False))

    print("\n[Exit-rule what-if, in R]")
    print(scenario_summary(table).to_string(index=False))

    losses = table[table["actual_R"] < 0]
    wins = table[table["actual_R"] > 0]
    print("\n[Excursions]")
    print(f"losses: {len(losses)}  with MFE>=0.5R: {(losses['mfe_R'] >= 0.5).sum()}  MFE>=1.0R: {(losses['mfe_R'] >= 1.0).sum()}")
    print(f"wins:   {len(wins)}  with MAE>=0.5R: {(wins['mae_R'] >= 0.5).sum()}  MAE>=0.8R: {(wins['mae_R'] >= 0.8).sum()}")
    print(f"exits inside NY open window 08:00-09:45: {int(table['exit_in_ny_open'].sum())} "
          f"(losses: {int(losses['exit_in_ny_open'].sum())})")

    print("\n[By judgment slot]")
    slot = table.groupby("slot").agg(n=("actual_R", "size"), sum_R=("actual_R", "sum"),
                                     win_rate=("actual_R", lambda s: round((s > 0).mean() * 100, 1)),
                                     avg_mae_R=("mae_R", "mean"), avg_mfe_R=("mfe_R", "mean")).round(2)
    print(slot.to_string())


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: python analysis/path_analysis.py <trade_log.csv> <ohlcv.csv> [horizon_hours]")
        raise SystemExit(2)
    horizon = float(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_HORIZON_HOURS
    print_report(sys.argv[1], sys.argv[2], horizon)
