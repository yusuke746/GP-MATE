from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd

PF_STAGE_THRESHOLD = 1.3
MAX_DD_STAGE_THRESHOLD = 20.0
MIN_TRADES_STAGE_THRESHOLD = 30

# Confidence-band edges for threshold calibration reporting.
CONFIDENCE_BAND_EDGES: tuple[float, ...] = (0.0, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 1.0)

PNL_COLUMN_CANDIDATES: tuple[str, ...] = (
    "pnl",
    "realized_pnl",
    "profit",
    "net_pnl",
)


def _safe_to_datetime(series: pd.Series) -> pd.Series:
    # format="ISO8601" accepts mixed ISO variants (with/without microseconds,
    # 'Z' suffix); without it pandas infers the format from the first row and
    # coerces every differently-formatted row to NaT.
    try:
        return pd.to_datetime(series, errors="coerce", utc=True, format="ISO8601")
    except (TypeError, ValueError):
        return pd.to_datetime(series, errors="coerce", utc=True)


def _resolve_pnl_column(df: pd.DataFrame) -> str | None:
    for column in PNL_COLUMN_CANDIDATES:
        if column in df.columns:
            return column
    return None


def _extract_hold_count(raw_df: pd.DataFrame) -> int:
    if raw_df.empty or "action" not in raw_df.columns:
        return 0
    action = raw_df["action"].astype(str).str.upper().str.strip()
    return int((action == "HOLD").sum())


def load_trades(csv_path: str) -> pd.DataFrame:
    """Load trade log and return only settled BUY/SELL rows.

    HOLD rows are excluded from the returned frame. Their count is stored in
    df.attrs["hold_count"] for downstream reporting.
    """
    path = Path(csv_path)
    if not path.exists():
        empty = pd.DataFrame(columns=["timestamp_utc", "action", "pnl"])
        empty.attrs["hold_count"] = 0
        return empty

    raw_df = pd.read_csv(path)
    if raw_df.empty:
        raw_df = pd.DataFrame(columns=["timestamp_utc", "action", "pnl"])

    hold_count = _extract_hold_count(raw_df)

    if "action" not in raw_df.columns:
        empty = pd.DataFrame(columns=["timestamp_utc", "action", "pnl"])
        empty.attrs["hold_count"] = hold_count
        return empty

    pnl_col = _resolve_pnl_column(raw_df)
    if pnl_col is None:
        # Existing schema may not have realized PnL yet. Return empty safely.
        empty = pd.DataFrame(columns=["timestamp_utc", "action", "pnl"])
        empty.attrs["hold_count"] = hold_count
        return empty

    df = raw_df.copy()
    df["action"] = df["action"].astype(str).str.upper().str.strip()
    df["pnl"] = pd.to_numeric(df[pnl_col], errors="coerce")

    if "timestamp_utc" in df.columns:
        df["timestamp_utc"] = _safe_to_datetime(df["timestamp_utc"])
    else:
        df["timestamp_utc"] = pd.NaT

    settled = df[df["action"].isin(["BUY", "SELL"]) & df["pnl"].notna()].copy()
    settled = settled.sort_values("timestamp_utc", kind="stable")
    settled.attrs["hold_count"] = hold_count
    return settled


def calc_metrics(df: pd.DataFrame) -> dict[str, Any]:
    """Calculate performance metrics from settled trades DataFrame."""
    hold_count = int(df.attrs.get("hold_count", 0))

    if df.empty:
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "profit_factor": None,
            "total_pnl": 0.0,
            "max_drawdown_pct": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "consecutive_loss": 0,
            "hold_count": hold_count,
        }

    pnl = pd.to_numeric(df.get("pnl", pd.Series([], dtype=float)), errors="coerce").fillna(0.0)

    total_trades = int(len(pnl))
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]

    win_rate = (len(wins) / total_trades * 100.0) if total_trades > 0 else 0.0
    gross_profit = float(wins.sum())
    gross_loss = float(losses.sum())
    total_pnl = float(pnl.sum())

    if gross_loss < 0:
        profit_factor: float | None = gross_profit / abs(gross_loss)
    elif gross_profit > 0:
        profit_factor = float("inf")
    else:
        profit_factor = None

    cumulative_pnl = pnl.cumsum()
    running_peak = cumulative_pnl.cummax()
    drawdown = running_peak - cumulative_pnl
    drawdown_pct = (drawdown / running_peak.abs().replace(0.0, pd.NA)) * 100.0
    max_drawdown_pct = float(drawdown_pct.fillna(0.0).max()) if not drawdown_pct.empty else 0.0

    avg_win = float(wins.mean()) if not wins.empty else 0.0
    avg_loss = float(losses.mean()) if not losses.empty else 0.0

    max_consecutive_loss = 0
    current = 0
    for value in pnl:
        if value < 0:
            current += 1
            if current > max_consecutive_loss:
                max_consecutive_loss = current
        else:
            current = 0

    return {
        "total_trades": total_trades,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "total_pnl": total_pnl,
        "max_drawdown_pct": max_drawdown_pct,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "consecutive_loss": max_consecutive_loss,
        "hold_count": hold_count,
    }


def daily_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return daily PnL/trade-count/win-rate summary."""
    if df.empty:
        return pd.DataFrame(columns=["date", "daily_pnl", "trades", "win_rate"])  # pragma: no cover

    work = df.copy()
    if "timestamp_utc" not in work.columns:
        work["timestamp_utc"] = pd.NaT
    if "pnl" not in work.columns:
        work["pnl"] = 0.0

    work["timestamp_utc"] = _safe_to_datetime(work["timestamp_utc"])
    work = work[work["timestamp_utc"].notna()].copy()
    if work.empty:
        return pd.DataFrame(columns=["date", "daily_pnl", "trades", "win_rate"])

    work["date"] = work["timestamp_utc"].dt.date
    work["is_win"] = pd.to_numeric(work["pnl"], errors="coerce").fillna(0.0) > 0

    grouped = work.groupby("date", as_index=False).agg(
        daily_pnl=("pnl", "sum"),
        trades=("pnl", "count"),
        wins=("is_win", "sum"),
    )
    grouped["win_rate"] = grouped.apply(
        lambda row: (float(row["wins"]) / float(row["trades"]) * 100.0) if float(row["trades"]) > 0 else 0.0,
        axis=1,
    )

    return grouped[["date", "daily_pnl", "trades", "win_rate"]]


def load_raw_log(csv_path: str) -> pd.DataFrame:
    """Load the full trade log without filtering (all row types)."""
    path = Path(csv_path)
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, dtype=str).fillna("")
    if "timestamp_utc" in df.columns:
        df["_ts"] = _safe_to_datetime(df["timestamp_utc"])
    else:
        df["_ts"] = pd.NaT
    return df


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    if name in df.columns:
        return df[name].astype(str)
    return pd.Series([""] * len(df), index=df.index, dtype=str)


def link_decisions_to_outcomes(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Join executed entry decisions to their realized (closed) PnL.

    Matching:
    1. Exact join on position_id when both rows carry it.
    2. Fallback: each closed row is matched to the latest earlier unmatched
       entry decision (valid because MAX_POSITIONS=1 keeps trades sequential).

    Returns a DataFrame with one row per executed decision:
    timestamp_utc, action, confidence, realized_pnl (NaN if still open /
    unmatched), matched_by ('position_id' | 'sequence' | '').
    """
    if raw_df.empty:
        return pd.DataFrame(
            columns=["timestamp_utc", "action", "confidence", "realized_pnl", "matched_by"]
        )

    action = _col(raw_df, "action").str.upper().str.strip()
    reasoning = _col(raw_df, "reasoning").str.strip()
    order_success = _col(raw_df, "order_success").str.lower() == "true"

    # Pending orders that were successfully placed count as decisions too;
    # unfilled ones simply stay in the open_or_unmatched column.
    decision_actions = ["BUY", "SELL", "BUY_LIMIT", "BUY_STOP", "SELL_LIMIT", "SELL_STOP"]
    decisions = raw_df[
        action.isin(decision_actions) & order_success & (reasoning != "closed_trade_sync")
    ].copy()
    outcomes = raw_df[
        (reasoning == "closed_trade_sync")
        & (pd.to_numeric(_col(raw_df, "pnl"), errors="coerce").notna())
    ].copy()

    decisions["_confidence"] = pd.to_numeric(_col(decisions, "confidence"), errors="coerce")
    decisions["_position_id"] = _col(decisions, "position_id").str.strip()
    decisions["_realized_pnl"] = float("nan")
    decisions["_matched_by"] = ""

    outcomes["_position_id"] = _col(outcomes, "position_id").str.strip()
    outcomes["_pnl"] = pd.to_numeric(_col(outcomes, "pnl"), errors="coerce")
    outcomes = outcomes.sort_values("_ts", kind="stable")

    matched_outcome_indices: set[Any] = set()

    # Pass 1: exact position_id join. A position may close via multiple deals
    # (partial closes); sum the realized PnL over all matching closed rows.
    for idx, decision in decisions.iterrows():
        pos_id = str(decision["_position_id"] or "")
        if not pos_id:
            continue
        hits = outcomes[(outcomes["_position_id"] == pos_id) & (~outcomes.index.isin(matched_outcome_indices))]
        if hits.empty:
            continue
        decisions.at[idx, "_realized_pnl"] = float(hits["_pnl"].sum())
        decisions.at[idx, "_matched_by"] = "position_id"
        matched_outcome_indices.update(hits.index.tolist())

    # Pass 2: sequential fallback for rows without a usable position_id.
    unmatched_decisions = decisions[decisions["_matched_by"] == ""].sort_values("_ts", kind="stable")
    for out_idx, outcome in outcomes.iterrows():
        if out_idx in matched_outcome_indices:
            continue
        candidates = unmatched_decisions[
            (unmatched_decisions["_matched_by"] == "")
            & (unmatched_decisions["_ts"].notna())
            & (unmatched_decisions["_ts"] <= outcome["_ts"])
        ]
        if candidates.empty:
            continue
        target_idx = candidates.index[-1]
        decisions.at[target_idx, "_realized_pnl"] = float(outcome["_pnl"])
        decisions.at[target_idx, "_matched_by"] = "sequence"
        unmatched_decisions.at[target_idx, "_matched_by"] = "sequence"
        matched_outcome_indices.add(out_idx)

    linked = pd.DataFrame(
        {
            "timestamp_utc": decisions["_ts"],
            "action": _col(decisions, "action").str.upper().str.strip(),
            "confidence": decisions["_confidence"],
            "realized_pnl": decisions["_realized_pnl"],
            "matched_by": decisions["_matched_by"],
        }
    ).sort_values("timestamp_utc", kind="stable")
    return linked.reset_index(drop=True)


def slot_summary(linked: pd.DataFrame) -> pd.DataFrame:
    """Aggregate realized results per judgment slot (NY time).

    Rows are labeled by the decision timestamp rounded down to the half hour
    in America/New_York, matching the judgment schedule (03:00/08:00/09:30/10:30).
    """
    columns = ["slot_ny", "decisions", "settled", "wins", "win_rate", "total_pnl", "profit_factor"]
    if linked.empty:
        return pd.DataFrame(columns=columns)

    work = linked.copy()
    work = work[work["timestamp_utc"].notna()]
    if work.empty:
        return pd.DataFrame(columns=columns)

    ny_times = work["timestamp_utc"].dt.tz_convert("America/New_York")
    work["slot_ny"] = ny_times.dt.strftime("%H:") + ny_times.dt.minute.map(
        lambda m: "30" if m >= 30 else "00"
    )

    rows: list[dict[str, Any]] = []
    for slot, slot_df in work.groupby("slot_ny"):
        settled = slot_df[slot_df["realized_pnl"].notna()]
        pnl = settled["realized_pnl"]
        wins = pnl[pnl > 0]
        losses = pnl[pnl < 0]
        if float(losses.sum()) < 0:
            pf: float | None = float(wins.sum()) / abs(float(losses.sum()))
        elif float(wins.sum()) > 0:
            pf = float("inf")
        else:
            pf = None
        rows.append(
            {
                "slot_ny": slot,
                "decisions": int(len(slot_df)),
                "settled": int(len(settled)),
                "wins": int(len(wins)),
                "win_rate": (len(wins) / len(settled) * 100.0) if len(settled) else 0.0,
                "total_pnl": float(pnl.sum()) if len(settled) else 0.0,
                "profit_factor": pf,
            }
        )
    return pd.DataFrame(rows, columns=columns).sort_values("slot_ny").reset_index(drop=True)


def _band_label(low: float, high: float) -> str:
    return f"[{low:.2f}-{high:.2f})"


def confidence_band_summary(
    linked: pd.DataFrame,
    edges: tuple[float, ...] = CONFIDENCE_BAND_EDGES,
) -> pd.DataFrame:
    """Aggregate realized results per confidence band.

    Only decisions with a realized outcome contribute to win_rate/PF; the
    `open_or_unmatched` column shows how many are excluded per band.
    """
    columns = [
        "band",
        "decisions",
        "settled",
        "open_or_unmatched",
        "wins",
        "win_rate",
        "total_pnl",
        "avg_pnl",
        "profit_factor",
    ]
    if linked.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, Any]] = []
    for low, high in zip(edges[:-1], edges[1:]):
        upper_inclusive = high >= edges[-1]
        conf = linked["confidence"]
        in_band = (conf >= low) & ((conf <= high) if upper_inclusive else (conf < high))
        band_df = linked[in_band]
        settled = band_df[band_df["realized_pnl"].notna()]
        pnl = settled["realized_pnl"]
        wins = pnl[pnl > 0]
        losses = pnl[pnl < 0]

        if float(losses.sum()) < 0:
            pf: float | None = float(wins.sum()) / abs(float(losses.sum()))
        elif float(wins.sum()) > 0:
            pf = float("inf")
        else:
            pf = None

        rows.append(
            {
                "band": _band_label(low, high),
                "decisions": int(len(band_df)),
                "settled": int(len(settled)),
                "open_or_unmatched": int(len(band_df) - len(settled)),
                "wins": int(len(wins)),
                "win_rate": (len(wins) / len(settled) * 100.0) if len(settled) else 0.0,
                "total_pnl": float(pnl.sum()) if len(settled) else 0.0,
                "avg_pnl": float(pnl.mean()) if len(settled) else 0.0,
                "profit_factor": pf,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def threshold_blocked_summary(
    raw_df: pd.DataFrame,
    edges: tuple[float, ...] = CONFIDENCE_BAND_EDGES,
) -> pd.DataFrame:
    """Count HOLD decisions per confidence band.

    `with_bias` counts HOLDs where the trader still reported a directional
    bias — the candidates a lower threshold would most likely have converted
    into trades.
    """
    columns = ["band", "holds", "with_bias"]
    if raw_df.empty:
        return pd.DataFrame(columns=columns)

    action = _col(raw_df, "action").str.upper().str.strip()
    reasoning = _col(raw_df, "reasoning").str.strip()
    holds = raw_df[(action == "HOLD") & (reasoning != "closed_trade_sync") & (reasoning != "breakeven_monitor")].copy()
    holds["_confidence"] = pd.to_numeric(_col(holds, "confidence"), errors="coerce")
    holds["_bias"] = _col(holds, "directional_bias").str.upper().str.strip()
    holds = holds[holds["_confidence"].notna()]

    rows: list[dict[str, Any]] = []
    for low, high in zip(edges[:-1], edges[1:]):
        upper_inclusive = high >= edges[-1]
        conf = holds["_confidence"]
        in_band = (conf >= low) & ((conf <= high) if upper_inclusive else (conf < high))
        band_df = holds[in_band]
        rows.append(
            {
                "band": _band_label(low, high),
                "holds": int(len(band_df)),
                "with_bias": int((band_df["_bias"].isin(["BULLISH", "BEARISH"])).sum()),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def print_confidence_report(csv_path: str) -> None:
    raw = load_raw_log(csv_path)
    linked = link_decisions_to_outcomes(raw)
    bands = confidence_band_summary(linked)
    blocked = threshold_blocked_summary(raw)

    settled_total = int(linked["realized_pnl"].notna().sum()) if not linked.empty else 0

    print("=== GP-MATE Confidence Calibration Report ===")
    print(f"CSV: {csv_path}")
    print("")
    print(f"Executed decisions : {len(linked)}")
    print(f"Settled (matched)  : {settled_total}")
    print("")
    print("[Realized results by confidence band]")
    if bands.empty or bands["decisions"].sum() == 0:
        print("No executed decisions yet")
    else:
        display = bands.copy()
        display["win_rate"] = display["win_rate"].map(lambda v: f"{v:.1f}%")
        display["profit_factor"] = display["profit_factor"].map(
            lambda v: "N/A" if v is None or pd.isna(v) else ("inf" if v == float("inf") else f"{v:.2f}")
        )
        print(display.to_string(index=False))
    print("")
    print("[Realized results by judgment slot (NY time)]")
    slots = slot_summary(linked)
    if slots.empty or slots["decisions"].sum() == 0:
        print("No executed decisions yet")
    else:
        slot_display = slots.copy()
        slot_display["win_rate"] = slot_display["win_rate"].map(lambda v: f"{v:.1f}%")
        slot_display["profit_factor"] = slot_display["profit_factor"].map(
            lambda v: "N/A" if v is None or pd.isna(v) else ("inf" if v == float("inf") else f"{v:.2f}")
        )
        print(slot_display.to_string(index=False))
    print("")
    print("[HOLD decisions by confidence band]")
    if blocked.empty or blocked["holds"].sum() == 0:
        print("No HOLD decisions yet")
    else:
        print(blocked.to_string(index=False))
    print("")
    if settled_total < MIN_TRADES_STAGE_THRESHOLD:
        print(
            f"注意: 決済済みサンプルが{settled_total}件と少なく（目安 {MIN_TRADES_STAGE_THRESHOLD}件以上）、"
            "帯別の勝率/PFはまだ統計的に信頼できません。閾値変更は保留を推奨。"
        )


def _status_line(name: str, ok: bool, detail: str) -> str:
    marker = "✓" if ok else "-"
    return f"{marker} {name}: {detail}"


def print_report(csv_path: str) -> None:
    trades = load_trades(csv_path)
    metrics = calc_metrics(trades)
    daily = daily_summary(trades)

    pf = metrics["profit_factor"]
    pf_display = "N/A" if pf is None else ("inf" if pf == float("inf") else f"{pf:.2f}")

    print("=== GP-MATE Performance Report ===")
    print(f"CSV: {csv_path}")
    print("")
    print(f"Total Trades      : {metrics['total_trades']}")
    print(f"Hold/Skip Count   : {metrics['hold_count']}")
    print(f"Win Rate          : {metrics['win_rate']:.2f}%")
    print(f"Profit Factor     : {pf_display}")
    print(f"Total PnL         : {metrics['total_pnl']:.2f}")
    print(f"Max Drawdown      : {metrics['max_drawdown_pct']:.2f}%")
    print(f"Average Win       : {metrics['avg_win']:.2f}")
    print(f"Average Loss      : {metrics['avg_loss']:.2f}")
    print(f"Max Consecutive L : {metrics['consecutive_loss']}")
    print("")

    pf_ok = pf is not None and pf > PF_STAGE_THRESHOLD
    dd_ok = metrics["max_drawdown_pct"] < MAX_DD_STAGE_THRESHOLD
    trades_ok = metrics["total_trades"] >= MIN_TRADES_STAGE_THRESHOLD

    print("[Stage Criteria]")
    print(_status_line("PF > 1.3", pf_ok, f"{pf_display}"))
    print(_status_line("MaxDD < 20%", dd_ok, f"{metrics['max_drawdown_pct']:.2f}%"))
    print(_status_line("Trades >= 30", trades_ok, str(metrics["total_trades"])))

    if pf_ok and dd_ok and trades_ok:
        print("=> 次Stageへの移行を検討可")
    else:
        print("=> まだ現Stageで検証継続")

    print("")
    print("[Daily Summary]")
    if daily.empty:
        print("No settled trades")
    else:
        print(daily.to_string(index=False))


if __name__ == "__main__":
    default_csv = Path(__file__).resolve().parent.parent / "logs" / "trade_log.csv"
    args = sys.argv[1:]
    csv_arg = next((a for a in args if not a.startswith("-")), str(default_csv))
    if "--confidence" in args:
        print_confidence_report(csv_arg)
    else:
        print_report(csv_arg)
        print("")
        print_confidence_report(csv_arg)
