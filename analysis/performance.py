from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

PF_STAGE_THRESHOLD = 1.3
MAX_DD_STAGE_THRESHOLD = 20.0
MIN_TRADES_STAGE_THRESHOLD = 30

PNL_COLUMN_CANDIDATES: tuple[str, ...] = (
    "pnl",
    "realized_pnl",
    "profit",
    "net_pnl",
)


def _safe_to_datetime(series: pd.Series) -> pd.Series:
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
    print_report(str(default_csv))
