"""Export OHLCV bars from the MT5 terminal to CSV for offline path analysis.

Run on the Windows machine where MT5 is installed, e.g.:

    python scripts/export_ohlcv.py --tf M5 --start 2026-06-20 --end 2026-09-06

The output feeds analysis/path_analysis.py. Bar timestamps are converted from
MT5 server wall-clock time to true UTC using MT5_SERVER_TIMEZONE (same rule as
closed-deal timestamps), so they line up with trade_log.csv.

If the export stops short of --start, raise "Max bars in chart" in the MT5
terminal options (Tools > Options > Charts) and run again.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import LOG_DIR, SYMBOL  # noqa: E402
from data import mt5_client  # noqa: E402
from data.mt5_client import (  # noqa: E402
    _deal_epoch_to_utc,
    _ensure_symbol,
    _timeframe_to_mt5_constant,
    connect,
    disconnect,
)

LOGGER = logging.getLogger("gp_mate.export_ohlcv")

CSV_COLUMNS = ("time_utc", "open", "high", "low", "close", "tick_volume", "spread")


def _parse_date(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def rates_to_rows(rates: Any) -> list[dict[str, Any]]:
    """Convert the MT5 structured array into CSV rows with true-UTC timestamps."""
    rows: list[dict[str, Any]] = []
    for rate in rates:
        rows.append(
            {
                "time_utc": _deal_epoch_to_utc(int(rate["time"])).isoformat(),
                "open": float(rate["open"]),
                "high": float(rate["high"]),
                "low": float(rate["low"]),
                "close": float(rate["close"]),
                "tick_volume": int(rate["tick_volume"]),
                "spread": int(rate["spread"]),
            }
        )
    return rows


def export_rates(symbol: str, tf: str, start: datetime, end: datetime) -> list[dict[str, Any]]:
    mt5 = mt5_client.mt5
    if mt5 is None:
        raise RuntimeError("MetaTrader5 package is not available (run this on the MT5 machine)")
    mt5_tf = _timeframe_to_mt5_constant(tf)
    if mt5_tf is None:
        raise ValueError(f"unsupported timeframe: {tf}")
    if not connect():
        raise RuntimeError("MT5 connect failed")
    try:
        if not _ensure_symbol(symbol):
            raise RuntimeError(f"symbol not available on MT5: {symbol}")
        # MT5 evaluates the range against server time; pad a day on both ends
        # so the requested UTC window is fully covered after conversion.
        rates = mt5.copy_rates_range(symbol, mt5_tf, start - timedelta(days=1), end + timedelta(days=1))
        if rates is None:
            raise RuntimeError(f"copy_rates_range failed: {mt5.last_error()}")
        rows = rates_to_rows(rates)
    finally:
        disconnect()

    start_iso = start.isoformat()
    end_iso = (end + timedelta(days=1)).isoformat()
    return [row for row in rows if start_iso <= row["time_utc"] < end_iso]


def write_csv(rows: list[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(CSV_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbol", default=SYMBOL)
    parser.add_argument("--tf", default="M5", help="M1 / M5 / M15 / M30 / H1 (default M5)")
    parser.add_argument("--start", required=True, help="YYYY-MM-DD (UTC, inclusive)")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD (UTC, inclusive)")
    parser.add_argument("--out", default="", help="output CSV path (default logs/ohlcv_<symbol>_<tf>.csv)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    start = _parse_date(args.start)
    end = _parse_date(args.end)
    if end < start:
        parser.error("--end must not be before --start")

    out_path = Path(args.out) if args.out else Path(LOG_DIR) / f"ohlcv_{args.symbol}_{args.tf}.csv"
    try:
        rows = export_rates(args.symbol, args.tf, start, end)
    except Exception as exc:
        LOGGER.error("export failed: %s", exc)
        return 1

    write_csv(rows, out_path)
    if rows:
        LOGGER.info("wrote %d bars (%s .. %s) to %s", len(rows), rows[0]["time_utc"], rows[-1]["time_utc"], out_path)
        if rows[0]["time_utc"][:10] > args.start:
            LOGGER.warning("history starts after --start; raise 'Max bars in chart' in MT5 and re-run")
    else:
        LOGGER.warning("no bars returned for the requested range")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
