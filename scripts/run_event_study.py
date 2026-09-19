"""Run the event-study harness on one or more exported OHLCV CSVs.

    python scripts/export_ohlcv.py --symbol GOLD# --tf H1 --start 2006-01-01 --end 2026-09-17
    python scripts/run_event_study.py --csv "logs/ohlcv_GOLD#_H1.csv" --csv logs/ohlcv_USDJPY_H1.csv

By default only the development period (up to --dev-end) is evaluated. The
holdout is opened with --unseal-holdout, and that should happen ONCE, after
the signal definitions and parameters are frozen.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.event_study import bonferroni_t, data_quality, evaluate, load_ohlcv, prepare  # noqa: E402
from research.signals import SIGNALS  # noqa: E402


def _symbol_from_path(path: Path) -> str:
    stem = path.stem
    return stem[len("ohlcv_"):] if stem.startswith("ohlcv_") else stem


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", action="append", required=True, help="export_ohlcv.py output (repeatable)")
    parser.add_argument("--signals", default=",".join(SIGNALS), help="comma separated; default all")
    parser.add_argument("--horizons", default="1,3,6,12")
    parser.add_argument("--tb-k", type=float, default=1.0, help="triple-barrier distance in ATR")
    parser.add_argument("--tb-horizon", type=int, default=6)
    parser.add_argument("--dev-end", default="2022-12-31")
    parser.add_argument("--unseal-holdout", action="store_true")
    parser.add_argument("--point", type=float, default=None, help="price per point (default: inferred)")
    parser.add_argument("--cost-points", type=float, default=None, help="fixed round-trip cost in points (default: CSV spread)")
    parser.add_argument("--min-events", type=int, default=30)
    parser.add_argument("--out", default="logs/event_study.csv")
    args = parser.parse_args()

    names = [s.strip() for s in args.signals.split(",") if s.strip()]
    unknown = [s for s in names if s not in SIGNALS]
    if unknown:
        parser.error(f"unknown signals: {unknown}; available: {list(SIGNALS)}")
    horizons = [int(h) for h in args.horizons.split(",") if h.strip()]

    pd.set_option("display.width", 200)
    tables: list[pd.DataFrame] = []
    for csv_path in (Path(p) for p in args.csv):
        symbol = _symbol_from_path(csv_path)
        raw = load_ohlcv(csv_path)
        print(f"\n=== {symbol}: {len(raw)} bars, {raw.index[0]} .. {raw.index[-1]}")
        quality = data_quality(raw)
        print(quality.to_string())
        if quality["thin"].any():
            print("WARNING: years marked thin have <80% of the typical bar count; treat them with suspicion.")
        prepared = prepare(raw, horizons, args.tb_k, args.tb_horizon, args.point, args.cost_points)
        print(f"point={prepared.point:g}  cost={prepared.cost_note}")
        print(f"median cost = {prepared.frame['cost_atr'].median():.3f} ATR per trade")
        for name in names:
            table = evaluate(prepared, SIGNALS[name], name, args.dev_end, args.unseal_holdout, args.min_events)
            if not table.empty:
                table.insert(0, "symbol", symbol)
                tables.append(table)

    if not tables:
        print("no segment reached --min-events")
        return 1
    result = pd.concat(tables, ignore_index=True)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False)

    headline = result[result["regime"] == "all"]
    print("\n=== headline (regime=all); full table incl. ADX splits in", out_path)
    print(headline.to_string(index=False))
    tests = len(result)
    print(
        f"\n{tests} segments were tested. With that many looks, |t_excess| needs to exceed "
        f"~{bonferroni_t(tests):.1f} (Bonferroni) before it is more than luck; ~2 is NOT enough."
    )
    if not args.unseal_holdout:
        print(f"Holdout (after {args.dev_end}) is sealed. Freeze definitions first, then run once with --unseal-holdout.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
