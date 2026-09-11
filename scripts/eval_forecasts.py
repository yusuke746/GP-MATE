"""Score forecast-only records: Brier / log loss, reliability, breakdowns,
baselines and a block-bootstrap comparison against the base-rate baseline.

    python scripts/eval_forecasts.py
    python scripts/eval_forecasts.py --csv logs/forecast_eval.csv
    python scripts/eval_forecasts.py --file logs/forecasts.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.forecast_eval import evaluate_file, format_report, write_csv  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--file", default="", help="forecasts.jsonl path (default logs/forecasts.jsonl)")
    parser.add_argument("--csv", default="", help="also write the tables to this CSV path")
    args = parser.parse_args()
    report = evaluate_file(Path(args.file) if args.file else None)
    print(format_report(report))
    if args.csv:
        write_csv(report, Path(args.csv))
        print(f"\ncsv written: {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
