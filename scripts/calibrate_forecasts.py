"""Fit a post-hoc calibration on the first half of forecasts.jsonl, test on the rest.

    python scripts/calibrate_forecasts.py [--file logs/forecasts.jsonl] [--fit-fraction 0.5]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.forecast_calibration import evaluate_calibration, format_calibration  # noqa: E402
from analysis.forecast_store import read_forecasts  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--file", default="")
    parser.add_argument("--fit-fraction", type=float, default=0.5)
    args = parser.parse_args()
    rows = read_forecasts(Path(args.file) if args.file else None)
    print(format_calibration(evaluate_calibration(rows, args.fit_fraction)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
