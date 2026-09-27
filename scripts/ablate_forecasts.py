"""Re-forecast archived forecast inputs with one report removed, then compare.

    python scripts/ablate_forecasts.py                 # run default conditions, then compare
    python scripts/ablate_forecasts.py --limit 20      # smoke test on the first 20 rows
    python scripts/ablate_forecasts.py --conditions no_sentiment,no_macro,technical_only,full_rerun
    python scripts/ablate_forecasts.py --compare-only  # no LLM calls, just the table

Resumable: already-done (forecast_id, condition) pairs are skipped.
Output: logs/forecast_ablation.jsonl. Nothing here trades.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.forecast_ablation import (  # noqa: E402
    CONDITIONS,
    DEFAULT_CONDITIONS,
    compare_conditions,
    format_comparison,
    read_ablation,
    run_ablation,
)
from analysis.forecast_store import read_forecasts  # noqa: E402
from config import MODEL_FORECAST  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--conditions", default=",".join(DEFAULT_CONDITIONS), help=f"comma-separated from {sorted(CONDITIONS)}")
    parser.add_argument("--limit", type=int, default=None, help="only the first N scorable forecasts (chronological)")
    parser.add_argument("--model", default=MODEL_FORECAST)
    parser.add_argument("--compare-only", action="store_true", help="skip LLM calls; print the comparison table")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    conditions = tuple(c.strip() for c in args.conditions.split(",") if c.strip())
    unknown = [c for c in conditions if c not in CONDITIONS]
    if unknown:
        parser.error(f"unknown conditions {unknown}; choose from {sorted(CONDITIONS)}")

    if not args.compare_only:
        counts = run_ablation(conditions, model=args.model, limit=args.limit)
        print(f"ablation run: {counts}")

    report = compare_conditions(read_forecasts(), read_ablation(), conditions)
    print(format_comparison(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
