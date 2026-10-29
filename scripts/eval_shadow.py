"""Per-instrument summary of shadow plans (and gold's production plans).

    python scripts/eval_shadow.py
    python scripts/eval_shadow.py --file logs/shadow_plans.jsonl --resolve

--resolve first labels open plans from MT5 bars (needs the terminal).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.shadow_eval import format_shadow_report, summarize_plans  # noqa: E402
from analysis.shadow_store import read_plans  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--file", default="", help="shadow_plans.jsonl path (default logs/shadow_plans.jsonl)")
    parser.add_argument("--resolve", action="store_true", help="resolve open plans from MT5 bars first")
    args = parser.parse_args()
    if args.resolve:
        from shadow import resolve_shadow_plans

        print("resolve:", resolve_shadow_plans())
    rows = read_plans(Path(args.file) if args.file else None)
    print(format_shadow_report(summarize_plans(rows)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
