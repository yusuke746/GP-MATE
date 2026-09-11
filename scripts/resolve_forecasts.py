"""Label forecast-only records whose horizon has elapsed (idempotent).

    python scripts/resolve_forecasts.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.forecast_pipeline import resolve_pending  # noqa: E402


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    counts = resolve_pending()
    print(counts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
