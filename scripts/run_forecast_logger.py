"""Forecast-only logger: record LLM triple-barrier probabilities every H1 bar.

No orders are placed. Runs FORECAST_DELAY_SEC after every UTC full hour,
skips weekends, then labels forecasts whose horizon has elapsed.

    python scripts/run_forecast_logger.py            # scheduler (foreground)
    python scripts/run_forecast_logger.py --once     # one cycle now, then exit
    python scripts/run_forecast_logger.py --no-resolve
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.forecast_pipeline import resolve_pending, run_once  # noqa: E402
from config import FORECAST_DELAY_SEC, FORECAST_HORIZONS, FORECAST_SESSION_FILTER, FORECAST_USE_DEBATE, MODEL_FORECAST  # noqa: E402

LOGGER = logging.getLogger("gp_mate.forecast_logger")
MISFIRE_GRACE_SECONDS = 600


def _cycle(resolve: bool) -> None:
    try:
        summary = run_once()
        LOGGER.info("forecast cycle: %s", summary)
    except Exception as exc:  # pragma: no cover - run_once is itself fail-safe
        LOGGER.exception("forecast cycle crashed: %s", exc)
    if resolve:
        try:
            LOGGER.info("resolve: %s", resolve_pending())
        except Exception as exc:  # pragma: no cover
            LOGGER.exception("resolve crashed: %s", exc)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--once", action="store_true", help="run a single cycle and exit")
    parser.add_argument("--no-resolve", action="store_true", help="do not label elapsed forecasts after each cycle")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    LOGGER.info(
        "forecast logger: model=%s horizons=%s debate=%s session_filter=%r delay=%ss",
        MODEL_FORECAST, FORECAST_HORIZONS, FORECAST_USE_DEBATE, FORECAST_SESSION_FILTER, FORECAST_DELAY_SEC,
    )

    if args.once:
        _cycle(resolve=not args.no_resolve)
        return 0

    try:
        from apscheduler.schedulers.blocking import BlockingScheduler
    except Exception as exc:
        LOGGER.error("APScheduler unavailable: %s", exc)
        return 1

    scheduler = BlockingScheduler(timezone="UTC")
    scheduler.add_job(
        _cycle,
        "cron",
        minute=FORECAST_DELAY_SEC // 60,
        second=FORECAST_DELAY_SEC % 60,
        kwargs={"resolve": not args.no_resolve},
        misfire_grace_time=MISFIRE_GRACE_SECONDS,
    )
    LOGGER.info("scheduled hourly at :%02d:%02d UTC", FORECAST_DELAY_SEC // 60, FORECAST_DELAY_SEC % 60)
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        LOGGER.info("forecast logger stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
