from __future__ import annotations

from datetime import datetime
import logging
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config import JST_TZ, MARKET_TIMEZONE_NAME, MARKET_TZ, NY_RUN_TIMES, STAGE
from data.mt5_client import get_account_info
from main import run_once

LOGGER = logging.getLogger("gp_mate.scheduler")
SCHEDULER_MISFIRE_GRACE_SECONDS = 30


def _trade_mode_name(trade_mode: int) -> str:
    if trade_mode == 0:
        return "DEMO"
    if trade_mode == 2:
        return "REAL"
    return f"OTHER({trade_mode})"


def _format_schedule_log(hour: int, minute: int, reference: datetime | None = None) -> str:
    current_ny = (reference or datetime.now(tz=MARKET_TZ)).astimezone(MARKET_TZ)
    scheduled_ny = datetime(
        current_ny.year,
        current_ny.month,
        current_ny.day,
        hour,
        minute,
        tzinfo=MARKET_TZ,
    )
    scheduled_jst = scheduled_ny.astimezone(JST_TZ)
    return (
        "Scheduled run_once at "
        f"{hour:02d}:{minute:02d} {MARKET_TIMEZONE_NAME} "
        f"(={scheduled_jst:%H:%M} JST current conversion)"
    )


def _job_wrapper() -> None:
    started = time.time()
    LOGGER.info("run_once start")
    try:
        result = run_once()
        LOGGER.info(
            "run_once done action=%s allowed=%s order_success=%s",
            result.get("action"),
            result.get("allowed"),
            result.get("order_success"),
        )
    except Exception as exc:
        LOGGER.exception("run_once error (scheduler continues): %s", exc)
    finally:
        elapsed = time.time() - started
        LOGGER.info("run_once end elapsed=%.2fs", elapsed)


def _confirm_real_stage_warning(trade_mode: int) -> bool:
    if trade_mode != 2 or STAGE <= 1:
        return True

    print("\n!!! WARNING !!!")
    print("リアル口座 かつ STAGE > 1 を検出しました。")
    print("想定以上のリスクで発注される可能性があります。")
    answer = input("本当にスケジューラ起動しますか? [y/N]: ").strip().lower()
    return answer == "y"


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    info = get_account_info()
    if not info.get("success"):
        LOGGER.error("接続失敗のため終了: %s", info.get("reason", ""))
        return 1

    account = info.get("data")
    if account is None:
        LOGGER.error("口座情報が空のため終了")
        return 1

    mode = _trade_mode_name(account.trade_mode)
    LOGGER.info("Account login=%s server=%s mode=%s stage=%s", account.login, account.server, mode, STAGE)
    if account.trade_mode == 2:
        LOGGER.warning("リアル口座を検出しました。発注リスクに注意してください。")

    if not _confirm_real_stage_warning(account.trade_mode):
        LOGGER.warning("ユーザーにより起動キャンセル")
        return 1

    try:
        from apscheduler.schedulers.blocking import BlockingScheduler
    except Exception as exc:
        LOGGER.error("APSchedulerが利用できません: %s", exc)
        LOGGER.error("pip install apscheduler を実行してください")
        return 1

    scheduler = BlockingScheduler()
    for hour, minute in NY_RUN_TIMES:
        scheduler.add_job(
            _job_wrapper,
            "cron",
            hour=hour,
            minute=minute,
            timezone=MARKET_TIMEZONE_NAME,
            misfire_grace_time=SCHEDULER_MISFIRE_GRACE_SECONDS,
        )
        LOGGER.info(_format_schedule_log(hour, minute))

    LOGGER.info("Scheduler started misfire_grace_time=%ss", SCHEDULER_MISFIRE_GRACE_SECONDS)
    try:
        scheduler.start()
    except KeyboardInterrupt:
        LOGGER.info("Scheduler stopped by user")
        return 0
    except Exception as exc:
        LOGGER.exception("Scheduler crashed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
