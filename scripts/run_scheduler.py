from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config import JUDGMENT_TIMES, STAGE
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


def _parse_judgment_time(value: str) -> tuple[int, int] | None:
    try:
        hh_str, mm_str = value.split(":", 1)
        hh = int(hh_str)
        mm = int(mm_str)
        if not (0 <= hh <= 23 and 0 <= mm <= 59):
            return None
        return hh, mm
    except Exception:
        return None


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
    for time_str in JUDGMENT_TIMES:
        parsed = _parse_judgment_time(time_str)
        if parsed is None:
            LOGGER.warning("Invalid JUDGMENT_TIMES value skipped: %s", time_str)
            continue
        hour, minute = parsed
        scheduler.add_job(
            _job_wrapper,
            "cron",
            hour=hour,
            minute=minute,
            misfire_grace_time=SCHEDULER_MISFIRE_GRACE_SECONDS,
        )
        LOGGER.info("Scheduled run_once at %02d:%02d", hour, minute)

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
