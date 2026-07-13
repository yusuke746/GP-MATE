from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config import BREAKEVEN_MONITOR_TIMES, STAGE, SYMBOL
from data.mt5_client import get_account_info, get_position_details
from main import _append_trade_log, manage_breakeven_for_position

LOGGER = logging.getLogger("gp_mate.breakeven_monitor")
SCHEDULER_MISFIRE_GRACE_SECONDS = 30


def _trade_mode_name(trade_mode: int) -> str:
    if trade_mode == 0:
        return "DEMO"
    if trade_mode == 2:
        return "REAL"
    return f"OTHER({trade_mode})"


def _parse_minute(value: str) -> int | None:
    try:
        minute = int(value)
    except Exception:
        return None
    if not (0 <= minute <= 59):
        return None
    return minute


def _confirm_real_stage_warning(trade_mode: int) -> bool:
    if trade_mode != 2 or STAGE <= 1:
        return True

    print("\n!!! WARNING !!!")
    print("リアル口座 かつ STAGE > 1 を検出しました。")
    print("SL 更新のみでも実口座操作です。")
    answer = input("本当に建値監視スケジューラ起動しますか? [y/N]: ").strip().lower()
    return answer == "y"


def _build_log_row(position_context: dict[str, Any], breakeven_log: dict[str, Any], execution_result: dict[str, Any], filter_reason: str, allowed: bool, timestamp_utc: str) -> dict[str, Any]:
    return {
        "timestamp_utc": timestamp_utc,
        "deal_id": "",
        "symbol": SYMBOL,
        "action": "HOLD",
        "entry_price": float(position_context.get("price_open", 0.0) or 0.0),
        "exit_price": "",
        "holding_seconds": "",
        "pnl": float(position_context.get("profit", 0.0) or 0.0),
        "confidence": "",
        "reasoning": "breakeven_monitor",
        "risk_level": "",
        "allowed": allowed,
        "filter_reason": filter_reason,
        "lot": float(position_context.get("volume", 0.0) or 0.0),
        "sl": float(position_context.get("sl", 0.0) or 0.0),
        "tp": float(position_context.get("tp", 0.0) or 0.0),
        "order_success": bool(execution_result.get("success", False)),
        "retcode": execution_result.get("retcode", ""),
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "analysis_model": "",
        "decision_model": "",
        "news_count": 0,
        "error": str(execution_result.get("reason", "")),
        "position_direction": str(position_context.get("type", "") or ""),
        "technical_signal": "",
        "debate_direction": "",
        "evaluate_action": "HOLD",
        "evaluate_confidence": "",
        "evaluate_reasoning": "",
        "evaluate_reasoning_len": "",
        **breakeven_log,
    }


def run_monitor_once() -> dict[str, Any]:
    result: dict[str, Any] = {
        "success": True,
        "checked_positions": 0,
        "moved_positions": 0,
        "reason": "OK",
    }
    try:
        positions = get_position_details(SYMBOL)
    except Exception as exc:
        LOGGER.warning("get_position_details failed; monitor exits safely: %s", exc)
        return {
            "success": False,
            "checked_positions": 0,
            "moved_positions": 0,
            "reason": str(exc),
        }

    if not positions:
        LOGGER.info("No open positions for %s", SYMBOL)
        return result

    result["checked_positions"] = len(positions)
    for position_context in positions:
        try:
            timestamp_utc = str(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
            execution_result, breakeven_log, filter_reason, allowed = manage_breakeven_for_position(
                position_context=position_context,
                now_iso=timestamp_utc,
            )
            if breakeven_log.get("breakeven_triggered"):
                _append_trade_log(
                    _build_log_row(
                        position_context=position_context,
                        breakeven_log=breakeven_log,
                        execution_result=execution_result,
                        filter_reason=filter_reason,
                        allowed=allowed,
                        timestamp_utc=timestamp_utc,
                    )
                )
            if bool(execution_result.get("success", False)):
                result["moved_positions"] = int(result["moved_positions"]) + 1
        except Exception as exc:
            LOGGER.warning("breakeven monitor skipped one position safely: %s", exc)

    return result


def _job_wrapper() -> None:
    started = time.time()
    LOGGER.info("run_monitor_once start")
    try:
        result = run_monitor_once()
        LOGGER.info(
            "run_monitor_once done success=%s checked_positions=%s moved_positions=%s",
            result.get("success"),
            result.get("checked_positions"),
            result.get("moved_positions"),
        )
    except Exception as exc:
        LOGGER.exception("run_monitor_once error (scheduler continues): %s", exc)
    finally:
        elapsed = time.time() - started
        LOGGER.info("run_monitor_once end elapsed=%.2fs", elapsed)


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
        LOGGER.warning("リアル口座を検出しました。SL 更新リスクに注意してください。")

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
    for minute_str in BREAKEVEN_MONITOR_TIMES:
        minute = _parse_minute(minute_str)
        if minute is None:
            LOGGER.warning("Invalid BREAKEVEN_MONITOR_TIMES value skipped: %s", minute_str)
            continue
        scheduler.add_job(
            _job_wrapper,
            "cron",
            minute=minute,
            misfire_grace_time=SCHEDULER_MISFIRE_GRACE_SECONDS,
        )
        LOGGER.info("Scheduled run_monitor_once at each hour minute=%02d", minute)

    LOGGER.info("Breakeven monitor scheduler started misfire_grace_time=%ss", SCHEDULER_MISFIRE_GRACE_SECONDS)
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