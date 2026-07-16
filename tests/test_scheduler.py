from __future__ import annotations

import importlib
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import main
from apscheduler.triggers.cron import CronTrigger
from risk.risk_manager import check_filters
from scripts import run_scheduler


def _build_ny_trigger(hour: int, minute: int) -> CronTrigger:
    return CronTrigger(hour=hour, minute=minute, timezone=run_scheduler.MARKET_TIMEZONE_NAME)


def _next_fire_jst(hour: int, minute: int, now: datetime) -> tuple[datetime, str]:
    fire_time = _build_ny_trigger(hour, minute).get_next_fire_time(None, now)
    assert fire_time is not None
    fire_jst = fire_time.astimezone(run_scheduler.JST_TZ)
    return fire_time, fire_jst.strftime("%H:%M")


def test_scheduler_runs_when_time_is_within_catchup_window(monkeypatch) -> None:
    monkeypatch.setattr(main, "NY_RUN_TIMES", ((9, 0),))
    now_local = datetime(2026, 7, 2, 9, 3, 0, tzinfo=ZoneInfo("America/New_York"))

    calls: list[dict[str, float]] = []

    def _fake_run_once(baseline_spread, consecutive_losses, daily_loss_pct):
        calls.append(
            {
                "consecutive_losses": float(consecutive_losses),
                "daily_loss_pct": float(daily_loss_pct),
            }
        )
        return {"action": "HOLD"}

    monkeypatch.setattr(main, "calc_today_risk_stats", lambda: (2, 0.015))
    monkeypatch.setattr(main, "run_once", _fake_run_once)

    executed_today: set[str] = set()
    main._run_scheduler_due_jobs(now_local=now_local, executed_today=executed_today, baseline_spread=10.0)

    assert len(calls) == 1
    assert calls[0]["consecutive_losses"] == 2.0
    assert calls[0]["daily_loss_pct"] == 0.015


def test_scheduler_does_not_double_execute_same_slot(monkeypatch) -> None:
    monkeypatch.setattr(main, "NY_RUN_TIMES", ((9, 0),))
    now_local = datetime(2026, 7, 2, 9, 1, 0, tzinfo=ZoneInfo("America/New_York"))

    count = {"n": 0}

    def _fake_run_once(baseline_spread, consecutive_losses, daily_loss_pct):
        count["n"] += 1
        return {"action": "HOLD"}

    monkeypatch.setattr(main, "calc_today_risk_stats", lambda: (0, 0.0))
    monkeypatch.setattr(main, "run_once", _fake_run_once)

    executed_today: set[str] = set()
    main._run_scheduler_due_jobs(now_local=now_local, executed_today=executed_today, baseline_spread=10.0)
    main._run_scheduler_due_jobs(now_local=now_local, executed_today=executed_today, baseline_spread=10.0)

    assert count["n"] == 1


def test_scheduler_continues_when_run_once_raises(monkeypatch) -> None:
    monkeypatch.setattr(main, "NY_RUN_TIMES", ((9, 0), (9, 1)))
    now_local = datetime(2026, 7, 2, 9, 1, 30, tzinfo=ZoneInfo("America/New_York"))

    calls = {"n": 0}

    def _fake_run_once(baseline_spread, consecutive_losses, daily_loss_pct):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("boom")
        return {"action": "HOLD"}

    monkeypatch.setattr(main, "calc_today_risk_stats", lambda: (0, 0.0))
    monkeypatch.setattr(main, "run_once", _fake_run_once)

    executed_today: set[str] = set()
    main._run_scheduler_due_jobs(now_local=now_local, executed_today=executed_today, baseline_spread=10.0)

    assert calls["n"] == 2


def test_run_scheduler_sets_misfire_grace_time(monkeypatch) -> None:
    monkeypatch.setattr(run_scheduler, "NY_RUN_TIMES", ((8, 0),))
    monkeypatch.setattr(run_scheduler, "STAGE", 1)
    monkeypatch.setattr(
        run_scheduler,
        "get_account_info",
        lambda: {
            "success": True,
            "data": type("A", (), {"login": 1, "server": "demo", "trade_mode": 0})(),
        },
    )

    captured: list[dict[str, object]] = []

    class FakeScheduler:
        def add_job(self, func, trigger, **kwargs):
            captured.append(
                {
                    "func": func,
                    "trigger": trigger,
                    "kwargs": kwargs,
                }
            )

        def start(self):
            raise KeyboardInterrupt()

    blocking = importlib.import_module("apscheduler.schedulers.blocking")
    monkeypatch.setattr(blocking, "BlockingScheduler", FakeScheduler)

    assert run_scheduler.main() == 0
    assert len(captured) == 1
    assert captured[0]["trigger"] == "cron"
    assert captured[0]["kwargs"] == {
        "hour": 8,
        "minute": 0,
        "timezone": run_scheduler.MARKET_TIMEZONE_NAME,
        "misfire_grace_time": run_scheduler.SCHEDULER_MISFIRE_GRACE_SECONDS,
    }


def test_format_schedule_log_uses_current_dst_conversion() -> None:
    reference = datetime(2026, 7, 15, 12, 0, tzinfo=ZoneInfo("UTC"))

    message = run_scheduler._format_schedule_log(8, 0, reference=reference)

    assert message == (
        "Scheduled run_once at 08:00 America/New_York "
        "(=21:00 JST current conversion)"
    )


def test_format_schedule_log_uses_winter_dst_conversion() -> None:
    reference = datetime(2026, 1, 15, 12, 0, tzinfo=ZoneInfo("UTC"))

    message = run_scheduler._format_schedule_log(10, 30, reference=reference)

    assert message == (
        "Scheduled run_once at 10:30 America/New_York "
        "(=00:30 JST current conversion)"
    )


def test_cron_trigger_next_fire_time_tracks_dst_boundaries() -> None:
    cases = (
        (
            datetime(2026, 10, 31, 0, 0, tzinfo=run_scheduler.MARKET_TZ),
            ("16:00", "21:00", "22:30", "23:30"),
        ),
        (
            datetime(2026, 11, 2, 0, 0, tzinfo=run_scheduler.MARKET_TZ),
            ("17:00", "22:00", "23:30", "00:30"),
        ),
        (
            datetime(2026, 3, 7, 0, 0, tzinfo=run_scheduler.MARKET_TZ),
            ("17:00", "22:00", "23:30", "00:30"),
        ),
        (
            datetime(2026, 3, 9, 0, 0, tzinfo=run_scheduler.MARKET_TZ),
            ("16:00", "21:00", "22:30", "23:30"),
        ),
    )

    for now, expected_jst_times in cases:
        actual_jst_times = []
        for hour, minute in run_scheduler.NY_RUN_TIMES:
            fire_time, jst_time = _next_fire_jst(hour, minute, now)
            assert fire_time.astimezone(run_scheduler.MARKET_TZ).strftime("%H:%M") == f"{hour:02d}:{minute:02d}"
            actual_jst_times.append(jst_time)
        assert tuple(actual_jst_times) == expected_jst_times


def test_cron_trigger_emits_one_occurrence_per_day_across_dst_end() -> None:
    start = datetime(2026, 10, 31, 0, 0, tzinfo=run_scheduler.MARKET_TZ)
    expected_by_day = (
        ("2026-10-31", "EDT", ("16:00", "21:00", "22:30", "23:30")),
        ("2026-11-01", "EST", ("17:00", "22:00", "23:30", "00:30")),
        ("2026-11-02", "EST", ("17:00", "22:00", "23:30", "00:30")),
    )

    actual_by_day: dict[str, list[tuple[str, str]]] = {}
    for hour, minute in run_scheduler.NY_RUN_TIMES:
        trigger = _build_ny_trigger(hour, minute)
        previous_fire_time = None
        now = start
        for _ in range(3):
            fire_time = trigger.get_next_fire_time(previous_fire_time, now)
            assert fire_time is not None
            fire_market = fire_time.astimezone(run_scheduler.MARKET_TZ)
            fire_jst = fire_time.astimezone(run_scheduler.JST_TZ)
            day_key = fire_market.strftime("%Y-%m-%d")
            actual_by_day.setdefault(day_key, []).append((fire_market.tzname() or "", fire_jst.strftime("%H:%M")))
            assert fire_market.strftime("%H:%M") == f"{hour:02d}:{minute:02d}"
            previous_fire_time = fire_time
            now = fire_time

    assert tuple(actual_by_day) == tuple(day for day, _, _ in expected_by_day)
    for day, expected_tz_name, expected_jst_times in expected_by_day:
        entries = actual_by_day[day]
        assert len(entries) == len(run_scheduler.NY_RUN_TIMES)
        assert all(tz_name == expected_tz_name for tz_name, _ in entries)
        assert tuple(jst_time for _, jst_time in entries) == expected_jst_times


def test_consecutive_loss_limit_blocks_after_five_losses(tmp_path: Path, monkeypatch) -> None:
    log_path = tmp_path / "trade_log.csv"
    now = datetime.now(UTC)
    ts = now.isoformat()
    log_path.write_text(
        "timestamp_utc,action,pnl,reasoning\n"
        f"{ts},SELL,-100,closed_trade_sync\n"
        f"{ts},SELL,-100,closed_trade_sync\n"
        f"{ts},SELL,-100,closed_trade_sync\n"
        f"{ts},SELL,-100,closed_trade_sync\n"
        f"{ts},SELL,-100,closed_trade_sync\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(main, "TRADE_LOG_PATH", log_path)
    monkeypatch.setattr(
        main,
        "get_account_info",
        lambda: {"success": True, "data": type("A", (), {"balance": 500000.0})()},
    )

    consecutive_losses, daily_loss_pct = main.calc_today_risk_stats()
    result = check_filters(
        confidence=0.9,
        spread=20.0,
        baseline_spread=15.0,
        is_news_soon=False,
        consecutive_losses=consecutive_losses,
        daily_loss_pct=daily_loss_pct,
    )

    assert not result.ok
    assert result.reason == "Consecutive loss limit reached"


def test_daily_loss_limit_blocks_when_over_three_percent(tmp_path: Path, monkeypatch) -> None:
    log_path = tmp_path / "trade_log.csv"
    now = datetime.now(UTC)
    ts = now.isoformat()
    log_path.write_text(
        "timestamp_utc,action,pnl,reasoning\n"
        f"{ts},SELL,-4000,closed_trade_sync\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(main, "TRADE_LOG_PATH", log_path)
    monkeypatch.setattr(
        main,
        "get_account_info",
        lambda: {"success": True, "data": type("A", (), {"balance": 100000.0})()},
    )

    consecutive_losses, daily_loss_pct = main.calc_today_risk_stats()
    result = check_filters(
        confidence=0.9,
        spread=20.0,
        baseline_spread=15.0,
        is_news_soon=False,
        consecutive_losses=consecutive_losses,
        daily_loss_pct=daily_loss_pct,
    )

    assert not result.ok
    assert result.reason == "Daily loss limit reached"


def test_consecutive_losses_reset_when_win_inserted(tmp_path: Path, monkeypatch) -> None:
    log_path = tmp_path / "trade_log.csv"
    now = datetime.now(UTC)
    ts = now.isoformat()
    log_path.write_text(
        "timestamp_utc,action,pnl,reasoning\n"
        f"{ts},SELL,-100,closed_trade_sync\n"
        f"{ts},SELL,-50,closed_trade_sync\n"
        f"{ts},BUY,20,closed_trade_sync\n"
        f"{ts},SELL,-10,closed_trade_sync\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(main, "TRADE_LOG_PATH", log_path)
    monkeypatch.setattr(
        main,
        "get_account_info",
        lambda: {"success": True, "data": type("A", (), {"balance": 500000.0})()},
    )

    consecutive_losses, _ = main.calc_today_risk_stats()
    assert consecutive_losses == 1


def test_calc_today_risk_stats_falls_back_safe_on_failure(tmp_path: Path, monkeypatch) -> None:
    log_path = tmp_path / "trade_log.csv"
    log_path.write_text(
        "timestamp_utc,action,pnl,reasoning\n"
        "not-a-datetime,SELL,-100,closed_trade_sync\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(main, "TRADE_LOG_PATH", log_path)

    consecutive_losses, daily_loss_pct = main.calc_today_risk_stats()

    assert consecutive_losses == main.CONSECUTIVE_LOSS_LIMIT
    assert daily_loss_pct == main.MAX_DAILY_LOSS_PCT


def test_calc_today_risk_stats_returns_zero_when_no_today_settlement(tmp_path: Path, monkeypatch) -> None:
    log_path = tmp_path / "trade_log.csv"
    now = datetime.now(UTC)
    ts = now.isoformat()
    log_path.write_text(
        "timestamp_utc,action,pnl,reasoning\n"
        f"{ts},HOLD,,\n"
        f"{ts},BUY,,\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(main, "TRADE_LOG_PATH", log_path)

    consecutive_losses, daily_loss_pct = main.calc_today_risk_stats()

    assert consecutive_losses == 0
    assert daily_loss_pct == 0.0
