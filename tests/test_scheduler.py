from __future__ import annotations

import importlib
from datetime import UTC, datetime
from pathlib import Path

import main
from risk.risk_manager import check_filters
from scripts import run_scheduler


def test_scheduler_runs_when_time_is_within_catchup_window(monkeypatch) -> None:
    monkeypatch.setattr(main, "JUDGMENT_TIMES", ("09:00",))
    now_local = datetime(2026, 7, 2, 9, 3, 0)

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
    monkeypatch.setattr(main, "JUDGMENT_TIMES", ("09:00",))
    now_local = datetime(2026, 7, 2, 9, 1, 0)

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
    monkeypatch.setattr(main, "JUDGMENT_TIMES", ("09:00", "09:01"))
    now_local = datetime(2026, 7, 2, 9, 1, 30)

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
    monkeypatch.setattr(run_scheduler, "JUDGMENT_TIMES", ("09:00",))
    monkeypatch.setattr(run_scheduler, "STAGE", 1)
    monkeypatch.setattr(
        run_scheduler,
        "get_account_info",
        lambda: {
            "success": True,
            "data": type("A", (), {"login": 1, "server": "demo", "trade_mode": 0})(),
        },
    )

    captured: dict[str, object] = {}

    class FakeScheduler:
        def add_job(self, func, trigger, **kwargs):
            captured["func"] = func
            captured["trigger"] = trigger
            captured["kwargs"] = kwargs

        def start(self):
            raise KeyboardInterrupt()

    blocking = importlib.import_module("apscheduler.schedulers.blocking")
    monkeypatch.setattr(blocking, "BlockingScheduler", FakeScheduler)

    assert run_scheduler.main() == 0
    assert captured["trigger"] == "cron"
    assert captured["kwargs"] == {
        "hour": 9,
        "minute": 0,
        "misfire_grace_time": run_scheduler.SCHEDULER_MISFIRE_GRACE_SECONDS,
    }


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
