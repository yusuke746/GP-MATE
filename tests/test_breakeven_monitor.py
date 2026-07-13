from __future__ import annotations

import csv
import importlib
from pathlib import Path

import pytest

import main
from scripts import run_breakeven_monitor


def test_run_monitor_once_exits_cleanly_without_positions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(run_breakeven_monitor, "get_position_details", lambda symbol: [])

    result = run_breakeven_monitor.run_monitor_once()

    assert result["success"] is True
    assert result["checked_positions"] == 0
    assert result["moved_positions"] == 0


def test_run_monitor_once_moves_breakeven_at_1r(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    log_path = tmp_path / "trade_log.csv"
    monkeypatch.setattr(main, "LOG_DIR", tmp_path)
    monkeypatch.setattr(main, "TRADE_LOG_PATH", log_path)
    monkeypatch.setattr(
        run_breakeven_monitor,
        "get_position_details",
        lambda symbol: [
            {
                "ticket": 321,
                "symbol": symbol,
                "type": "BUY",
                "volume": 0.01,
                "price_open": 100.0,
                "price_current": 105.0,
                "sl": 95.0,
                "tp": 110.0,
                "profit": 50.0,
            }
        ],
    )
    modify_calls: list[tuple[int, float]] = []
    monkeypatch.setattr(main, "modify_sl", lambda ticket, new_sl: modify_calls.append((ticket, new_sl)) or {"success": True, "retcode": 0})

    result = run_breakeven_monitor.run_monitor_once()

    assert result["success"] is True
    assert result["checked_positions"] == 1
    assert result["moved_positions"] == 1
    assert modify_calls == [(321, 100.0 + main.BREAKEVEN_BUFFER)]
    rows = list(csv.DictReader(log_path.open("r", encoding="utf-8")))
    assert len(rows) == 1
    assert rows[0]["breakeven_triggered"] == "True"
    assert rows[0]["breakeven_reason"] == "MOVED"


def test_run_monitor_once_does_not_move_before_1r(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        run_breakeven_monitor,
        "get_position_details",
        lambda symbol: [
            {
                "ticket": 321,
                "symbol": symbol,
                "type": "BUY",
                "volume": 0.01,
                "price_open": 100.0,
                "price_current": 104.9,
                "sl": 95.0,
                "tp": 110.0,
                "profit": 49.0,
            }
        ],
    )
    monkeypatch.setattr(main, "modify_sl", lambda ticket, new_sl: {"success": True, "retcode": 0})

    result = run_breakeven_monitor.run_monitor_once()

    assert result["success"] is True
    assert result["checked_positions"] == 1
    assert result["moved_positions"] == 0


def test_run_monitor_once_price_fetch_failure_exits_safely(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(run_breakeven_monitor, "get_position_details", lambda symbol: (_ for _ in ()).throw(RuntimeError("mt5 error")))

    result = run_breakeven_monitor.run_monitor_once()

    assert result["success"] is False
    assert result["checked_positions"] == 0
    assert result["moved_positions"] == 0


def test_monitor_scheduler_sets_expected_minutes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(run_breakeven_monitor, "BREAKEVEN_MONITOR_TIMES", ("07", "22", "37", "52"))
    monkeypatch.setattr(run_breakeven_monitor, "STAGE", 1)
    monkeypatch.setattr(
        run_breakeven_monitor,
        "get_account_info",
        lambda: {
            "success": True,
            "data": type("A", (), {"login": 1, "server": "demo", "trade_mode": 0})(),
        },
    )

    captured: list[dict[str, object]] = []

    class FakeScheduler:
        def add_job(self, func, trigger, **kwargs):
            captured.append({"func": func, "trigger": trigger, "kwargs": kwargs})

        def start(self):
            raise KeyboardInterrupt()

    blocking = importlib.import_module("apscheduler.schedulers.blocking")
    monkeypatch.setattr(blocking, "BlockingScheduler", FakeScheduler)

    assert run_breakeven_monitor.main() == 0
    assert [item["kwargs"] for item in captured] == [
        {"minute": 7, "misfire_grace_time": run_breakeven_monitor.SCHEDULER_MISFIRE_GRACE_SECONDS},
        {"minute": 22, "misfire_grace_time": run_breakeven_monitor.SCHEDULER_MISFIRE_GRACE_SECONDS},
        {"minute": 37, "misfire_grace_time": run_breakeven_monitor.SCHEDULER_MISFIRE_GRACE_SECONDS},
        {"minute": 52, "misfire_grace_time": run_breakeven_monitor.SCHEDULER_MISFIRE_GRACE_SECONDS},
    ]