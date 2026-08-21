from __future__ import annotations

import csv
import importlib
from pathlib import Path

import pytest

import main
from scripts import run_breakeven_monitor


def test_run_monitor_once_exits_cleanly_without_positions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(run_breakeven_monitor, "get_position_details", lambda symbol: [])

    monkeypatch.setattr(run_breakeven_monitor, "_is_weekend_flat_window", lambda: False)
    monkeypatch.setattr(run_breakeven_monitor, "_is_pending_flat_window", lambda: False)
    monkeypatch.setattr(run_breakeven_monitor, "is_high_impact_soon", lambda minutes: False)
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

    monkeypatch.setattr(run_breakeven_monitor, "_is_weekend_flat_window", lambda: False)
    monkeypatch.setattr(run_breakeven_monitor, "_is_pending_flat_window", lambda: False)
    monkeypatch.setattr(run_breakeven_monitor, "is_high_impact_soon", lambda minutes: False)
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

    monkeypatch.setattr(run_breakeven_monitor, "_is_weekend_flat_window", lambda: False)
    monkeypatch.setattr(run_breakeven_monitor, "_is_pending_flat_window", lambda: False)
    monkeypatch.setattr(run_breakeven_monitor, "is_high_impact_soon", lambda minutes: False)
    result = run_breakeven_monitor.run_monitor_once()

    assert result["success"] is True
    assert result["checked_positions"] == 1
    assert result["moved_positions"] == 0


def test_run_monitor_once_price_fetch_failure_exits_safely(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(run_breakeven_monitor, "get_position_details", lambda symbol: (_ for _ in ()).throw(RuntimeError("mt5 error")))

    monkeypatch.setattr(run_breakeven_monitor, "_is_weekend_flat_window", lambda: False)
    monkeypatch.setattr(run_breakeven_monitor, "_is_pending_flat_window", lambda: False)
    monkeypatch.setattr(run_breakeven_monitor, "is_high_impact_soon", lambda minutes: False)
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

def test_run_monitor_once_force_closes_positions_in_weekend_flat_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log_path = tmp_path / "trade_log.csv"
    monkeypatch.setattr(main, "LOG_DIR", tmp_path)
    monkeypatch.setattr(main, "TRADE_LOG_PATH", log_path)
    monkeypatch.setattr(run_breakeven_monitor, "_is_weekend_flat_window", lambda: True)
    monkeypatch.setattr(
        run_breakeven_monitor,
        "get_position_details",
        lambda symbol: [
            {
                "ticket": 555,
                "symbol": symbol,
                "type": "BUY",
                "volume": 0.02,
                "price_open": 4356.25,
                "price_current": 4342.0,
                "sl": 4325.43,
                "tp": 4417.9,
                "profit": -4498.0,
            }
        ],
    )

    close_calls: list[int] = []
    monkeypatch.setattr(
        run_breakeven_monitor,
        "close_position",
        lambda ticket: close_calls.append(ticket) or {"success": True, "retcode": 0, "deal": 999},
    )
    modify_calls: list[tuple[int, float]] = []
    monkeypatch.setattr(main, "modify_sl", lambda ticket, new_sl: modify_calls.append((ticket, new_sl)) or {"success": True, "retcode": 0})

    result = run_breakeven_monitor.run_monitor_once()

    assert close_calls == [555]
    assert modify_calls == []
    assert result["closed_positions"] == 1
    assert result["reason"] == "WEEKEND_FLAT"
    rows = list(csv.DictReader(log_path.open("r", encoding="utf-8")))
    assert len(rows) == 1
    assert rows[0]["action"] == "CLOSE"
    assert rows[0]["reasoning"] == "weekend_flat_close"
    assert rows[0]["order_success"] == "True"
    assert rows[0]["position_id"] == "555"


def test_run_monitor_once_reports_failure_when_weekend_close_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log_path = tmp_path / "trade_log.csv"
    monkeypatch.setattr(main, "LOG_DIR", tmp_path)
    monkeypatch.setattr(main, "TRADE_LOG_PATH", log_path)
    monkeypatch.setattr(run_breakeven_monitor, "_is_weekend_flat_window", lambda: True)
    monkeypatch.setattr(
        run_breakeven_monitor,
        "get_position_details",
        lambda symbol: [
            {
                "ticket": 556,
                "symbol": symbol,
                "type": "SELL",
                "volume": 0.01,
                "price_open": 100.0,
                "price_current": 99.0,
                "sl": 105.0,
                "tp": 90.0,
                "profit": 10.0,
            }
        ],
    )
    monkeypatch.setattr(
        run_breakeven_monitor,
        "close_position",
        lambda ticket: {"success": False, "retcode": None, "reason": "market closed"},
    )

    result = run_breakeven_monitor.run_monitor_once()

    assert result["success"] is False
    assert result["closed_positions"] == 0
    rows = list(csv.DictReader(log_path.open("r", encoding="utf-8")))
    assert rows[0]["order_success"] == "False"
    assert rows[0]["error"] == "market closed"


def test_run_monitor_once_cancels_pendings_in_weekend_flat_window(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(run_breakeven_monitor, "_is_weekend_flat_window", lambda: True)
    monkeypatch.setattr(run_breakeven_monitor, "get_position_details", lambda symbol: [])

    calls: list[str] = []
    monkeypatch.setattr(
        run_breakeven_monitor,
        "cancel_pending_orders",
        lambda symbol: calls.append(symbol) or {"success": True, "canceled": 2, "reason": "OK"},
    )

    result = run_breakeven_monitor.run_monitor_once()

    assert calls == [run_breakeven_monitor.SYMBOL]
    assert result["success"] is True
    assert result["checked_positions"] == 0


def test_run_monitor_once_cancels_pendings_at_daily_cutoff_but_keeps_breakeven(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log_path = tmp_path / "trade_log.csv"
    monkeypatch.setattr(main, "LOG_DIR", tmp_path)
    monkeypatch.setattr(main, "TRADE_LOG_PATH", log_path)
    monkeypatch.setattr(run_breakeven_monitor, "_is_weekend_flat_window", lambda: False)
    monkeypatch.setattr(run_breakeven_monitor, "_is_pending_flat_window", lambda: True)

    cancel_calls: list[str] = []
    monkeypatch.setattr(
        run_breakeven_monitor,
        "cancel_pending_orders",
        lambda symbol: cancel_calls.append(symbol) or {"success": True, "canceled": 1, "reason": "OK"},
    )
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
    monkeypatch.setattr(
        main, "modify_sl", lambda ticket, new_sl: modify_calls.append((ticket, new_sl)) or {"success": True, "retcode": 0}
    )

    result = run_breakeven_monitor.run_monitor_once()

    # Pendings are cancelled, but open positions are still breakeven-managed.
    assert cancel_calls == [run_breakeven_monitor.SYMBOL]
    assert result["moved_positions"] == 1
    assert modify_calls == [(321, 100.0 + main.BREAKEVEN_BUFFER)]


def test_run_monitor_once_cancels_pendings_before_high_impact_news(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(run_breakeven_monitor, "_is_weekend_flat_window", lambda: False)
    monkeypatch.setattr(run_breakeven_monitor, "_is_pending_flat_window", lambda: False)
    monkeypatch.setattr(run_breakeven_monitor, "is_high_impact_soon", lambda minutes: True)
    monkeypatch.setattr(run_breakeven_monitor, "get_position_details", lambda symbol: [])

    calls: list[str] = []
    monkeypatch.setattr(
        run_breakeven_monitor,
        "cancel_pending_orders",
        lambda symbol: calls.append(symbol) or {"success": True, "canceled": 1, "reason": "OK"},
    )

    result = run_breakeven_monitor.run_monitor_once()

    assert calls == [run_breakeven_monitor.SYMBOL]
    assert result["success"] is True
