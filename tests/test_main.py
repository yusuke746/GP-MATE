from __future__ import annotations

import csv
from pathlib import Path

import main


def test_run_once_with_missing_baseline_logs_hold(tmp_path: Path, monkeypatch) -> None:
    log_path = tmp_path / "trade_log.csv"
    monkeypatch.setattr(main, "LOG_DIR", tmp_path)
    monkeypatch.setattr(main, "TRADE_LOG_PATH", log_path)
    monkeypatch.setattr(main, "sync_closed_trades", lambda: 0)

    result = main.run_once(baseline_spread=0.0)

    assert result["action"] == "HOLD"
    assert log_path.exists()

    rows = list(csv.DictReader(log_path.open("r", encoding="utf-8")))
    assert len(rows) == 1
    assert rows[0]["action"] == "HOLD"
