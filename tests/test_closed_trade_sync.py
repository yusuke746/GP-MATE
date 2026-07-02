from __future__ import annotations

import csv
from pathlib import Path

import main


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fp:
        return list(csv.DictReader(fp))


def test_sync_closed_trades_appends_pnl_rows(tmp_path: Path, monkeypatch) -> None:
    log_path = tmp_path / "trade_log.csv"
    state_path = tmp_path / "closed_deal_state.json"

    monkeypatch.setattr(main, "LOG_DIR", tmp_path)
    monkeypatch.setattr(main, "TRADE_LOG_PATH", log_path)
    monkeypatch.setattr(main, "CLOSED_DEAL_STATE_PATH", state_path)

    monkeypatch.setattr(
        main,
        "get_closed_deals",
        lambda symbol, since: [
            {
                "deal_id": 101,
                "symbol": symbol,
                "action": "SELL",
                "entry_price": 2300.0,
                "exit_price": 2310.0,
                "lot": 0.1,
                "profit": 12.5,
                "holding_seconds": 3600,
                "time_utc": "2026-07-02T00:00:00+00:00",
            }
        ],
    )

    appended = main.sync_closed_trades()

    assert appended == 1
    rows = _read_rows(log_path)
    assert len(rows) == 1
    assert rows[0]["deal_id"] == "101"
    assert rows[0]["pnl"] == "12.5"


def test_sync_closed_trades_deduplicates_deal_id(tmp_path: Path, monkeypatch) -> None:
    log_path = tmp_path / "trade_log.csv"
    state_path = tmp_path / "closed_deal_state.json"

    monkeypatch.setattr(main, "LOG_DIR", tmp_path)
    monkeypatch.setattr(main, "TRADE_LOG_PATH", log_path)
    monkeypatch.setattr(main, "CLOSED_DEAL_STATE_PATH", state_path)

    deal_payload = [
        {
            "deal_id": 999,
            "symbol": "GOLD#",
            "action": "BUY",
            "entry_price": 2290.0,
            "exit_price": 2305.0,
            "lot": 0.2,
            "profit": 20.0,
            "holding_seconds": 1200,
            "time_utc": "2026-07-02T01:00:00+00:00",
        }
    ]
    monkeypatch.setattr(main, "get_closed_deals", lambda symbol, since: deal_payload)

    first = main.sync_closed_trades()
    second = main.sync_closed_trades()

    assert first == 1
    assert second == 0

    rows = _read_rows(log_path)
    assert len(rows) == 1
    assert rows[0]["deal_id"] == "999"
