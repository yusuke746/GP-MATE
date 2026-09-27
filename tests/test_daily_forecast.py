from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd

from scripts import run_daily_forecast as daily


def _frame() -> pd.DataFrame:
    dates = pd.bdate_range("2026-05-01", periods=80)
    close = np.linspace(100.0, 120.0, len(dates))
    return pd.DataFrame(
        {
            "time": (dates.astype("int64") // 10**9).astype(int),
            "open": close - 0.2,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "tick_volume": [100] * 79 + [1],
        }
    )


def test_parse_probabilities_rejects_invalid_values() -> None:
    parsed, error = daily.parse_probabilities({"p_up_1d": 0.6, "p_up_10d": 0.7, "p_highvol_10d": 0.4})
    assert parsed == {"p_up_1d": 0.6, "p_up_10d": 0.7, "p_highvol_10d": 0.4}
    assert error == ""
    parsed, error = daily.parse_probabilities({"p_up_1d": 1.2, "p_up_10d": 0.7, "p_highvol_10d": 0.4})
    assert parsed is None and "outside_0_1" in error


def test_load_confirmed_d1_drops_forming_bar() -> None:
    mt5 = Mock()
    mt5.TIMEFRAME_D1 = 1
    mt5.symbol_select.return_value = True
    mt5.copy_rates_from_pos.return_value = _frame().to_records(index=False)
    frame = daily.load_confirmed_d1(mt5, "GOLD#")
    assert len(frame) == 79
    assert float(frame["close"].iloc[-1]) != 120.0


def test_run_writes_once_and_deduplicates(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "daily_forecasts.jsonl"
    frame = _frame().iloc[:-1].copy()
    frame["bar_date"] = pd.to_datetime(frame["time"], unit="s").dt.date.astype(str)
    monkeypatch.setattr(daily, "universe", lambda: ["GOLD#", "US500Cash#"])
    monkeypatch.setattr(daily, "load_confirmed_d1", lambda mt5, symbol, force=False: frame.copy())
    monkeypatch.setattr(daily, "fetch_news_with_meta", lambda **kwargs: ([{"title": "Fed update"}], {"feeds_live": 1}))

    result = SimpleNamespace(
        ok=True,
        payload={"p_up_1d": 0.55, "p_up_10d": 0.6, "p_highvol_10d": 0.4, "key_reason": "材料は中立"},
        model="model",
        error="",
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )
    client = Mock()
    client.call_json.return_value = result

    first = daily.run(Mock(), client, "model", output=output)
    second = daily.run(Mock(), client, "model", output=output)

    assert first == {"symbols": 2, "written": 2, "skipped": 0, "failed": 0, "news_count": 1}
    assert second == {"symbols": 2, "written": 0, "skipped": 2, "failed": 0, "news_count": 1}
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    assert client.call_json.call_count == 2
    assert all(row["normal_vol_60d"] > 0 for row in rows)