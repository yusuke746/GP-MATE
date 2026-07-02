from __future__ import annotations

import csv
from pathlib import Path

import main
from data import mt5_client


def test_get_baseline_spread_uses_median(monkeypatch) -> None:
    samples = [10.0, 11.0, 9.0, 100.0, 10.0]
    it = iter(samples)

    monkeypatch.setattr(mt5_client, "get_spread", lambda symbol: next(it))
    baseline = mt5_client.get_baseline_spread("GOLD#", samples=len(samples), interval_sec=0.0)

    assert baseline == 10.0


def test_run_once_holds_when_baseline_calibration_fails(tmp_path: Path, monkeypatch) -> None:
    log_path = tmp_path / "trade_log.csv"
    monkeypatch.setattr(main, "LOG_DIR", tmp_path)
    monkeypatch.setattr(main, "TRADE_LOG_PATH", log_path)
    monkeypatch.setattr(main, "get_baseline_spread", lambda symbol, samples, interval_sec: None)

    result = main.run_once()

    assert result["action"] == "HOLD"
    assert result["filter_reason"] == "Baseline spread calibration failed"

    rows = list(csv.DictReader(log_path.open("r", encoding="utf-8")))
    assert rows[-1]["action"] == "HOLD"


def test_run_once_holds_when_spread_exceeds_multiplier(tmp_path: Path, monkeypatch) -> None:
    log_path = tmp_path / "trade_log.csv"
    monkeypatch.setattr(main, "LOG_DIR", tmp_path)
    monkeypatch.setattr(main, "TRADE_LOG_PATH", log_path)

    monkeypatch.setattr(main, "is_high_impact_soon", lambda minutes: False)
    monkeypatch.setattr(main, "get_positions", lambda symbol: [])

    def _dummy_df() -> object:
        import pandas as pd

        return pd.DataFrame(
            [
                {
                    "close": 2300.0,
                    "rsi_14": 55.0,
                    "macd": 1.0,
                    "macd_signal": 0.8,
                    "macd_hist": 0.2,
                    "bb_upper": 2310.0,
                    "bb_mid": 2300.0,
                    "bb_lower": 2290.0,
                    "atr_14": 10.0,
                    "recent_high_20": 2320.0,
                    "recent_low_20": 2280.0,
                }
            ]
        )

    monkeypatch.setattr(main, "get_rates", lambda symbol, tf, count: _dummy_df())
    monkeypatch.setattr(main, "add_indicators", lambda df: df)

    monkeypatch.setattr(main, "fetch_news", lambda hours=24: [])
    monkeypatch.setattr(
        main,
        "analyze_technical",
        lambda payload: {"signal": "BUY", "_meta": {"usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}},
    )
    monkeypatch.setattr(
        main,
        "analyze_sentiment",
        lambda items: {"score": 0.1, "_meta": {"usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}},
    )
    monkeypatch.setattr(
        main,
        "run_debate",
        lambda t, s: {"_meta": {"usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}},
    )
    monkeypatch.setattr(
        main,
        "decide_trade",
        lambda t, s, d: {
            "action": "BUY",
            "confidence": 0.9,
            "reasoning": "test",
            "risk_level": "LOW",
            "_meta": {"usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}},
        },
    )

    class _Account:
        balance = 500_000

    monkeypatch.setattr(main, "get_account_info", lambda: {"success": True, "data": _Account()})
    monkeypatch.setattr(
        main,
        "build_risk_plan",
        lambda action, entry_price, atr, balance_jpy: {
            "ok": True,
            "action": "BUY",
            "lot": 0.1,
            "sl": 2285.0,
            "tp": 2330.0,
        },
    )
    monkeypatch.setattr(main, "get_spread", lambda symbol: 25.0)
    monkeypatch.setattr(
        main,
        "send_order",
        lambda symbol, action, lot, sl, tp: {"success": True, "retcode": 0},
    )

    result = main.run_once(baseline_spread=10.0)

    assert result["action"] == "HOLD"
    assert result["allowed"] is False
    assert result["filter_reason"] == "Spread too high"
