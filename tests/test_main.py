from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import main
import pandas as pd
import pytest


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


def _patch_run_once_common(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    log_path = tmp_path / "trade_log.csv"
    monkeypatch.setattr(main, "LOG_DIR", tmp_path)
    monkeypatch.setattr(main, "TRADE_LOG_PATH", log_path)
    monkeypatch.setattr(main, "sync_closed_trades", lambda: 0)
    monkeypatch.setattr(main, "calc_today_risk_stats", lambda: (0, 0.0))
    monkeypatch.setattr(main, "is_high_impact_soon", lambda minutes: False)
    monkeypatch.setattr(main, "get_positions", lambda symbol: [])

    rates = pd.DataFrame(
        [
            {
                "close": 100.0,
                "rsi_14": 58.0,
                "macd": 1.0,
                "macd_signal": 0.5,
                "macd_hist": 0.5,
                "bb_upper": 110.0,
                "bb_mid": 100.0,
                "bb_lower": 90.0,
                "atr_14": 2.0,
                "recent_high_20": 111.0,
                "recent_low_20": 89.0,
            }
        ]
    )
    monkeypatch.setattr(main, "get_rates", lambda symbol, tf, bars: rates)
    monkeypatch.setattr(main, "add_indicators", lambda df: df)

    monkeypatch.setattr(main, "fetch_news", lambda hours: [])
    monkeypatch.setattr(main, "get_macro_data", lambda force_refresh=False: {})
    monkeypatch.setattr(
        main,
        "analyze_macro_environment",
        lambda macro_data: {"macro_bias": "NEUTRAL", "_meta": {"usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}},
    )
    monkeypatch.setattr(
        main,
        "analyze_technical",
        lambda payload: {
            "signal": "BUY",
            "trend": "UP",
            "alignment": "ALIGNED",
            "rsi_14": 58.0,
            "_meta": {"usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3}},
        },
    )
    monkeypatch.setattr(
        main,
        "analyze_sentiment",
        lambda news: {
            "score": 0.2,
            "sentiment": "BULLISH",
            "_meta": {"usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3}},
        },
    )
    monkeypatch.setattr(
        main,
        "decide_trade",
        lambda technical_report, sentiment_report, debate_report: {
            "action": "BUY",
            "confidence": 0.8,
            "reasoning": "test",
            "risk_level": "MID",
            "_meta": {
                "model": "decision-test",
                "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
            },
        },
    )

    monkeypatch.setattr(main, "get_spread", lambda symbol: 10.0)
    monkeypatch.setattr(main, "check_filters", lambda **kwargs: SimpleNamespace(ok=True, reason="OK"))
    monkeypatch.setattr(
        main,
        "get_account_info",
        lambda: {"success": True, "data": type("A", (), {"balance": 500000.0})()},
    )
    monkeypatch.setattr(
        main,
        "build_risk_plan",
        lambda action, entry_price, atr, balance_jpy: {
            "ok": True,
            "action": "BUY",
            "lot": 0.01,
            "sl": 95.0,
            "tp": 110.0,
        },
    )
    monkeypatch.setattr(main, "send_order", lambda symbol, action, lot, sl, tp: {"success": True, "retcode": "0"})
    return log_path


def _read_single_row(log_path: Path) -> dict[str, str]:
    rows = list(csv.DictReader(log_path.open("r", encoding="utf-8")))
    assert len(rows) == 1
    return rows[0]


def test_run_once_logs_debate_fields_when_executed(tmp_path: Path, monkeypatch) -> None:
    log_path = _patch_run_once_common(monkeypatch, tmp_path)

    monkeypatch.setattr(
        main,
        "should_execute_debate",
        lambda technical_report, sentiment_report, macro_report: {
            "should_debate": True,
            "reason": "議論実行（通常判定）",
            "technical_direction": "BUY",
            "sentiment_direction": "BULLISH",
            "macro_direction": "NEUTRAL",
            "alignment": "ALIGNED",
            "estimated_confidence": 0.61,
        },
    )
    monkeypatch.setattr(
        main,
        "run_debate_graph",
        lambda technical_report, sentiment_report, macro_report: {
            "judge_summary": {
                "conflicts": ["方向感"],
                "stronger_side": "bull",
                "confidence_shift": {"bull": [0.5, 0.62], "bear": [0.5, 0.48]},
            },
            "_meta": {
                "ok": True,
                "judge_ok": True,
                "judge_error": "",
                "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18},
            },
        },
    )

    result = main.run_once(baseline_spread=10.0)

    assert result["action"] == "BUY"
    row = _read_single_row(log_path)
    assert row["action"] == "BUY"
    assert row["debate_executed"] == "True"
    assert row["skip_reason"] == ""
    assert row["stronger_side"] == "bull"
    assert json.loads(row["conflicts"]) == ["方向感"]
    assert json.loads(row["confidence_shift"]) == {"bull": [0.5, 0.62], "bear": [0.5, 0.48]}
    assert row["debate_tokens"] == "18"
    assert row["judge_parse_ok"] == "True"
    assert row["judge_error"] == ""
    assert row["technical_direction"] == "BUY"
    assert row["estimated_confidence"] == "0.61"
    assert row["prompt_tokens"] != ""
    assert row["completion_tokens"] != ""
    assert row["total_tokens"] != ""


def test_run_once_logs_skip_reason_when_debate_skipped(tmp_path: Path, monkeypatch) -> None:
    log_path = _patch_run_once_common(monkeypatch, tmp_path)

    monkeypatch.setattr(
        main,
        "should_execute_debate",
        lambda technical_report, sentiment_report, macro_report: {
            "should_debate": False,
            "reason": "議論スキップ（明確なトレンドのため）",
            "technical_direction": "BUY",
            "sentiment_direction": "BULLISH",
            "macro_direction": "NEUTRAL",
            "alignment": "ALIGNED",
            "estimated_confidence": 0.74,
        },
    )
    monkeypatch.setattr(
        main,
        "build_skipped_debate_report",
        lambda reason: {
            "judge_summary": {
                "conflicts": [reason],
                "stronger_side": "neutral",
                "confidence_shift": {"bull": [], "bear": []},
            },
            "_meta": {
                "debate_executed": False,
                "skip_reason": reason,
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            },
        },
    )

    result = main.run_once(baseline_spread=10.0)

    assert result["action"] == "BUY"
    row = _read_single_row(log_path)
    assert row["debate_executed"] == "False"
    assert row["skip_reason"] == "議論スキップ（明確なトレンドのため）"
    assert row["stronger_side"] == "neutral"
    assert json.loads(row["conflicts"]) == ["議論スキップ（明確なトレンドのため）"]
    assert row["debate_tokens"] == "0"
    assert row["judge_parse_ok"] == ""


def test_run_once_logs_judge_parse_failure_flag(tmp_path: Path, monkeypatch) -> None:
    log_path = _patch_run_once_common(monkeypatch, tmp_path)

    monkeypatch.setattr(
        main,
        "should_execute_debate",
        lambda technical_report, sentiment_report, macro_report: {
            "should_debate": True,
            "reason": "議論実行（通常判定）",
            "technical_direction": "BUY",
            "sentiment_direction": "BEARISH",
            "macro_direction": "NEUTRAL",
            "alignment": "MIXED",
            "estimated_confidence": 0.58,
        },
    )
    monkeypatch.setattr(
        main,
        "run_debate_graph",
        lambda technical_report, sentiment_report, macro_report: {
            "judge_summary": {
                "conflicts": ["judge parse error"],
                "stronger_side": "neutral",
                "confidence_shift": {"bull": [0.5], "bear": [0.5]},
            },
            "_meta": {
                "ok": False,
                "judge_ok": False,
                "judge_error": "judge json parse error",
                "usage": {"prompt_tokens": 5, "completion_tokens": 4, "total_tokens": 9},
            },
        },
    )

    main.run_once(baseline_spread=10.0)

    row = _read_single_row(log_path)
    assert row["debate_executed"] == "True"
    assert row["judge_parse_ok"] == "False"
    assert row["judge_error"] == "judge json parse error"


def test_run_once_continues_when_debate_log_serialization_fails(tmp_path: Path, monkeypatch) -> None:
    log_path = _patch_run_once_common(monkeypatch, tmp_path)

    monkeypatch.setattr(
        main,
        "should_execute_debate",
        lambda technical_report, sentiment_report, macro_report: {
            "should_debate": True,
            "reason": "議論実行（通常判定）",
            "technical_direction": "BUY",
            "sentiment_direction": "MIXED",
            "macro_direction": "NEUTRAL",
            "alignment": "DIVERGENT",
            "estimated_confidence": 0.55,
        },
    )
    monkeypatch.setattr(
        main,
        "run_debate_graph",
        lambda technical_report, sentiment_report, macro_report: {
            "judge_summary": {
                "conflicts": {"non_serializable"},
                "stronger_side": "bull",
                "confidence_shift": {"bad": {1, 2, 3}},
            },
            "_meta": {
                "ok": True,
                "judge_ok": True,
                "judge_error": "",
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        },
    )

    result = main.run_once(baseline_spread=10.0)

    assert result["action"] == "BUY"
    row = _read_single_row(log_path)
    assert row["conflicts"] == "[]"
    assert row["confidence_shift"] == "{}"
