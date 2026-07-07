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


def test_run_once_uses_evaluate_position_when_position_exists(tmp_path: Path, monkeypatch) -> None:
    log_path = _patch_run_once_common(monkeypatch, tmp_path)
    monkeypatch.setattr(main, "get_positions", lambda symbol: [{"ticket": 123}])
    monkeypatch.setattr(
        main,
        "get_position_details",
        lambda symbol: [
            {
                "ticket": 123,
                "symbol": symbol,
                "type": "BUY",
                "volume": 0.01,
                "price_open": 100.0,
                "price_current": 101.0,
                "sl": 95.0,
                "tp": 110.0,
                "profit": 10.0,
            }
        ],
    )
    called = {"evaluate": 0, "close": 0, "decide": 0, "modify": 0}
    def _fake_evaluate(**kwargs):
        called["evaluate"] += 1
        assert kwargs["confidence_threshold"] == main.CLOSE_CONFIDENCE_THRESHOLD
        return {
            "action": "HOLD",
            "confidence": 0.7,
            "reasoning": "keep",
            "risk_level": "MID",
            "_meta": {"model": "eval-test", "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}},
        }

    monkeypatch.setattr(main, "evaluate_position", _fake_evaluate)
    monkeypatch.setattr(main, "close_position", lambda ticket: called.__setitem__("close", called["close"] + 1) or {"success": True, "retcode": 0})
    monkeypatch.setattr(main, "modify_sl", lambda ticket, new_sl: called.__setitem__("modify", called["modify"] + 1) or {"success": True, "retcode": 0})
    monkeypatch.setattr(
        main,
        "decide_trade",
        lambda technical_report, sentiment_report, debate_report: called.__setitem__("decide", called["decide"] + 1) or {
            "action": "BUY",
            "confidence": 0.8,
            "reasoning": "test",
            "risk_level": "MID",
            "_meta": {"model": "decision-test", "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}},
        },
    )

    result = main.run_once(baseline_spread=10.0)

    assert result["action"] == "HOLD"
    assert called["evaluate"] == 1
    assert called["close"] == 0
    assert called["decide"] == 0
    assert called["modify"] == 0
    row = _read_single_row(log_path)
    assert row["decision_model"] == "eval-test"
    assert row["breakeven_triggered"] == "False"
    assert row["breakeven_ticket"] == "123"
    assert row["breakeven_entry_price"] == "100.0"
    assert row["breakeven_initial_sl"] == "95.0"
    assert row["breakeven_current_price"] == "101.0"
    assert row["breakeven_reason"] == "NOT_TRIGGERED_OR_ALREADY_MOVED"


def test_run_once_moves_sl_to_breakeven_on_hold_at_1r(tmp_path: Path, monkeypatch) -> None:
    log_path = _patch_run_once_common(monkeypatch, tmp_path)
    monkeypatch.setattr(main, "get_positions", lambda symbol: [{"ticket": 321}])
    monkeypatch.setattr(
        main,
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
    monkeypatch.setattr(
        main,
        "evaluate_position",
        lambda **kwargs: {
            "action": "HOLD",
            "confidence": 0.9,
            "reasoning": "hold and protect",
            "risk_level": "LOW",
            "_meta": {"model": "eval-test", "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}},
        },
    )

    modify_calls: list[tuple[int, float]] = []

    def _fake_modify(ticket: int, new_sl: float) -> dict[str, object]:
        modify_calls.append((ticket, new_sl))
        return {"success": True, "retcode": 0}

    monkeypatch.setattr(main, "modify_sl", _fake_modify)
    monkeypatch.setattr(main, "close_position", lambda ticket: {"success": True, "retcode": 0})

    result = main.run_once(baseline_spread=10.0)

    assert result["action"] == "HOLD"
    assert modify_calls == [(321, 100.0 + main.BREAKEVEN_BUFFER)]
    row = _read_single_row(log_path)
    assert row["filter_reason"] == "Position hold + breakeven moved"
    assert row["order_success"] == "True"
    assert row["breakeven_triggered"] == "True"
    assert row["breakeven_new_sl"] == str(100.0 + main.BREAKEVEN_BUFFER)
    assert row["breakeven_time"] != ""
    assert row["breakeven_ticket"] == "321"
    assert row["breakeven_entry_price"] == "100.0"
    assert row["breakeven_initial_sl"] == "95.0"
    assert row["breakeven_trigger_price"] == "105.0"
    assert row["breakeven_current_price"] == "105.0"
    assert row["breakeven_modify_success"] == "True"
    assert row["breakeven_reason"] == "MOVED"


def test_run_once_closes_position_when_evaluate_position_returns_close(tmp_path: Path, monkeypatch) -> None:
    log_path = _patch_run_once_common(monkeypatch, tmp_path)
    monkeypatch.setattr(main, "get_positions", lambda symbol: [{"ticket": 456}])
    monkeypatch.setattr(
        main,
        "get_position_details",
        lambda symbol: [
            {
                "ticket": 456,
                "symbol": symbol,
                "type": "SELL",
                "volume": 0.02,
                "price_open": 100.0,
                "price_current": 99.0,
                "sl": 105.0,
                "tp": 95.0,
                "profit": 20.0,
            }
        ],
    )
    monkeypatch.setattr(
        main,
        "evaluate_position",
        lambda **kwargs: {
            "action": "CLOSE",
            "confidence": 0.82,
            "reasoning": "reverse strong",
            "risk_level": "MID",
            "_meta": {"model": "eval-test", "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}},
        },
    )
    close_calls: list[int] = []
    modify_calls: list[tuple[int, float]] = []
    monkeypatch.setattr(
        main,
        "close_position",
        lambda ticket: close_calls.append(ticket) or {"success": True, "retcode": 0, "deal": 999},
    )
    monkeypatch.setattr(
        main,
        "modify_sl",
        lambda ticket, new_sl: modify_calls.append((ticket, new_sl)) or {"success": True, "retcode": 0},
    )

    result = main.run_once(baseline_spread=10.0)

    assert result["action"] == "CLOSE"
    assert close_calls == [456]
    assert modify_calls == []
    row = _read_single_row(log_path)
    assert row["action"] == "CLOSE"
    assert row["order_success"] == "True"
    assert row["position_direction"] == "SELL"
    assert row["technical_signal"] == "BUY"
    assert row["evaluate_action"] == "CLOSE"
    assert row["evaluate_confidence"] == "0.82"
    assert row["evaluate_reasoning"] == "reverse strong"
    assert row["evaluate_reasoning_len"] == str(len("reverse strong"))
    assert row["breakeven_triggered"] == "False"
    assert row["breakeven_reason"] == "NOT_HOLD_ACTION"


def test_run_once_same_direction_position_holds_without_close(tmp_path: Path, monkeypatch) -> None:
    log_path = _patch_run_once_common(monkeypatch, tmp_path)
    monkeypatch.setattr(main, "get_positions", lambda symbol: [{"ticket": 777}])
    monkeypatch.setattr(
        main,
        "get_position_details",
        lambda symbol: [
            {
                "ticket": 777,
                "symbol": symbol,
                "type": "BUY",
                "volume": 0.03,
                "price_open": 100.0,
                "price_current": 102.0,
                "sl": 96.0,
                "tp": 108.0,
                "profit": 35.0,
            }
        ],
    )

    called = {"close": 0}

    def _fake_evaluate(**kwargs):
        assert kwargs["position_context"]["type"] == "BUY"
        assert kwargs["technical_report"]["signal"] == "BUY"
        return {
            "action": "HOLD",
            "confidence": 0.9,
            "reasoning": "同方向のため保持",
            "risk_level": "LOW",
            "_meta": {"model": "eval-test", "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}},
        }

    monkeypatch.setattr(main, "evaluate_position", _fake_evaluate)
    monkeypatch.setattr(
        main,
        "close_position",
        lambda ticket: called.__setitem__("close", called["close"] + 1) or {"success": True, "retcode": 0},
    )

    result = main.run_once(baseline_spread=10.0)

    assert result["action"] == "HOLD"
    assert called["close"] == 0
    row = _read_single_row(log_path)
    assert row["action"] == "HOLD"


def test_run_once_without_position_keeps_decide_trade_flow(tmp_path: Path, monkeypatch) -> None:
    log_path = _patch_run_once_common(monkeypatch, tmp_path)
    called = {"evaluate": 0, "decide": 0}
    monkeypatch.setattr(
        main,
        "evaluate_position",
        lambda **kwargs: called.__setitem__("evaluate", called["evaluate"] + 1) or {"action": "HOLD"},
    )
    monkeypatch.setattr(
        main,
        "decide_trade",
        lambda technical_report, sentiment_report, debate_report: called.__setitem__("decide", called["decide"] + 1) or {
            "action": "BUY",
            "confidence": 0.8,
            "reasoning": "test",
            "risk_level": "MID",
            "_meta": {"model": "decision-test", "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}},
        },
    )

    result = main.run_once(baseline_spread=10.0)

    assert result["action"] == "BUY"
    assert called["evaluate"] == 0
    assert called["decide"] == 1
    row = _read_single_row(log_path)
    assert row["action"] == "BUY"
