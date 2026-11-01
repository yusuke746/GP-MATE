from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

import main
import shadow
from analysis import shadow_store
from analysis.shadow_eval import format_shadow_report, summarize_plans
from analysis.shadow_store import build_plan_record, next_slot_utc
from instruments import EURUSD, GOLD, USDJPY, instrument_for, parse_symbol_list


def test_instrument_lookup_ignores_broker_suffixes_and_falls_back_to_gold() -> None:
    assert instrument_for("GOLD#") is GOLD and instrument_for("XAUUSD.r") is GOLD and instrument_for("gold") is GOLD
    assert instrument_for("USDJPY#") is USDJPY and instrument_for("usdjpym") is USDJPY
    assert instrument_for("EURUSD.pro") is EURUSD
    assert instrument_for("GBPUSD") is GOLD  # unknown: gold profile keeps the pipeline running
    assert parse_symbol_list("USDJPY, eurusd, FOO, usdjpy") == ("USDJPY", "EURUSD")
    assert parse_symbol_list("") == ()
    assert USDJPY.structure_buffer < GOLD.structure_buffer and EURUSD.round_step_minor == 0.005


def test_analyst_prompts_are_filled_per_instrument_and_default_to_gold() -> None:
    from agents import macro_analyst, panel_debate, sentiment, technical

    for module, fn in ((technical, technical.system_prompt_for), (sentiment, sentiment.system_prompt_for), (macro_analyst, macro_analyst.system_prompt_for)):
        assert fn(None) == module.SYSTEM_PROMPT and "GOLD(XAU/USD)" in module.SYSTEM_PROMPT
        filled = fn(USDJPY)
        assert "ドル円(USD/JPY)" in filled and "GOLD" not in filled and "__" not in filled
    assert "金価格" in sentiment.SYSTEM_PROMPT and "金価格" not in sentiment.system_prompt_for(EURUSD)
    assert "ユーロ高・ドル安" in macro_analyst.system_prompt_for(EURUSD)
    assert panel_debate.panel_system_prompt_for(None) == panel_debate.PANEL_SYSTEM_PROMPT
    assert "ドル円(USD/JPY)" in panel_debate.panel_system_prompt_for(USDJPY) and "__ROLE__" in panel_debate.panel_system_prompt_for(USDJPY)


def test_build_market_reports_passes_instrument_to_news_calendar_and_analysts(monkeypatch: pytest.MonkeyPatch) -> None:
    rates = pd.DataFrame([{"close": 150.0, "high": 150.2, "low": 149.8, "rsi_14": 50.0, "macd": 0.0, "macd_signal": 0.0, "macd_hist": 0.0, "bb_upper": 151.0, "bb_mid": 150.0, "bb_lower": 149.0, "atr_14": 0.3, "recent_high_20": 151.0, "recent_low_20": 149.0}] * main.MIN_CONFIRMED_BARS)
    seen: dict[str, Any] = {}
    monkeypatch.setattr(main, "get_rates", lambda symbol, tf, bars: seen.setdefault("symbols", set()).add(symbol) or rates)
    monkeypatch.setattr(main, "add_indicators", lambda df: df)
    monkeypatch.setattr(main, "fetch_news_with_meta", lambda hours, keywords=None: seen.__setitem__("keywords", keywords) or ([], {"feeds_total": 1, "feeds_live": 1}))
    monkeypatch.setattr(main, "get_macro_data", lambda force_refresh=False: {"_meta": {"ok": True}})
    monkeypatch.setattr(main, "build_macro_inputs", lambda macro_data, **kwargs: seen.__setitem__("calendar_fetcher", kwargs.get("calendar_fetcher")) or macro_data)
    monkeypatch.setattr(main, "fetch_calendar_events", lambda currencies=None: seen.__setitem__("currencies", currencies) or [])
    monkeypatch.setattr(main, "analyze_macro_environment", lambda data, instrument=None: seen.__setitem__("macro_instrument", instrument) or {"macro_bias": "NEUTRAL", "_meta": {"ok": True, "usage": {}}})
    monkeypatch.setattr(main, "analyze_technical", lambda payload, instrument=None: seen.__setitem__("technical_instrument", instrument) or {"signal": "NEUTRAL", "_meta": {"usage": {}}})
    monkeypatch.setattr(main, "analyze_sentiment", lambda news, instrument=None: seen.__setitem__("sentiment_instrument", instrument) or {"score": 0.0, "_meta": {"usage": {}}})
    monkeypatch.setattr(main, "sentiment_for_feed_state", lambda news, meta: None)

    main._build_market_reports(datetime(2026, 10, 20, 12, 0, tzinfo=UTC), instrument=USDJPY, symbol="USDJPY#")

    assert seen["symbols"] == {"USDJPY#"}
    assert "yen" in seen["keywords"] and "gold" not in seen["keywords"]
    seen["calendar_fetcher"]()
    assert seen["currencies"] == {"USD", "JPY"}
    assert seen["macro_instrument"] is USDJPY and seen["technical_instrument"] is USDJPY and seen["sentiment_instrument"] is USDJPY

    # The gold path is unchanged: default keywords, default calendar, analysts called without an instrument.
    seen.clear()
    monkeypatch.setattr(main, "fetch_news_with_meta", lambda hours: seen.__setitem__("default_news", True) or ([], {"feeds_total": 1, "feeds_live": 1}))
    monkeypatch.setattr(main, "build_macro_inputs", lambda macro_data: seen.__setitem__("default_macro", True) or macro_data)
    monkeypatch.setattr(main, "analyze_macro_environment", lambda data: {"macro_bias": "NEUTRAL", "_meta": {"ok": True, "usage": {}}})
    monkeypatch.setattr(main, "analyze_technical", lambda payload: {"signal": "NEUTRAL", "_meta": {"usage": {}}})
    monkeypatch.setattr(main, "analyze_sentiment", lambda news: {"score": 0.0, "_meta": {"usage": {}}})
    main._build_market_reports(datetime(2026, 10, 20, 12, 0, tzinfo=UTC))
    assert seen == {"symbols": {main.SYMBOL}, "default_news": True, "default_macro": True}


def test_round_numbers_scale_with_the_instrument() -> None:
    assert 150.0 in main._build_round_numbers(150.2, instrument=USDJPY) and 150.5 in main._build_round_numbers(150.2, instrument=USDJPY)
    assert max(main._build_round_numbers(150.2, instrument=USDJPY)) <= 156.5
    assert 1.085 in main._build_round_numbers(1.0843, instrument=EURUSD)
    assert 4100.0 in main._build_round_numbers(4123.0)  # gold default unchanged


def _trader(action: str, **extra: Any) -> dict[str, Any]:
    return {"action": action, "reasoning": "r", "directional_bias": "BEARISH", "trade_horizon": "INTRADAY", **extra}


def test_build_plan_record_market_pending_and_skips() -> None:
    market = build_plan_record(symbol="USDJPY#", instrument_key="USDJPY", ts_utc="2026-10-20T12:00:00+00:00", source="shadow", trader_report=_trader("SELL"), risk_plan={"ok": True, "sl": 150.5, "tp": 149.4, "net_rr": 1.7, "sl_source": "suggested", "tp_source": "suggested"}, pending=None, pending_status="", final_action="SELL", current_price=150.0, atr_used=0.3, atr_timeframe="H1", trade_horizon="INTRADAY", valid_until_utc=None, placed=True)
    assert market["plan_type"] == "MARKET" and market["status"] == "OPEN" and market["order_type"] == "SELL"
    assert market["entry"] == 150.0 and market["sl_distance"] == 0.5 and market["tp_distance"] == 0.6 and market["valid_until_utc"] is None

    pending = build_plan_record(symbol="USDJPY#", instrument_key="USDJPY", ts_utc="t", source="shadow", trader_report=_trader("HOLD"), risk_plan={"ok": True, "sl": 150.9, "tp": 149.8, "net_rr": 1.9}, pending={"type": "SELL_LIMIT", "price": 150.5}, pending_status="placed", final_action="HOLD", current_price=150.0, atr_used=0.3, atr_timeframe="H1", trade_horizon="INTRADAY", valid_until_utc="2026-10-20T13:30:00+00:00", placed=True)
    assert pending["plan_type"] == "PENDING" and pending["status"] == "OPEN" and pending["order_type"] == "SELL_LIMIT" and pending["entry"] == 150.5

    skipped = build_plan_record(symbol="USDJPY#", instrument_key="USDJPY", ts_utc="t", source="shadow", trader_report=_trader("HOLD"), risk_plan={"ok": False, "reason": "low_rr"}, pending={"type": "SELL_LIMIT", "price": 150.5}, pending_status="skipped_low_rr", final_action="HOLD", current_price=150.0, atr_used=0.3, atr_timeframe="H1", trade_horizon="INTRADAY", valid_until_utc=None, placed=False)
    assert skipped["plan_type"] == "NONE" and skipped["status"] == "NO_PLAN" and skipped["skip_reason"] == "skipped_low_rr" and skipped["entry"] == 150.5

    nothing = build_plan_record(symbol="GOLD#", instrument_key="XAUUSD", ts_utc="t", source="production", trader_report=_trader("HOLD"), risk_plan={}, pending=None, pending_status="none_proposed", final_action="HOLD", current_price=4100.0, atr_used=12.0, atr_timeframe="H1", trade_horizon="INTRADAY", valid_until_utc=None, placed=False)
    assert nothing["status"] == "NO_PLAN" and nothing["skip_reason"] == "none_proposed"
    blocked = build_plan_record(symbol="GOLD#", instrument_key="XAUUSD", ts_utc="t", source="production", trader_report=_trader("BUY"), risk_plan={"ok": False, "reason": "risk_cap", "sl": 1.0, "tp": 2.0}, pending=None, pending_status="", final_action="HOLD", current_price=4100.0, atr_used=12.0, atr_timeframe="H1", trade_horizon="INTRADAY", valid_until_utc=None, placed=False)
    assert blocked["status"] == "NO_PLAN" and blocked["skip_reason"] == "risk_cap"


def _bars(start: datetime, rows: list[tuple[float, float, float]]) -> list[dict[str, Any]]:
    return [{"time": start + timedelta(hours=i), "high": h, "low": lo, "close": c} for i, (h, lo, c) in enumerate(rows)]


def test_resolve_plan_pending_fill_then_tp_sl_ambiguous_and_not_filled() -> None:
    t0 = datetime(2026, 10, 20, 12, 0, tzinfo=UTC)  # Tuesday
    plan = {"plan_type": "PENDING", "order_type": "SELL_LIMIT", "entry": 150.5, "sl": 150.9, "tp": 149.7, "valid_until_utc": (t0 + timedelta(hours=4)).isoformat()}
    # bar0 no touch, bar1 fills (high 150.55), bar2 drifts, bar3 hits TP
    bars = _bars(t0, [(150.3, 149.9, 150.1), (150.55, 150.0, 150.2), (150.6, 150.1, 150.3), (150.4, 149.6, 149.8)])
    label = shadow.resolve_plan(plan, bars, t0 + timedelta(hours=6))
    assert label["outcome"] == "TP" and label["r_result"] == 2.0 and label["bars_held"] == 3
    assert label["filled_at_utc"] == (t0 + timedelta(hours=1)).isoformat() and label["mfe_r"] >= 2.0 and label["mae_r"] == 0.25

    sl_bars = _bars(t0, [(150.55, 150.0, 150.2), (151.0, 150.2, 150.9)])
    assert shadow.resolve_plan(plan, sl_bars, t0 + timedelta(hours=6))["outcome"] == "SL"
    assert shadow.resolve_plan(plan, sl_bars, t0 + timedelta(hours=6))["r_result"] == -1.0

    both = _bars(t0, [(150.55, 150.0, 150.2), (151.0, 149.6, 150.0)])
    assert shadow.resolve_plan(plan, both, t0 + timedelta(hours=6))["outcome"] == "AMBIGUOUS_SL"

    never = _bars(t0, [(150.3, 149.9, 150.1)] * 6)
    assert shadow.resolve_plan(plan, never, t0 + timedelta(hours=2)) is None  # still valid
    assert shadow.resolve_plan(plan, never, t0 + timedelta(hours=6))["outcome"] == "NOT_FILLED"
    # a touch AFTER the validity window does not count
    late = _bars(t0, [(150.3, 149.9, 150.1)] * 4 + [(150.7, 150.0, 150.2)])
    assert shadow.resolve_plan(plan, late, t0 + timedelta(hours=8))["outcome"] == "NOT_FILLED"

    running = _bars(t0, [(150.55, 150.0, 150.2), (150.6, 150.1, 150.3)])
    assert shadow.resolve_plan(plan, running, t0 + timedelta(hours=3)) is None


def test_resolve_plan_market_fills_at_first_bar_and_flattens_before_the_weekend() -> None:
    friday = datetime(2026, 10, 23, 19, 0, tzinfo=UTC)  # 15:00 NY (EDT)
    plan = {"plan_type": "MARKET", "order_type": "BUY", "entry": 100.0, "sl": 99.0, "tp": 102.0, "valid_until_utc": None}
    bars = _bars(friday, [(100.5, 99.8, 100.4), (100.6, 100.0, 100.3), (100.7, 100.1, 100.6)])  # third bar = 17:00 NY -> flat
    label = shadow.resolve_plan(plan, bars, friday + timedelta(hours=5))
    assert label["outcome"] == "WEEKEND_FLAT" and label["r_result"] == 0.6 and label["bars_held"] == 3

    timeout_bars = _bars(datetime(2026, 10, 19, 0, 0, tzinfo=UTC), [(100.5, 99.5, 100.2)] * (shadow.MAX_HOLD_BARS + 2))
    label = shadow.resolve_plan(plan, timeout_bars, datetime(2026, 10, 23, 0, 0, tzinfo=UTC))
    assert label["outcome"] == "TIMEOUT" and label["bars_held"] == shadow.MAX_HOLD_BARS and label["r_result"] == 0.2


def test_resolve_shadow_plans_labels_open_rows_from_bars(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shadow_store, "LOG_DIR", tmp_path)
    t0 = datetime(2026, 10, 20, 12, 0, tzinfo=UTC)
    shadow_store.append_plan({"symbol": "USDJPY#", "instrument": "USDJPY", "ts_utc": t0.isoformat(), "plan_type": "PENDING", "order_type": "BUY_LIMIT", "entry": 149.8, "sl": 149.5, "tp": 150.4, "status": "OPEN", "valid_until_utc": (t0 + timedelta(hours=3)).isoformat()})
    shadow_store.append_plan({"symbol": "USDJPY#", "instrument": "USDJPY", "ts_utc": t0.isoformat(), "plan_type": "NONE", "status": "NO_PLAN"})
    frame = pd.DataFrame(
        [
            {"time": t0 - timedelta(hours=2), "open": 150.0, "high": 150.2, "low": 149.9, "close": 150.0},
            {"time": t0, "open": 150.0, "high": 150.1, "low": 149.75, "close": 149.9},
            {"time": t0 + timedelta(hours=1), "open": 149.9, "high": 150.5, "low": 149.85, "close": 150.4},
            {"time": t0 + timedelta(hours=2), "open": 150.4, "high": 150.5, "low": 150.3, "close": 150.4},
        ]
    )
    monkeypatch.setattr(shadow, "bar_times_to_utc", lambda f: f)
    monkeypatch.setattr(shadow, "drop_forming_bars", lambda f, now, tf: f)

    counts = shadow.resolve_shadow_plans(now=t0 + timedelta(hours=5), rates_fetcher=lambda symbol, tf, count: frame)

    assert counts == {"total": 2, "open": 1, "resolved": 1, "waiting": 0, "errors": 0}
    rows = shadow_store.read_plans()
    assert rows[0]["status"] == "RESOLVED" and rows[0]["outcome"] == "TP" and rows[0]["r_result"] == 2.0
    assert rows[1]["status"] == "NO_PLAN"
    # idempotent
    assert shadow.resolve_shadow_plans(now=t0 + timedelta(hours=6), rates_fetcher=lambda *a: frame)["open"] == 0


def test_run_shadow_once_records_a_pending_plan_without_trading(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shadow_store, "LOG_DIR", tmp_path)
    frame = pd.DataFrame([{"close": 150.0, "atr_14": 0.3}] * 3)
    h4 = pd.DataFrame([{"close": 150.0, "atr_14": 0.8}] * 3)
    monkeypatch.setattr(shadow, "resolve_broker_symbol", lambda candidates: "USDJPY#")
    monkeypatch.setattr(main, "_build_market_reports", lambda now, instrument=None, symbol=None: (frame, h4, frame, [], {"macro_bias": "BEARISH", "_meta": {"usage": {}}}, {"signal": "SELL", "_meta": {"usage": {}}}, {"score": -0.5, "_meta": {"usage": {}}}, {}))
    monkeypatch.setattr(main, "_insufficient_bars_reason", lambda meta: "")
    monkeypatch.setattr(main, "_is_past_pending_placement_cutoff", lambda reference=None: False)
    debate = {"regime_summary": {"regime": "TREND", "direction_if_trend": "DOWN", "entry_style": "LIMIT_PULLBACK", "panel_consensus_type": "UNANIMOUS", "source": "judge"}, "_meta": {"ok": True, "usage": {}}}
    monkeypatch.setattr(main, "_build_debate_and_decision_reports", lambda t, s, m, symbol=None, instrument=None: ({"should_debate": True}, debate, None))
    monkeypatch.setattr(shadow, "get_spread_price", lambda symbol: 0.01)
    seen: dict[str, Any] = {}

    def _decide(technical, sentiment, debate_report, macro_report=None, recent_context=None, **kwargs):
        seen.update(kwargs)
        return {"action": "HOLD", "reasoning": "戻り売り", "directional_bias": "BEARISH", "trade_horizon": "INTRADAY", "pending_orders": [{"type": "SELL_LIMIT", "price": 150.3, "tp": 149.6, "sl": 150.7}], "_meta": {"usage": {"total_tokens": 10}}}

    monkeypatch.setattr(shadow, "decide_trade", _decide)
    monkeypatch.setattr(shadow, "resolve_shadow_plans", lambda now=None, rates_fetcher=None: {"total": 1})

    result = shadow.run_shadow_once("USDJPY", now=datetime(2026, 10, 20, 12, 0, tzinfo=UTC))

    assert result["ok"] is True and seen["instrument"] is USDJPY and seen["symbol"] == "USDJPY#" and seen["spread_usd"] == 0.01
    record = shadow_store.read_plans()[0]
    assert record["source"] == "shadow" and record["plan_type"] == "PENDING" and record["status"] == "OPEN"
    assert record["entry"] == 150.3 and record["sl"] > 150.3 > record["tp"] and record["regime"] == "TREND"
    assert record["valid_until_utc"] > record["ts_utc"] and record["atr_used"] == 0.3
    # SL used the instrument's buffer (0.05), not gold's 2.0: 150.7 + 0.05 = 150.75 within the H1 box [0.3, 0.45]
    assert record["sl"] == 150.75 and record["sl_source"] == "suggested"

    monkeypatch.setattr(shadow, "shadow_instruments", lambda: [USDJPY])
    cycle = shadow.run_shadow_cycle(now=datetime(2026, 10, 20, 12, 0, tzinfo=UTC))
    assert cycle["USDJPY"]["ok"] is True and cycle["resolve"] == {"total": 1}


def test_run_shadow_once_fails_safely_when_symbol_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shadow, "resolve_broker_symbol", lambda candidates: None)
    assert shadow.run_shadow_once("EURUSD")["ok"] is False
    assert shadow.run_shadow_once("GBPJPY")["ok"] is False


def test_next_slot_utc_picks_the_following_slot_across_midnight() -> None:
    slots = ((3, 0), (8, 0), (20, 0))
    now = datetime(2026, 10, 20, 12, 30, tzinfo=UTC)  # 08:30 NY -> next is 20:00 NY = 00:00 UTC next day
    assert next_slot_utc(now, slots, main.MARKET_TZ) == datetime(2026, 10, 21, 0, 0, tzinfo=UTC)
    late = datetime(2026, 10, 21, 1, 0, tzinfo=UTC)  # 21:00 NY -> 03:00 NY = 07:00 UTC
    assert next_slot_utc(late, slots, main.MARKET_TZ) == datetime(2026, 10, 21, 7, 0, tzinfo=UTC)


def test_resolve_broker_symbol_uses_the_broker_variant(monkeypatch: pytest.MonkeyPatch) -> None:
    from data import mt5_client

    class _Mt5:
        def symbols_get(self):
            return [SimpleNamespace(name=n) for n in ("GOLD#", "USDJPY#", "USDJPYm", "EURUSD#")]

        def symbol_info(self, name):
            return object() if name != "USDJPY#" else None  # the shortest variant is unavailable here

    monkeypatch.setattr(mt5_client, "mt5", _Mt5())
    monkeypatch.setattr(mt5_client, "connect", lambda: True)
    monkeypatch.setattr(mt5_client, "disconnect", lambda: None)
    assert mt5_client.resolve_broker_symbol(("USDJPY",)) == "USDJPYm"
    assert mt5_client.resolve_broker_symbol(("EURUSD",)) == "EURUSD#"
    assert mt5_client.resolve_broker_symbol(("GBPJPY",)) is None


def test_scheduler_shadow_slot_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import run_breakeven_monitor  # noqa: F401  (import path sanity)
    from scripts import run_scheduler

    monkeypatch.setattr(run_scheduler, "SHADOW_SYMBOLS", ())
    assert run_scheduler._shadow_slot_due() is False
    monkeypatch.setattr(run_scheduler, "SHADOW_SYMBOLS", ("USDJPY",))
    monkeypatch.setattr(run_scheduler, "SHADOW_JUDGMENT_TIMES_NY", ())
    assert run_scheduler._shadow_slot_due() is True
    monkeypatch.setattr(run_scheduler, "SHADOW_JUDGMENT_TIMES_NY", ((8, 0),))
    assert run_scheduler._shadow_slot_due(datetime(2026, 10, 20, 8, 3, tzinfo=main.MARKET_TZ)) is True
    assert run_scheduler._shadow_slot_due(datetime(2026, 10, 20, 9, 30, tzinfo=main.MARKET_TZ)) is False


def test_summary_and_report_over_mixed_plans() -> None:
    rows = [
        {"instrument": "USDJPY", "plan_type": "PENDING", "status": "RESOLVED", "outcome": "TP", "r_result": 1.8, "mfe_r": 1.9, "net_rr": 1.7, "trade_horizon": "INTRADAY"},
        {"instrument": "USDJPY", "plan_type": "PENDING", "status": "RESOLVED", "outcome": "SL", "r_result": -1.0, "mfe_r": 0.4, "net_rr": 1.6, "trade_horizon": "INTRADAY"},
        {"instrument": "USDJPY", "plan_type": "PENDING", "status": "RESOLVED", "outcome": "NOT_FILLED", "r_result": 0.0, "net_rr": 1.6, "trade_horizon": "SWING"},
        {"instrument": "USDJPY", "plan_type": "NONE", "status": "NO_PLAN", "skip_reason": "skipped_low_rr"},
        {"instrument": "XAUUSD", "plan_type": "MARKET", "status": "OPEN", "net_rr": 2.0, "trade_horizon": "INTRADAY"},
    ]
    summary = summarize_plans(rows)
    assert summary["USDJPY"]["judgments"] == 4 and summary["USDJPY"]["plans"] == 3 and summary["USDJPY"]["filled"] == 2
    assert summary["USDJPY"]["fill_rate"] == round(2 / 3, 3) and summary["USDJPY"]["win_rate"] == 0.5 and summary["USDJPY"]["expectancy_r"] == 0.4
    assert summary["USDJPY"]["reached_1r_rate"] == 0.5 and summary["USDJPY"]["skip_reasons"] == {"skipped_low_rr": 1}
    assert summary["XAUUSD"]["open"] == 1 and summary["XAUUSD"]["expectancy_r"] is None
    text = format_shadow_report(summary)
    assert "USDJPY" in text and "XAUUSD" in text and "exp_R" in text
    assert "(no plans recorded yet)" in format_shadow_report({})


def test_run_once_records_gold_judgment_as_a_production_plan(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.test_main import _action_aware_risk_plan, _patch_run_once_common, _trader_stub

    _patch_run_once_common(monkeypatch, tmp_path)
    _action_aware_risk_plan(monkeypatch)
    monkeypatch.setattr(shadow_store, "LOG_DIR", tmp_path)
    monkeypatch.setattr(main, "send_order", lambda symbol, action, lot, sl, tp: {"success": True, "retcode": "0", "order": 1})
    monkeypatch.setattr(main, "decide_trade", _trader_stub(trade_horizon="INTRADAY"))

    result = main.run_once(baseline_spread=10.0)

    assert result["action"] == "BUY"
    rows = shadow_store.read_plans()
    assert len(rows) == 1 and rows[0]["source"] == "production" and rows[0]["instrument"] == "XAUUSD"
    assert rows[0]["plan_type"] == "MARKET" and rows[0]["status"] == "OPEN" and rows[0]["placed"] is True
    assert rows[0]["entry"] == 100.0 and rows[0]["sl"] == 95.0 and rows[0]["tp"] == 110.0
