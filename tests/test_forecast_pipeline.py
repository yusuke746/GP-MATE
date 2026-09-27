from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from unittest.mock import Mock

import pandas as pd
import pytest

import main
from analysis import forecast_pipeline as fp
from analysis import forecast_store
from data import mt5_client

NOW = datetime(2026, 9, 9, 14, 1, 30, tzinfo=UTC)  # Wednesday, 90s after the 14:00 bar opened


def _h1_frame(end_open: datetime, bars: int = 30, close: float = 4363.5, atr: float = 21.27) -> pd.DataFrame:
    times = [end_open - timedelta(hours=bars - 1 - i) for i in range(bars)]
    rows = []
    for i, t in enumerate(times):
        c = close - (bars - 1 - i) * 0.5  # gently rising into the forecast bar
        rows.append({"time": t, "open": c - 0.2, "high": c + 3.0, "low": c - 3.0, "close": c, "tick_volume": 1,
                     "atr_14": atr, "rsi_14": 50.0, "macd": 0.0, "macd_signal": 0.0, "macd_hist": 0.0,
                     "bb_upper": c + 20, "bb_mid": c, "bb_lower": c - 20, "adx_14": 20.0,
                     "recent_high_20": c + 30, "recent_low_20": c - 30})
    frame = pd.DataFrame(rows)
    frame["time"] = pd.to_datetime(frame["time"], utc=True)
    # The last row is the FORMING bar (opened at end_open); MT5 returns it too.
    frame.loc[frame.index[-1], "close"] = 9999.0  # must never be used as P0
    return frame


def _forecaster_client(p_up=0.55, p_down=0.25, p_timeout=0.2) -> Mock:
    result = Mock()
    result.ok = True
    result.payload = {"p_up": p_up, "p_down": p_down, "p_timeout": p_timeout, "key_reason": "支持帯反発"}
    result.model = "gpt-5.6-terra"
    result.error = ""
    result.usage = Mock(prompt_tokens=6000, completion_tokens=120, total_tokens=6120)
    client = Mock()
    client.call_json.return_value = result
    return client


def _patch_common(monkeypatch: pytest.MonkeyPatch, tmp_path, now: datetime = NOW) -> None:
    monkeypatch.setattr(forecast_store, "LOG_DIR", tmp_path)
    monkeypatch.setattr(mt5_client, "SERVER_TZ", None)
    forming_open = now.replace(minute=0, second=0, microsecond=0)
    frames = {
        "H1": _h1_frame(forming_open),
        "H4": _h1_frame(forming_open.replace(hour=(forming_open.hour // 4) * 4), bars=10),
        "D1": _h1_frame(forming_open.replace(hour=0), bars=10),
    }
    monkeypatch.setattr(fp, "get_rates", lambda symbol, tf, count: frames[tf].copy())
    monkeypatch.setattr(fp, "add_indicators", lambda df: df)
    monkeypatch.setattr(
        fp,
        "build_reports",
        lambda frames, use_debate=False: {
            "technical": {"signal": "SELL", "_meta": {"usage": {"prompt_tokens": 800, "completion_tokens": 200, "total_tokens": 1000}}},
            "sentiment": {"score": -0.6, "_meta": {"usage": {"prompt_tokens": 4000, "completion_tokens": 300, "total_tokens": 4300}}},
            "macro": {"macro_bias": "NEUTRAL", "_meta": {"usage": {"prompt_tokens": 2300, "completion_tokens": 700, "total_tokens": 3000}}},
            "debate": None,
            "news_count": 15,
        },
    )


def test_drop_forming_bars_keeps_only_closed_bars() -> None:
    frame = _h1_frame(datetime(2026, 9, 9, 14, 0, tzinfo=UTC), bars=5)
    kept = fp.drop_forming_bars(frame, NOW, "H1")
    assert len(kept) == 4
    assert pd.Timestamp(kept["time"].iloc[-1]) == pd.Timestamp("2026-09-09T13:00:00Z")


def test_bar_times_to_utc_reinterprets_server_wall_clock(monkeypatch) -> None:
    from zoneinfo import ZoneInfo

    monkeypatch.setattr(mt5_client, "SERVER_TZ", ZoneInfo("Europe/Athens"))
    frame = pd.DataFrame({"time": pd.to_datetime(["2026-09-09T16:00:00Z"]), "close": [1.0]})
    converted = fp.bar_times_to_utc(frame)
    assert pd.Timestamp(converted["time"].iloc[0]) == pd.Timestamp("2026-09-09T13:00:00Z")  # EEST = UTC+3


def test_session_filter_parsing_and_window() -> None:
    assert fp.session_allows(NOW, "") is True
    # 14:01 UTC = 10:01 NY (EDT)
    assert fp.session_allows(NOW, "08:00-16:00") is True
    assert fp.session_allows(NOW, "11:00-16:00") is False
    assert fp.session_allows(NOW, "22:00-11:00") is True  # overnight window
    assert fp.parse_session_filter("garbage") is None


def test_run_once_logs_forecast_from_confirmed_bar(tmp_path, monkeypatch) -> None:
    _patch_common(monkeypatch, tmp_path)
    client = _forecaster_client()

    summary = fp.run_once(NOW, client=client, horizons=(6,), use_debate=False, samples=1, model="gpt-5.6-terra")

    assert summary["ok"] is True and summary["written"] == 1
    rows = forecast_store.read_forecasts()
    assert len(rows) == 1
    row = rows[0]
    assert row["bar_time_utc"] == "2026-09-09T13:00:00+00:00"
    assert row["ts_utc"] == "2026-09-09T14:00:00+00:00"
    assert row["p0"] == 4363.0  # confirmed (13:00) bar close, not the forming bar's 9999
    assert row["atr_h1"] == 21.27
    assert row["barrier_up"] == round(4363.0 + 21.27, 5) and row["barrier_down"] == round(4363.0 - 21.27, 5)
    assert row["p_up"] == 0.55 and row["probs_valid"] is True and row["ok"] is True
    assert row["outcome"] is None and row["horizon_bars"] == 6
    assert row["momentum_3"] == 1.5
    assert row["ny_hour"] == 10 and row["weekday"] == 2
    assert row["usage"]["total_tokens"] == 6120
    assert row["usage_total"]["total_tokens"] == 6120 + 1000 + 4300 + 3000
    assert len(row["input_hash"]) == 64
    archived = json.loads(open(row["inputs_path"], encoding="utf-8").read())
    assert archived["payload"]["task"]["p0_close"] == 4363.0
    assert archived["reports"]["technical"]["signal"] == "SELL"
    # The LLM saw the same task numbers.
    sent = json.loads(client.call_json.call_args.kwargs["user_prompt"])
    assert sent["task"]["barrier_up"] == row["barrier_up"]


def test_run_once_is_idempotent_per_bar_and_writes_one_row_per_horizon(tmp_path, monkeypatch) -> None:
    _patch_common(monkeypatch, tmp_path)
    client = _forecaster_client()
    first = fp.run_once(NOW, client=client, horizons=(4, 12))
    assert first["written"] == 2
    second = fp.run_once(NOW + timedelta(minutes=5), client=client, horizons=(4, 12))
    assert second["skipped"] == "already_logged" and second["written"] == 0
    assert sorted(r["horizon_bars"] for r in forecast_store.read_forecasts()) == [4, 12]


def test_run_once_skips_weekend_and_session_filter(tmp_path, monkeypatch) -> None:
    _patch_common(monkeypatch, tmp_path)
    saturday = datetime(2026, 9, 12, 14, 1, 30, tzinfo=UTC)
    assert fp.run_once(saturday, client=_forecaster_client())["skipped"] == "market_closed"
    monkeypatch.setattr(fp, "FORECAST_SESSION_FILTER", "11:00-16:00")
    assert fp.run_once(NOW, client=_forecaster_client())["skipped"] == "outside_session_filter"
    assert forecast_store.read_forecasts() == []


def test_run_once_records_llm_failure_without_raising(tmp_path, monkeypatch) -> None:
    _patch_common(monkeypatch, tmp_path)
    client = Mock()
    client.call_json.side_effect = RuntimeError("api down")
    summary = fp.run_once(NOW, client=client, horizons=(6,))
    assert summary["written"] == 1
    row = forecast_store.read_forecasts()[0]
    assert row["ok"] is False and "api down" in row["error"] and row["p_up"] is None


def test_run_once_fails_safe_when_rates_raise(tmp_path, monkeypatch) -> None:
    _patch_common(monkeypatch, tmp_path)
    monkeypatch.setattr(fp, "get_rates", lambda symbol, tf, count: (_ for _ in ()).throw(RuntimeError("mt5 down")))
    summary = fp.run_once(NOW, client=_forecaster_client())
    assert summary["ok"] is False and "mt5 down" in summary["error"]
    assert forecast_store.read_forecasts() == []


def test_resolve_pending_labels_from_later_bars(tmp_path, monkeypatch) -> None:
    _patch_common(monkeypatch, tmp_path)
    fp.run_once(NOW, client=_forecaster_client(), horizons=(6,))
    row = forecast_store.read_forecasts()[0]
    barrier_up = row["barrier_up"]

    # Bars after the forecast: 14:00..19:00 plus a forming 20:00 bar. The 16:00 bar crosses the upper barrier.
    later = _h1_frame(datetime(2026, 9, 9, 20, 0, tzinfo=UTC), bars=40)
    later["high"] = later["close"] + 3.0
    mask = later["time"] == pd.Timestamp("2026-09-09T16:00:00Z")
    later.loc[mask, "high"] = barrier_up + 0.5

    # Too early: horizon has not elapsed.
    early = fp.resolve_pending(datetime(2026, 9, 9, 18, 0, tzinfo=UTC), rates_fetcher=lambda s, tf, n: later.copy())
    assert early["pending"] == 0 and forecast_store.read_forecasts()[0]["outcome"] is None

    done = fp.resolve_pending(datetime(2026, 9, 9, 20, 1, tzinfo=UTC), rates_fetcher=lambda s, tf, n: later.copy())
    assert done["pending"] == 1 and done["resolved"] == 1
    labelled = forecast_store.read_forecasts()[0]
    assert labelled["outcome"] == "UP" and labelled["bars_to_hit"] == 3
    assert labelled["resolved_at_utc"].startswith("2026-09-09T20:01")

    # Idempotent: nothing pending, nothing changed.
    again = fp.resolve_pending(datetime(2026, 9, 9, 21, 0, tzinfo=UTC), rates_fetcher=lambda s, tf, n: later.copy())
    assert again["pending"] == 0
    assert forecast_store.read_forecasts()[0]["outcome"] == "UP"


def test_resolve_pending_waits_when_bars_missing(tmp_path, monkeypatch) -> None:
    _patch_common(monkeypatch, tmp_path)
    fp.run_once(NOW, client=_forecaster_client(), horizons=(6,))
    # Only two bars after the forecast and no barrier hit -> keep waiting.
    later = _h1_frame(datetime(2026, 9, 9, 16, 0, tzinfo=UTC), bars=20)
    counts = fp.resolve_pending(datetime(2026, 9, 10, 2, 0, tzinfo=UTC), rates_fetcher=lambda s, tf, n: later.copy())
    assert counts["pending"] == 1 and counts["waiting_bars"] == 1 and counts["resolved"] == 0


def test_build_reports_uses_confirmed_frames_and_optional_debate(monkeypatch) -> None:
    forming_open = NOW.replace(minute=0, second=0, microsecond=0)
    frames = {tf: fp.drop_forming_bars(_h1_frame(forming_open, bars=30), NOW, "H1") for tf in ("D1", "H4", "H1")}
    monkeypatch.setattr(fp, "build_horizontal_levels", lambda **kwargs: {"supports": [], "resistances": []})
    monkeypatch.setattr(main, "_build_tp_reference_only", lambda **kwargs: {"levels": {}})
    monkeypatch.setattr(fp, "fetch_news_with_meta", lambda hours: ([{"title": "gold"}], {"feeds_total": 4, "feeds_live": 4}))
    monkeypatch.setattr(fp, "get_macro_data", lambda force_refresh=False: {"_meta": {"ok": True}})
    monkeypatch.setattr(fp, "build_macro_inputs", lambda data: {**data, "recent_releases": []})
    monkeypatch.setattr(fp, "analyze_macro_environment", lambda data: {"macro_bias": "NEUTRAL"})
    captured: dict = {}

    def _fake_technical(payload):
        captured["technical"] = payload
        return {"signal": "SELL"}

    monkeypatch.setattr(fp, "analyze_technical", _fake_technical)
    monkeypatch.setattr(fp, "analyze_sentiment", lambda items: {"score": -0.5, "n": len(items)})
    debate_calls: list = []

    import agents.debate_graph as dg

    monkeypatch.setattr(dg, "run_debate_graph", lambda t, s, m, **kwargs: debate_calls.append((t, s, m, kwargs)) or {"judge_summary": {"stronger_side": "bear"}})

    reports = fp.build_reports(frames, use_debate=False)
    assert reports["technical"]["signal"] == "SELL" and reports["debate"] is None
    # The rule-based regime read rides on the technical report (ADX 20 on every frame -> not a trend).
    assert reports["technical"]["regime"]["source"] == "rule_based"
    assert reports["technical"]["regime"]["regime"] in {"RANGE", "TRANSITION"}
    assert reports["sentiment"]["feed_meta"]["feeds_live"] == 4
    assert captured["technical"]["direction_context"]["h1"]["close"] == 4363.0  # confirmed close
    assert debate_calls == []

    monkeypatch.setattr(fp, "FORECAST_DEBATE_AXIS", "regime")
    reports = fp.build_reports(frames, use_debate=True)
    assert reports["debate"]["judge_summary"]["stronger_side"] == "bear"
    assert len(debate_calls) == 1
    _, _, _, kwargs = debate_calls[0]
    assert kwargs["axis"] == "regime"
    assert kwargs["regime_hint"] == reports["technical"]["regime"]


def _resolved_row(i: int, outcome: str, horizon: int = 6) -> dict:
    return {"ts_utc": f"2026-09-{1 + i // 24:02d}T{i % 24:02d}:00:00+00:00", "outcome": outcome, "horizon_bars": horizon}


def test_reference_base_rates_requires_enough_resolved_rows_of_same_horizon() -> None:
    rows = [_resolved_row(i, "UP") for i in range(fp.REFERENCE_BASE_RATE_MIN_ROWS - 1)]
    assert fp.reference_base_rates(rows, 6) is None
    rows.append(_resolved_row(99, "DOWN", horizon=12))  # other horizon does not count
    rows.append(_resolved_row(98, None))  # unresolved does not count
    rows.append(_resolved_row(97, "AMBIGUOUS"))
    assert fp.reference_base_rates(rows, 6) is None
    rows.append(_resolved_row(96, "TIMEOUT"))
    rates = fp.reference_base_rates(rows, 6)
    assert rates["n"] == fp.REFERENCE_BASE_RATE_MIN_ROWS
    assert abs(rates["p_up"] + rates["p_down"] + rates["p_timeout"] - 1.0) < 1e-9
    assert rates["p_down"] == 0.0 and rates["p_timeout"] == 1 / fp.REFERENCE_BASE_RATE_MIN_ROWS


def test_reference_base_rates_uses_most_recent_window_only() -> None:
    old = [_resolved_row(i, "UP") for i in range(100)]
    recent = [_resolved_row(100 + i, "DOWN") for i in range(fp.REFERENCE_BASE_RATE_WINDOW)]
    rates = fp.reference_base_rates(recent + old, 6)  # unsorted on purpose
    assert rates["n"] == fp.REFERENCE_BASE_RATE_WINDOW
    assert rates["p_down"] == 1.0 and rates["p_up"] == 0.0


def test_run_once_records_anchor_regime_and_axis(tmp_path, monkeypatch) -> None:
    _patch_common(monkeypatch, tmp_path)
    monkeypatch.setattr(
        fp,
        "build_reports",
        lambda frames, use_debate=False: {
            "technical": {"signal": "SELL", "regime": {"regime": "RANGE", "direction": "NEUTRAL", "confidence": 0.8, "entry_style": "LIMIT_FADE", "evidence": []}},
            "sentiment": {"score": -0.6},
            "macro": {"macro_bias": "NEUTRAL"},
            "debate": {"axis": "regime", "regime_summary": {"regime": "RANGE"}, "judge_summary": {"stronger_side": "neutral"}} if use_debate else None,
            "news_count": 3,
        },
    )
    seed = [{**_resolved_row(i, "TIMEOUT" if i % 2 else "UP"), "bar_time_utc": f"seed{i}"} for i in range(40)]
    forecast_store.write_forecasts(seed)
    client = _forecaster_client()

    fp.run_once(NOW, client=client, horizons=(6,), use_debate=False)
    row = [r for r in forecast_store.read_forecasts() if r.get("forecast_id")][0]
    assert row["reference_base_rates"] == {"p_up": 0.5, "p_down": 0.0, "p_timeout": 0.5, "n": 40}
    assert row["regime"] == "RANGE" and row["debate_axis"] == "" and row["used_debate"] is False
    sent = json.loads(client.call_json.call_args.kwargs["user_prompt"])
    assert sent["task"]["reference_base_rates"]["p_up"] == 0.5
    assert sent["task"]["regime"]["regime"] == "RANGE"
    assert "debate" not in sent

    monkeypatch.setattr(fp, "FORECAST_DEBATE_AXIS", "regime")
    fp.run_once(NOW + timedelta(hours=1), client=client, horizons=(6,), use_debate=True)
    latest = [r for r in forecast_store.read_forecasts() if r.get("forecast_id")][-1]
    assert latest["used_debate"] is True and latest["debate_axis"] == "regime"
    sent = json.loads(client.call_json.call_args.kwargs["user_prompt"])
    assert sent["debate"]["regime_summary"]["regime"] == "RANGE"
