from __future__ import annotations

import pytest

from risk.risk_manager import (
    build_risk_plan,
    calc_lot,
    calc_sl_tp,
    check_filters,
)


def test_calc_lot_minimum_floor() -> None:
    # Small accounts floor at the broker minimum lot so they can still trade,
    # even though effective risk then exceeds the configured risk_pct.
    lot = calc_lot(balance_jpy=500_000, risk_pct=0.01, sl_distance_usd=1000, jpy_usd_rate=155.0)
    assert lot == 0.01


def test_calc_lot_regular_case() -> None:
    lot = calc_lot(balance_jpy=500_000, risk_pct=0.01, sl_distance_usd=10, jpy_usd_rate=155.0)
    assert lot > 0
    assert round(lot, 2) == lot


def test_calc_lot_uses_given_fx_rate() -> None:
    # A weaker JPY (higher USDJPY) means less USD risk budget -> smaller lot.
    lot_low_rate = calc_lot(balance_jpy=500_000, risk_pct=0.01, sl_distance_usd=10, jpy_usd_rate=100.0)
    lot_high_rate = calc_lot(balance_jpy=500_000, risk_pct=0.01, sl_distance_usd=10, jpy_usd_rate=200.0)
    assert lot_low_rate > lot_high_rate


def test_calc_lot_invalid_inputs_return_zero() -> None:
    assert calc_lot(balance_jpy=0, risk_pct=0.01, sl_distance_usd=10) == 0.0
    assert calc_lot(balance_jpy=500_000, risk_pct=0.0, sl_distance_usd=10) == 0.0
    assert calc_lot(balance_jpy=500_000, risk_pct=0.01, sl_distance_usd=0) == 0.0
    assert calc_lot(balance_jpy=500_000, risk_pct=0.01, sl_distance_usd=10, jpy_usd_rate=0.0) == 0.0


def test_calc_sl_tp_buy() -> None:
    sl, tp = calc_sl_tp(entry_price=2300.0, atr=10.0, action="BUY", atr_mult=1.5, rr=2.0)
    assert sl == 2285.0
    assert tp == 2330.0


def test_calc_sl_tp_sell() -> None:
    sl, tp = calc_sl_tp(entry_price=2300.0, atr=10.0, action="SELL", atr_mult=1.5, rr=2.0)
    assert sl == 2315.0
    assert tp == 2270.0


def test_calc_sl_tp_invalid_action_raises() -> None:
    with pytest.raises(ValueError):
        calc_sl_tp(entry_price=2300.0, atr=10.0, action="HOLD")


def test_check_filters_confidence_block() -> None:
    result = check_filters(
        confidence=0.5,
        spread=20,
        baseline_spread=15,
        is_news_soon=False,
        consecutive_losses=0,
        daily_loss_pct=0.0,
    )
    assert not result.ok


def test_check_filters_spread_block() -> None:
    result = check_filters(
        confidence=0.9,
        spread=31,
        baseline_spread=15,
        is_news_soon=False,
        consecutive_losses=0,
        daily_loss_pct=0.0,
    )
    assert not result.ok


def test_check_filters_news_block() -> None:
    result = check_filters(
        confidence=0.9,
        spread=20,
        baseline_spread=15,
        is_news_soon=True,
        consecutive_losses=0,
        daily_loss_pct=0.0,
    )
    assert not result.ok


def test_check_filters_consecutive_loss_block() -> None:
    result = check_filters(
        confidence=0.9,
        spread=20,
        baseline_spread=15,
        is_news_soon=False,
        consecutive_losses=3,
        daily_loss_pct=0.0,
    )
    assert not result.ok


def test_check_filters_daily_loss_block() -> None:
    result = check_filters(
        confidence=0.9,
        spread=20,
        baseline_spread=15,
        is_news_soon=False,
        consecutive_losses=0,
        daily_loss_pct=0.03,
    )
    assert not result.ok


def test_check_filters_ok() -> None:
    result = check_filters(
        confidence=0.9,
        spread=20,
        baseline_spread=15,
        is_news_soon=False,
        consecutive_losses=0,
        daily_loss_pct=0.01,
    )
    assert result.ok


def test_build_risk_plan_invalid_action_falls_back_to_hold() -> None:
    plan = build_risk_plan(action="HOLD", entry_price=2300.0, atr=10.0, balance_jpy=500_000)
    assert not plan["ok"]
    assert plan["action"] == "HOLD"


def test_build_risk_plan_success_buy() -> None:
    plan = build_risk_plan(action="BUY", entry_price=2300.0, atr=10.0, balance_jpy=500_000)
    assert plan["ok"]
    assert plan["action"] == "BUY"
    assert float(plan["lot"]) > 0
    assert float(plan["tp"]) > float(plan["sl"])


def test_build_risk_plan_uses_default_2r_take_profit() -> None:
    plan = build_risk_plan(action="BUY", entry_price=2300.0, atr=10.0, balance_jpy=500_000)
    assert plan["ok"]
    sl_distance = 2300.0 - float(plan["sl"])
    tp_distance = float(plan["tp"]) - 2300.0
    assert sl_distance == 15.0
    assert tp_distance == 30.0


def test_build_risk_plan_suggested_tp_none_falls_back_to_2r() -> None:
    plan = build_risk_plan(
        action="BUY",
        entry_price=2300.0,
        atr=10.0,
        balance_jpy=500_000,
        suggested_tp=None,
    )
    assert plan["ok"]
    assert float(plan["tp"]) == float(plan["tp_2r"])
    assert plan["tp_source"] == "fallback_2r"


def test_build_risk_plan_buy_uses_suggested_when_inside_2r() -> None:
    plan = build_risk_plan(
        action="BUY",
        entry_price=2300.0,
        atr=10.0,
        balance_jpy=500_000,
        suggested_tp=2320.0,
    )
    assert plan["ok"]
    assert float(plan["tp"]) == 2320.0
    assert plan["tp_source"] == "suggested"


def test_build_risk_plan_buy_caps_suggested_when_beyond_2r() -> None:
    plan = build_risk_plan(
        action="BUY",
        entry_price=2300.0,
        atr=10.0,
        balance_jpy=500_000,
        suggested_tp=2350.0,
    )
    assert plan["ok"]
    assert float(plan["tp"]) == float(plan["tp_2r"])
    assert float(plan["tp"]) == 2330.0
    assert plan["tp_source"] == "suggested_capped_2r"


def test_build_risk_plan_sell_uses_suggested_when_inside_2r() -> None:
    plan = build_risk_plan(
        action="SELL",
        entry_price=2300.0,
        atr=10.0,
        balance_jpy=500_000,
        suggested_tp=2280.0,
    )
    assert plan["ok"]
    assert float(plan["tp"]) == 2280.0
    assert plan["tp_source"] == "suggested"


def test_build_risk_plan_sell_caps_suggested_when_beyond_2r() -> None:
    plan = build_risk_plan(
        action="SELL",
        entry_price=2300.0,
        atr=10.0,
        balance_jpy=500_000,
        suggested_tp=2240.0,
    )
    assert plan["ok"]
    assert float(plan["tp"]) == float(plan["tp_2r"])
    assert float(plan["tp"]) == 2270.0
    assert plan["tp_source"] == "suggested_capped_2r"


def test_build_risk_plan_buy_rejects_wrong_direction_suggested_tp() -> None:
    plan = build_risk_plan(
        action="BUY",
        entry_price=2300.0,
        atr=10.0,
        balance_jpy=500_000,
        suggested_tp=2299.0,
    )
    assert plan["ok"]
    assert float(plan["tp"]) == float(plan["tp_2r"])
    assert plan["tp_source"] == "fallback_2r"


def test_build_risk_plan_sell_rejects_wrong_direction_suggested_tp() -> None:
    plan = build_risk_plan(
        action="SELL",
        entry_price=2300.0,
        atr=10.0,
        balance_jpy=500_000,
        suggested_tp=2301.0,
    )
    assert plan["ok"]
    assert float(plan["tp"]) == float(plan["tp_2r"])
    assert plan["tp_source"] == "fallback_2r"


def test_build_risk_plan_passes_fx_rate_to_lot_calc() -> None:
    plan_low = build_risk_plan(
        action="BUY", entry_price=2300.0, atr=10.0, balance_jpy=5_000_000, jpy_usd_rate=100.0
    )
    plan_high = build_risk_plan(
        action="BUY", entry_price=2300.0, atr=10.0, balance_jpy=5_000_000, jpy_usd_rate=200.0
    )
    assert plan_low["ok"] and plan_high["ok"]
    assert float(plan_low["lot"]) > float(plan_high["lot"])


def test_build_risk_plan_small_balance_floors_to_min_lot() -> None:
    plan = build_risk_plan(action="BUY", entry_price=2300.0, atr=100.0, balance_jpy=10_000)
    assert plan["ok"]
    assert plan["action"] == "BUY"
    assert float(plan["lot"]) == 0.01


def test_build_risk_plan_sl_is_unchanged_with_suggested_tp() -> None:
    baseline = build_risk_plan(action="BUY", entry_price=2300.0, atr=10.0, balance_jpy=500_000)
    adjusted = build_risk_plan(
        action="BUY",
        entry_price=2300.0,
        atr=10.0,
        balance_jpy=500_000,
        suggested_tp=2320.0,
    )
    assert baseline["ok"] and adjusted["ok"]
    assert float(baseline["sl"]) == float(adjusted["sl"])


def test_build_risk_plan_adopts_structural_sl_with_wick_buffer() -> None:
    # Baseline SL distance = ATR(15) x 1.5 = 22.5.
    # suggested_sl is the structural LEVEL; the stop is placed
    # SL_STRUCTURE_BUFFER_USD (default 2.0) beyond it: 4446 -> 4444.
    # Buffered distance 21 is inside [15.0, 22.5] -> adopted.
    plan = build_risk_plan(
        action="BUY", entry_price=4465.0, atr=15.0, balance_jpy=5_000_000, suggested_sl=4446.0
    )
    assert plan["ok"]
    assert plan["sl"] == 4444.0
    assert plan["sl_source"] == "suggested"
    # The 2R box stays anchored to the ATR baseline, NOT the adopted SL:
    # 4465 + 22.5*2 = 4510.
    assert float(plan["tp_2r"]) == 4510.0


def test_build_risk_plan_sell_buffer_extends_above_level() -> None:
    # SELL: level 4484 -> stop at 4486 (level + buffer), distance 21.
    plan = build_risk_plan(
        action="SELL", entry_price=4465.0, atr=15.0, balance_jpy=5_000_000, suggested_sl=4484.0
    )
    assert plan["sl_source"] == "suggested"
    assert plan["sl"] == 4486.0


def test_build_risk_plan_rejects_structural_sl_outside_baseline() -> None:
    # Too close even after buffer: 4455 -> 4453, distance 12 < 1.0*ATR(15).
    too_close = build_risk_plan(
        action="BUY", entry_price=4465.0, atr=15.0, balance_jpy=5_000_000, suggested_sl=4455.0
    )
    assert too_close["sl_source"] == "fallback_atr"
    assert float(too_close["sl"]) == 4465.0 - 15.0 * 1.5

    # Deeper than the ATR baseline after buffer: 4444 -> 4442, distance 23 > 22.5.
    deeper = build_risk_plan(
        action="BUY", entry_price=4465.0, atr=15.0, balance_jpy=5_000_000, suggested_sl=4444.0
    )
    assert deeper["sl_source"] == "fallback_atr"
    assert float(deeper["sl"]) == 4465.0 - 15.0 * 1.5

    # Wrong side for SELL.
    wrong_side = build_risk_plan(
        action="SELL", entry_price=4465.0, atr=15.0, balance_jpy=5_000_000, suggested_sl=4446.0
    )
    assert wrong_side["sl_source"] == "fallback_atr"


def test_build_risk_plan_structural_sl_resizes_lot() -> None:
    # Tighter SL -> larger lot at the same account risk (both inside baseline).
    narrow = build_risk_plan(
        action="BUY", entry_price=4465.0, atr=15.0, balance_jpy=5_000_000, suggested_sl=4449.0
    )
    wide = build_risk_plan(
        action="BUY", entry_price=4465.0, atr=15.0, balance_jpy=5_000_000, suggested_sl=4445.0
    )
    assert narrow["sl_source"] == "suggested" and wide["sl_source"] == "suggested"
    assert float(wide["lot"]) < float(narrow["lot"])
