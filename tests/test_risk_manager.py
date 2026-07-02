from __future__ import annotations

import pytest

from risk.risk_manager import (
    build_risk_plan,
    calc_lot,
    calc_sl_tp,
    check_filters,
)


def test_calc_lot_minimum_floor() -> None:
    lot = calc_lot(balance_jpy=500_000, risk_pct=0.01, sl_distance_usd=1000)
    assert lot == 0.01


def test_calc_lot_regular_case() -> None:
    lot = calc_lot(balance_jpy=500_000, risk_pct=0.01, sl_distance_usd=10)
    assert lot > 0
    assert round(lot, 2) == lot


def test_calc_lot_invalid_inputs_return_zero() -> None:
    assert calc_lot(balance_jpy=0, risk_pct=0.01, sl_distance_usd=10) == 0.0
    assert calc_lot(balance_jpy=500_000, risk_pct=0.0, sl_distance_usd=10) == 0.0
    assert calc_lot(balance_jpy=500_000, risk_pct=0.01, sl_distance_usd=0) == 0.0


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
