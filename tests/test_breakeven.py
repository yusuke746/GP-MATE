from __future__ import annotations

from risk.breakeven import should_move_to_breakeven


def test_buy_moves_at_1r_with_buffer() -> None:
    should_move, new_sl = should_move_to_breakeven(
        entry=100.0,
        initial_sl=95.0,
        current_price=105.0,
        current_sl=95.0,
        side="BUY",
        buffer=0.1,
    )
    assert should_move is True
    assert new_sl == 100.1


def test_sell_moves_at_1r_with_buffer() -> None:
    should_move, new_sl = should_move_to_breakeven(
        entry=100.0,
        initial_sl=105.0,
        current_price=95.0,
        current_sl=105.0,
        side="SELL",
        buffer=0.1,
    )
    assert should_move is True
    assert new_sl == 99.9


def test_buy_boundary_before_at_after_1r() -> None:
    before = should_move_to_breakeven(100.0, 95.0, 104.99999, 95.0, "BUY", 0.1)
    at = should_move_to_breakeven(100.0, 95.0, 105.0, 95.0, "BUY", 0.1)
    after = should_move_to_breakeven(100.0, 95.0, 105.00001, 95.0, "BUY", 0.1)

    assert before == (False, None)
    assert at == (True, 100.1)
    assert after == (True, 100.1)


def test_sell_boundary_before_at_after_1r() -> None:
    before = should_move_to_breakeven(100.0, 105.0, 95.00001, 105.0, "SELL", 0.1)
    at = should_move_to_breakeven(100.0, 105.0, 95.0, 105.0, "SELL", 0.1)
    after = should_move_to_breakeven(100.0, 105.0, 94.99999, 105.0, "SELL", 0.1)

    assert before == (False, None)
    assert at == (True, 99.9)
    assert after == (True, 99.9)


def test_skip_when_already_at_or_better_than_breakeven() -> None:
    buy_skip = should_move_to_breakeven(100.0, 95.0, 110.0, 100.1, "BUY", 0.1)
    sell_skip = should_move_to_breakeven(100.0, 105.0, 90.0, 99.9, "SELL", 0.1)

    assert buy_skip == (False, None)
    assert sell_skip == (False, None)


def test_safety_guard_never_moves_to_worse_direction() -> None:
    # BUY: current SL already above candidate new SL -> never move down.
    buy_worse = should_move_to_breakeven(100.0, 95.0, 106.0, 100.2, "BUY", 0.1)
    # SELL: current SL already below candidate new SL -> never move up.
    sell_worse = should_move_to_breakeven(100.0, 105.0, 94.0, 99.8, "SELL", 0.1)

    assert buy_worse == (False, None)
    assert sell_worse == (False, None)

def test_partial_close_volume_floors_to_the_step_and_keeps_a_legal_remainder() -> None:
    from risk.breakeven import partial_close_volume

    assert partial_close_volume(0.03, 0.5) == 0.01  # 0.015 -> floor 0.01, remainder 0.02
    assert partial_close_volume(0.02, 0.5) == 0.01
    assert partial_close_volume(0.04, 0.5) == 0.02
    assert partial_close_volume(0.10, 0.5) == 0.05
    assert partial_close_volume(0.01, 0.5) == 0.0  # cannot split a minimum lot
    assert partial_close_volume(0.03, 0.0) == 0.0  # feature off
    assert partial_close_volume(0.03, 1.0) == 0.02  # "all" still leaves the minimum running
    assert partial_close_volume(0.02, 0.9) == 0.01
    assert partial_close_volume("x", 0.5) == 0.0
    assert partial_close_volume(0.0, 0.5) == 0.0
