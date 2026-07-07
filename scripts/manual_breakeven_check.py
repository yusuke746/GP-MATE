from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config import BREAKEVEN_BUFFER, SYMBOL
from data.mt5_client import get_account_info, get_position_details, modify_sl
from risk.breakeven import should_move_to_breakeven


def _trade_mode_name(trade_mode: int) -> str:
    if trade_mode == 0:
        return "DEMO"
    if trade_mode == 2:
        return "REAL"
    return f"OTHER({trade_mode})"


def _confirm(prompt: str) -> bool:
    answer = input(f"{prompt} [y/N]: ").strip().lower()
    return answer == "y"


def _fmt_price(value: Any) -> str:
    try:
        return f"{float(value):.5f}"
    except Exception:
        return str(value)


def _print_position(position: dict[str, Any]) -> None:
    print(
        "position: "
        f"ticket={position.get('ticket')} "
        f"type={position.get('type')} "
        f"volume={position.get('volume')} "
        f"entry={_fmt_price(position.get('price_open'))} "
        f"current={_fmt_price(position.get('price_current'))} "
        f"sl={_fmt_price(position.get('sl'))} "
        f"tp={_fmt_price(position.get('tp'))} "
        f"profit={position.get('profit')}"
    )


def _find_by_ticket(positions: list[dict[str, Any]], ticket: int) -> dict[str, Any] | None:
    for position in positions:
        if int(position.get("ticket") or 0) == ticket:
            return position
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Manual breakeven stop check")
    parser.add_argument("--symbol", default=SYMBOL)
    parser.add_argument("--ticket", type=int, default=0, help="Target ticket. Default: first open position")
    args = parser.parse_args()

    symbol = str(args.symbol or SYMBOL)
    target_ticket = int(args.ticket or 0)

    print("=== GP-MATE Manual Breakeven Check ===")
    info = get_account_info()
    if not info.get("success"):
        print("MT5接続に失敗しました。", info.get("reason", ""))
        return 1

    account = info.get("data")
    if account is None:
        print("口座情報が取得できませんでした。")
        return 1

    mode = _trade_mode_name(account.trade_mode)
    print(f"login   : {account.login}")
    print(f"server  : {account.server}")
    print(f"mode    : {mode}")
    print(f"symbol  : {symbol}")
    print(f"buffer  : {BREAKEVEN_BUFFER}")

    if account.trade_mode != 0:
        print("デモ口座ではありません。安全のため中断します。")
        return 1

    positions = get_position_details(symbol)
    if not positions:
        print("対象シンボルの保有ポジションがありません。")
        return 1

    print(f"open positions: {len(positions)}")
    for position in positions:
        _print_position(position)

    target = positions[0] if target_ticket <= 0 else _find_by_ticket(positions, target_ticket)
    if target is None:
        print(f"ticket={target_ticket} は見つかりませんでした。")
        return 1

    ticket = int(target.get("ticket") or 0)
    side = str(target.get("type") or "")
    entry = float(target.get("price_open") or 0.0)
    current_price = float(target.get("price_current") or 0.0)
    current_sl = float(target.get("sl") or 0.0)

    should_move, new_sl = should_move_to_breakeven(
        entry=entry,
        initial_sl=current_sl,
        current_price=current_price,
        current_sl=current_sl,
        side=side,
        buffer=BREAKEVEN_BUFFER,
    )

    risk_r = abs(entry - current_sl)
    trigger_price = (entry + risk_r) if side == "BUY" else (entry - risk_r)

    print("\n[Decision Preview]")
    print(f"ticket           : {ticket}")
    print(f"side             : {side}")
    print(f"entry            : {_fmt_price(entry)}")
    print(f"current_price    : {_fmt_price(current_price)}")
    print(f"current_sl       : {_fmt_price(current_sl)}")
    print(f"R                : {_fmt_price(risk_r)}")
    print(f"trigger_price    : {_fmt_price(trigger_price)}")
    print(f"should_move      : {should_move}")
    print(f"new_sl(candidate): {_fmt_price(new_sl)}")

    if not should_move or new_sl is None:
        print("建値更新条件を満たしていないため、更新は実行しません。")
        return 0

    if not _confirm("buildされた new_sl で modify_sl を実行しますか?"):
        print("ユーザーによりキャンセルしました。")
        return 0

    result = modify_sl(ticket, float(new_sl))
    print(f"modify_sl result: {result}")

    refreshed = get_position_details(symbol)
    updated = _find_by_ticket(refreshed, ticket)
    if updated is not None:
        print("\n[After Update]")
        _print_position(updated)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())