from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config import SYMBOL
from data import mt5_client
from data.mt5_client import close_position, get_account_info, get_position_details, modify_sl, send_order


def _trade_mode_name(trade_mode: int) -> str:
    if trade_mode == 0:
        return "DEMO"
    if trade_mode == 2:
        return "REAL"
    return f"OTHER({trade_mode})"


def _confirm(prompt: str) -> bool:
    answer = input(f"{prompt} [y/N]: ").strip().lower()
    return answer == "y"


def _wait_for_review(message: str) -> bool:
    return _confirm(f"{message} 続行しますか?")


def _fmt_price(value: Any) -> str:
    try:
        return f"{float(value):.5f}"
    except Exception:
        return str(value)


def _fmt_pnl(value: Any) -> str:
    try:
        return f"{float(value):.2f}"
    except Exception:
        return str(value)


def _get_symbol_info(symbol: str) -> Any | None:
    if not mt5_client.connect():
        return None

    try:
        if mt5_client.mt5 is None:
            return None

        info = mt5_client.mt5.symbol_info(symbol)
        if info is None:
            return None

        if not getattr(info, "visible", False):
            mt5_client.mt5.symbol_select(symbol, True)
            info = mt5_client.mt5.symbol_info(symbol)

        return info
    finally:
        mt5_client.disconnect()


def _get_min_lot(symbol: str) -> float | None:
    info = _get_symbol_info(symbol)
    if info is None:
        return None

    volume_min = float(getattr(info, "volume_min", 0.0) or 0.0)
    if volume_min <= 0:
        return None
    return volume_min


def _build_buy_order_plan(symbol: str) -> dict[str, float] | None:
    if not mt5_client.connect():
        return None

    try:
        if mt5_client.mt5 is None:
            return None

        info = mt5_client.mt5.symbol_info(symbol)
        if info is None:
            return None
        if not getattr(info, "visible", False):
            mt5_client.mt5.symbol_select(symbol, True)
            info = mt5_client.mt5.symbol_info(symbol)
            if info is None:
                return None

        tick = mt5_client.mt5.symbol_info_tick(symbol)
        if tick is None:
            return None

        point = float(getattr(info, "point", 0.0) or 0.0)
        if point <= 0:
            point = 0.01

        stops_level = int(getattr(info, "trade_stops_level", 0) or 0)
        distance = max(point * 500.0, point * max(20, stops_level * 2))
        lot = float(getattr(info, "volume_min", 0.0) or 0.0)
        if lot <= 0:
            return None

        entry_price = float(tick.ask)
        sl = round(entry_price - distance, 5)
        tp = round(entry_price + (distance * 2.0), 5)
        return {
            "lot": lot,
            "entry_price": entry_price,
            "sl": sl,
            "tp": tp,
            "distance": distance,
            "point": point,
        }
    finally:
        mt5_client.disconnect()


def _print_positions(positions: list[dict[str, Any]]) -> None:
    if not positions:
        print("保有ポジションはありません。")
        return

    print(f"保有ポジション数: {len(positions)}")
    for position in positions:
        print(
            " - "
            f"ticket={position.get('ticket')} "
            f"symbol={position.get('symbol')} "
            f"type={position.get('type')} "
            f"volume={position.get('volume')} "
            f"open={_fmt_price(position.get('price_open'))} "
            f"current={_fmt_price(position.get('price_current'))} "
            f"sl={_fmt_price(position.get('sl'))} "
            f"tp={_fmt_price(position.get('tp'))} "
            f"profit={_fmt_pnl(position.get('profit'))}"
        )


def _find_position_by_ticket(positions: list[dict[str, Any]], ticket: int) -> dict[str, Any] | None:
    for position in positions:
        if int(position.get("ticket") or 0) == ticket:
            return position
    return None


def _detect_new_ticket(before: list[dict[str, Any]], after: list[dict[str, Any]]) -> int | None:
    before_tickets = {int(position.get("ticket") or 0) for position in before}
    for position in after:
        ticket = int(position.get("ticket") or 0)
        if ticket > 0 and ticket not in before_tickets:
            return ticket
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Manual MT5 position I/O verification for GP-MATE")
    parser.add_argument("--symbol", default=SYMBOL, help="Target symbol to inspect and test")
    args = parser.parse_args()

    symbol = str(args.symbol or SYMBOL)

    print("=== GP-MATE Manual Position Check ===")
    print("このスクリプトは手動検証専用です。各ステップで必ず目視確認してください。")

    print("\n[Step A] 接続確認")
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
    print(f"balance : {account.balance:.2f} {account.currency}")
    print("重要: デモ口座であることを必ず目視確認してください。")

    if account.trade_mode != 0:
        print("デモ口座ではありません。安全のため停止します。")
        return 1

    if not _wait_for_review("デモ口座であることを確認しました。"):
        print("ユーザー確認で停止しました。")
        return 0

    print("\n[Step B] 現在の保有確認")
    positions_before = get_position_details(symbol)
    print(f"symbol: {symbol}")
    _print_positions(positions_before)
    if not _wait_for_review("取得結果を目視確認しました。"):
        print("ユーザー確認で停止しました。")
        return 0

    print("\n[Step C] 小ロットで新規BUY建て")
    min_lot = _get_min_lot(symbol)
    order_plan = _build_buy_order_plan(symbol)
    if min_lot is None or order_plan is None:
        print("最小ロットまたは発注プランを取得できませんでした。")
        return 1

    print(f"最小ロット: {min_lot}")
    print(
        "発注予定: "
        f"BUY {symbol} lot={order_plan['lot']} "
        f"entry_est={_fmt_price(order_plan['entry_price'])} "
        f"sl={_fmt_price(order_plan['sl'])} "
        f"tp={_fmt_price(order_plan['tp'])}"
    )

    created_ticket: int | None = None
    positions_after_open = positions_before
    if _confirm("新規BUYを実行しますか?"):
        order_result = send_order(
            symbol=symbol,
            action="BUY",
            lot=float(order_plan["lot"]),
            sl=float(order_plan["sl"]),
            tp=float(order_plan["tp"]),
        )
        print(f"send_order result: {order_result}")
        positions_after_open = get_position_details(symbol)
        _print_positions(positions_after_open)
        created_ticket = _detect_new_ticket(positions_before, positions_after_open)
        if created_ticket is not None:
            print(f"新規ポジション ticket: {created_ticket}")
        else:
            print("新規 ticket を自動特定できませんでした。既存保有と照合してください。")
    else:
        print("新規建てはスキップしました。")

    if created_ticket is None:
        print("\n建てたポジションがないため、Step D と Step E はスキップします。")
        return 0

    if not _wait_for_review("新規建て結果を確認しました。"):
        print("ユーザー確認で停止しました。")
        return 0

    print("\n[Step D] SL修正の確認")
    opened_position = _find_position_by_ticket(positions_after_open, created_ticket)
    if opened_position is None:
        print("建てたポジションを再取得できませんでした。")
        return 1

    current_sl = float(opened_position.get("sl") or 0.0)
    entry_price = float(opened_position.get("price_open") or 0.0)
    proposed_sl = round((current_sl + entry_price) / 2.0, 5) if current_sl > 0 and entry_price > 0 else 0.0
    if proposed_sl <= 0 or proposed_sl == current_sl:
        print("修正用SLを安全に計算できませんでした。")
        return 1

    print(
        "SL修正予定: "
        f"ticket={created_ticket} current_sl={_fmt_price(current_sl)} new_sl={_fmt_price(proposed_sl)}"
    )
    if _confirm("SLを修正しますか?"):
        modify_result = modify_sl(created_ticket, proposed_sl)
        print(f"modify_sl result: {modify_result}")
        positions_after_modify = get_position_details(symbol)
        _print_positions(positions_after_modify)
    else:
        print("SL修正はスキップしました。")
        positions_after_modify = positions_after_open

    if not _wait_for_review("SL修正結果を確認しました。"):
        print("ユーザー確認で停止しました。")
        return 0

    print("\n[Step E] 決済の確認")
    print(f"決済予定 ticket: {created_ticket}")
    if _confirm("ポジションを成行決済しますか?"):
        close_result = close_position(created_ticket)
        print(f"close_position result: {close_result}")
        positions_after_close = get_position_details(symbol)
        _print_positions(positions_after_close)
    else:
        print("決済はスキップしました。")
        positions_after_close = positions_after_modify

    if _find_position_by_ticket(positions_after_close, created_ticket) is None:
        print("対象ポジションは一覧から消えています。")
    else:
        print("対象ポジションが残っています。MT5画面と照合してください。")

    print("\n手動検証を終了します。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())