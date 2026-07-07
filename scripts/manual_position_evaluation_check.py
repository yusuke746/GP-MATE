from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from agents.data.fred_client import get_macro_data
from agents.debate_graph import build_skipped_debate_report, run_debate_graph, should_execute_debate
from agents.evaluate_position import evaluate_position
from agents.macro_analyst import analyze_macro_environment
from agents.sentiment import analyze_sentiment
from agents.technical import analyze_technical
from config import CLOSE_CONFIDENCE_THRESHOLD, SYMBOL
from data import mt5_client
from data.mt5_client import close_position, get_account_info, get_position_details, get_rates, send_order
from data.news_client import fetch_news
from indicators.ta_calc import add_indicators


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


def _extract_latest_features(tf_df: Any) -> dict[str, Any]:
    if tf_df is None or tf_df.empty:
        return {}
    latest = tf_df.iloc[-1]
    return {
        "close": float(latest.get("close", 0.0)),
        "rsi_14": float(latest.get("rsi_14", 50.0)),
        "macd": float(latest.get("macd", 0.0)),
        "macd_signal": float(latest.get("macd_signal", 0.0)),
        "macd_hist": float(latest.get("macd_hist", 0.0)),
        "bb_upper": float(latest.get("bb_upper", 0.0)),
        "bb_mid": float(latest.get("bb_mid", 0.0)),
        "bb_lower": float(latest.get("bb_lower", 0.0)),
        "atr_14": float(latest.get("atr_14", 0.0)),
        "recent_high_20": float(latest.get("recent_high_20", 0.0)),
        "recent_low_20": float(latest.get("recent_low_20", 0.0)),
    }


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


def _build_order_plan(symbol: str, action: str) -> dict[str, float] | None:
    normalized = action.upper().strip()
    if normalized not in {"BUY", "SELL"}:
        return None

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

        if normalized == "BUY":
            entry_price = float(tick.ask)
            sl = round(entry_price - distance, 5)
            tp = round(entry_price + (distance * 2.0), 5)
        else:
            entry_price = float(tick.bid)
            sl = round(entry_price + distance, 5)
            tp = round(entry_price - (distance * 2.0), 5)

        return {
            "lot": lot,
            "entry_price": entry_price,
            "sl": sl,
            "tp": tp,
            "distance": distance,
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


def _direction_from_debate(debate_report: dict[str, Any]) -> str:
    judge_summary = debate_report.get("judge_summary", {}) if isinstance(debate_report, dict) else {}
    if not isinstance(judge_summary, dict):
        return "NEUTRAL"

    stronger = str(judge_summary.get("stronger_side", "neutral") or "neutral").lower()
    if stronger == "bull":
        return "BUY"
    if stronger == "bear":
        return "SELL"
    return "NEUTRAL"


def _position_context(position: dict[str, Any], position_count: int) -> dict[str, Any]:
    return {
        "ticket": int(position.get("ticket") or 0),
        "symbol": str(position.get("symbol") or SYMBOL),
        "type": str(position.get("type") or "UNKNOWN"),
        "volume": float(position.get("volume") or 0.0),
        "price_open": float(position.get("price_open") or 0.0),
        "price_current": float(position.get("price_current") or 0.0),
        "sl": float(position.get("sl") or 0.0),
        "tp": float(position.get("tp") or 0.0),
        "profit": float(position.get("profit") or 0.0),
        "position_count": int(position_count),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Manual evaluate_position validation with live LLM")
    parser.add_argument("--symbol", default=SYMBOL, help="Target symbol")
    parser.add_argument("--entry-action", default="BUY", choices=["BUY", "SELL"], help="Entry direction for test position")
    args = parser.parse_args()

    symbol = str(args.symbol or SYMBOL)
    entry_action = str(args.entry_action or "BUY").upper()

    print("=== GP-MATE Manual Position Evaluation Check ===")
    print("目的: 実LLMが保有方向を踏まえて HOLD/CLOSE を判断するかを目視確認する")

    print("\n[Step A] 安全確認")
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
    print(f"symbol  : {symbol}")

    if account.trade_mode != 0:
        print("デモ口座ではありません。安全のため中断します。")
        return 1

    if not _wait_for_review("デモ口座であることを確認しました。"):
        print("ユーザー確認で停止しました。")
        return 0

    print("\n[Step B] 検証用ポジション建玉")
    positions_before = get_position_details(symbol)
    print("建玉前の保有:")
    _print_positions(positions_before)

    order_plan = _build_order_plan(symbol, entry_action)
    if order_plan is None:
        print("発注プランを取得できませんでした。")
        return 1

    print(
        "発注予定: "
        f"{entry_action} {symbol} lot={order_plan['lot']} "
        f"entry_est={_fmt_price(order_plan['entry_price'])} "
        f"sl={_fmt_price(order_plan['sl'])} tp={_fmt_price(order_plan['tp'])}"
    )

    if not _confirm("検証用ポジションを建てますか?"):
        print("建玉をスキップしたため終了します。")
        return 0

    order_result = send_order(
        symbol=symbol,
        action=entry_action,
        lot=float(order_plan["lot"]),
        sl=float(order_plan["sl"]),
        tp=float(order_plan["tp"]),
    )
    print(f"send_order result: {order_result}")

    positions_after_open = get_position_details(symbol)
    print("建玉後の保有:")
    _print_positions(positions_after_open)

    created_ticket = _detect_new_ticket(positions_before, positions_after_open)
    if created_ticket is None:
        print("新規ticketを自動特定できませんでした。安全のため中断します。")
        return 1

    target_position = _find_position_by_ticket(positions_after_open, created_ticket)
    if target_position is None:
        print("建玉したポジション詳細を再取得できませんでした。")
        return 1

    if not _wait_for_review("建玉結果を目視確認しました。"):
        print("ユーザー確認で停止しました。")
        return 0

    print("\n[Step C] 保有評価フロー（実LLM）")
    print("価格/分析/議論を実行し、evaluate_position を呼びます。")
    if not _confirm("評価フローを実行しますか?"):
        print("評価フローをスキップして終了します。")
        return 0

    d1 = add_indicators(get_rates(symbol, "D1", 300))
    h4 = add_indicators(get_rates(symbol, "H4", 300))
    h1 = add_indicators(get_rates(symbol, "H1", 300))
    if h4.empty or h1.empty:
        print("価格データが不足しているため評価を中断します。")
        return 1

    news_items = fetch_news(hours=24)
    macro_data = get_macro_data(force_refresh=False)
    macro_report = analyze_macro_environment(macro_data)
    technical_report = analyze_technical(
        {
            "d1": _extract_latest_features(d1) if not d1.empty else {},
            "h4": _extract_latest_features(h4),
            "h1": _extract_latest_features(h1),
        }
    )
    sentiment_report = analyze_sentiment(news_items)

    gate = should_execute_debate(technical_report, sentiment_report, macro_report)
    if gate["should_debate"]:
        debate_report = run_debate_graph(technical_report, sentiment_report, macro_report)
    else:
        debate_report = build_skipped_debate_report(gate["reason"])

    position_ctx = _position_context(target_position, len(positions_after_open))
    evaluation_report = evaluate_position(
        position_context=position_ctx,
        technical_report=technical_report,
        sentiment_report=sentiment_report,
        debate_report=debate_report,
        confidence_threshold=CLOSE_CONFIDENCE_THRESHOLD,
    )

    technical_signal = str(technical_report.get("signal", "NEUTRAL") or "NEUTRAL")
    debate_direction = _direction_from_debate(debate_report)
    action = str(evaluation_report.get("action", "HOLD") or "HOLD")
    confidence = float(evaluation_report.get("confidence", 0.0) or 0.0)
    reasoning = str(evaluation_report.get("reasoning", "") or "")

    print("\n--- 評価結果（目視確認用）---")
    print(f"保有方向            : {position_ctx['type']}")
    print(f"テクニカル方向      : {technical_signal}")
    print(f"議論結論方向        : {debate_direction}")
    print(f"議論実行            : {gate['should_debate']}")
    print(f"議論ゲート理由      : {gate['reason']}")
    print(f"evaluate action      : {action}")
    print(f"evaluate confidence  : {confidence:.2f}")
    print(f"close threshold      : {CLOSE_CONFIDENCE_THRESHOLD:.2f}")
    print("reasoning (全文):")
    print(reasoning)

    print("\n[目視確認ポイント]")
    print("1) AIは保有方向を認識しているか?")
    print("2) 逆方向/同方向を、reasoning内で整合的に説明しているか?")
    print("3) actionとconfidenceが閾値ルール(0.7)に整合しているか?")

    if not _wait_for_review("評価結果を確認しました。"):
        print("ユーザー確認で停止しました。")
        return 0

    print("\n[Step D] actionに応じた決済（任意）")
    positions_after_eval = get_position_details(symbol)
    if action == "CLOSE":
        print("evaluate_position は CLOSE を返しました。")
        if _confirm("このポジションを close_position で決済しますか?"):
            close_result = close_position(created_ticket)
            print(f"close_position result: {close_result}")
            positions_after_eval = get_position_details(symbol)
            _print_positions(positions_after_eval)
        else:
            print("CLOSE提案を見送りました（決済未実行）。")
    else:
        print("evaluate_position は HOLD を返しました。保持します。")

    print("\n[Step E] 後片付け（残存ポジション確認）")
    positions_final = get_position_details(symbol)
    _print_positions(positions_final)

    remaining = _find_position_by_ticket(positions_final, created_ticket)
    if remaining is not None:
        print("警告: 検証で作成したポジションが残っています。")
        if _confirm("残存ポジションを決済しますか?"):
            cleanup_result = close_position(created_ticket)
            print(f"cleanup close result: {cleanup_result}")
            final_after_cleanup = get_position_details(symbol)
            _print_positions(final_after_cleanup)
        else:
            print("警告: ポジションを残したまま終了します。")
    else:
        print("検証で作成したポジションは残っていません。")

    print("\n手動検証を終了します。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())