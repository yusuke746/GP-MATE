from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import main as strategy_main
from data.mt5_client import get_account_info


def _trade_mode_name(trade_mode: int) -> str:
    if trade_mode == 0:
        return "DEMO"
    if trade_mode == 2:
        return "REAL"
    return f"OTHER({trade_mode})"


def _confirm_order(action: str, symbol: str, lot: float, sl: float, tp: float) -> bool:
    prompt = (
        f"発注確認: {action} {symbol} lot={lot} sl={sl} tp={tp} を実行しますか? [y/N]: "
    )
    answer = input(prompt).strip().lower()
    return answer == "y"


def main() -> int:
    runtime_notice = strategy_main.python_runtime_notice()
    if runtime_notice:
        print(f"[Runtime Notice] {runtime_notice}")

    info = get_account_info()
    if not info.get("success"):
        print("接続失敗。手動実行を中止します。", info.get("reason", ""))
        return 1

    account = info.get("data")
    if account is None:
        print("口座情報が取得できないため中止します。")
        return 1

    mode = _trade_mode_name(account.trade_mode)
    print("=== GP-MATE Manual Run ===")
    print(f"Account: login={account.login} server={account.server} mode={mode}")
    print("注意: デモ口座であることを確認してから続行してください。")

    original_send_order = strategy_main.send_order

    def guarded_send_order(symbol: str, action: str, lot: float, sl: float, tp: float) -> dict[str, Any]:
        if not _confirm_order(action=action, symbol=symbol, lot=lot, sl=sl, tp=tp):
            return {
                "success": False,
                "retcode": None,
                "reason": "User canceled order",
            }
        return original_send_order(symbol=symbol, action=action, lot=lot, sl=sl, tp=tp)

    strategy_main.send_order = guarded_send_order
    try:
        result = strategy_main.run_once()
    finally:
        strategy_main.send_order = original_send_order

    print("\nResult:")
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    print(f"\nLog: {strategy_main.TRADE_LOG_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
