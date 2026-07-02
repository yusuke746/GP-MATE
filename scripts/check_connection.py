from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config import MODEL_ANALYSIS, MODEL_DECISION, SYMBOL
from data import mt5_client
from data.mt5_client import connect, disconnect, get_account_info, get_rates
from data.news_client import check_rss_feeds_health


def _status(ok: bool) -> str:
    return "[OK]" if ok else "[NG]"


def _trade_mode_name(trade_mode: int) -> str:
    if trade_mode == 0:
        return "DEMO"
    if trade_mode == 2:
        return "REAL"
    return f"OTHER({trade_mode})"


def _check_mt5_connect() -> tuple[bool, str]:
    ok = connect()
    if not ok:
        return False, "MT5接続に失敗。MT5起動/ログイン情報/.env(MT5_PATH, MT5_LOGIN)を確認"
    return True, "MT5接続成功"


def _check_symbol_name() -> tuple[bool, str]:
    try:
        if mt5_client.mt5 is None:
            return False, "MetaTrader5モジュールが利用不可"

        symbols = mt5_client.mt5.symbols_get()
        if not symbols:
            return False, "symbols_get()が空。接続状態または口座権限を確認"

        names = [s.name for s in symbols if getattr(s, "name", "")]
        candidates = [n for n in names if "gold" in n.lower() or "xau" in n.lower()]
        if not candidates:
            return False, "GOLD系銘柄が見つかりません"

        first = ", ".join(candidates[:10])
        configured = SYMBOL.strip().upper()
        candidate_set = {name.strip().upper() for name in candidates}
        if configured not in candidate_set:
            hint = (
                f"検出銘柄: {first} | 現在SYMBOL={SYMBOL} "
                "[WARN] configのSYMBOLと検出銘柄が不一致です。"
            )
            return True, hint

        hint = f"検出銘柄: {first} | 現在SYMBOL={SYMBOL}"
        return True, hint
    except Exception as exc:
        return False, f"symbols_get確認失敗: {exc}"


def _check_rates() -> tuple[bool, str]:
    try:
        df = get_rates(SYMBOL, "H1", 5)
        if df.empty:
            return False, f"{SYMBOL}の価格取得に失敗"

        latest = df.tail(3)[["time", "open", "high", "low", "close"]]
        return True, f"価格取得OK\n{latest.to_string(index=False)}"
    except Exception as exc:
        return False, f"get_rates失敗: {exc}"


def _check_account_info() -> tuple[bool, str]:
    info = get_account_info()
    if not info.get("success"):
        return False, str(info.get("reason", "口座情報取得失敗"))

    account = info.get("data")
    if account is None:
        return False, "account_info dataが空"

    message = (
        f"login={account.login}, server={account.server}, "
        f"mode={_trade_mode_name(account.trade_mode)}, balance={account.balance:.2f} {account.currency}"
    )
    return True, message


def _check_openai() -> tuple[bool, str]:
    try:
        from openai import OpenAI  # type: ignore[import-not-found]
        from config import OPENAI_API_KEY

        if not OPENAI_API_KEY:
            return False, "OPENAI_API_KEYが未設定"

        client = OpenAI(api_key=OPENAI_API_KEY)
        response: Any = client.responses.create(
            model=MODEL_ANALYSIS,
            input="ping",
            max_output_tokens=16,
        )
        model = str(getattr(response, "model", MODEL_ANALYSIS))
        output = str(getattr(response, "output_text", "")).strip()
        if not output:
            output = "(empty output)"
        return True, f"OpenAI応答OK model={model}, text={output[:80]}"
    except Exception as exc:
        return False, f"OpenAI疎通失敗: {exc}"


def _check_openai_decision_function_call() -> tuple[bool, str]:
    try:
        from openai import OpenAI  # type: ignore[import-not-found]
        from config import OPENAI_API_KEY

        if not OPENAI_API_KEY:
            return False, "OPENAI_API_KEYが未設定"

        client = OpenAI(api_key=OPENAI_API_KEY)
        response: Any = client.responses.create(
            model=MODEL_DECISION,
            input="Function Calling互換確認。place_trade_orderを呼び出してください。",
            tools=[
                {
                    "type": "function",
                    "name": "place_trade_order",
                    "description": "分析結果に基づき売買判断を実行する",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "action": {"type": "string", "enum": ["BUY", "SELL", "HOLD"]},
                            "symbol": {"type": "string"},
                            "confidence": {"type": "number"},
                            "reasoning": {"type": "string"},
                            "risk_level": {"type": "string", "enum": ["LOW", "MID", "HIGH"]},
                        },
                        "required": ["action", "symbol", "confidence", "reasoning"],
                    },
                }
            ],
            tool_choice={"type": "function", "name": "place_trade_order"},
        )

        model = str(getattr(response, "model", MODEL_DECISION))
        output_items = getattr(response, "output", []) or []
        for item in output_items:
            if getattr(item, "type", "") == "function_call" and getattr(item, "name", "") == "place_trade_order":
                return True, f"Decision Function Call OK model={model}, function=place_trade_order"

        return False, f"Function Callが返りませんでした model={model}"
    except Exception as exc:
        return False, f"Decision Function Call疎通失敗: {exc}"


def _check_rss_feeds() -> tuple[bool, str]:
    results = check_rss_feeds_health()
    if not results:
        return False, "RSS_FEEDSが未設定"

    lines: list[str] = []
    has_live = False
    for item in results:
        ok = bool(item.get("ok"))
        status = item.get("status_code")
        url = str(item.get("url", ""))
        tag = "OK" if ok else "NG"
        if ok:
            has_live = True
        lines.append(f"[{tag}] {status if status is not None else '-'} {url}")

    return has_live, "\n".join(lines)


def main() -> int:
    print("=== GP-MATE Connection Check (No Trading) ===")
    results: list[tuple[str, bool, str]] = []

    mt5_ok, mt5_msg = _check_mt5_connect()
    results.append(("MT5 connect", mt5_ok, mt5_msg))

    if mt5_ok:
        sym_ok, sym_msg = _check_symbol_name()
        results.append(("symbols_get", sym_ok, sym_msg))

        rates_ok, rates_msg = _check_rates()
        results.append(("get_rates", rates_ok, rates_msg))

        acc_ok, acc_msg = _check_account_info()
        results.append(("get_account_info", acc_ok, acc_msg))

        disconnect()

    openai_ok, openai_msg = _check_openai()
    results.append(("OpenAI", openai_ok, openai_msg))

    fc_ok, fc_msg = _check_openai_decision_function_call()
    results.append(("OpenAI Decision FC", fc_ok, fc_msg))

    rss_ok, rss_msg = _check_rss_feeds()
    results.append(("RSS feeds", rss_ok, rss_msg))

    print("")
    has_ng = False
    for name, ok, msg in results:
        print(f"{_status(ok)} {name}: {msg}")
        if not ok:
            has_ng = True

    print("")
    if has_ng:
        print("接続確認にNGがあります。上記ヒントを先に解消してください。")
        return 1

    print("全項目OK。次は scripts/run_manual.py で手動1回実行できます。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
