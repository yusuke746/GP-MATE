from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

load_dotenv(BASE_DIR / ".env")

from agents.data.fred_client import MacroData, get_macro_data
from agents.macro_analyst import MacroAnalysisResult, analyze_macro_environment


SERIES_LABELS: dict[str, str] = {
    "dxy": "DTWEXBGS (広義ドル指数)",
    "real_rate": "DFII10 (10年実質金利)",
    "us10y": "DGS10 (米10年債利回り)",
    "breakeven": "T10YIE (期待インフレ率10年)",
    "fed_funds": "FEDFUNDS (FF金利)",
}


def _fmt_num(value: float | None) -> str:
    if value is None:
        return "None"
    return f"{value:.4f}"


def _print_series(name: str, data: dict[str, Any]) -> None:
    label = SERIES_LABELS.get(name, name)
    value = data.get("value")
    change_30d = data.get("change_30d")
    direction = data.get("direction")
    print(f"- {label}")
    print(f"  value      : {_fmt_num(value if isinstance(value, (int, float)) else None)}")
    print(f"  change_30d : {_fmt_num(change_30d if isinstance(change_30d, (int, float)) else None)}")
    print(f"  direction  : {direction}")


def _print_fred_data(fred_data: MacroData) -> None:
    meta = fred_data.get("_meta", {})
    print("\n=== FRED Data ===")
    print(f"as_of: {fred_data.get('as_of', '')}")
    print(
        "_meta: "
        f"ok={meta.get('ok')} "
        f"cached={meta.get('cached')} "
        f"source={meta.get('source')} "
        f"fetched_at={meta.get('fetched_at')}"
    )
    if not bool(meta.get("ok")):
        print(f"error: {meta.get('error', '')}")

    _print_series("dxy", fred_data.get("dxy", {}))
    _print_series("real_rate", fred_data.get("real_rate", {}))
    _print_series("us10y", fred_data.get("us10y", {}))
    _print_series("breakeven", fred_data.get("breakeven", {}))
    _print_series("fed_funds", fred_data.get("fed_funds", {}))


def _print_macro_result(result: MacroAnalysisResult) -> None:
    meta = result.get("_meta", {})
    print("\n=== Macro Analyst Output ===")
    print(f"macro_bias: {result.get('macro_bias')}")
    print(f"confidence: {float(result.get('confidence', 0.0) or 0.0):.3f}")
    print(f"key_drivers: {result.get('key_drivers', [])}")
    print(f"reasoning: {result.get('reasoning', '')}")
    print(
        "_meta: "
        f"ok={meta.get('ok')} "
        f"model={meta.get('model')} "
        f"error={meta.get('error')}"
    )
    usage = meta.get("usage", {})
    print(
        "usage: "
        f"prompt_tokens={usage.get('prompt_tokens', 0)} "
        f"completion_tokens={usage.get('completion_tokens', 0)} "
        f"total_tokens={usage.get('total_tokens', 0)}"
    )


def _print_visual_checks(fred_data: MacroData, result: MacroAnalysisResult) -> None:
    dxy_direction = str(fred_data.get("dxy", {}).get("direction", "FLAT") or "FLAT")
    real_rate_value = fred_data.get("real_rate", {}).get("value")
    dxy_value = fred_data.get("dxy", {}).get("value")
    key_drivers_text = " | ".join(result.get("key_drivers", []))
    reasoning_text = str(result.get("reasoning", ""))

    print("\n=== Visual Checks ===")
    print(
        "- 実質金利2.20%前後でも単純にBEARISHへ倒れていないか: "
        f"{'OK' if result.get('macro_bias') != 'BEARISH' or dxy_direction == 'UP' else 'CHECK'}"
    )
    print(
        "- reasoningに逆相関崩壊の理解があるか: "
        f"{'OK' if '逆相関' in reasoning_text else 'CHECK'}"
    )
    print(
        "- DXYを水準ではなく方向で見ているか: "
        f"{'OK' if 'DTWEXBGS' in key_drivers_text else 'CHECK'}"
    )
    print(
        "- key_driversにDXY方向が主軸としてあるか: "
        f"{'OK' if 'ドル' in key_drivers_text else 'CHECK'}"
    )
    print(
        "- macro_biasが環境に対して妥当か: "
        f"bias={result.get('macro_bias')} / dxy={_fmt_num(dxy_value if isinstance(dxy_value, (int, float)) else None)} / real_rate={_fmt_num(real_rate_value if isinstance(real_rate_value, (int, float)) else None)}"
    )


def main() -> int:
    print("=== GP-MATE Macro Analyst Check ===")
    print("注意: データ取得と分析のみ。発注は行いません。")
    print("実行は1回だけを想定しています。")

    fred_data = get_macro_data(force_refresh=True)
    _print_fred_data(fred_data)

    result = analyze_macro_environment(fred_data)
    _print_macro_result(result)
    _print_visual_checks(fred_data, result)

    if not bool(fred_data.get("_meta", {}).get("ok", False)):
        print("\n結果: FRED取得失敗のため、マクロ分析は安全側フォールバックでした。")
        return 1

    if not bool(result.get("_meta", {}).get("ok", False)):
        print("\n結果: マクロ分析官はLLM失敗のため、安全側フォールバックでした。")
        return 1

    print("\n結果: FREDデータとマクロ分析官の出力は正常です。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
