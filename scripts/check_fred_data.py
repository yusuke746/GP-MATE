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


SERIES_HINTS: dict[str, str] = {
    "dxy": "DXY(DTWEXBGS): 2026年時点でおおよそ100前後か",
    "real_rate": "実質金利(DFII10): おおよそ1-2%台か",
    "us10y": "10年債(DGS10): おおよそ4%前後か",
    "breakeven": "期待インフレ率(T10YIE): おおよそ2%前後か",
    "fed_funds": "FF金利(FEDFUNDS): 政策金利として妥当な範囲か",
}


def _fmt_num(value: float | None) -> str:
    if value is None:
        return "None"
    return f"{value:.4f}"


def _print_series(name: str, label: str, data: dict[str, Any]) -> None:
    value = data.get("value")
    change_30d = data.get("change_30d")
    direction = data.get("direction")
    print(f"- {label}")
    print(f"  value      : {_fmt_num(value if isinstance(value, (int, float)) else None)}")
    print(f"  change_30d : {_fmt_num(change_30d if isinstance(change_30d, (int, float)) else None)}")
    print(f"  direction  : {direction}")
    print(f"  hint       : {SERIES_HINTS.get(name, '')}")


def _print_result(title: str, result: MacroData) -> None:
    meta = result.get("_meta", {})
    print(f"\n=== {title} ===")
    print(f"as_of: {result.get('as_of', '')}")
    print(
        "_meta: "
        f"ok={meta.get('ok')} "
        f"cached={meta.get('cached')} "
        f"source={meta.get('source')} "
        f"fetched_at={meta.get('fetched_at')}"
    )
    if not bool(meta.get("ok")):
        print(f"error: {meta.get('error', '')}")

    _print_series("dxy", "dxy / DTWEXBGS", result.get("dxy", {}))
    _print_series("real_rate", "real_rate / DFII10", result.get("real_rate", {}))
    _print_series("us10y", "us10y / DGS10", result.get("us10y", {}))
    _print_series("breakeven", "breakeven / T10YIE", result.get("breakeven", {}))
    _print_series("fed_funds", "fed_funds / FEDFUNDS", result.get("fed_funds", {}))


def _print_cache_summary(first: MacroData, second: MacroData) -> None:
    first_meta = first.get("_meta", {})
    second_meta = second.get("_meta", {})
    print("\n=== Cache Check ===")
    print(f"1st call ok={first_meta.get('ok')} cached={first_meta.get('cached')}")
    print(f"2nd call ok={second_meta.get('ok')} cached={second_meta.get('cached')}")
    if bool(first_meta.get("ok")) and bool(second_meta.get("ok")):
        if bool(second_meta.get("cached")):
            print("result: 2回目はキャッシュ返却。プロセス内でFRED実リクエストは1回のみの想定。")
        else:
            print("result: 2回目がキャッシュではありません。キャッシュ動作を再確認してください。")


def main() -> int:
    print("=== GP-MATE FRED Data Check ===")
    print("注意: データ取得のみ。発注は行いません。")

    first = get_macro_data(force_refresh=True)
    _print_result("First fetch", first)

    second = get_macro_data()
    _print_result("Second fetch", second)
    _print_cache_summary(first, second)

    return 0 if bool(first.get("_meta", {}).get("ok")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
