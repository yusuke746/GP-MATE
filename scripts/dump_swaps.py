"""MT5から現在のスワップ仕様を logs/swap_info.json に書き出す（baseline.py が読む）。
    python scripts/dump_swaps.py
対象は logs/ohlcv_*_D1.csv がある銘柄。"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data import mt5_client  # noqa: E402
from data.mt5_client import connect, disconnect  # noqa: E402

symbols = [f.stem[len("ohlcv_"):-len("_D1")] for f in Path("logs").glob("ohlcv_*_D1.csv")]
if not connect():
    raise SystemExit("MT5 connect failed")
try:
    out = {}
    for sym in symbols:
        mt5_client.mt5.symbol_select(sym, True)
        i, t = mt5_client.mt5.symbol_info(sym), mt5_client.mt5.symbol_info_tick(sym)
        out[sym] = {"swap_long": i.swap_long, "swap_short": i.swap_short, "swap_mode": i.swap_mode,
                    "swap_rollover3days": i.swap_rollover3days, "point": i.point,
                    "contract_size": i.trade_contract_size, "price": (t.bid + t.ask) / 2}
        print(sym, out[sym])
    Path("logs/swap_info.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
finally:
    disconnect()
