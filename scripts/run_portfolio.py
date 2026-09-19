"""
run_portfolio.py — 日足ポートフォリオ（トレンド: metals/fx_usd/energy ＋ 指数押し目買い）の日次実行。

    python scripts/run_portfolio.py                 # ドライラン（発注しない。まずこれを数週間）
    python scripts/run_portfolio.py --live          # 実発注
    python scripts/run_portfolio.py --reset-halt    # 停止状態を手動解除

シグナルとサイズ計算は scripts/baseline.py をそのまま呼ぶ（バックテストと同一ロジック）。
このスクリプトがやるのは「目標ウェイト → ロット換算 → 現在建玉との差分を発注」だけ。

■ 必ず GP-MATE とは別の口座・別のMT5ターミナルで動かすこと
  GP-MATE の get_positions はマジックナンバーで絞らないため、同じ口座で GOLD# を持つと
  GP-MATE の建値移動やポジション評価がこちらの建玉を触る。接続先は環境変数で指定:
    PF_MT5_PATH / PF_MT5_LOGIN / PF_MT5_PASSWORD / PF_MT5_SERVER

■ 実行タイミング: 月〜金、サーバー時間 01:10 頃（指数の取引再開後）= 日本時間 夏 07:10 / 冬 08:10。
  差分発注なので同じ日に複数回走らせても安全。
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline as bl  # noqa: E402

MAGIC = 20260919                 # GP-MATE(20260702)と別
BARS = 700                       # シグナル・ボラ推定に使う日足本数
MAX_GROSS = 5.0                  # 総想定元本 / 資金 の上限
HALT_DD = 0.30                   # 資金がピークから30%減で全決済して停止（手動解除）
EXEC_BAND = 0.10                 # 目標と現在の差がこの比率未満なら発注しない
STATE = Path("logs/portfolio_state.json")
RUNLOG = Path("logs/portfolio_runs.jsonl")


def universe() -> list[str]:
    """バックテストと同じ銘柄集合: logs/ にD1のCSVがある銘柄からFXクロスを除いたもの（--core と同じ規則）。"""
    syms = [f.stem[len("ohlcv_"):-len("_D1")] for f in sorted(Path("logs").glob("ohlcv_*_D1.csv"))]
    return [s for s in syms if bl.group_of(s) != "fx_cross"]


def load_live(mt5, symbols: list[str], force: bool) -> pd.DataFrame:
    closes = {}
    for sym in symbols:
        if not mt5.symbol_select(sym, True):
            print(f"  ! {sym}: 選択できない → 除外"); continue
        rates = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_D1, 0, BARS)
        if rates is None or len(rates) < 300:
            print(f"  ! {sym}: 日足が不足 → 除外"); continue
        df = pd.DataFrame(rates)
        # 最後の足は形成中のはず。出来高が大きければ確定足を捨てることになるので止める。
        if not force and df["tick_volume"].iloc[-1] > 0.5 * df["tick_volume"].iloc[-30:-1].median():
            raise SystemExit(f"{sym}: 最新の日足がほぼ確定している。日次ロールオーバー直後に実行すること（--force で無視）")
        df = df.iloc[:-1]
        df.index = pd.to_datetime(df["time"], unit="s").dt.normalize()   # MT5の足時刻はサーバー時間
        df = df[~df.index.duplicated(keep="last")]
        closes[sym] = df["close"]
        bl.OPENS[sym] = df["open"]
    return pd.DataFrame(closes).sort_index()


def target_weights(closes: pd.DataFrame) -> pd.Series:
    """6バリエーションの平均（バックテストのCOMBOと同じ）。"""
    ws = [bl.backtest(closes, fn, p)["next_weights"] for _, fn, params in bl.SIGNALS for p in params]
    w = pd.DataFrame(ws).mean()
    gross = w.abs().sum()
    return w * (MAX_GROSS / gross) if gross > MAX_GROSS else w


def lot_value(mt5, sym: str) -> float:
    """1ロットの想定元本（口座通貨建て）。通貨換算はMT5に任せる。"""
    t = mt5.symbol_info_tick(sym)
    px = (t.bid + t.ask) / 2
    p = mt5.order_calc_profit(mt5.ORDER_TYPE_BUY, sym, 1.0, px, px * 1.01)
    if p is None or p <= 0:
        raise RuntimeError(f"{sym}: order_calc_profit 失敗 {mt5.last_error()}")
    return p / 0.01


def to_lots(mt5, sym: str, notional: float) -> float:
    i = mt5.symbol_info(sym)
    raw = notional / lot_value(mt5, sym)
    lots = round(round(raw / i.volume_step) * i.volume_step, 8)
    if abs(lots) < i.volume_min:
        return 0.0
    return float(np.sign(lots) * min(abs(lots), i.volume_max))


def my_positions(mt5) -> dict[str, list]:
    out: dict[str, list] = {}
    for p in mt5.positions_get() or []:
        if p.magic == MAGIC:
            out.setdefault(p.symbol, []).append(p)
    return out


def send(mt5, sym: str, lots: float, ticket: int | None = None) -> dict:
    """lots>0 で買い、<0 で売り。ticket 指定時はその建玉の（部分）決済。"""
    t = mt5.symbol_info_tick(sym)
    req = {"action": mt5.TRADE_ACTION_DEAL, "symbol": sym, "volume": abs(lots),
           "type": mt5.ORDER_TYPE_BUY if lots > 0 else mt5.ORDER_TYPE_SELL,
           "price": t.ask if lots > 0 else t.bid, "deviation": 30, "magic": MAGIC,
           "comment": "PF", "type_time": mt5.ORDER_TIME_GTC, "type_filling": mt5.ORDER_FILLING_IOC}
    if ticket is not None:
        req["position"] = ticket
    r = mt5.order_send(req)
    ok = r is not None and r.retcode == mt5.TRADE_RETCODE_DONE
    return {"symbol": sym, "lots": lots, "close_ticket": ticket, "ok": ok,
            "retcode": None if r is None else r.retcode, "comment": None if r is None else r.comment}


def rebalance(mt5, sym: str, target: float, positions: list, live: bool) -> list[dict]:
    """両建て口座でも動くよう、減らす時は既存建玉を古い順に決済し、増やす時だけ新規に建てる。"""
    signed = lambda p: p.volume if p.type == mt5.POSITION_TYPE_BUY else -p.volume
    current = round(sum(signed(p) for p in positions), 8)
    step = mt5.symbol_info(sym).volume_step
    diff = round(target - current, 8)
    same_side = target * current > 0
    if abs(diff) < step or (same_side and abs(diff) < EXEC_BAND * max(abs(target), abs(current))):
        return []
    plan: list[tuple[float, int | None]] = []
    to_close = abs(current) if not same_side else max(0.0, abs(current) - abs(target))
    for p in sorted(positions, key=lambda p: p.time):
        if to_close < step / 2:
            break
        v = round(min(p.volume, to_close), 8)
        plan.append((-np.sign(signed(p)) * v, p.ticket)); to_close -= v
    to_open = target if not same_side else np.sign(target) * max(0.0, abs(target) - abs(current))
    if abs(to_open) >= step:
        plan.append((round(to_open, 8), None))
    if not live:
        return [{"symbol": sym, "lots": l, "close_ticket": tk, "ok": None, "dry_run": True} for l, tk in plan]
    return [send(mt5, sym, l, tk) for l, tk in plan]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--force", action="store_true", help="日足の確定チェックを無視")
    ap.add_argument("--reset-halt", action="store_true")
    a = ap.parse_args()

    import MetaTrader5 as mt5
    kw = {k: v for k, v in {"path": os.getenv("PF_MT5_PATH"), "password": os.getenv("PF_MT5_PASSWORD"),
                            "server": os.getenv("PF_MT5_SERVER")}.items() if v}
    if os.getenv("PF_MT5_LOGIN"):
        kw["login"] = int(os.environ["PF_MT5_LOGIN"])
    if a.live and "login" not in kw:
        raise SystemExit("--live には PF_MT5_LOGIN 等の指定が必須（GP-MATEの口座に誤発注しないため）")
    if not mt5.initialize(**kw):
        raise SystemExit(f"MT5 initialize failed: {mt5.last_error()}")
    try:
        acc = mt5.account_info()
        if "login" in kw and acc.login != kw["login"]:
            raise SystemExit(f"接続先の口座 {acc.login} が PF_MT5_LOGIN と違う。中止。")
        state = json.loads(STATE.read_text()) if STATE.exists() else {"peak": acc.equity, "halted": False}
        if a.reset_halt:
            state = {"peak": acc.equity, "halted": False}
        state["peak"] = max(state["peak"], acc.equity)
        dd = 1 - acc.equity / state["peak"]
        print(f"口座 {acc.login} / 資金 {acc.equity:,.0f} {acc.currency} / ピーク比 -{dd*100:.1f}% / {'LIVE' if a.live else 'DRY-RUN'}")

        bl.DIP_MODE = True
        bl.load_swaps()
        closes = load_live(mt5, universe(), a.force)
        weights = target_weights(closes)
        if dd >= HALT_DD or state["halted"]:
            state["halted"] = True
            weights[:] = 0.0
            print(f"  !! ドローダウン停止中（-{HALT_DD*100:.0f}%到達）。全決済。解除は --reset-halt")

        held, rows, orders = my_positions(mt5), [], []
        for sym, w in weights.items():
            lots = to_lots(mt5, sym, w * acc.equity)
            cur = sum(p.volume if p.type == mt5.POSITION_TYPE_BUY else -p.volume for p in held.get(sym, []))
            rows.append({"symbol": sym, "group": bl.group_of(sym), "目標w": round(w, 3),
                         "目標lot": lots, "現在lot": round(cur, 2),
                         "丸め落ち": bool(lots == 0 and abs(w) > 1e-6)})
            orders += rebalance(mt5, sym, lots, held.get(sym, []), a.live)
        for sym in set(held) - set(weights.index):          # 対象外になった銘柄の残り建玉
            orders += rebalance(mt5, sym, 0.0, held[sym], a.live)

        table = pd.DataFrame(rows).set_index("symbol")
        print(table.to_string())
        lost = table.loc[table["丸め落ち"], "目標w"].abs().sum()
        print(f"\n総想定元本 {weights.abs().sum():.2f}倍 / 最小ロット未満で持てない分 {lost:.2f}倍"
              + ("  ← 大きい場合は資金に対して銘柄が粗い" if lost > 0.2 * weights.abs().sum() else ""))
        print(f"発注 {len(orders)} 件:")
        for o in orders:
            print("  ", o)

        STATE.parent.mkdir(exist_ok=True)
        STATE.write_text(json.dumps(state))
        with RUNLOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": datetime.now(timezone.utc).isoformat(), "live": a.live,
                                "equity": acc.equity, "last_bar": str(closes.index[-1].date()),
                                "weights": weights.round(4).to_dict(), "orders": orders},
                               ensure_ascii=False, default=str) + "\n")
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    main()
