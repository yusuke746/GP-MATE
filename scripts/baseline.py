"""
baseline.py — 多銘柄・日足トレンドフォローのベースライン計測（コスト込み）。

目的: 「これを超えなければ採用しない」という基準線を作る。
      SMC・ニュースセンチメント等は、後でこの結果に対する上乗せとして評価する。

使い方:
    python baseline.py                 # logs/ohlcv_*_D1.csv を全部使う
    python baseline.py --synthetic     # MT5なしで動作確認（ランダムデータ）
    python baseline.py --plot          # 資産曲線を baseline_equity.png に保存

設計:
  - シグナルは終値で確定 → 翌日のリターンに適用（先読みなし）
  - サイズ = 目標ボラ / 実現ボラ（ボラティリティ・ターゲティング）、レバ上限あり
  - コスト = 売買回転 × 片道コスト + 保有 × スワップ年率
  - パラメータは少数。複数の期間で走らせ、特定の値に依存していないか見る
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# ───────────── 設定 ─────────────
PORT_TARGET_VOL = 0.10      # ポートフォリオ目標ボラ（年率）
VOL_LOOKBACK = 30           # 実現ボラの半減期（日）
MAX_LEV_PER_ASSET = 2.0     # 1銘柄あたりの想定元本/資金の上限
REBALANCE_BAND = 0.20       # 目標ウェイトから20%以上ズレたら調整（無駄な回転を抑える）
ANN = 252

# 片道コスト(bps)。未指定の銘柄はCSVのspread列から自動推定する。
COST_BPS = {"default": 3.0}   # 銘柄名をキーに手動指定すれば自動推定より優先
# スワップ年率（保有元本に対する%）。XMは買い・売りとも不利なことが多いので両方向に課す仮値。
# 実値は MT5 の仕様から換算して上書きすること。
SWAP_ANNUAL = {"default": 0.04}   # logs/swap_info.json が無い時だけ使う仮値（買い売り両方向に課す）
SWAP_LS: dict = {}                # sym -> (買いの年率コスト, 売りの年率コスト)。負ならスワップ受取


def load_swaps(path="logs/swap_info.json"):
    """scripts/dump_swaps.py の出力を年率コストに換算。現在のスワップを全期間に適用する近似
    （過去の金利水準は反映されない）。"""
    import json
    f = Path(path)
    if not f.exists():
        print("  ! logs/swap_info.json なし → スワップは仮値 4%/年・両方向（かなり悲観的）")
        return
    for sym, m in json.loads(f.read_text(encoding="utf-8")).items():
        px, mode = m["price"], m["swap_mode"]
        if mode == 1:                      # ポイント建て（XMの標準）。水曜3倍込みで年365日分
            conv = lambda v: -v * m["point"] * 365 / px
        elif mode in (5, 6):               # 年率%建て
            conv = lambda v: -v / 100
        elif mode == 3:                    # 1ロットあたりの金額（証拠金通貨建て）。指数・原油CFDは証拠金通貨=建値通貨と仮定
            conv = lambda v: -v * 365 / (m["contract_size"] * px)
        elif mode == 0:
            conv = lambda v: 0.0
        else:
            print(f"  ! {sym}: swap_mode={mode} は未対応 → 仮値を使用"); continue
        SWAP_LS[sym] = (conv(m["swap_long"]), conv(m["swap_short"]))
        print(f"  swap[{sym}] 買い {SWAP_LS[sym][0]*100:+.2f}%/年  売り {SWAP_LS[sym][1]*100:+.2f}%/年 （+がコスト）")

def group_of(sym: str) -> str:
    """相関の高い銘柄をまとめるグループ。リスクはグループ間で均等、グループ内で均等に配る。"""
    u = sym.rstrip("#").upper()
    if any(k in u for k in ("GOLD", "SILVER", "XAU", "XAG", "XPD", "XPT")):
        return "metals"
    if any(k in u for k in ("OIL", "BRENT", "NGAS")):
        return "energy"
    if len(u) == 6 and u.isalpha():
        return "fx_usd" if "USD" in u else "fx_cross"
    return "index_dip" if DIP_MODE else "index"


PORT_VOL_HALFLIFE = 60      # ポートフォリオ全体のボラ調整に使う実現ボラの半減期（日）
MAX_PORT_SCALE = 3.0

# 走らせるシグナル（名前, 関数, パラメータ）
def sig_donchian(close: pd.Series, n: int) -> pd.Series:
    """n日高値ブレイクで買い、n/2日安値割れで手仕舞い（売りは対称）。"""
    hi_in, lo_in = close.rolling(n).max().shift(1), close.rolling(n).min().shift(1)
    hi_out, lo_out = close.rolling(n // 2).max().shift(1), close.rolling(n // 2).min().shift(1)
    pos = np.zeros(len(close))
    c = close.values
    for i in range(1, len(c)):
        p = pos[i - 1]
        if p == 0:
            p = 1 if c[i] > hi_in.iloc[i] else (-1 if c[i] < lo_in.iloc[i] else 0)
        elif p == 1 and c[i] < lo_out.iloc[i]:
            p = -1 if c[i] < lo_in.iloc[i] else 0
        elif p == -1 and c[i] > hi_out.iloc[i]:
            p = 1 if c[i] > hi_in.iloc[i] else 0
        pos[i] = p
    return pd.Series(pos, index=close.index)


def sig_ma(close: pd.Series, n: int) -> pd.Series:
    """EMA(n/4) と EMA(n) の位置関係。"""
    fast, slow = close.ewm(span=n // 4).mean(), close.ewm(span=n).mean()
    s = np.sign(fast - slow)
    s.iloc[:n] = 0
    return s


DIP_HOLD = 10   # 営業日。イベントスタディの h=10 に対応
DIP_MODE = False
OPENS: dict = {}


def sig_dip(close: pd.Series) -> pd.Series:
    """指数の押し目買い: 終値が BB(20, 2.0) 下限を初めて割った日に買い、DIP_HOLD 日保有。買いのみ。
    research/signals.py の bb_reversion（買い側）と同じ定義。ここは結果を見て調整しないこと。"""
    basis = close.rolling(20, min_periods=20).mean()
    lower = basis - 2.0 * close.rolling(20, min_periods=20).std(ddof=0)
    below = (close < lower).fillna(False)
    trigger = below & ~below.shift(1, fill_value=False)
    return (trigger.astype(float).rolling(DIP_HOLD, min_periods=1).max() > 0).astype(float)


SIGNALS = [("donchian", sig_donchian, [50, 100, 200]),
           ("ma", sig_ma, [80, 160, 320])]


# ───────────── データ ─────────────
def load_data(folder="logs") -> pd.DataFrame:
    """GP-MATE の scripts/export_ohlcv.py が出す logs/ohlcv_<symbol>_D1.csv を読む。
    CSVのspread列（ポイント）から片道コストを推定し、COST_BPSに手動指定が無い銘柄に使う。"""
    closes = {}
    for f in sorted(Path(folder).glob("ohlcv_*_D1.csv")):
        sym = f.stem[len("ohlcv_"):-len("_D1")]
        df = pd.read_csv(f)
        df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True)
        df = df.drop_duplicates("time_utc").set_index("time_utc").sort_index()
        # D1足の開始はサーバー0時 = UTC 21/22時（前日）。+3hしてサーバー日付に戻す。
        df.index = (df.index.tz_localize(None) + pd.Timedelta(hours=3)).normalize()
        df = df[df.index.weekday < 5]          # 日曜の短い足を除外
        df = df[~df.index.duplicated(keep="last")]
        closes[sym] = df["close"]
        OPENS[sym] = df["open"]
        sp = df["spread"][df["spread"] > 0].iloc[-500:]
        if sym not in COST_BPS and len(sp):
            digits = next(d for d in range(7) if np.allclose(df["close"].iloc[-500:] * 10**d,
                                                             np.round(df["close"].iloc[-500:] * 10**d), atol=1e-6))
            half_spread_bps = sp.median() * 10.0**-digits / df["close"].iloc[-1] / 2 * 1e4
            COST_BPS[sym] = round(half_spread_bps + 1.0, 2)   # +1bps は滑りの見込み
            print(f"  cost[{sym}] = {COST_BPS[sym]} bps/片道（CSVスプレッドから推定）")
    if not closes:
        raise SystemExit("logs/ohlcv_*_D1.csv がない。先に export_ohlcv.py --tf D1 を実行するか --synthetic で確認。")
    return pd.DataFrame(closes).sort_index()


def synthetic(n_assets=8, years=20, seed=1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2006-01-02", periods=years * ANN)
    out = {}
    for k in range(n_assets):
        vol = rng.uniform(0.08, 0.30) / np.sqrt(ANN)
        # ゆっくり変わるドリフト（弱いトレンド性）＋ノイズ
        drift = pd.Series(rng.normal(0, 1, len(idx))).ewm(span=250).mean().values * vol * 0.6
        out[f"SYN{k}"] = 100 * np.exp(np.cumsum(drift + rng.normal(0, vol, len(idx))))
    return pd.DataFrame(out, index=idx)


# ───────────── バックテスト ─────────────
def backtest(closes: pd.DataFrame, sig_fn, param) -> dict:
    rets = closes.pct_change()
    groups = {sym: group_of(sym) for sym in closes.columns}
    n_groups = len(set(groups.values()))
    n_in = pd.Series(groups).value_counts()
    # グループ間は無相関、グループ内は完全相関とみなす保守的な配分。残りのズレは後段の全体ボラ調整で吸収
    target_of = {sym: PORT_TARGET_VOL / np.sqrt(n_groups) / n_in[g] for sym, g in groups.items()}

    pnl, turnover, gross, spread_c, swap_c, weights = {}, {}, {}, {}, {}, {}
    for sym in closes.columns:
        c = closes[sym].dropna()
        r = c.pct_change()                 # 他銘柄の休場日でNaNが混ざらないよう銘柄単体で計算
        vol = r.ewm(halflife=VOL_LOOKBACK, min_periods=VOL_LOOKBACK).std() * np.sqrt(ANN)
        is_dip = groups[sym] == "index_dip"
        raw_sig = sig_dip(c) if is_dip else sig_fn(c, param)
        target = (raw_sig * target_of[sym] / vol).clip(-MAX_LEV_PER_ASSET, MAX_LEV_PER_ASSET).fillna(0)

        # バンド付きリバランス: 向きが変わるか、ズレが大きい時だけ建玉を変える
        w = np.zeros(len(target)); t = target.values
        for i in range(1, len(t)):
            prev = w[i - 1]
            if np.sign(t[i]) != np.sign(prev) or abs(t[i] - prev) > REBALANCE_BAND * max(abs(prev), 1e-9):
                w[i] = t[i]
            else:
                w[i] = prev
        w = pd.Series(w, index=c.index)
        weights[sym] = w

        held = w.shift(1).fillna(0)                   # 終値で決めて翌日から保有
        if is_dip and sym in OPENS:                   # 押し目は翌日始値で入る（シグナル当日終値→翌始値の窓は取らない）
            o = OPENS[sym].reindex(c.index)
            entry_day = (held > 0) & (held.shift(1).fillna(0) == 0)
            r = r.where(~entry_day, c / o - 1)
        trade = w.diff().abs().fillna(0)
        cost = trade.shift(1).fillna(0) * COST_BPS.get(sym, COST_BPS["default"]) / 1e4
        if sym in SWAP_LS:
            cl, cs = SWAP_LS[sym]
            swap = (held.clip(lower=0) * cl + (-held).clip(lower=0) * cs) / ANN
        else:
            swap = held.abs() * SWAP_ANNUAL.get(sym, SWAP_ANNUAL["default"]) / ANN
        gross[sym], spread_c[sym], swap_c[sym] = held * r, cost, swap
        pnl[sym] = held * r - cost - swap
        turnover[sym] = trade.sum() / (len(c) / ANN)

    pnl = pd.DataFrame(pnl).fillna(0)
    rv = pnl.sum(axis=1).ewm(halflife=PORT_VOL_HALFLIFE, min_periods=PORT_VOL_HALFLIFE).std() * np.sqrt(ANN)
    scale_raw = (PORT_TARGET_VOL / rv).clip(upper=MAX_PORT_SCALE).fillna(1.0)
    scale = scale_raw.shift(1).fillna(1.0)
    pnl = pnl.mul(scale, axis=0)
    tot = lambda d: pd.DataFrame(d).fillna(0).mul(scale, axis=0).sum(axis=1)
    return {"pnl": pnl, "port": pnl.sum(axis=1), "turnover": pd.Series(turnover),
            "gross": tot(gross), "spread": tot(spread_c), "swap": tot(swap_c),
            "gross_by": pd.DataFrame(gross).fillna(0).mul(scale, axis=0),
            "swap_by": pd.DataFrame(swap_c).fillna(0).mul(scale, axis=0), "groups": groups,
            # 実運用用: 直近確定足で決まった「次の日に持つべき」ウェイト（資金比の想定元本）
            "next_weights": {sym: float(w.iloc[-1]) * float(scale_raw.iloc[-1]) for sym, w in weights.items()}}


def stats(r: pd.Series) -> dict:
    r = r[r.ne(0).idxmax():] if r.ne(0).any() else r   # 最初の建玉以降
    eq = (1 + r).cumprod()
    dd = eq / eq.cummax() - 1
    under = (dd < 0).astype(int)
    longest = int(under.groupby((under == 0).cumsum()).sum().max())
    yrs = len(r) / ANN
    vol = r.std() * np.sqrt(ANN)
    return {"CAGR%": round((eq.iloc[-1] ** (1 / yrs) - 1) * 100, 2),
            "Vol%": round(vol * 100, 2),
            "Sharpe": round(r.mean() * ANN / vol, 2) if vol > 0 else 0.0,
            "MaxDD%": round(dd.min() * 100, 1),
            "最長DD(日)": longest}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--core", action="store_true", help="FXクロスを除外（USDストレートと情報が重複するため）")
    ap.add_argument("--dip", action="store_true", help="指数をトレンドではなく押し目買い(BB下限→10日保有)で運用")
    ap.add_argument("--end", default=None, help="この日付までで評価（例: 2022-12-31）")
    a = ap.parse_args()
    global DIP_MODE
    DIP_MODE = a.dip

    closes = synthetic() if a.synthetic else load_data()
    if not a.synthetic:
        load_swaps()
    if a.core:
        closes = closes[[c for c in closes.columns if group_of(c) != "fx_cross"]]
    if a.end:
        closes = closes.loc[:a.end]
    print("グループ:", pd.Series({c: group_of(c) for c in closes.columns}).value_counts().to_dict())
    print(f"銘柄数 {closes.shape[1]} / 期間 {closes.index[0].date()} → {closes.index[-1].date()}\n")

    rows, ports, parts = [], {}, {"gross": {}, "spread": {}, "swap": {}}
    for name, fn, params in SIGNALS:
        for p in params:
            res = backtest(closes, fn, p)
            key = f"{name}_{p}"
            ports[key] = res["port"]
            for k in parts:
                parts[k][key] = res[k]
            rows.append({"signal": key, **stats(res["port"]),
                         "年間回転(倍)": round(res["turnover"].sum(), 1)})
    summary = pd.DataFrame(rows).set_index("signal")

    # 全バリエーション等ウェイト合成 = 特定パラメータに賭けない版（これを基準線とする）
    combo = pd.DataFrame(ports).mean(axis=1)
    summary.loc["COMBO(全平均)"] = {**stats(combo), "年間回転(倍)": np.nan}
    print("=== パラメータ別（コスト込み） ===")
    print(summary.to_string(), "\n")

    # コスト分解: どこでエッジが消えているか
    g, sp, sw = (pd.DataFrame(parts[k]).mean(axis=1) for k in ("gross", "spread", "swap"))
    print("=== COMBO コスト分解 ===")
    print(pd.DataFrame({"コスト前": stats(g), "スプレッド後": stats(g - sp), "スワップ後(最終)": stats(g - sp - sw)}).to_string())
    print(f"年間コスト: スプレッド {sp.mean()*ANN*100:.2f}% / スワップ {sw.mean()*ANN*100:.2f}%\n")

    # 銘柄別の寄与（代表としてdonchian_100）
    rep = backtest(closes, sig_donchian, 100)["pnl"]
    r100 = backtest(closes, sig_donchian, 100)
    per = pd.DataFrame({s: stats(rep[s]) for s in rep.columns}).T[["Sharpe", "MaxDD%"]]
    per["コスト前Sharpe"] = [stats(r100["gross_by"][s])["Sharpe"] for s in rep.columns]
    per["スワップ%/年"] = [round(r100["swap_by"][s].mean() * ANN * 100, 2) for s in rep.columns]
    print("=== 銘柄別 (donchian_100) ===")
    print(per.to_string(), "\n")

    gsr = pd.Series(r100["groups"])
    print("=== グループ別 (donchian_100) ===")
    print(pd.DataFrame({g: {"銘柄数": int((gsr == g).sum()),
                            "コスト前Sharpe": stats(r100["gross_by"][gsr[gsr == g].index].sum(axis=1))["Sharpe"],
                            "最終Sharpe": stats(rep[gsr[gsr == g].index].sum(axis=1))["Sharpe"]}
                        for g in sorted(gsr.unique())}).T.to_string(), "\n")

    gp = pd.DataFrame({g: rep[gsr[gsr == g].index].sum(axis=1) for g in sorted(gsr.unique())})
    if gp.shape[1] > 1:
        print("=== グループ間の月次相関 (donchian_100) ===")
        print(gp.resample("ME").sum().corr().round(2).to_string(), "\n")

    yearly = combo.groupby(combo.index.year).apply(lambda x: round(((1 + x).prod() - 1) * 100, 1))
    print("=== COMBO 年次リターン% ===")
    print(yearly.to_string(), "\n")

    # 前半/後半の分割: エッジが時代で消えていないか
    mid = len(combo) // 2
    print("=== COMBO 前半 / 後半 ===")
    print(pd.DataFrame({"前半": stats(combo.iloc[:mid]), "後半": stats(combo.iloc[mid:])}).to_string())

    summary.to_csv("baseline_summary.csv")
    combo.to_csv("baseline_combo_returns.csv", header=["ret"])
    if a.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ax = (1 + pd.DataFrame(ports)).cumprod().plot(figsize=(11, 6), alpha=0.5, lw=1)
        (1 + combo).cumprod().plot(ax=ax, color="black", lw=2, label="COMBO")
        ax.set_yscale("log"); ax.legend(); ax.set_title("Baseline trend following (after costs)")
        plt.savefig("baseline_equity.png", dpi=120, bbox_inches="tight")
        print("\nbaseline_equity.png を保存")


if __name__ == "__main__":
    main()
