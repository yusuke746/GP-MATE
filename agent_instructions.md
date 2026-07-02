# GP-MATE 実装設計書

> **GPT-Powered Multi-Agent Trading EA**
> LLMマルチエージェントによるGOLD自動売買システム
> 本設計書はVSCode上のAI coding agentが実装するための仕様書です。

---

## 0. プロジェクト概要

| 項目           | 内容                                            |
| -------------- | ----------------------------------------------- |
| プロジェクト名 | GP-MATE（GPT-Powered Multi-Agent Trading EA）   |
| 目的           | LLMマルチエージェントによるGOLD自動売買システム |
| 対象銘柄       | XAU/USD（GOLD）単一                             |
| ブローカー     | XM Trading（MT5）                               |
| 時間軸         | H4でトレンド判断 → H1でエントリー              |
| 初期資金       | 500,000円                                       |
| LLM            | GPTシリーズ（分析=mini / 最終判断=上位モデル）  |
| 言語           | Python 3.11+                                    |
| 月次API予算    | 5,000円以内                                     |
| 最重要方針     | 資金を溶かさないことを全機能で最優先            |

---

## 1. 技術スタック

```
言語        : Python 3.11+
MT5連携     : MetaTrader5 (pip install MetaTrader5)
LLM         : openai (Function Calling使用)
指標計算    : TA-Lib または pandas-ta
データ処理  : pandas, numpy
ニュース    : requests (NewsAPI/RSS), feedparser
スケジューラ: APScheduler または cron
エージェント: LangGraph（マルチエージェント制御・任意）
設定管理    : python-dotenv (.env でAPIキー管理)
ログ        : logging + CSV出力
```

---

## 2. システムアーキテクチャ

```
┌─────────────────────────────────────────────┐
│              スケジューラ（1日4回起動）             │
└───────────────────┬─────────────────────────┘
                    ↓
┌─────────── L1 データ取得（無料） ───────────┐
│  MT5価格  │  経済指標  │  ニュースRSS/API      │
└───────────────────┬─────────────────────────┘
                    ↓
┌─────────── L2 整形・特徴量 ─────────────────┐
│  TA-Lib（RSI/MACD/BB/ATR）  │  プロンプト生成    │
└───────────────────┬─────────────────────────┘
                    ↓
┌─────────── L3 分析エージェント【mini】 ──────┐
│  テクニカル │ センチメント(FinGPT併用) │ ファンダ │
└───────────────────┬─────────────────────────┘
                    ↓
┌─────────── L3.5 議論【mini・2ラウンド】 ─────┐
│      強気(Bull) ⇄ Bear(弱気) 2ラウンド        │
└───────────────────┬─────────────────────────┘
                    ↓
┌─────────── L4 最終意思決定【上位モデル】 ─────┐
│  トレーダー判断 → BUY/SELL/HOLD + 根拠         │
└───────────────────┬─────────────────────────┘
                    ↓
┌─────────── L4.5 リスク管理【コード】 ─────────┐
│ ATRベースSL/TP │ ロット計算 │ 過剰取引ブロック   │
└───────────────────┬─────────────────────────┘
                    ↓
┌─────────── L5 執行 ─────────────────────────┐
│  MT5 order_send │ 約定確認 │ 全ログ記録         │
└─────────────────────────────────────────────┘
```

---

## 3. ディレクトリ構成

```
gp_mate/
├── .env                    # APIキー等（gitignore対象）
├── config.py               # 全設定の一元管理
├── main.py                 # エントリポイント・スケジューラ
├── data/
│   ├── mt5_client.py       # MT5接続・価格取得・発注
│   └── news_client.py      # ニュース取得・フィルタ
├── indicators/
│   └── ta_calc.py          # RSI/MACD/BB/ATR算出
├── agents/
│   ├── base.py             # LLM呼び出し共通処理
│   ├── technical.py        # テクニカルアナリスト
│   ├── sentiment.py        # センチメントアナリスト
│   ├── debate.py           # Bull/Bear 2ラウンド議論
│   └── trader.py           # 最終判断（Function Calling）
├── risk/
│   └── risk_manager.py     # ロット計算・SL/TP・安全フィルタ
├── backtest/
│   └── timecapsule.py      # Time-Capsule検証（動作確認用）
├── logs/
│   └── trade_log.csv       # 全判断・約定記録
└── tests/
    └── test_*.py           # 各モジュールの単体テスト
```

---

## 4. モジュール仕様

### 4-1. config.py

```python
# 銘柄・時間軸
SYMBOL = "GOLD"              # XMのGOLD銘柄名（要MT5確認: "XAUUSD"の可能性あり）
TIMEFRAME_TREND = "H4"       # トレンド判断
TIMEFRAME_ENTRY = "H1"       # エントリー

# リスク管理（堅実モード）
RISK_PERCENT = 0.01          # 1トレードリスク1%
MAX_POSITIONS = 1            # 最大同時ポジション
CONFIDENCE_THRESHOLD = 0.6   # これ未満はHOLD
MAX_DAILY_LOSS_PCT = 0.03    # 日次損失上限3%
CONSECUTIVE_LOSS_LIMIT = 3   # 3連敗で当日停止

# SL/TP
ATR_MULTIPLIER_SL = 1.5      # SL = ATR × 1.5
RISK_REWARD_RATIO = 2.0      # TP = SL × 2.0

# 実行制御
NEWS_FILTER_MINUTES = 15     # 重要指標±15分は新規禁止
JUDGMENT_TIMES = ["09:00", "16:00", "21:00", "23:30"]  # 1日4回

# LLMモデル（単価確定後に設定）
MODEL_ANALYSIS = "gpt-5.4-mini"   # 分析・議論用
MODEL_DECISION = "gpt-5.5"        # 最終判断用
MAX_NEWS_ITEMS = 15               # ニュース最大件数

# 攻めレバー（Stage制・初期はStage1）
STAGE = 1                         # 1:リスク1% / 2:2% / 3:3-5%+複利
```

### 4-2. data/mt5_client.py

| 関数                                        | 入力         | 出力      | 処理                   |
| ------------------------------------------- | ------------ | --------- | ---------------------- |
| `connect()`                               | -            | bool      | MT5初期化・ログイン    |
| `get_rates(symbol, tf, count)`            | 銘柄,足,本数 | DataFrame | 価格取得（copy_rates） |
| `get_spread(symbol)`                      | 銘柄         | float     | 現在スプレッド         |
| `get_positions(symbol)`                   | 銘柄         | list      | 保有ポジション         |
| `send_order(symbol, action, lot, sl, tp)` | 発注情報     | result    | 発注実行               |
| `get_account_info()`                      | -            | dict      | 残高・証拠金           |

**発注リクエスト仕様：**

```python
import MetaTrader5 as mt5

def send_order(symbol, action, lot, sl, tp):
    order_type = mt5.ORDER_TYPE_BUY if action == "BUY" else mt5.ORDER_TYPE_SELL
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol, "volume": lot, "type": order_type,
        "sl": sl, "tp": tp, "deviation": 20,
        "magic": 20260702, "comment": "GP-MATE",
    }
    return mt5.order_send(request)
```

### 4-3. data/news_client.py

| 関数                                   | 処理                                                        |
| -------------------------------------- | ----------------------------------------------------------- |
| `fetch_news(hours=24, max_items=15)` | NewsAPI+RSSから取得→キーワードフィルタ→重複除去→上位15件 |
| `is_high_impact_soon(minutes=15)`    | 重要指標カレンダー取得→±15分判定                          |
| `get_macro_data()`                   | FRED APIから金利・CPI取得（日次キャッシュ）                 |

**GOLD特化キーワード：**

```python
GOLD_KEYWORDS = ["gold", "xau", "fed", "fomc", "rate", "rates",
                 "inflation", "cpi", "pce", "dollar", "powell",
                 "treasury", "yield", "geopolitical"]
```

**無料ニュースソース構成（3層）：**

| 層                 | ソース                      | 取得内容                | 費用   |
| ------------------ | --------------------------- | ----------------------- | ------ |
| L1：経済指標       | ForexFactory RSS / investpy | 指標カレンダー・結果    | 無料   |
| L2：ニュース見出し | NewsAPI（無料枠）+ RSS      | 金・ドル・FRB関連見出し | 無料枠 |
| L3：マクロ数値     | FRED API / Alpha Vantage    | 金利・CPI等の生数値     | 無料   |

### 4-4. indicators/ta_calc.py

```python
# 入力: 価格DataFrame → 出力: 指標を付与したDataFrame
- RSI(14)
- MACD(12, 26, 9)
- ボリンジャーバンド(20, 2)
- ATR(14)          # SL/TP計算の核
- 直近高値/安値(直近20本)
```

### 4-5. agents/（マルチエージェント）

**共通仕様（base.py）：**

- LLM呼び出しラッパー（モデル指定・リトライ・トークン記録）
- 全エージェントJSON出力を強制（パース確実性のため）
- エラー時は安全側（HOLD）にフォールバック

| ファイル     | 役割                    | モデル                | 出力          |
| ------------ | ----------------------- | --------------------- | ------------- |
| technical.py | テクニカル分析          | mini                  | JSON          |
| sentiment.py | センチメント分析        | mini（+FinGPT前処理） | JSON          |
| debate.py    | Bull/Bear 2ラウンド議論 | mini                  | JSON          |
| trader.py    | 最終判断                | 上位モデル            | Function Call |

#### プロンプト設計

**① テクニカルアナリスト**

```
【役割】あなたは経験20年のテクニカルアナリストです。
【入力】XAU/USD 4時間足・1時間足の価格データと以下の指標:
  RSI(14), MACD(12,26,9), ボリンジャーバンド(20,2), ATR(14), 直近高値/安値
【タスク】
  1. 現在のトレンド（上昇/下降/レンジ）を判定
  2. 過熱感・反転シグナルの有無
  3. 重要な価格帯（サポート/レジスタンス）
【出力】JSON形式:
  { "trend": "...", "signal": "BUY/SELL/NEUTRAL",
    "key_levels": {...}, "reasoning": "根拠を2文で" }
【禁止】未来の価格を推測しないこと。与えられたデータのみで判断。
```

**② センチメントアナリスト**

```
【役割】あなたは金融ニュースのセンチメント専門家です。
【入力】直近24時間のXAU/USD・ドル・金利関連ニュース見出し（最大15件）
【タスク】
  1. 各ニュースを 強気/中立/弱気 に分類
  2. 全体センチメントスコア（-1.0〜+1.0）を算出
  3. 最も影響が大きい材料を1つ抽出
【出力】JSON: { "score": 0.0, "dominant_news": "...", "reasoning": "..." }
【注意】強気バイアスに注意し、悪材料を過小評価しないこと。
```

**③ Bull（強気）リサーチャー**

```
【役割】買いポジションを正当化する強気派。
【入力】①②のレポート
【タスク】買うべき理由を最も説得力ある形で主張。ただし事実に基づくこと。
【出力】{ "bull_case": "...", "conviction": 0.0-1.0 }
```

**④ Bear（弱気）リサーチャー**

```
【役割】売り／見送りを正当化する弱気派。楽観論の穴を突く。
【入力】①②のレポート＋Bullの主張
【タスク】リスク・下落要因・Bull論の弱点を指摘。
【出力】{ "bear_case": "...", "conviction": 0.0-1.0 }
```

**⑤ トレーダー（最終判断）**

```
【役割】あなたは最終決定権を持つ責任者。Bull/Bearの議論を裁定する。
【入力】①②③④すべて
【タスク】
  1. 両論を天秤にかけ BUY/SELL/HOLD を決定
  2. 確信度を出す（confidence < 0.6 なら必ずHOLD）
【出力】place_trade_order 関数を呼び出す
  （action, confidence, reasoning, risk_level）
【鉄則】迷ったらHOLD。資金を守ることが最優先。
```

#### 議論制御（debate.py）

```
Round1: Bull主張(①②入力) → Bear反論(①②+Bull入力)
Round2: Bull再反論(全入力) → Bear再反論(全入力)
→ 議論全体をtrader.pyへ
```

#### Function Callスキーマ（trader.py）

```json
{
  "name": "place_trade_order",
  "description": "分析結果に基づき売買判断を実行する",
  "parameters": {
    "type": "object",
    "properties": {
      "action":     {"type": "string", "enum": ["BUY", "SELL", "HOLD"]},
      "symbol":     {"type": "string"},
      "confidence": {"type": "number", "description": "0-1の確信度"},
      "reasoning":  {"type": "string", "description": "判断根拠（日本語）"},
      "risk_level": {"type": "string", "enum": ["LOW", "MID", "HIGH"]}
    },
    "required": ["action", "symbol", "confidence", "reasoning"]
  }
}
```

> SL/TP・ロットはここに含めない（risk_manager.pyが計算）。

### 4-6. risk/risk_manager.py

**LLM不使用・純粋計算：**

```python
def calc_lot(balance, risk_pct, sl_distance_usd, contract_size=100):
    """SL幅からロット自動計算。最小0.01保証"""
    risk_amount_usd = (balance / 155) * risk_pct   # 円→ドル概算
    loss_per_lot = sl_distance_usd * contract_size # 1ロットの損失
    lot = risk_amount_usd / loss_per_lot
    return max(0.01, round(lot, 2))

def calc_sl_tp(entry_price, atr, action, atr_mult=1.5, rr=2.0):
    """ATRベースでSL/TP算出。BUY/SELLで方向切替"""
    sl_distance = atr * atr_mult
    tp_distance = sl_distance * rr
    if action == "BUY":
        sl = entry_price - sl_distance
        tp = entry_price + tp_distance
    else:  # SELL
        sl = entry_price + sl_distance
        tp = entry_price - tp_distance
    return sl, tp

def check_filters(confidence, spread, is_news_soon,
                  consecutive_losses, daily_loss_pct):
    """全安全フィルタ判定。1つでもNGならFalse（＝取引しない）"""
    # confidence < 0.6 → False
    # spread > 平常×2 → False
    # is_news_soon → False
    # consecutive_losses >= 3 → False
    # daily_loss_pct >= 3% → False
```

**安全フィルタ一覧：**

| フィルタ   | 条件                     |
| ---------- | ------------------------ |
| 確信度     | confidence < 0.6 → HOLD |
| スプレッド | 平常時の2倍超 → 見送り  |
| 経済指標   | ±15分 → 新規禁止       |
| 連敗       | 3連敗 → 当日停止        |
| 日次損失   | -3%到達 → 当日停止      |

### 4-7. main.py（処理フロー）

```
1. 起動チェック（フィルタ）
   - 経済指標±15分か？ → Yes: 即終了
   - 3連敗中 or 日次損失-3%到達？ → Yes: 即終了
   - スプレッド正常か？ → No: 終了
2. データ取得（MT5 + ニュース）
3. 特徴量生成（ta_calc）
4. 分析エージェント（technical + sentiment）
5. 議論（debate 2ラウンド）
6. 最終判断（trader → Function Call）
7. リスク管理（risk_manager: フィルタ→ロット→SL/TP）
8. 執行（mt5_client.send_order）
9. ログ記録（trade_log.csv）
```

### 4-8. backtest/timecapsule.py（動作確認・バグ検出用）

> **位置づけ**：損益予測ではなく、ロジックの動作確認・バグ検出が目的。

```python
def run_timecapsule_test(start_date, end_date):
    """
    - 判断日ループで未来データを封印して渡す
    - 本番と同じエージェント経路を通す
    - 目的: エラー・例外・パース失敗・発注ロジック不整合の検出
    - コスト対策: miniのみ使用・期間は短期・結果キャッシュ
    """

def get_timecapsule_data(decision_date):
    """判断日より前のデータだけを返す（未来を封印）"""
    prices = load_prices(end=decision_date - 1)      # 前日まで
    news   = load_news(end=decision_date)            # 判断時刻まで
    return prices, news
```

**検出したいバグの例：**

- JSONパース失敗
- SL/TP計算の符号ミス（BUY/SELL逆）
- ロット計算のゼロ除算・最小ロット割れ
- フィルタのすり抜け
- MT5発注リクエストの形式エラー

---

## 5. リスク管理パラメータ（堅実モード）

| パラメータ           | 設定値                 | 理由                     |
| -------------------- | ---------------------- | ------------------------ |
| 1トレードリスク割合  | 1.0%（=5,000円）       | 連敗耐性を最優先         |
| 最大同時ポジション   | 1                      | GOLD単一で集中管理       |
| 実効レバレッジ       | 5倍以下                | 破産確率を抑制           |
| ストップロス         | ATR(14) × 1.5         | ノイズで狩られにくい距離 |
| テイクプロフィット   | SL幅 × 2.0（RR 1:2）  | 勝率34%超で収支プラス    |
| トレーリングストップ | 含み益がSL幅超で建値へ | 利益を守る               |

**攻めレバー（エッジ確認後に段階解放）：**

```
Stage 1（堅実）: リスク1%             ★初期はここ
Stage 2（標準）: リスク2%             手応え確認後
Stage 3（攻め）: リスク3-5% + 複利ON   高リターン挑戦
```

移行条件：直前ステージで「最大ドローダウン許容内」かつ「プロフィットファクター > 1.3」。

---

## 6. 実装着手順

```
Step1: config.py + .env         # 土台
Step2: data/mt5_client.py       # MT5接続確認（最重要・ここで疎通）
Step3: indicators/ta_calc.py    # 指標計算（単体テスト容易）
Step4: risk/risk_manager.py     # 純粋計算（LLM不要・テスト容易）
Step5: data/news_client.py      # ニュース取得
Step6: agents/ 各種             # LLM部（ここでAPI課金開始）
Step7: main.py                  # 全結合
Step8: backtest/timecapsule.py  # 動作確認・バグ出し
Step9: デモ口座でフォワードテスト
```

> Step4までAPI課金ゼロで動作確認可能。LLM部は後回しにしてコストを抑える。

---

## 7. AI Agentへの実装指示メモ

- **型ヒント必須**：全関数に型アノテーションを付与
- **例外処理**：MT5接続断・API失敗時は必ずHOLD（安全側）にフォールバック
- **ログ徹底**：全エージェントの入出力・判断根拠をCSV記録
- **APIキーは.env**：ハードコード禁止
- **XM銘柄名確認**：`SYMBOL="GOLD"`はブローカーで名称が違う場合あり（"XAUUSD"等）。`mt5.symbols_get()`で実名確認するコードを入れる
- **単体テスト**：risk_manager・ta_calcは特にテストを厚く（計算ミスは致命的）
- **トークン記録**：各API呼び出しのトークン数をログし、月次コストを可視化

---

## 8. 運用ロードマップ

```
Step1: Time-Capsuleでバグ検出・動作確認
   ↓
Step2: デモ口座フォワードテスト（1〜2か月）★実弾前の必須関門
   ↓
Step3: 実弾・堅実モード（リスク1%）で運用開始
   ↓（PF>1.3, 最大DD<20%を確認）
Step4: Stage2（リスク2%）へ
   ↓（安定を確認）
Step5: Stage3（リスク3-5%＋複利）→ 高リターンに挑戦
```

---

## 9. 重要な留意事項

1. **強気バイアス**：LLMは楽観的判断に偏りやすい。Bull/Bearの議論機構で緩和。
2. **数値計算の弱さ**：ポジションサイジング・SL/TPは必ずコード側で計算。
3. **API遅延**：スキャルピング不可。H1〜H4のスイング寄りで運用。
4. **バックテストの限界**：LLMは学習で未来を知る（Look-ahead Bias）。損益予測ではなくバグ検出目的で使用。
5. **実弾前にデモ必須**：バックテストを軽視するなら、デモ・少額での生きた検証を必ず1つ挟む。
6. **金融リスク**：本設計は投資成果を保証しない。必ずデモ・少額から段階的に検証。

---

## 参考技術

- TradingAgents: Multi-Agents LLM Financial Trading Framework (arXiv:2412.20138)
- FinGPT (AI4Finance Foundation)
- FINSABER: LLMトレーディングの現実的評価 (arXiv:2508.17565)
- OpenAI Function Calling Guide
- MetaTrader5 Python統合 公式ドキュメント
