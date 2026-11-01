# GP-MATE

GP-MATE is a GPT-powered multi-agent trading EA for XAU/USD (GOLD) on MT5.
The system prioritizes capital protection and uses a staged workflow for safe operation.

## Overview

- Symbol: XAU/USD (XM symbol auto-detected, currently GOLD#)
- Timeframes: H4 for trend, H1 for entries
- Architecture: three analysts (technical, macro, sentiment) give their own reads, discuss trend vs reversal as a panel, then a trader decides
- Risk-first policy: fail-safe HOLD on uncertainty or failures. The trader is not asked for a confidence number and no confidence threshold exists; HOLD is how it expresses doubt, and only market/risk facts (spread, news window, losing streak, daily loss, RR, pending distance, time windows) can block its decision

## Project Structure

- `data/`: MT5 integration, market/news data access
- `agents/`: LLM agents for analysis, debate, and final decision
- `backtest/`: time-capsule validation and bug-detection flows
- `analysis/`: performance metrics and reporting scripts
- `scripts/`: operation scripts (`check_connection`, `run_manual`, `run_scheduler`)

## Setup

- Recommended Python: 3.12 or 3.13
- Python 3.14+: supported with safe fallback, but some LangChain internals may emit compatibility warnings.

1. Create and activate virtual environment (PowerShell):
   - `python -m venv .venv`
   - `.\.venv\Scripts\Activate.ps1`
2. Upgrade pip and install dependencies:
   - `python -m pip install --upgrade pip`
   - `pip install -r requirements.txt`
3. Copy `.env.example` to `.env`.
4. Fill required values in `.env`:
   - MT5 credentials (`MT5_LOGIN`, `MT5_PASSWORD`, `MT5_SERVER`, `MT5_PATH`)
   - API keys (`OPENAI_API_KEY`, optional `NEWS_API_KEY`, `FRED_API_KEY`)
   - Judgment schedule defaults to `America/New_York` 08:00 / 09:30 / 10:30 and follows DST automatically; set `JUDGMENT_TIMES_NY=20:00,22:00,03:00,08:00,09:30,10:30` to add Asia / London slots. Pending orders live until the next slot re-plans.

## Run Flow (Safe 3-Step)

1. Connection check (no trading):
   - `python scripts/check_connection.py`
   - Data-source health (RSS feeds, economic calendar, FRED, Yahoo same-day yields, CFTC COT, GLD holdings, MT5 dollar index):
     `python scripts/check_data_sources.py`
2. Manual single run (with order confirmation):
   - `python scripts/run_manual.py`
3. Automated schedule run:
   - `python scripts/run_scheduler.py`

## Forecast-only logger (research, no trading)

**Status (2026-10-18): paused.** After 501 scored forecasts the LLM's Brier
score (0.603) equals the rolling base-rate anchor (0.600, CI crosses zero),
the panel debate made it worse within the same week (0.676 vs 0.540), and no
single analyst report changes the score. The question "which of ±1 ATR is hit
first within 6 H1 bars" is answered: not better than the base rate. Restart
the logger only with a task shaped like the actual trades (entry at a level,
SL and TP as the barriers).

Records the LLM's triple-barrier probabilities (`p_up` / `p_down` / `p_timeout`)
every confirmed H1 bar, labels them later from realised bars, and scores
calibration (Brier / log loss / reliability) against uniform, base-rate and
momentum baselines. It never places orders and does not touch `main.py` or
the trade log.

- `python scripts/run_forecast_logger.py` — hourly scheduler (`--once` for a single cycle)
- `python scripts/resolve_forecasts.py` — label elapsed forecasts (idempotent; the logger also does this after each cycle)
- `python scripts/eval_forecasts.py [--csv out.csv]` — calibration report
- Data: `logs/forecasts.jsonl` (one row per forecast) and `logs/forecast_inputs/<id>.json` (full inputs for ablations)
- Settings: `FORECAST_*` and `MODEL_FORECAST` in `.env` (defaults: K=1.0 ATR both sides, 6 bars, 24h, no debate, 1 sample)
- The forecaster is anchored on the realised UP/DOWN/TIMEOUT frequencies of recent resolved forecasts (`task.reference_base_rates`, from 30 rows) and receives the rule-based regime read (`task.regime`)
- `python scripts/calibrate_forecasts.py` — fits shrink / shrink+tilt transforms on the first half of the record and tests them on the second half; a fitted weight near 0 means the probabilities carry no usable signal
- `python scripts/ablate_forecasts.py` — re-forecasts archived inputs with reports removed to attribute the error; `no_debate` (default) runs only on forecasts whose input had a debate
- The evaluation also reports the deviation from the base-rate anchor (direction accuracy of the move, its size, Brier vs the anchor); once anchored, raw `p_up` vs `p_down` mostly echoes the anchor, so read the deviation

## Analysts and the panel debate

The three analysts are asked for their own reading and nothing in code
re-decides it. The technical analyst receives the multi-timeframe indicator
snapshots, the horizontal levels and price-structure facts from
`indicators/structure.py` (recent swing highs/lows and unfilled fair value
gaps) and answers with D1 / execution trend, alignment, a `regime_view`
(TREND_CONTINUATION / MEAN_REVERSION / UNCLEAR), key prices and an
invalidation price. The macro analyst receives the FRED / dollar-index /
positioning / release data with provenance notes only (what a series is and how
fresh it is, not what it means for gold) and answers with `macro_bias`,
US Treasury yields (5y/10y/30y) come same-day from Yahoo Finance (`^FVX`,
`^TNX`, `^TYX`; no key; the 2-year stays on FRED) with `change_1d` and
`stale_minutes`, because the FRED closes
arrive a day late and the 2026-10-08 losses were taken against a bond-market
reversal that FRED could not show yet; the FRED values stay under `*_fred`.
The analyst answers with `macro_bias`,
`regime_view` and an invalidation condition; it is not asked for a confidence
number. The sentiment analyst is asked what is *new* in the headlines and
whether it supports continuation or reversal. Rule-based reads survive only as
fail-safes: a technical report falling back to them is marked
`source=rule_based_fallback`, a macro analyst that does not answer is NEUTRAL.

With `DEBATE_AXIS=panel` the same three analysts discuss one question, "trend
continuation or reversal?", with no assigned sides. Each reads the other two
reports and the statements so far, may agree, disagree or change their view,
and states what would change it. A chair summarises agreements, conflicts, the
panel regime (TREND / RANGE / TRANSITION), `entry_style`, key levels and the
consensus (UNANIMOUS / MAJORITY / SPLIT); if the chair fails, a plain majority
of the stated views is used and marked `vote_fallback`. The trader is told to
set a directional bias and pending orders only when the panel points the same
way, and otherwise to stay flat. A few entries per day is the intended pace.
Pending orders follow the panel regime rather than a self-reported bias
number: in TREND only orders on the trend side are accepted, in RANGE limit
fades at the band edges are accepted on either side with no directional bias,
in TRANSITION none. The chair follows the majority (two analysts for
continuation is TREND, two for mean reversion is RANGE); a chair that still
says TRANSITION is recorded with `chair_overrode_majority`.

## Data integrity and fail-safes

- **Closed bars only.** `data/confirmed_bars.py` fetches a margin of extra
  bars, converts MT5 server time to UTC, drops the forming bar and keeps the
  last 300 closed bars before any indicator, swing, gap or level is computed.
  Both the trading loop and the forecast logger use it. The trade log records
  `last_closed_bar_h1/h4/d1`, `dropped_open_bar`, `closed_bar_count` and
  `bar_age_seconds`; fewer than 60 closed bars on any timeframe is a HOLD.
- **Chair failure is a HOLD.** The panel chair gets one retry; if its output
  is still missing or outside the enums, `judge_status=FAILED`, the report is
  not ok and the trading loop holds. The analysts' majority vote is kept for
  the log only (`regime_source=vote_fallback_log_only`).
- **Symmetric panel rounds.** In round 1 every analyst sees the three reports
  and no statements, so speaking order cannot anchor anyone; round 2 (optional)
  shows everyone the same round-1 statements. `panel_agreement` (0.8 / 0.6 /
  0.4 for unanimous / majority / split, with `panel_votes_available` and the
  vote distribution) is an agreement measure, not a probability, and is never
  used as an order threshold.
- **Feed state before news count.** Dead feeds (`news_feed_health=BAD`) make
  the sentiment evidence INSUFFICIENT and the trader holds; healthy feeds with
  no items are `NO_NEWS` (neutral) and trading continues.
- **RR net of spread.** `net_rr = (tp - spread) / (sl + spread)` must reach
  `MIN_RISK_REWARD_RATIO`; `gross_rr`, `spread_cost`, `net_rr` and
  `rr_rejection_reason` are logged.

## Levels by id, four decision layers, counter-evidence

- **Price levels by id.** `indicators/price_levels.py` builds one catalogue per
  cycle from the horizontal levels, swings, unfilled FVGs, previous-day
  high/low, moving averages and round numbers, each tagged with a stable
  `level_id` (e.g. `H4_SWING_LOW_1`, `D1_CLUSTER_SUPPORT_2`, `ROUND_4400`) and
  its distance in ATR. The technical analyst, the panel members, the chair
  and the trader refer to levels by id. A raw price is snapped to the nearest
  candidate within 0.3 ATR; a pending order whose trigger matches no candidate
  is dropped (`pending_status=skipped_unanchored_price:*`). TP/SL ids and the
  pending trigger id are logged (`tp_level_id`, `sl_level_id`,
  `pending_level_id`).
- **Four layers in the log.** `market_state` (TREND / RANGE / TRANSITION /
  UNKNOWN), `direction` (UP / DOWN / NEUTRAL), `setup` (PULLBACK / BREAKOUT /
  FADE / NONE) and `executability` (EXECUTABLE / WAIT / BLOCKED) with
  `executability_reason`, so "the market is trending" and "an order can be
  sent now" are recorded separately. A failed chair is `UNKNOWN`, never the
  vote.
- **Counter-evidence and data quality.** Every analyst reports
  `counter_evidence`, `data_quality` (GOOD / PARTIAL / POOR) and an optional
  `abstain_reason` (which forces a neutral / UNCLEAR view); the panel members
  do the same and the chair sees them. `analyst_data_quality` summarises this
  per row.
- **H1 role.** The technical analyst reports `h4_trend`, `h1_trend`, `h1_role`
  (IMPULSE / PULLBACK / REVERSAL_ATTEMPT / NOISE) and
  `timeframe_relationship`, so an H1 dip inside an H4 uptrend can be read as a
  pullback instead of a divergence.

## Trade horizon (INTRADAY / SWING)

The panel's thesis is usually a D1/H4 read, but every execution rule used to be
measured in the H1 ATR, so a swing idea was given an hourly-sized stop and was
stopped by noise before the idea could play out. The trader now declares
`trade_horizon` with every decision:

- `INTRADAY` (default when absent): SL box (1.0–1.5 ATR), 2R TP cap and the
  pending-trigger distance (0.1–3.0 ATR) are measured in the **H1 ATR**.
- `SWING`: the same rules are measured in the **H4 ATR** (falls back to H1 when
  no H4 ATR is available; the log says which one was used).

The prompt only states these facts; which horizon fits is the trader's call.
The lot still targets `RISK_PERCENT`; because the 0.01-lot floor can make a
wide (SWING) stop risk more than that, the realised risk is reported as
`effective_risk_pct` and the order is skipped with `skipped_risk_cap` when it
exceeds `MAX_RISK_PERCENT` (default 0.02). The horizon and ATR of every sent
order are stored per ticket in `logs/trade_meta.json`, so the position
evaluator sees `position.trade_horizon` / `atr_at_entry` and judges the
holding on the timescale it was planned on (`UNKNOWN` for older tickets).
Trade-log columns: `trade_horizon`, `atr_timeframe`, `atr_used`,
`effective_risk_pct`.
**Exit shape.** When price first reaches +1R the breakeven monitor closes
`PARTIAL_CLOSE_FRACTION` of the position (default half, floored to the 0.01
lot step; a 0.01-lot position cannot be split) and then moves the stop to
breakeven; the remainder runs to the target. Once per ticket, recorded in
`logs/trade_meta.json` and the `partial_close_*` columns. The position
evaluator also receives `position.invalidation` (where the stop sits relative
to the panel's reversal level, `stop_decides_first`) and `r_multiple_now`.

The trader also receives `execution_geometry`: for each horizon the SL box,
the 2R TP cap, the smallest TP distance that still clears the net-RR floor
after spread, and the pending-trigger distance range, all in USD, so a plan
can be checked against the rules before it is proposed. Its `recent_context`
shows the last 24h of its own decisions and the realized closes of the last
7 days (side, entry/exit price, horizon, entry basis).

## Other instruments: shadow mode

The method is written for XAU/USD but runs on other dollar pairs through
`instruments.py` profiles (USDJPY, EURUSD): the analysts' wording and what
BULLISH means, the headline keywords, the calendar currencies, and the
price-unit constants (round-number steps, confluence proximities, the wick
buffer behind a structural stop). Set `SHADOW_SYMBOLS=USDJPY,EURUSD` and the
scheduler runs the full pipeline for those instruments after each gold
judgment **without placing orders**, recording the plan (market or pending:
entry, SL, TP, validity until the next slot) to `logs/shadow_plans.jsonl`.
Gold's production judgments are recorded there too (no extra LLM calls).
`shadow.resolve_shadow_plans` labels each plan from realised H1 bars: was a
pending plan filled before it expired, and did the trade reach TP or SL first
(timeout after 72 bars, Friday flat mirrors production).
`python scripts/eval_shadow.py [--resolve]` prints, per instrument, the plan
rate, fill rate, win rate, expectancy in R and how often +1R was reached.
Cost: each shadow instrument is roughly one production run per slot;
`SHADOW_JUDGMENT_TIMES_NY` limits the slots.

## Regime (Trend vs Range)

`indicators/regime.py` classifies every cycle as TREND / RANGE / TRANSITION from
ADX (H4/H1/D1), multi-timeframe alignment, H4 Bollinger width and D1 extension,
and proposes an `entry_style` (STOP_BREAKOUT / LIMIT_PULLBACK / LIMIT_FADE / NONE).
The read is attached to the technical report, handed to the debate as
`regime_hint`, to the trader (order type must fit the regime before direction)
and to the forecast logger, and is written to the trade log
(`regime`, `regime_confidence`, `entry_style`, `regime_source`).

`DEBATE_AXIS` selects the debate format: `panel` (see above), `direction`
(legacy Bull vs Bear, default until the forecast A/B is done) or `regime`
(legacy Trend advocate vs Range advocate). The forecast logger uses
`FORECAST_DEBATE_AXIS` (default `panel`) so formats can be compared on Brier
score before production is switched with `DEBATE_AXIS=panel`.

## Tests

- Run all tests:
  - `python -m pytest tests -q`

## Security Notes

- Never commit `.env`, logs, CSVs, or state files.
- `.gitignore` is configured to block sensitive files.
- Verify `git status` before every commit.

## Disclaimer

This software is for research and automation support only.
Trading involves financial risk. Validate on demo accounts first, then move to live trading at your own responsibility.
