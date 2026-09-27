"""Log daily probabilistic forecasts for every portfolio symbol.

Runs once after the D1 rollover. A second daily invocation is safe: valid
forecasts are deduplicated by (bar_date, symbol), while failed calls retry.
This script never places orders.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

load_dotenv(ROOT / ".env")

import baseline as bl  # noqa: E402
from agents.base import LLMClient  # noqa: E402
from config import MODEL_ANALYSIS  # noqa: E402
from data.news_client import fetch_news_with_meta  # noqa: E402

LOGGER = logging.getLogger("gp_mate.daily_forecast")
OUTPUT = ROOT / "logs" / "daily_forecasts.jsonl"
BARS = 100
NEWS_HOURS = 24
NEWS_MAX_ITEMS = 40
PROB_KEYS = ("p_up_1d", "p_up_10d", "p_highvol_10d")

SYSTEM_PROMPT = """You are a calibrated market probability forecaster.
Use only the supplied point-in-time news headlines and confirmed daily price data.
For the named instrument, estimate three independent probabilities:
- p_up_1d: the next completed D1 close is above p0.
- p_up_10d: the D1 close exactly 10 completed trading bars later is above p0.
- p_highvol_10d: annualized standard deviation of the next 10 close-to-close
  returns exceeds the supplied trailing 60-day annualized volatility.
Return JSON only with p_up_1d, p_up_10d, p_highvol_10d in [0,1], and
key_reason as one concise Japanese sentence. Do not force confidence; use values
near 0.5 when evidence is weak."""


def universe(folder: Path = ROOT / "logs") -> list[str]:
    symbols = [f.stem[len("ohlcv_"):-len("_D1")] for f in sorted(folder.glob("ohlcv_*_D1.csv"))]
    return [symbol for symbol in symbols if bl.group_of(symbol) != "fx_cross"]


def read_records(path: Path = OUTPUT) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def completed_keys(rows: list[dict[str, Any]]) -> set[tuple[str, str]]:
    return {
        (str(row.get("bar_date", "")), str(row.get("symbol", "")))
        for row in rows
        if row.get("ok") and all(row.get(key) is not None for key in PROB_KEYS)
    }


def load_confirmed_d1(mt5: Any, symbol: str, force: bool = False) -> pd.DataFrame:
    if not mt5.symbol_select(symbol, True):
        raise RuntimeError(f"symbol_select failed: {mt5.last_error()}")
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_D1, 0, BARS)
    if rates is None or len(rates) < 75:
        raise RuntimeError(f"D1 bars unavailable: {mt5.last_error()}")
    frame = pd.DataFrame(rates)
    if not force and frame["tick_volume"].iloc[-1] > 0.5 * frame["tick_volume"].iloc[-30:-1].median():
        raise RuntimeError("latest D1 bar appears nearly complete; run after rollover")
    frame = frame.iloc[:-1].copy()
    frame["bar_date"] = pd.to_datetime(frame["time"], unit="s").dt.date.astype(str)
    return frame.drop_duplicates("bar_date", keep="last").reset_index(drop=True)


def price_context(frame: pd.DataFrame) -> dict[str, Any]:
    close = frame["close"].astype(float)
    previous = close.shift(1)
    true_range = pd.concat(
        [
            frame["high"].astype(float) - frame["low"].astype(float),
            (frame["high"].astype(float) - previous).abs(),
            (frame["low"].astype(float) - previous).abs(),
        ],
        axis=1,
    ).max(axis=1)
    returns = close.pct_change()
    atr = float(true_range.rolling(14).mean().iloc[-1])
    normal_vol = float(returns.iloc[-60:].std(ddof=1) * np.sqrt(252))
    recent = []
    for index in frame.index[-30:]:
        recent.append(
            {
                "date": str(frame.at[index, "bar_date"]),
                "open": round(float(frame.at[index, "open"]), 6),
                "high": round(float(frame.at[index, "high"]), 6),
                "low": round(float(frame.at[index, "low"]), 6),
                "close": round(float(frame.at[index, "close"]), 6),
            }
        )
    return {
        "bar_date": str(frame["bar_date"].iloc[-1]),
        "p0": float(close.iloc[-1]),
        "atr_d1": atr,
        "normal_vol_60d": normal_vol,
        "recent_d1": recent,
    }


def parse_probabilities(payload: dict[str, Any]) -> tuple[dict[str, float] | None, str]:
    parsed: dict[str, float] = {}
    for key in PROB_KEYS:
        try:
            value = float(payload.get(key))
        except (TypeError, ValueError):
            return None, f"{key}: missing_or_non_numeric"
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            return None, f"{key}: outside_0_1"
        parsed[key] = round(value, 6)
    return parsed, ""


def forecast_one(
    client: Any,
    symbol: str,
    context: dict[str, Any],
    news: list[dict[str, Any]],
    model: str,
) -> dict[str, Any]:
    prompt = {
        "as_of_bar": context["bar_date"],
        "symbol": symbol,
        "p0": context["p0"],
        "atr_d1": context["atr_d1"],
        "normal_vol_60d_annualized": context["normal_vol_60d"],
        "recent_confirmed_d1": context["recent_d1"],
        "news_last_24h": [
            {"title": item.get("title", ""), "published_at": item.get("published_at", ""), "source": item.get("source", "")}
            for item in news
        ],
    }
    fallback = {**{key: None for key in PROB_KEYS}, "key_reason": ""}
    result = client.call_json(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=json.dumps(prompt, ensure_ascii=False),
        model=model,
        fallback_payload=fallback,
    )
    probabilities, invalid_reason = parse_probabilities(result.payload) if result.ok else (None, result.error)
    usage = getattr(result, "usage", None)
    return {
        "ok": bool(result.ok and probabilities is not None),
        **(probabilities or {key: None for key in PROB_KEYS}),
        "model": str(getattr(result, "model", model) or model),
        "key_reason": str(result.payload.get("key_reason", "") or ""),
        "error": invalid_reason,
        "usage": {
            "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
            "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
            "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
        },
    }


def append_record(record: dict[str, Any], path: Path = OUTPUT) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def run(mt5: Any, client: Any, model: str, force: bool = False, output: Path = OUTPUT) -> dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    existing = completed_keys(read_records(output))
    news, news_meta = fetch_news_with_meta(hours=NEWS_HOURS, max_items=NEWS_MAX_ITEMS, keywords=None)
    summary = {"symbols": 0, "written": 0, "skipped": 0, "failed": 0, "news_count": len(news)}
    for symbol in universe():
        summary["symbols"] += 1
        try:
            context = price_context(load_confirmed_d1(mt5, symbol, force=force))
            key = (context["bar_date"], symbol)
            if key in existing:
                summary["skipped"] += 1
                continue
            forecast = forecast_one(client, symbol, context, news, model)
            record = {
                "ts_utc": now,
                "bar_date": context["bar_date"],
                "symbol": symbol,
                "p_up_1d": forecast["p_up_1d"],
                "p_up_10d": forecast["p_up_10d"],
                "p_highvol_10d": forecast["p_highvol_10d"],
                "p0": context["p0"],
                "atr_d1": round(context["atr_d1"], 8),
                "normal_vol_60d": round(context["normal_vol_60d"], 8),
                "model": forecast["model"],
                "key_reason": forecast["key_reason"],
                "ok": forecast["ok"],
                "error": forecast["error"],
                "news_count": len(news),
                "news_meta": news_meta,
                "usage": forecast["usage"],
            }
            append_record(record, output)
            summary["written"] += 1
            if not forecast["ok"]:
                summary["failed"] += 1
        except Exception as exc:
            LOGGER.exception("%s: daily forecast failed", symbol)
            summary["failed"] += 1
            append_record({"ts_utc": now, "bar_date": None, "symbol": symbol, "ok": False, "error": str(exc)}, output)
            summary["written"] += 1
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="日足の確定チェックを無視")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    import MetaTrader5 as mt5

    kwargs = {
        key: value
        for key, value in {
            "path": os.getenv("PF_MT5_PATH"),
            "password": os.getenv("PF_MT5_PASSWORD"),
            "server": os.getenv("PF_MT5_SERVER"),
        }.items()
        if value
    }
    if os.getenv("PF_MT5_LOGIN"):
        kwargs["login"] = int(os.environ["PF_MT5_LOGIN"])
    if "login" not in kwargs:
        raise SystemExit("PF_MT5_LOGIN 等の指定が必須")
    if not mt5.initialize(**kwargs):
        raise SystemExit(f"MT5 initialize failed: {mt5.last_error()}")
    try:
        account = mt5.account_info()
        if account is None or account.login != kwargs["login"]:
            raise SystemExit("PF_MT5_LOGIN と接続口座が一致しない。中止。")
        summary = run(mt5, LLMClient(), MODEL_ANALYSIS, force=args.force)
        print(json.dumps(summary, ensure_ascii=False))
        return 1 if summary["failed"] else 0
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())