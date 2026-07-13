from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from dotenv import load_dotenv

BASE_DIR: Final[Path] = Path(__file__).resolve().parent
ENV_PATH: Final[Path] = BASE_DIR / ".env"

# Keep RSS sources in code to simplify deployment and avoid extra env keys.
DEFAULT_RSS_FEEDS: Final[tuple[str, ...]] = (
    "https://www.forexlive.com/feed/news",
    "https://www.investing.com/rss/news_285.rss",
    "https://www.marketwatch.com/rss/topstories",
)

load_dotenv(ENV_PATH)


@dataclass(frozen=True)
class Settings:
    symbol: str
    timeframe_trend: str
    timeframe_entry: str

    risk_percent: float
    max_positions: int
    confidence_threshold: float
    close_confidence_threshold: float
    max_daily_loss_pct: float
    consecutive_loss_limit: int

    atr_multiplier_sl: float
    risk_reward_ratio: float
    breakeven_buffer: float
    breakeven_monitor_times: tuple[str, ...]

    news_filter_minutes: int
    spread_multiplier_limit: float
    spread_samples: int
    spread_sample_interval: float
    judgment_times: tuple[str, ...]

    model_analysis: str
    model_decision: str
    max_news_items: int
    rss_feeds: tuple[str, ...]
    stage: int

    mt5_login: int | None
    mt5_password: str
    mt5_server: str
    mt5_path: str

    openai_api_key: str
    news_api_key: str
    fred_api_key: str


def _get_env_str(name: str, default: str = "") -> str:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip()


def _get_env_int(name: str, default: int) -> int:
    value = _get_env_str(name)
    if value == "":
        return default
    return int(value)


def _get_env_float(name: str, default: float) -> float:
    value = _get_env_str(name)
    if value == "":
        return default
    return float(value)


def _parse_judgment_times(value: str) -> tuple[str, ...]:
    times = [item.strip() for item in value.split(",") if item.strip()]
    if not times:
        return ("09:00", "16:00", "21:00", "23:30")
    return tuple(times)


def _parse_breakeven_monitor_times(value: str) -> tuple[str, ...]:
    times = [item.strip() for item in value.split(",") if item.strip()]
    if not times:
        return ("07", "22", "37", "52")
    return tuple(times)


def _normalize_symbol_name(symbol: str) -> str:
    return "".join(ch for ch in symbol.lower() if ch.isalnum())


def _resolve_symbol_name(
    preferred_symbol: str,
    mt5_login: int | None,
    mt5_password: str,
    mt5_server: str,
    mt5_path: str,
) -> str:
    try:
        import MetaTrader5 as mt5  # type: ignore[import-not-found]
    except Exception:
        return preferred_symbol

    init_kwargs: dict[str, str] = {}
    if mt5_path:
        init_kwargs["path"] = mt5_path

    try:
        if not mt5.initialize(**init_kwargs):
            return preferred_symbol

        if mt5_login is not None and mt5_password and mt5_server:
            if not mt5.login(login=mt5_login, password=mt5_password, server=mt5_server):
                return preferred_symbol

        symbols = mt5.symbols_get()
        if not symbols:
            return preferred_symbol

        names = [s.name for s in symbols if getattr(s, "name", "")]
        if not names:
            return preferred_symbol

        # Priority: explicit preference -> common XM variants -> normalized match.
        candidates = [preferred_symbol, "GOLD", "XAUUSD", "GOLD#", "XAUUSD#", "gold#"]
        lowered_to_original = {name.lower(): name for name in names}

        for candidate in candidates:
            found = lowered_to_original.get(candidate.lower())
            if found:
                return found

        normalized_map = {_normalize_symbol_name(name): name for name in names}
        for normalized_candidate in (
            _normalize_symbol_name(preferred_symbol),
            "gold",
            "xauusd",
        ):
            found = normalized_map.get(normalized_candidate)
            if found:
                return found

        # Last resort: partial match for broker suffix symbols like GOLDm, XAUUSD.r, etc.
        partial_matches = [
            name
            for name in names
            if "gold" in name.lower() or "xauusd" in name.lower()
        ]
        if partial_matches:
            return sorted(partial_matches, key=len)[0]

        return preferred_symbol
    except Exception:
        return preferred_symbol
    finally:
        try:
            mt5.shutdown()
        except Exception:
            pass


def load_settings() -> Settings:
    mt5_login_raw = _get_env_str("MT5_LOGIN")
    mt5_login = int(mt5_login_raw) if mt5_login_raw else None
    mt5_password = _get_env_str("MT5_PASSWORD", "")
    mt5_server = _get_env_str("MT5_SERVER", "")
    mt5_path = _get_env_str("MT5_PATH", "")

    preferred_symbol = _get_env_str("SYMBOL", "GOLD#")
    resolved_symbol = _resolve_symbol_name(
        preferred_symbol=preferred_symbol,
        mt5_login=mt5_login,
        mt5_password=mt5_password,
        mt5_server=mt5_server,
        mt5_path=mt5_path,
    )

    return Settings(
        symbol=resolved_symbol,
        timeframe_trend=_get_env_str("TIMEFRAME_TREND", "H4"),
        timeframe_entry=_get_env_str("TIMEFRAME_ENTRY", "H1"),
        risk_percent=_get_env_float("RISK_PERCENT", 0.01),
        max_positions=_get_env_int("MAX_POSITIONS", 1),
        confidence_threshold=_get_env_float("CONFIDENCE_THRESHOLD", 0.6),
        close_confidence_threshold=_get_env_float("CLOSE_CONFIDENCE_THRESHOLD", 0.7),
        max_daily_loss_pct=_get_env_float("MAX_DAILY_LOSS_PCT", 0.03),
        consecutive_loss_limit=_get_env_int("CONSECUTIVE_LOSS_LIMIT", 3),
        atr_multiplier_sl=_get_env_float("ATR_MULTIPLIER_SL", 1.5),
        risk_reward_ratio=_get_env_float("RISK_REWARD_RATIO", 2.0),
        breakeven_buffer=_get_env_float("BREAKEVEN_BUFFER", 0.1),
        breakeven_monitor_times=_parse_breakeven_monitor_times(
            _get_env_str("BREAKEVEN_MONITOR_TIMES", "07,22,37,52")
        ),
        news_filter_minutes=_get_env_int("NEWS_FILTER_MINUTES", 15),
        spread_multiplier_limit=2.0,
        spread_samples=20,
        spread_sample_interval=0.5,
        judgment_times=_parse_judgment_times(
            _get_env_str("JUDGMENT_TIMES", "09:00,16:00,21:00,23:30")
        ),
        model_analysis=_get_env_str("MODEL_ANALYSIS", "gpt-5.4-mini-2026-03-17"),
        model_decision=_get_env_str("MODEL_DECISION", "gpt-5.5-2026-04-23"),
        max_news_items=_get_env_int("MAX_NEWS_ITEMS", 15),
        rss_feeds=DEFAULT_RSS_FEEDS,
        stage=_get_env_int("STAGE", 1),
        mt5_login=mt5_login,
        mt5_password=mt5_password,
        mt5_server=mt5_server,
        mt5_path=mt5_path,
        openai_api_key=_get_env_str("OPENAI_API_KEY", ""),
        news_api_key=_get_env_str("NEWS_API_KEY", ""),
        fred_api_key=_get_env_str("FRED_API_KEY", ""),
    )


settings = load_settings()

# Backward-compatible module-level aliases.
SYMBOL: Final[str] = settings.symbol
TIMEFRAME_TREND: Final[str] = settings.timeframe_trend
TIMEFRAME_ENTRY: Final[str] = settings.timeframe_entry

RISK_PERCENT: Final[float] = settings.risk_percent
MAX_POSITIONS: Final[int] = settings.max_positions
CONFIDENCE_THRESHOLD: Final[float] = settings.confidence_threshold
CLOSE_CONFIDENCE_THRESHOLD: Final[float] = settings.close_confidence_threshold
MAX_DAILY_LOSS_PCT: Final[float] = settings.max_daily_loss_pct
CONSECUTIVE_LOSS_LIMIT: Final[int] = settings.consecutive_loss_limit

ATR_MULTIPLIER_SL: Final[float] = settings.atr_multiplier_sl
RISK_REWARD_RATIO: Final[float] = settings.risk_reward_ratio
BREAKEVEN_BUFFER: Final[float] = settings.breakeven_buffer
BREAKEVEN_MONITOR_TIMES: Final[tuple[str, ...]] = settings.breakeven_monitor_times

NEWS_FILTER_MINUTES: Final[int] = settings.news_filter_minutes
SPREAD_MULTIPLIER_LIMIT: Final[float] = settings.spread_multiplier_limit
SPREAD_SAMPLES: Final[int] = settings.spread_samples
SPREAD_SAMPLE_INTERVAL: Final[float] = settings.spread_sample_interval
JUDGMENT_TIMES: Final[tuple[str, ...]] = settings.judgment_times

MODEL_ANALYSIS: Final[str] = settings.model_analysis
MODEL_DECISION: Final[str] = settings.model_decision
MAX_NEWS_ITEMS: Final[int] = settings.max_news_items
RSS_FEEDS: Final[tuple[str, ...]] = settings.rss_feeds
STAGE: Final[int] = settings.stage

MT5_LOGIN: Final[int | None] = settings.mt5_login
MT5_PASSWORD: Final[str] = settings.mt5_password
MT5_SERVER: Final[str] = settings.mt5_server
MT5_PATH: Final[str] = settings.mt5_path

OPENAI_API_KEY: Final[str] = settings.openai_api_key
NEWS_API_KEY: Final[str] = settings.news_api_key
FRED_API_KEY: Final[str] = settings.fred_api_key
