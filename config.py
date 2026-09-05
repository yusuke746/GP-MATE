from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Final
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

BASE_DIR: Final[Path] = Path(__file__).resolve().parent
ENV_PATH: Final[Path] = BASE_DIR / ".env"

# Default RSS sources. Google News aggregates Reuters/Bloomberg/CNBC headlines
# behind a bot-tolerant endpoint, so it is the backbone; the rest are
# gold/FX specialists. Override with RSS_FEEDS=url1,url2 in .env after checking
# scripts/check_data_sources.py (marketwatch topstories was dropped: it rarely
# carried a gold headline and investing.com intermittently blocks scrapers).
DEFAULT_RSS_FEEDS: Final[tuple[str, ...]] = (
    "https://news.google.com/rss/search?q=gold+price+OR+XAUUSD+OR+bullion&hl=en-US&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=Fed+OR+%22treasury+yields%22+OR+%22dollar+index%22&hl=en-US&gl=US&ceid=US:en",
    "https://www.kitco.com/rss/category/commodities",
    "https://www.fxstreet.com/rss/news",
    "https://www.forexlive.com/feed/news",
    "https://www.investing.com/rss/news_285.rss",
)


def _parse_feed_list(value: str) -> tuple[str, ...]:
    feeds = tuple(item.strip() for item in value.split(",") if item.strip())
    return feeds if feeds else DEFAULT_RSS_FEEDS

load_dotenv(ENV_PATH)

MARKET_TIMEZONE_NAME: Final[str] = "America/New_York"
JST_TIMEZONE_NAME: Final[str] = "Asia/Tokyo"
# Judgment slots (America/New_York). NY session only: the London-open slot
# (03:00, later 04:00) was retired after 10 weeks of live data showed it was
# the sole losing slot -- 14 settled, 21% win rate, -44,533 JPY, PF 0.26 --
# while the three NY slots combined ran 36 settled, 58% win rate, PF 1.81.
# Positions opened in London were repeatedly stopped out during the NY open
# (08:00-09:30) before the next judgment could re-evaluate them.
NY_RUN_TIMES: Final[tuple[tuple[int, int], ...]] = (
    (8, 0),
    (9, 30),
    (10, 30),
)
MARKET_TZ: Final[ZoneInfo] = ZoneInfo(MARKET_TIMEZONE_NAME)
JST_TZ: Final[ZoneInfo] = ZoneInfo(JST_TIMEZONE_NAME)


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
    macro_debate_conf_threshold: float
    macro_bias_carry_threshold: float
    macro_against_close_threshold: float

    atr_multiplier_sl: float
    risk_reward_ratio: float
    breakeven_buffer: float
    breakeven_monitor_times: tuple[str, ...]

    news_filter_minutes: int
    calendar_timezone: str
    jpy_usd_rate_fallback: float
    friday_flat_time_ny: tuple[int, int]
    daily_pending_cutoff_ny: tuple[int, int]
    monday_open_ny: tuple[int, int]
    sl_structure_buffer_usd: float
    spread_multiplier_limit: float
    spread_samples: int
    spread_sample_interval: float

    model_analysis: str
    model_decision: str
    model_debate: str
    max_news_items: int
    rss_feeds: tuple[str, ...]
    stage: int

    mt5_login: int | None
    mt5_password: str
    mt5_server: str
    mt5_path: str
    mt5_server_timezone: str

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


def _parse_hhmm(value: str, default: tuple[int, int]) -> tuple[int, int]:
    try:
        hour_text, minute_text = value.strip().split(":")
        hour = int(hour_text)
        minute = int(minute_text)
    except Exception:
        return default
    if 0 <= hour <= 23 and 0 <= minute <= 59:
        return (hour, minute)
    return default


def _parse_breakeven_monitor_times(value: str) -> tuple[str, ...]:
    times = [item.strip() for item in value.split(",") if item.strip()]
    if not times:
        return ("07", "22", "37", "52")
    return tuple(times)


def current_judgment_times_jst(reference: datetime | None = None) -> tuple[str, ...]:
    current_ny = (reference or datetime.now(tz=MARKET_TZ)).astimezone(MARKET_TZ)
    judgment_times: list[str] = []
    for hour, minute in NY_RUN_TIMES:
        scheduled_ny = datetime(
            current_ny.year,
            current_ny.month,
            current_ny.day,
            hour,
            minute,
            tzinfo=MARKET_TZ,
        )
        judgment_times.append(scheduled_ny.astimezone(JST_TZ).strftime("%H:%M"))
    return tuple(judgment_times)


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
        macro_debate_conf_threshold=_get_env_float("MACRO_DEBATE_CONF_THRESHOLD", 0.65),
        macro_bias_carry_threshold=_get_env_float("MACRO_BIAS_CARRY_THRESHOLD", 0.65),
        macro_against_close_threshold=_get_env_float("MACRO_AGAINST_CLOSE_THRESHOLD", 0.70),
        atr_multiplier_sl=_get_env_float("ATR_MULTIPLIER_SL", 1.5),
        risk_reward_ratio=_get_env_float("RISK_REWARD_RATIO", 2.0),
        breakeven_buffer=_get_env_float("BREAKEVEN_BUFFER", 0.1),
        breakeven_monitor_times=_parse_breakeven_monitor_times(
            _get_env_str("BREAKEVEN_MONITOR_TIMES", "07,22,37,52")
        ),
        news_filter_minutes=_get_env_int("NEWS_FILTER_MINUTES", 15),
        calendar_timezone=_get_env_str("CALENDAR_TIMEZONE", "UTC"),
        jpy_usd_rate_fallback=_get_env_float("JPY_USD_RATE_FALLBACK", 155.0),
        friday_flat_time_ny=_parse_hhmm(_get_env_str("FRIDAY_FLAT_TIME_NY", "16:30"), (16, 30)),
        daily_pending_cutoff_ny=_parse_hhmm(_get_env_str("DAILY_PENDING_CUTOFF_NY", "16:45"), (16, 45)),
        monday_open_ny=_parse_hhmm(_get_env_str("MONDAY_OPEN_NY", "08:00"), (8, 0)),
        sl_structure_buffer_usd=_get_env_float("SL_STRUCTURE_BUFFER_USD", 2.0),
        spread_multiplier_limit=2.0,
        spread_samples=20,
        spread_sample_interval=0.5,
        model_analysis=_get_env_str("MODEL_ANALYSIS", "gpt-5.6-terra"),
        model_decision=_get_env_str("MODEL_DECISION", "gpt-5.6-sol"),
        model_debate=_get_env_str("MODEL_DEBATE", "gpt-5.6-terra"),
        max_news_items=_get_env_int("MAX_NEWS_ITEMS", 15),
        rss_feeds=_parse_feed_list(_get_env_str("RSS_FEEDS", "")),
        stage=_get_env_int("STAGE", 1),
        mt5_login=mt5_login,
        mt5_password=mt5_password,
        mt5_server=mt5_server,
        mt5_path=mt5_path,
        mt5_server_timezone=_get_env_str("MT5_SERVER_TIMEZONE", ""),
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
MACRO_DEBATE_CONF_THRESHOLD: Final[float] = settings.macro_debate_conf_threshold
MACRO_BIAS_CARRY_THRESHOLD: Final[float] = settings.macro_bias_carry_threshold
MACRO_AGAINST_CLOSE_THRESHOLD: Final[float] = settings.macro_against_close_threshold

ATR_MULTIPLIER_SL: Final[float] = settings.atr_multiplier_sl
RISK_REWARD_RATIO: Final[float] = settings.risk_reward_ratio
BREAKEVEN_BUFFER: Final[float] = settings.breakeven_buffer
BREAKEVEN_MONITOR_TIMES: Final[tuple[str, ...]] = settings.breakeven_monitor_times

NEWS_FILTER_MINUTES: Final[int] = settings.news_filter_minutes
CALENDAR_TIMEZONE_NAME: Final[str] = settings.calendar_timezone
JPY_USD_RATE_FALLBACK: Final[float] = settings.jpy_usd_rate_fallback
FRIDAY_FLAT_TIME_NY: Final[tuple[int, int]] = settings.friday_flat_time_ny
DAILY_PENDING_CUTOFF_NY: Final[tuple[int, int]] = settings.daily_pending_cutoff_ny
MONDAY_OPEN_NY: Final[tuple[int, int]] = settings.monday_open_ny
SL_STRUCTURE_BUFFER_USD: Final[float] = settings.sl_structure_buffer_usd
SPREAD_MULTIPLIER_LIMIT: Final[float] = settings.spread_multiplier_limit
SPREAD_SAMPLES: Final[int] = settings.spread_samples
SPREAD_SAMPLE_INTERVAL: Final[float] = settings.spread_sample_interval

MODEL_ANALYSIS: Final[str] = settings.model_analysis
MODEL_DECISION: Final[str] = settings.model_decision
MODEL_DEBATE: Final[str] = settings.model_debate
MAX_NEWS_ITEMS: Final[int] = settings.max_news_items
RSS_FEEDS: Final[tuple[str, ...]] = settings.rss_feeds
STAGE: Final[int] = settings.stage

MT5_LOGIN: Final[int | None] = settings.mt5_login
MT5_PASSWORD: Final[str] = settings.mt5_password
MT5_SERVER: Final[str] = settings.mt5_server
MT5_PATH: Final[str] = settings.mt5_path
MT5_SERVER_TIMEZONE: Final[str] = settings.mt5_server_timezone

OPENAI_API_KEY: Final[str] = settings.openai_api_key
NEWS_API_KEY: Final[str] = settings.news_api_key
FRED_API_KEY: Final[str] = settings.fred_api_key

# --------------------------------------------------------------------------- #
# Gold-specific external data (all optional; every fetcher fails safe)
# --------------------------------------------------------------------------- #
# CFTC Disaggregated Futures-Only report via the public Socrata API.
COT_DATASET_URL: Final[str] = _get_env_str(
    "COT_DATASET_URL", "https://publicreporting.cftc.gov/resource/72hh-3qpy.json"
)
COT_MARKET_NAME: Final[str] = _get_env_str("COT_MARKET_NAME", "GOLD - COMMODITY EXCHANGE INC.")
# SPDR Gold Shares daily holdings archive (CSV).
GLD_HOLDINGS_URL: Final[str] = _get_env_str(
    "GLD_HOLDINGS_URL", "https://www.spdrgoldshares.com/assets/dynamic/GLD/GLD_US_archive_EN.csv"
)
# Broker symbol candidates for a tradable dollar index; when none exists the
# index is synthesised from the major USD pairs (see mt5_client).
DXY_SYMBOL_CANDIDATES: Final[tuple[str, ...]] = tuple(
    s.strip() for s in _get_env_str("DXY_SYMBOL_CANDIDATES", "USDX,USDX#,DXY,USDOLLAR,DX").split(",") if s.strip()
)
# Hours of economic releases (back) and scheduled events (ahead) handed to the analysts.
RELEASES_LOOKBACK_HOURS: Final[int] = _get_env_int("RELEASES_LOOKBACK_HOURS", 48)
EVENTS_LOOKAHEAD_HOURS: Final[int] = _get_env_int("EVENTS_LOOKAHEAD_HOURS", 24)
