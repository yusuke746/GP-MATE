"""Instrument profiles: everything in the pipeline that is not "gold".

The decision pipeline was written for XAU/USD. Running the same method on
other dollar pairs needs, per instrument, the words the analysts are told
(what BULLISH means), the headline keywords, the calendar currencies, and the
price-unit constants (round-number steps, confluence proximities, the wick
buffer behind a structural stop) that were hard-coded in USD for gold.

Profiles are static; the broker symbol name (``GOLD#``, ``USDJPY#``, ``EURUSDm``)
is resolved at runtime (``resolve_broker_symbol``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final


@dataclass(frozen=True)
class Instrument:
    key: str
    label: str              # how the analysts are told what they are looking at
    asset_noun: str         # "金価格" / "ドル円" in the questions asked
    bias_meaning: str       # what BULLISH / BEARISH means for this price
    news_keywords: tuple[str, ...]
    calendar_currencies: frozenset[str]
    price_decimals: int
    round_step_minor: float
    round_step_major: float
    round_scan_range: float
    tp_proximity_round: float
    tp_proximity_prev_day: float
    tp_proximity_ma: float
    structure_buffer: float  # wick buffer added beyond a structural SL, in price units
    aliases: tuple[str, ...] = field(default_factory=tuple)

    @property
    def symbol_candidates(self) -> tuple[str, ...]:
        return (self.key, *self.aliases)


_COMMON_USD_KEYWORDS: Final[tuple[str, ...]] = (
    "fed", "fomc", "rate", "rates", "inflation", "cpi", "pce", "dollar", "powell",
    "treasury", "yield", "payrolls", "nonfarm", "jobs report", "central bank", "tariff",
)

GOLD: Final[Instrument] = Instrument(
    key="XAUUSD",
    label="GOLD(XAU/USD)",
    asset_noun="金価格",
    bias_meaning="BULLISH は金価格(XAU/USD)の上昇、BEARISH は下落",
    news_keywords=(
        "gold", "xau", "geopolitical", "bullion", "precious metal", "safe haven", "safe-haven", "etf",
        *_COMMON_USD_KEYWORDS,
    ),
    calendar_currencies=frozenset({"USD", "XAU"}),
    price_decimals=2,
    round_step_minor=50.0,
    round_step_major=100.0,
    round_scan_range=300.0,
    tp_proximity_round=5.0,
    tp_proximity_prev_day=8.0,
    tp_proximity_ma=8.0,
    structure_buffer=2.0,
    aliases=("GOLD", "GOLD#", "XAUUSD#"),
)

USDJPY: Final[Instrument] = Instrument(
    key="USDJPY",
    label="ドル円(USD/JPY)",
    asset_noun="ドル円(USD/JPY)",
    bias_meaning="BULLISH は USD/JPY の上昇(ドル高・円安)、BEARISH は下落(ドル安・円高)",
    news_keywords=(
        "usd/jpy", "usdjpy", "yen", "japan", "boj", "bank of japan", "ueda", "intervention", "jgb",
        *_COMMON_USD_KEYWORDS,
    ),
    calendar_currencies=frozenset({"USD", "JPY"}),
    price_decimals=3,
    # Typical H1 ATR ~0.3 vs gold ~13: the USD constants scale by ~1/45.
    round_step_minor=0.5,
    round_step_major=1.0,
    round_scan_range=6.0,
    tp_proximity_round=0.1,
    tp_proximity_prev_day=0.15,
    tp_proximity_ma=0.15,
    structure_buffer=0.05,
    aliases=("USDJPY#",),
)

EURUSD: Final[Instrument] = Instrument(
    key="EURUSD",
    label="ユーロドル(EUR/USD)",
    asset_noun="ユーロドル(EUR/USD)",
    bias_meaning="BULLISH は EUR/USD の上昇(ユーロ高・ドル安)、BEARISH は下落(ユーロ安・ドル高)",
    news_keywords=(
        "eur/usd", "eurusd", "euro", "ecb", "lagarde", "eurozone", "euro zone", "bund",
        *_COMMON_USD_KEYWORDS,
    ),
    calendar_currencies=frozenset({"USD", "EUR"}),
    price_decimals=5,
    # Typical H1 ATR ~0.002: the USD constants scale by ~1/6000.
    round_step_minor=0.005,
    round_step_major=0.01,
    round_scan_range=0.05,
    tp_proximity_round=0.0008,
    tp_proximity_prev_day=0.0012,
    tp_proximity_ma=0.0012,
    structure_buffer=0.0004,
    aliases=("EURUSD#",),
)

INSTRUMENTS: Final[dict[str, Instrument]] = {inst.key: inst for inst in (GOLD, USDJPY, EURUSD)}


def _normalize(name: str) -> str:
    return "".join(ch for ch in str(name or "").lower() if ch.isalnum())


def instrument_for(symbol: str) -> Instrument:
    """The profile for a configured or broker symbol name (suffixes ignored).

    Unknown symbols fall back to the gold profile with the given key, so the
    pipeline keeps working, just with gold-sized USD constants.
    """
    normalized = _normalize(symbol)
    for inst in INSTRUMENTS.values():
        for candidate in inst.symbol_candidates:
            if normalized == _normalize(candidate):
                return inst
    for inst in INSTRUMENTS.values():
        if normalized.startswith(_normalize(inst.key)):
            return inst
    if normalized.startswith("gold") or normalized.startswith("xau"):
        return GOLD
    return GOLD


def parse_symbol_list(value: str) -> tuple[str, ...]:
    """'USDJPY, EURUSD' -> ('USDJPY', 'EURUSD'); unknown keys are dropped."""
    out: list[str] = []
    for raw in str(value or "").split(","):
        key = raw.strip().upper()
        if not key:
            continue
        inst = instrument_for(key)
        if _normalize(inst.key) != _normalize(key) and key not in inst.aliases:
            continue
        if inst.key not in out:
            out.append(inst.key)
    return tuple(out)
