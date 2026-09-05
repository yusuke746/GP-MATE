"""Gold-specific positioning data: CFTC COT (managed money) and SPDR GLD holdings.

Both sources are free, unauthenticated and slow-moving (COT weekly, GLD daily),
so results are cached per calendar day. Every function fails safe: on any
network or parse problem it returns a dict with ``_meta.ok = False`` and an
error string, never raises.
"""

from __future__ import annotations

import copy
import csv
import io
import logging
from datetime import date, datetime
from threading import Lock
from typing import Any, Final

import requests

from config import COT_DATASET_URL, COT_MARKET_NAME, GLD_HOLDINGS_URL

LOGGER = logging.getLogger(__name__)

REQUEST_TIMEOUT_SECONDS: Final[float] = 20.0
HEADERS: Final[dict[str, str]] = {"User-Agent": "Mozilla/5.0 (GP-MATE data fetch)"}
COT_WEEKS: Final[int] = 60
COT_EXTREME_PCT_HIGH: Final[float] = 85.0
COT_EXTREME_PCT_LOW: Final[float] = 15.0

_CACHE_LOCK = Lock()
_DAILY_CACHE: dict[str, dict[str, Any]] = {}


def _today_key() -> str:
    return datetime.now().date().isoformat()


def _meta(ok: bool, source: str, error: str = "", cached: bool = False) -> dict[str, Any]:
    return {"ok": ok, "source": source, "error": error, "cached": cached, "fetched_at": _today_key()}


def _to_float(value: Any) -> float | None:
    try:
        text = str(value).replace(",", "").strip()
        if not text:
            return None
        return float(text)
    except (TypeError, ValueError):
        return None


def _percentile_rank(values: list[float], target: float) -> float:
    if not values:
        return 50.0
    below = sum(1 for v in values if v < target)
    equal = sum(1 for v in values if v == target)
    return 100.0 * (below + 0.5 * equal) / len(values)


# --------------------------------------------------------------------------- #
# CFTC Commitments of Traders (Disaggregated, futures only)
# --------------------------------------------------------------------------- #
def parse_cot_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Socrata rows (any order) -> managed-money net positioning summary."""
    points: list[tuple[date, float, float, float]] = []
    for row in rows:
        raw_date = str(row.get("report_date_as_yyyy_mm_dd") or "")[:10]
        long_ = _to_float(row.get("m_money_positions_long_all"))
        short = _to_float(row.get("m_money_positions_short_all"))
        oi = _to_float(row.get("open_interest_all"))
        if not raw_date or long_ is None or short is None:
            continue
        try:
            report_date = date.fromisoformat(raw_date)
        except ValueError:
            continue
        points.append((report_date, long_, short, oi or 0.0))
    if not points:
        return {"_meta": _meta(False, "cftc_cot", "no parsable rows")}

    points.sort(key=lambda p: p[0])
    nets = [p[1] - p[2] for p in points]
    latest_date, latest_long, latest_short, latest_oi = points[-1]
    net = nets[-1]
    change_1w = net - nets[-2] if len(nets) >= 2 else None
    change_4w = net - nets[-5] if len(nets) >= 5 else None
    pct = _percentile_rank(nets, net)
    if pct >= COT_EXTREME_PCT_HIGH:
        crowding = "CROWDED_LONG"
    elif pct <= COT_EXTREME_PCT_LOW:
        crowding = "CROWDED_SHORT"
    else:
        crowding = "NORMAL"
    return {
        "report_date": latest_date.isoformat(),
        "managed_money_long": latest_long,
        "managed_money_short": latest_short,
        "managed_money_net": net,
        "net_change_1w": change_1w,
        "net_change_4w": change_4w,
        "net_percentile_window": round(pct, 1),
        "window_weeks": len(nets),
        "open_interest": latest_oi,
        "net_pct_of_oi": round(100.0 * net / latest_oi, 1) if latest_oi else None,
        "crowding": crowding,
        "_meta": _meta(True, "cftc_cot"),
    }


def fetch_cot_gold(weeks: int = COT_WEEKS) -> dict[str, Any]:
    params = {
        "market_and_exchange_names": COT_MARKET_NAME,
        "$order": "report_date_as_yyyy_mm_dd DESC",
        "$limit": str(max(8, weeks)),
        "$select": ",".join(
            [
                "report_date_as_yyyy_mm_dd",
                "m_money_positions_long_all",
                "m_money_positions_short_all",
                "open_interest_all",
            ]
        ),
    }
    try:
        response = requests.get(COT_DATASET_URL, params=params, headers=HEADERS, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        rows = response.json()
        if not isinstance(rows, list):
            return {"_meta": _meta(False, "cftc_cot", "unexpected payload shape")}
        return parse_cot_rows(rows)
    except Exception as exc:
        LOGGER.warning("COT fetch failed: %s", exc)
        return {"_meta": _meta(False, "cftc_cot", str(exc))}


# --------------------------------------------------------------------------- #
# SPDR Gold Shares (GLD) holdings
# --------------------------------------------------------------------------- #
GLD_DATE_FORMATS = (
    "%d-%b-%Y", "%d-%B-%Y", "%d-%b-%y", "%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y",
    "%m/%d/%y", "%b %d, %Y", "%d %b %Y", "%Y%m%d",
)


def _parse_gld_date(text: str) -> date | None:
    text = text.strip().strip('"').strip()
    for fmt in GLD_DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def _sample_lines(text: str, count: int = 3, width: int = 160) -> str:
    lines = [line for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n") if line.strip()]
    return " | ".join(line[:width] for line in lines[:count])


def parse_gld_csv(text: str) -> dict[str, Any]:
    """SPDR archive CSV -> tonnes held with 5-day / 30-day changes.

    The file carries preamble lines before the header; the header row is the
    first row containing a 'Date' cell, and the tonnes column is the one whose
    header mentions 'Tonnes'.
    """
    # The archive is served with bare CR line endings (and occasionally CRLF
    # inside quoted cells); the csv module rejects that unless normalised.
    normalized = text.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
    reader = csv.reader(io.StringIO(normalized, newline=""))
    header: list[str] | None = None
    date_idx = tonnes_idx = -1
    points: list[tuple[date, float]] = []
    for row in reader:
        if header is None:
            lowered = [cell.strip().strip("\ufeff").lower() for cell in row]
            # Header = a row with a 'Date' cell AND a 'Tonnes' cell; preamble
            # rows may mention "date" in prose without being the header.
            date_cells = [i for i, cell in enumerate(lowered) if cell == "date" or cell.startswith("date")]
            tonnes_candidates = [i for i, cell in enumerate(lowered) if "tonne" in cell]
            if date_cells and tonnes_candidates:
                header = row
                date_idx = date_cells[0]
                tonnes_idx = tonnes_candidates[0]
            continue
        if len(row) <= max(date_idx, tonnes_idx):
            continue
        parsed_date = _parse_gld_date(row[date_idx])
        tonnes = _to_float(row[tonnes_idx])
        if parsed_date is None or tonnes is None:
            continue
        points.append((parsed_date, tonnes))

    if header is None:
        return {"_meta": _meta(False, "spdr_gld", f"no Date/Tonnes header found; file starts: {_sample_lines(text)}")}
    if not points:
        return {"_meta": _meta(False, "spdr_gld", f"header found but no parsable rows; file starts: {_sample_lines(text, 4)}")}
    points.sort(key=lambda p: p[0])
    latest_date, latest = points[-1]
    change_5d = latest - points[-6][1] if len(points) >= 6 else None
    ref_30 = None
    for d, v in points:
        if (latest_date - d).days >= 30:
            ref_30 = v
    change_30d = latest - ref_30 if ref_30 is not None else None

    def _dir(x: float | None) -> str:
        if x is None or abs(x) < 1e-9:
            return "FLAT"
        return "UP" if x > 0 else "DOWN"

    return {
        "as_of": latest_date.isoformat(),
        "tonnes": round(latest, 2),
        "change_5d": round(change_5d, 2) if change_5d is not None else None,
        "change_30d": round(change_30d, 2) if change_30d is not None else None,
        "direction_5d": _dir(change_5d),
        "direction_30d": _dir(change_30d),
        "_meta": _meta(True, "spdr_gld"),
    }


def fetch_gld_holdings() -> dict[str, Any]:
    try:
        response = requests.get(GLD_HOLDINGS_URL, headers=HEADERS, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        return parse_gld_csv(response.text)
    except Exception as exc:
        LOGGER.warning("GLD holdings fetch failed: %s", exc)
        return {"_meta": _meta(False, "spdr_gld", str(exc))}


# --------------------------------------------------------------------------- #
# Combined, cached per day
# --------------------------------------------------------------------------- #
def get_positioning(force_refresh: bool = False) -> dict[str, Any]:
    """{"cot": ..., "gld": ..., "_meta": ...}; partial success is still ok."""
    today = _today_key()
    with _CACHE_LOCK:
        if not force_refresh and today in _DAILY_CACHE:
            cached = copy.deepcopy(_DAILY_CACHE[today])
            cached["_meta"]["cached"] = True
            return cached

    cot = fetch_cot_gold()
    gld = fetch_gld_holdings()
    ok = bool(cot.get("_meta", {}).get("ok")) or bool(gld.get("_meta", {}).get("ok"))
    errors = [
        f"{name}: {block.get('_meta', {}).get('error')}"
        for name, block in (("cot", cot), ("gld", gld))
        if not block.get("_meta", {}).get("ok")
    ]
    result = {"cot": cot, "gld": gld, "_meta": _meta(ok, "positioning", "; ".join(errors))}

    with _CACHE_LOCK:
        if ok:
            _DAILY_CACHE[today] = copy.deepcopy(result)
    return result
