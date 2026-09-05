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
import json
import logging
import re
from pathlib import Path
from datetime import date, datetime
from threading import Lock
from typing import Any, Final

import requests

from config import COT_DATASET_URL, COT_MARKET_NAME, GLD_HOLDINGS_URLS, LOG_DIR

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
    return summarize_gld_points(points, source="spdr_gld:csv")


def summarize_gld_points(points: list[tuple[date, float]], source: str = "spdr_gld") -> dict[str, Any]:
    """(date, tonnes) points -> latest level with 5-observation / 30-day changes."""
    if not points:
        return {"_meta": _meta(False, source, "no points")}
    merged = sorted(dict(points).items())
    latest_date, latest = merged[-1]
    change_5d = latest - merged[-6][1] if len(merged) >= 6 else None
    ref_30 = None
    for d, v in merged:
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
        "history_points": len(merged),
        "_meta": _meta(True, source),
    }


def gld_points_from_csv(text: str) -> list[tuple[date, float]]:
    """Points only (empty on failure); parse_gld_csv wraps this with a summary."""
    result = parse_gld_csv(text)
    if not result.get("_meta", {}).get("ok"):
        return []
    # Re-run the light parse to recover the points (parse_gld_csv keeps them private).
    return _gld_points_from_csv_rows(text)


def _gld_points_from_csv_rows(text: str) -> list[tuple[date, float]]:
    normalized = text.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
    reader = csv.reader(io.StringIO(normalized, newline=""))
    header_found = False
    date_idx = tonnes_idx = -1
    points: list[tuple[date, float]] = []
    for row in reader:
        if not header_found:
            lowered = [cell.strip().strip("\ufeff").lower() for cell in row]
            date_cells = [i for i, cell in enumerate(lowered) if cell == "date" or cell.startswith("date")]
            tonnes_candidates = [i for i, cell in enumerate(lowered) if "tonne" in cell]
            if date_cells and tonnes_candidates:
                header_found, date_idx, tonnes_idx = True, date_cells[0], tonnes_candidates[0]
            continue
        if len(row) <= max(date_idx, tonnes_idx):
            continue
        parsed_date = _parse_gld_date(row[date_idx])
        tonnes = _to_float(row[tonnes_idx])
        if parsed_date is not None and tonnes is not None:
            points.append((parsed_date, tonnes))
    return points


TROY_OUNCES_PER_TONNE = 32150.7466
_XLSX_DATE_RE = re.compile(r"(\d{1,2}[-/ ][A-Za-z]{3}[-/ ]\d{2,4}|\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4}|[A-Za-z]{3} \d{1,2}, \d{4})")


def gld_point_from_xlsx(content: bytes) -> tuple[tuple[date, float] | None, str]:
    """Current tonnes from an SSGA-style daily holdings workbook.

    Scans every cell: the tonnes figure is the first number on a row whose
    label mentions 'tonnes' (or 'ounces', converted); the as-of date is the
    first date-like value in the sheet (SSGA writes 'Holdings: 04-Sep-2026').
    Returns (None, reason) when the workbook has no such cells.
    """
    try:
        import openpyxl  # type: ignore[import-not-found]
    except Exception:
        return None, "openpyxl not installed"
    try:
        workbook = openpyxl.load_workbook(io.BytesIO(content), read_only=True, data_only=True)
    except Exception as exc:
        return None, f"xlsx unreadable: {exc}"

    as_of: date | None = None
    tonnes: float | None = None

    def _num(cell: Any) -> float | None:
        if isinstance(cell, bool):
            return None
        if isinstance(cell, (int, float)):
            return float(cell)
        return _to_float(cell) if isinstance(cell, str) else None

    for sheet in workbook.worksheets:
        header_col: tuple[int, float] | None = None  # (column index, ounce->tonne divisor)
        for row in sheet.iter_rows(values_only=True):
            if all(c is None for c in row):
                continue
            for cell in row:
                if as_of is None:
                    if isinstance(cell, datetime):
                        as_of = cell.date()
                    elif isinstance(cell, str):
                        match = _XLSX_DATE_RE.search(cell)
                        if match:
                            as_of = _parse_gld_date(match.group(1))
            if tonnes is None:
                # Layout A: header row names the column, a later row carries the value.
                if header_col is not None:
                    value = _num(row[header_col[0]]) if header_col[0] < len(row) else None
                    if value is not None:
                        tonnes = value / header_col[1]
                        continue
                # Layout B: label and value on the same row.
                numbers = [v for v in (_num(c) for c in row) if v is not None]
                for idx, cell in enumerate(row):
                    if not isinstance(cell, str):
                        continue
                    label = cell.lower()
                    divisor = 1.0 if "tonne" in label else (TROY_OUNCES_PER_TONNE if ("ounce" in label or "oz" in label.split()) else None)
                    if divisor is None:
                        continue
                    if numbers:
                        tonnes = numbers[0] / divisor
                    else:
                        header_col = (idx, divisor)
                    break
            if as_of is not None and tonnes is not None:
                break
        if as_of is not None and tonnes is not None:
            break
    if tonnes is None:
        return None, "no tonnes/ounces cell in workbook"
    if as_of is None:
        as_of = datetime.now().date()
    return (as_of, round(tonnes, 2)), ""


def _history_path() -> Path:
    return Path(LOG_DIR) / "gld_holdings_history.json"


def _load_history() -> list[tuple[date, float]]:
    try:
        raw = json.loads(_history_path().read_text(encoding="utf-8"))
    except Exception:
        return []
    points: list[tuple[date, float]] = []
    for key, value in (raw.items() if isinstance(raw, dict) else []):
        try:
            points.append((date.fromisoformat(str(key)), float(value)))
        except (TypeError, ValueError):
            continue
    return points


def _save_history(points: list[tuple[date, float]], keep_days: int = 400) -> None:
    try:
        merged = dict(sorted(dict(points).items()))
        if merged:
            cutoff = max(merged) .toordinal() - keep_days
            merged = {d: v for d, v in merged.items() if d.toordinal() >= cutoff}
        path = _history_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({d.isoformat(): v for d, v in merged.items()}, indent=0), encoding="utf-8")
    except Exception as exc:  # pragma: no cover - disk problems must not break trading
        LOGGER.warning("GLD history save failed: %s", exc)


def _sniff(content: bytes) -> str:
    head = content.lstrip()[:8]
    if head.startswith(b"%PDF"):
        return "pdf"
    if head.startswith(b"PK"):
        return "xlsx"
    if head.startswith(b"<"):
        return "html"
    if head.startswith(b"\xd0\xcf\x11\xe0"):
        return "xls"
    return "text"


def fetch_gld_holdings(urls: tuple[str, ...] | None = None) -> dict[str, Any]:
    """Try each candidate URL; accept CSV (full history) or xlsx (today's
    level); merge into the on-disk history so 5d/30d changes survive a source
    that only publishes the current value."""
    candidates = urls if urls is not None else GLD_HOLDINGS_URLS
    failures: list[str] = []
    new_points: list[tuple[date, float]] = []
    source = ""
    for url in candidates:
        try:
            response = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT_SECONDS)
            response.raise_for_status()
        except Exception as exc:
            failures.append(f"{url}: {exc}")
            continue
        kind = _sniff(response.content)
        if kind in {"pdf", "html", "xls"}:
            failures.append(f"{url}: returned {kind}, not data")
            continue
        if kind == "xlsx":
            point, reason = gld_point_from_xlsx(response.content)
            if point is None:
                failures.append(f"{url}: {reason}")
                continue
            new_points, source = [point], f"spdr_gld:xlsx({url.rsplit('/', 1)[-1]})"
            break
        points = _gld_points_from_csv_rows(response.text)
        if not points:
            failures.append(f"{url}: no Date/Tonnes rows; starts: {_sample_lines(response.text, 2, 80)}")
            continue
        new_points, source = points, "spdr_gld:csv"
        break

    history = _load_history()
    if new_points:
        history = sorted(dict(history + new_points).items())
        _save_history(history)
    elif history:
        source = "spdr_gld:history_only"
        LOGGER.warning("GLD holdings: all sources failed, using on-disk history: %s", "; ".join(failures))
    else:
        LOGGER.warning("GLD holdings fetch failed: %s", "; ".join(failures))
        return {"_meta": _meta(False, "spdr_gld", "; ".join(failures) or "no candidate URLs")}

    result = summarize_gld_points(history, source=source)
    if failures:
        result["_meta"]["error"] = "; ".join(failures)
    return result


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
