"""Probe every external data source GP-MATE relies on and print a health table.

Run on the production machine (network + MT5 available):

    python scripts/check_data_sources.py            # everything
    python scripts/check_data_sources.py --no-mt5   # skip the MT5 dollar index

Use it after changing RSS_FEEDS / COT_* / GLD_HOLDINGS_URL in .env, or when
the trade log shows news_feeds_live dropping.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import FRED_API_KEY, RSS_FEEDS  # noqa: E402
from agents.data.fred_client import SERIES_MAP, get_macro_data  # noqa: E402
from agents.data.positioning import fetch_cot_gold, fetch_gld_holdings  # noqa: E402
from agents.data.releases import build_recent_releases, build_upcoming_events  # noqa: E402
from data.news_client import GOLD_KEYWORDS, _contains_keywords, _extract_rss_items, _safe_get, fetch_calendar_events  # noqa: E402


def _line(name: str, ok: bool, detail: str) -> None:
    print(f"{'OK ' if ok else 'NG '} {name:<28} {detail}")


def check_feeds() -> None:
    print("\n[RSS feeds]")
    for url in RSS_FEEDS:
        response = _safe_get(url)
        if response is None:
            _line(url[:60], False, "unreachable / non-200")
            continue
        items = _extract_rss_items(response.text, source=url)
        keyword = [i for i in items if _contains_keywords(str(i.get("title", "")), GOLD_KEYWORDS)]
        dated = [i for i in keyword if i.get("published_at")]
        newest = max((str(i.get("published_at")) for i in items if i.get("published_at")), default="-")
        _line(url[:60], bool(dated), f"items={len(items)} gold_kw={len(keyword)} dated={len(dated)} newest={newest[:16]}")


def check_calendar() -> None:
    print("\n[Economic calendar]")
    events = fetch_calendar_events()
    if events is None:
        _line("forexfactory xml", False, "unreachable or unparsable")
        return
    now = datetime.now(UTC)
    with_forecast = sum(1 for e in events if e.get("forecast"))
    with_actual = sum(1 for e in events if e.get("actual"))
    _line("forexfactory xml", True, f"high-impact USD/XAU events this week={len(events)} forecast={with_forecast} actual={with_actual}")
    releases = build_recent_releases(events, now=now, lookback_hours=72)
    _line("recent releases (72h)", True, f"{len(releases)} events, actual resolved for {sum(1 for r in releases if r.get('actual') is not None)}")
    for r in releases[:6]:
        print(f"      {r['time_utc'][:16]} {r['title'][:36]:<36} fc={r['forecast']} act={r['actual']} ({r['actual_source']}) surprise={r['surprise']}")
    upcoming = build_upcoming_events(events, now=now, lookahead_hours=48)
    for e in upcoming[:4]:
        print(f"      next: {e['title'][:40]} in {e['hours_ahead']}h (fc={e['forecast']})")


def check_fred() -> None:
    print("\n[FRED]")
    if not FRED_API_KEY:
        _line("FRED_API_KEY", False, "missing")
        return
    data = get_macro_data(force_refresh=True)
    meta = data.get("_meta", {})
    _line("get_macro_data", bool(meta.get("ok")), f"error={meta.get('error') or '-'} warnings={meta.get('warnings')}")
    for key in SERIES_MAP:
        block = data.get(key, {})
        print(f"      {key:<10} {SERIES_MAP[key]:<9} value={block.get('value')} 30d={block.get('change_30d')} 5d={block.get('change_5d')} as_of={block.get('as_of')}")


def check_positioning() -> None:
    print("\n[Positioning]")
    cot = fetch_cot_gold()
    _line("CFTC COT (managed money)", bool(cot.get("_meta", {}).get("ok")),
          f"report={cot.get('report_date')} net={cot.get('managed_money_net')} 4w={cot.get('net_change_4w')} pct={cot.get('net_percentile_window')} {cot.get('crowding', cot.get('_meta', {}).get('error'))}")
    gld = fetch_gld_holdings()
    _line("SPDR GLD holdings", bool(gld.get("_meta", {}).get("ok")),
          f"as_of={gld.get('as_of')} tonnes={gld.get('tonnes')} 5d={gld.get('change_5d')} 30d={gld.get('change_30d')} {gld.get('_meta', {}).get('error') or ''}")


def check_mt5_dollar_index() -> None:
    print("\n[MT5 dollar index]")
    from data.mt5_client import get_dollar_index_snapshot

    snap = get_dollar_index_snapshot()
    ok = bool(snap.get("_meta", {}).get("ok"))
    _line("dollar index", ok, json.dumps({k: v for k, v in snap.items() if k != "_meta"}, default=str) if ok else snap.get("_meta", {}).get("error", ""))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-mt5", action="store_true", help="skip the MT5 dollar-index probe")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s %(message)s")

    check_feeds()
    check_calendar()
    check_fred()
    check_positioning()
    if not args.no_mt5:
        check_mt5_dollar_index()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
