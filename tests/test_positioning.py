from __future__ import annotations

from unittest.mock import Mock, patch

from agents.data import positioning


def _cot_rows() -> list[dict[str, str]]:
    rows = []
    # 12 weeks: net long climbing from 100k to 210k -> latest is the window max.
    for week in range(12):
        rows.append(
            {
                "report_date_as_yyyy_mm_dd": f"2026-{6 + week // 4:02d}-{1 + 7 * (week % 4):02d}T00:00:00.000",
                "m_money_positions_long_all": str(150_000 + 10_000 * week),
                "m_money_positions_short_all": "50000",
                "open_interest_all": "500000",
            }
        )
    return rows


def test_parse_cot_rows_flags_crowded_long() -> None:
    result = positioning.parse_cot_rows(list(reversed(_cot_rows())))
    assert result["_meta"]["ok"] is True
    assert result["report_date"] == "2026-08-22"
    assert result["managed_money_net"] == 210_000.0
    assert result["net_change_1w"] == 10_000.0
    assert result["net_change_4w"] == 40_000.0
    assert result["crowding"] == "CROWDED_LONG"
    assert result["net_pct_of_oi"] == 42.0


def test_parse_cot_rows_handles_garbage() -> None:
    result = positioning.parse_cot_rows([{"foo": "bar"}])
    assert result["_meta"]["ok"] is False


def test_fetch_cot_gold_fails_safe_on_network_error() -> None:
    with patch("agents.data.positioning.requests.get", side_effect=Exception("boom")):
        result = positioning.fetch_cot_gold()
    assert result["_meta"]["ok"] is False
    assert "boom" in result["_meta"]["error"]


GLD_CSV = """SPDR Gold Trust,,,
"Some preamble line",,,
Date,GLD Close,LBMA Gold Price,"Total Net Asset Value Tonnes in the Trust as at 4.15 p.m. NYT",NAV
01-Jul-2026,320.1,4100.0,"950.12",320
""" + "\n".join(
    f"{day:02d}-Aug-2026,325.0,4300.0,\"{950 + day}.00\",325" for day in range(1, 32)
)


def test_parse_gld_csv_reads_tonnes_and_changes() -> None:
    result = positioning.parse_gld_csv(GLD_CSV)
    assert result["_meta"]["ok"] is True
    assert result["as_of"] == "2026-08-31"
    assert result["tonnes"] == 981.0
    assert result["change_5d"] == 5.0
    assert result["direction_5d"] == "UP"
    assert result["change_30d"] == 30.0  # most recent point at least 30 days old (01-Aug)
    assert result["direction_30d"] == "UP"


def test_parse_gld_csv_without_tonnes_column_fails_safe() -> None:
    result = positioning.parse_gld_csv("Date,Close\n01-Jul-2026,1\n")
    assert result["_meta"]["ok"] is False


def test_get_positioning_is_ok_when_either_source_works(monkeypatch) -> None:
    positioning._DAILY_CACHE.clear()
    monkeypatch.setattr(positioning, "fetch_cot_gold", lambda: {"_meta": {"ok": False, "error": "down"}})
    monkeypatch.setattr(
        positioning,
        "fetch_gld_holdings",
        lambda: {"tonnes": 900.0, "direction_5d": "UP", "_meta": {"ok": True, "error": ""}},
    )
    result = positioning.get_positioning(force_refresh=True)
    assert result["_meta"]["ok"] is True
    assert "cot: down" in result["_meta"]["error"]
    assert result["gld"]["tonnes"] == 900.0

    # Same-day cache hit.
    monkeypatch.setattr(positioning, "fetch_gld_holdings", lambda: {"_meta": {"ok": False, "error": "x"}})
    cached = positioning.get_positioning()
    assert cached["_meta"]["cached"] is True
    assert cached["gld"]["tonnes"] == 900.0


def test_fetch_gld_holdings_uses_browser_ua(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(positioning, "LOG_DIR", tmp_path)
    response = Mock()
    response.raise_for_status = Mock()
    response.text = GLD_CSV
    response.content = GLD_CSV.encode("utf-8")
    with patch("agents.data.positioning.requests.get", return_value=response) as mock_get:
        result = positioning.fetch_gld_holdings()
    assert result["_meta"]["ok"] is True
    assert "User-Agent" in mock_get.call_args.kwargs["headers"]
