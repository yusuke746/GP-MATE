from __future__ import annotations

import io
import json
from datetime import date, datetime
from unittest.mock import Mock, patch

import openpyxl

from agents.data import positioning

CSV = "Date,Close,Total Net Asset Value Tonnes in the Trust,NAV\n" + "\n".join(
    f"{d:02d}-Aug-2026,300,{900 + d}.0,300" for d in range(1, 12)
) + "\n"


def _response(content: bytes) -> Mock:
    response = Mock()
    response.raise_for_status = Mock()
    response.content = content
    response.text = content.decode("utf-8", errors="replace")
    return response


def _xlsx(rows: list[list[object]]) -> bytes:
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    for row in rows:
        sheet.append(row)
    buffer = io.BytesIO()
    workbook.save(buffer)
    return buffer.getvalue()


def test_sniff_detects_pdf_html_xlsx_text() -> None:
    assert positioning._sniff(b"%PDF-1.5 ...") == "pdf"
    assert positioning._sniff(b"<!doctype html>") == "html"
    assert positioning._sniff(_xlsx([["a"]])) == "xlsx"
    assert positioning._sniff(b"Date,Tonnes\n") == "text"


def test_fetch_skips_pdf_and_uses_csv_candidate(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(positioning, "LOG_DIR", tmp_path)
    responses = {"https://a/pdf": _response(b"%PDF-1.5 factsheet"), "https://b/csv": _response(CSV.encode())}
    with patch("agents.data.positioning.requests.get", side_effect=lambda url, **kw: responses[url]):
        result = positioning.fetch_gld_holdings(urls=("https://a/pdf", "https://b/csv"))

    assert result["_meta"]["ok"] is True
    assert result["_meta"]["source"] == "spdr_gld:csv"
    assert result["tonnes"] == 911.0
    assert result["change_5d"] == 5.0
    assert "returned pdf" in result["_meta"]["error"]
    saved = json.loads((tmp_path / "gld_holdings_history.json").read_text())
    assert saved["2026-08-11"] == 911.0


def test_xlsx_point_reads_tonnes_and_as_of_date() -> None:
    content = _xlsx(
        [
            ["Fund Name:", "SPDR Gold Shares"],
            ["Holdings:", "04-Sep-2026"],
            ["Name", "Ticker", "Weight", "Tonnes in the Trust"],
            ["GOLD", "GLD", 100.0, 981.23],
        ]
    )
    point, reason = positioning.gld_point_from_xlsx(content)
    assert reason == ""
    assert point == (date(2026, 9, 4), 981.23)


def test_xlsx_point_converts_ounces_and_reads_datetime_cells() -> None:
    content = _xlsx([["As of", datetime(2026, 9, 4)], ["Ounces of Gold in the Trust", 31_539_000.0]])
    point, _ = positioning.gld_point_from_xlsx(content)
    assert point is not None
    assert point[0] == date(2026, 9, 4)
    assert abs(point[1] - 981.0) < 0.5


def test_xlsx_without_tonnes_reports_reason() -> None:
    point, reason = positioning.gld_point_from_xlsx(_xlsx([["Name", "Weight"], ["GOLD", 100.0]]))
    assert point is None
    assert "no tonnes" in reason


def test_xlsx_daily_level_accumulates_into_history(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(positioning, "LOG_DIR", tmp_path)
    # Six prior days already on disk from earlier runs.
    (tmp_path / "gld_holdings_history.json").write_text(
        json.dumps({f"2026-08-{d:02d}": 900.0 + d for d in range(25, 31)}), encoding="utf-8"
    )
    content = _xlsx([["Holdings:", "31-Aug-2026"], ["Tonnes", 940.0]])
    with patch("agents.data.positioning.requests.get", return_value=_response(content)):
        result = positioning.fetch_gld_holdings(urls=("https://ssga/x.xlsx",))

    assert result["_meta"]["ok"] is True
    assert result["_meta"]["source"].startswith("spdr_gld:xlsx")
    assert result["as_of"] == "2026-08-31"
    assert result["tonnes"] == 940.0
    assert result["change_5d"] == 940.0 - 926.0
    assert result["history_points"] == 7


def test_all_sources_down_falls_back_to_history(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(positioning, "LOG_DIR", tmp_path)
    (tmp_path / "gld_holdings_history.json").write_text(json.dumps({"2026-08-30": 930.0, "2026-08-31": 940.0}))
    with patch("agents.data.positioning.requests.get", side_effect=Exception("down")):
        result = positioning.fetch_gld_holdings(urls=("https://x",))
    assert result["_meta"]["ok"] is True
    assert result["_meta"]["source"] == "spdr_gld:history_only"
    assert result["tonnes"] == 940.0
    assert "down" in result["_meta"]["error"]


def test_all_sources_down_without_history_fails_safe(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(positioning, "LOG_DIR", tmp_path)
    with patch("agents.data.positioning.requests.get", return_value=_response(b"%PDF-1.5")):
        result = positioning.fetch_gld_holdings(urls=("https://x",))
    assert result["_meta"]["ok"] is False
    assert "returned pdf" in result["_meta"]["error"]
