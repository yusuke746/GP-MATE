"""Append-only JSONL store for forecast-only records plus per-forecast input archives."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from config import LOG_DIR

FORECASTS_FILENAME = "forecasts.jsonl"
INPUTS_DIRNAME = "forecast_inputs"


def forecasts_path() -> Path:
    return Path(LOG_DIR) / FORECASTS_FILENAME


def inputs_dir() -> Path:
    return Path(LOG_DIR) / INPUTS_DIRNAME


def append_forecast(record: dict[str, Any], path: Path | None = None) -> None:
    target = path or forecasts_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def read_forecasts(path: Path | None = None) -> list[dict[str, Any]]:
    target = path or forecasts_path()
    if not target.exists():
        return []
    rows: list[dict[str, Any]] = []
    with target.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def write_forecasts(records: list[dict[str, Any]], path: Path | None = None) -> None:
    """Rewrite the whole file atomically (temp file + replace)."""
    target = path or forecasts_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".forecasts-", suffix=".tmp", dir=str(target.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fp:
            for record in records:
                fp.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        os.replace(tmp_name, target)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def save_inputs(forecast_id: str, payload: dict[str, Any], directory: Path | None = None) -> str:
    target_dir = directory or inputs_dir()
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{forecast_id}.json"
    target.write_text(json.dumps(payload, ensure_ascii=False, default=str, indent=1), encoding="utf-8")
    return str(target)
