from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any, Callable


@dataclass(frozen=True)
class TimeCapsuleInputs:
    prices_h4: Any
    prices_h1: Any
    news_items: list[dict[str, Any]]


@dataclass(frozen=True)
class TimeCapsuleResult:
    decision_date: date
    ok: bool
    action: str
    reason: str


def get_timecapsule_data(
    decision_date: datetime,
    load_prices: Callable[[str, datetime], Any],
    load_news: Callable[[datetime], list[dict[str, Any]]],
) -> TimeCapsuleInputs:
    """Return only data available before decision time to avoid look-ahead bias."""
    cutoff = decision_date - timedelta(seconds=1)
    prices_h4 = load_prices("H4", cutoff)
    prices_h1 = load_prices("H1", cutoff)
    news_items = load_news(decision_date)
    return TimeCapsuleInputs(prices_h4=prices_h4, prices_h1=prices_h1, news_items=news_items)


def run_timecapsule_test(
    start_date: date,
    end_date: date,
    evaluate_once: Callable[[datetime], dict[str, Any]],
) -> list[TimeCapsuleResult]:
    """Run a light validation loop for bug detection.

    This is for stability checks (exceptions, parsing, consistency), not for PnL estimation.
    """
    if end_date < start_date:
        return []

    results: list[TimeCapsuleResult] = []
    current = start_date

    while current <= end_date:
        decision_dt = datetime.combine(current, datetime.min.time(), tzinfo=UTC)
        try:
            payload = evaluate_once(decision_dt)
            action = str(payload.get("action", "HOLD"))
            results.append(
                TimeCapsuleResult(
                    decision_date=current,
                    ok=True,
                    action=action,
                    reason="OK",
                )
            )
        except Exception as exc:
            results.append(
                TimeCapsuleResult(
                    decision_date=current,
                    ok=False,
                    action="HOLD",
                    reason=str(exc),
                )
            )

        current += timedelta(days=1)

    return results
