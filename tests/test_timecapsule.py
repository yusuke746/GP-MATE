from __future__ import annotations

from datetime import UTC, date, datetime

from backtest.timecapsule import get_timecapsule_data, run_timecapsule_test


def test_get_timecapsule_data_calls_sources_with_cutoff() -> None:
    captured: dict[str, datetime] = {}

    def load_prices(tf: str, end_dt: datetime):
        captured[tf] = end_dt
        return [{"close": 1.0}]

    def load_news(end_dt: datetime):
        captured["news"] = end_dt
        return [{"title": "gold"}]

    decision = datetime(2026, 7, 2, 12, 0, tzinfo=UTC)
    result = get_timecapsule_data(decision, load_prices, load_news)

    assert "H4" in captured
    assert "H1" in captured
    assert "news" in captured
    assert result.news_items[0]["title"] == "gold"


def test_run_timecapsule_test_handles_success_and_failure() -> None:
    def evaluate_once(dt: datetime):
        if dt.date().day % 2 == 0:
            return {"action": "BUY"}
        raise RuntimeError("boom")

    results = run_timecapsule_test(
        start_date=date(2026, 7, 1),
        end_date=date(2026, 7, 3),
        evaluate_once=evaluate_once,
    )

    assert len(results) == 3
    assert any(item.ok for item in results)
    assert any(not item.ok for item in results)
