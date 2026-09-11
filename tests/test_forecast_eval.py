from __future__ import annotations

import numpy as np

from analysis import forecast_eval as fe


def _row(ts: str, p_up: float, p_down: float, outcome: str, momentum: float = 1.0, **extra) -> dict:
    row = {
        "ts_utc": ts,
        "ok": True,
        "probs_valid": True,
        "p_up": p_up,
        "p_down": p_down,
        "p_timeout": round(1.0 - p_up - p_down, 6),
        "outcome": outcome,
        "momentum_3": momentum,
        "model": "m",
        "used_debate": False,
        "horizon_bars": 6,
    }
    row.update(extra)
    return row


def _series(n: int, up_prob: float, outcome_fn) -> list[dict]:
    rows = []
    for i in range(n):
        ts = f"2026-09-{1 + i // 24:02d}T{i % 24:02d}:00:00+00:00"
        rows.append(_row(ts, up_prob, (1 - up_prob) / 2, outcome_fn(i)))
    return rows


def test_brier_and_logloss_extremes() -> None:
    perfect = [_row("2026-09-01T00:00:00+00:00", 1.0, 0.0, "UP")]
    assert fe.brier_per_row(fe.llm_probs(perfect), ["UP"])[0] == 0.0
    uniform = fe.baseline_uniform(perfect)
    assert abs(fe.brier_per_row(uniform, ["UP"])[0] - 2.0 / 3.0) < 1e-9
    assert abs(fe.logloss_per_row(uniform, ["UP"])[0] - np.log(3)) < 1e-9


def test_scorable_rows_excludes_failed_invalid_ambiguous_and_unresolved() -> None:
    rows = [
        _row("2026-09-01T00:00:00+00:00", 0.5, 0.3, "UP"),
        _row("2026-09-01T01:00:00+00:00", 0.5, 0.3, "AMBIGUOUS"),
        _row("2026-09-01T02:00:00+00:00", 0.5, 0.3, None),
        _row("2026-09-01T03:00:00+00:00", 0.5, 0.3, "DOWN", ok=False),
        _row("2026-09-01T04:00:00+00:00", 0.5, 0.3, "DOWN", probs_valid=False),
    ]
    assert len(fe.scorable_rows(rows)) == 1
    counts = fe.counts_summary(rows)
    assert counts == {"total": 5, "llm_failed": 1, "invalid_probs": 1, "resolved": 4, "ambiguous": 1, "scorable": 1}


def test_baselines_base_rate_and_momentum() -> None:
    rows = _series(10, 0.5, lambda i: "UP" if i < 6 else "DOWN")
    rates = fe.base_rates(rows)
    assert list(np.round(rates, 2)) == [0.6, 0.4, 0.0]
    rows[0]["momentum_3"] = -1.0
    rows[1]["momentum_3"] = 0.0
    mom = fe.baseline_momentum(rows)
    assert list(mom[0]) == [0.25, 0.45, 0.30]
    assert list(np.round(mom[1], 2)) == [0.6, 0.4, 0.0]
    assert list(mom[2]) == [0.45, 0.25, 0.30]


def test_reliability_table_bins_predictions() -> None:
    rows = _series(20, 0.65, lambda i: "UP" if i % 2 == 0 else "DOWN")
    table = fe.reliability_table(rows, "p_up", "UP")
    bin_06 = next(r for r in table if r["bin"] == "[0.6,0.7)")
    assert bin_06["n"] == 20 and bin_06["mean_pred"] == 0.65 and bin_06["actual_rate"] == 0.5
    assert all(r["n"] == 0 for r in table if r["bin"] != "[0.6,0.7)")


def test_block_bootstrap_ci_brackets_mean_and_is_deterministic() -> None:
    values = np.array([0.1, 0.2, 0.3, 0.4] * 20)
    lo, hi = fe.block_bootstrap_ci(values, block_size=12, rounds=200, seed=1)
    assert lo - 1e-9 <= values.mean() <= hi + 1e-9
    assert (lo, hi) == fe.block_bootstrap_ci(values, block_size=12, rounds=200, seed=1)
    assert fe.block_bootstrap_ci(np.array([]))[0] != fe.block_bootstrap_ci(np.array([]))[0]  # nan


def test_evaluate_reports_significant_improvement_for_sharp_forecaster() -> None:
    # Forecaster that always knows the answer with p=0.9 beats the base rate.
    def outcome(i: int) -> str:
        return ("UP", "DOWN", "TIMEOUT")[i % 3]

    rows = []
    for i in range(120):
        o = outcome(i)
        probs = {"UP": (0.9, 0.05), "DOWN": (0.05, 0.9), "TIMEOUT": (0.05, 0.05)}[o]
        rows.append(_row(f"2026-09-{1 + i // 24:02d}T{i % 24:02d}:00:00+00:00", probs[0], probs[1], o))
    report = fe.evaluate(rows)
    assert report["llm"]["brier"] < report["baselines"]["base_rate"]
    assert report["bootstrap"]["significantly_better"] is True
    text = fe.format_report(report)
    assert "significantly better than base rate: YES" in text
    assert "ブロック" in text  # caveat present
    assert any(r["group"].startswith("NY ") for r in report["by_ny_2h"])


def test_evaluate_handles_no_scorable_rows() -> None:
    report = fe.evaluate([_row("2026-09-01T00:00:00+00:00", 0.5, 0.3, None)])
    assert "llm" not in report
    assert "(no scorable rows yet)" in fe.format_report(report)


def test_write_csv(tmp_path) -> None:
    rows = _series(30, 0.5, lambda i: "UP" if i % 2 else "DOWN")
    report = fe.evaluate(rows)
    out = tmp_path / "eval.csv"
    fe.write_csv(report, out)
    text = out.read_text(encoding="utf-8")
    assert "reliability_p_up" in text and "by_ny_2h" in text
