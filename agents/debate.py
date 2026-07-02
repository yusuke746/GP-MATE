from __future__ import annotations

from typing import Any

from agents.debate_graph import run_debate_graph


def run_debate(
    technical_report: dict[str, Any],
    sentiment_report: dict[str, Any],
) -> dict[str, Any]:
    """Backward-compatible wrapper for legacy tests and call sites.

    Internally delegates to the LangGraph-based debate engine.
    """
    report = run_debate_graph(technical_report, sentiment_report)

    bull_args = list(report.get("bull_arguments", []))
    bear_args = list(report.get("bear_arguments", []))

    round1 = {
        "bull": {
            "bull_case": bull_args[0] if len(bull_args) >= 1 else "",
            "conviction": float(report.get("bull_confidence", 0.0) or 0.0),
        },
        "bear": {
            "bear_case": bear_args[0] if len(bear_args) >= 1 else "",
            "conviction": float(report.get("bear_confidence", 0.0) or 0.0),
        },
    }

    round2 = {
        "bull": {
            "bull_case": bull_args[1] if len(bull_args) >= 2 else round1["bull"]["bull_case"],
            "conviction": float(report.get("bull_confidence", 0.0) or 0.0),
        },
        "bear": {
            "bear_case": bear_args[1] if len(bear_args) >= 2 else round1["bear"]["bear_case"],
            "conviction": float(report.get("bear_confidence", 0.0) or 0.0),
        },
    }

    return {
        "round1": round1,
        "round2": round2,
        "judge_summary": str(report.get("judge_summary", "") or ""),
        "_meta": dict(report.get("_meta", {})),
    }
