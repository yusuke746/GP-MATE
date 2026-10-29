"""Summaries over shadow plans: per instrument, how often the method produced
a plan, how often a pending plan was filled, and what the plans returned in R."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 3) if values else None


def summarize_plans(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_symbol: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_symbol[str(row.get("instrument") or row.get("symbol") or "?")].append(row)

    out: dict[str, dict[str, Any]] = {}
    for key, items in sorted(by_symbol.items()):
        judgments = len(items)
        plans = [r for r in items if r.get("plan_type") in {"MARKET", "PENDING"}]
        resolved = [r for r in plans if r.get("status") == "RESOLVED"]
        open_plans = [r for r in plans if r.get("status") == "OPEN"]
        filled = [r for r in resolved if r.get("outcome") not in {"NOT_FILLED", None}]
        outcomes: dict[str, int] = defaultdict(int)
        for r in resolved:
            outcomes[str(r.get("outcome"))] += 1
        r_results = [float(r.get("r_result") or 0.0) for r in filled]
        wins = [r for r in filled if float(r.get("r_result") or 0.0) > 0]
        skips: dict[str, int] = defaultdict(int)
        for r in items:
            if r.get("plan_type") == "NONE":
                skips[str(r.get("skip_reason") or "?").split(":")[0]] += 1
        mfe = [float(r["mfe_r"]) for r in filled if r.get("mfe_r") is not None]
        reached_1r = sum(1 for v in mfe if v >= 1.0)
        out[key] = {
            "judgments": judgments,
            "plans": len(plans),
            "plan_rate": round(len(plans) / judgments, 3) if judgments else None,
            "open": len(open_plans),
            "resolved": len(resolved),
            "filled": len(filled),
            "fill_rate": round(len(filled) / len(resolved), 3) if resolved else None,
            "outcomes": dict(sorted(outcomes.items())),
            "win_rate": round(len(wins) / len(filled), 3) if filled else None,
            "expectancy_r": _mean(r_results),
            "sum_r": round(sum(r_results), 3) if r_results else None,
            "mean_mfe_r": _mean(mfe),
            "reached_1r_rate": round(reached_1r / len(mfe), 3) if mfe else None,
            "mean_net_rr_planned": _mean([float(r["net_rr"]) for r in plans if r.get("net_rr") is not None]),
            "skip_reasons": dict(sorted(skips.items(), key=lambda kv: -kv[1])[:6]),
            "horizons": dict(sorted(Counter(str(r.get("trade_horizon")) for r in plans).items())),
        }
    return out


def format_shadow_report(summary: dict[str, dict[str, Any]]) -> str:
    if not summary:
        return "=== GP-MATE Shadow plans ===\n(no plans recorded yet)"
    lines = ["=== GP-MATE Shadow plans (per instrument) ==="]
    header = f"{'instrument':<10} {'judg':>5} {'plans':>5} {'rate':>5} {'open':>4} {'res':>4} {'fill':>5} {'win':>5} {'exp_R':>6} {'sum_R':>6} {'mfe':>5} {'>=1R':>5}"
    lines.append(header)
    for key, s in summary.items():
        f = lambda v, w: (f"{v:.{w}f}" if isinstance(v, (int, float)) and v is not None else "-")
        lines.append(
            f"{key:<10} {s['judgments']:>5} {s['plans']:>5} {f(s['plan_rate'], 2):>5} {s['open']:>4} {s['resolved']:>4} "
            f"{f(s['fill_rate'], 2):>5} {f(s['win_rate'], 2):>5} {f(s['expectancy_r'], 2):>6} {f(s['sum_r'], 1):>6} {f(s['mean_mfe_r'], 2):>5} {f(s['reached_1r_rate'], 2):>5}"
        )
    lines.append("")
    for key, s in summary.items():
        lines.append(f"[{key}] outcomes={s['outcomes']} horizons={s['horizons']} planned_net_rr={s['mean_net_rr_planned']}")
        lines.append(f"[{key}] skip_reasons={s['skip_reasons']}")
    lines.append("")
    lines.append("exp_R = mean result per filled plan in R (TP=+tp/sl, SL=-1, timeout/weekend=mark-to-market). "
                 "fill = filled / resolved. >=1R = share of filled plans whose best excursion reached +1R (the partial-close point).")
    return "\n".join(lines)
