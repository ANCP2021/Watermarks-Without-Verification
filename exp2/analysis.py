from __future__ import annotations

import math

import numpy as np

from exp1.common import ARMS
from exp1.stats import bootstrap_mean_ci, ci_covers, ci_within

from .settings import FPR_TARGETS, PASS1_MARGIN, PREFIX_LENGTHS

UNPARSEABLE = "unparseable"
HARNESS_ERROR = "harness_error"
DELTAS = {"delta_wm": ("A", "B1"), "delta_ctrl": ("B2", "B1")}


def complete_triples(generations: dict[tuple, dict]) -> list[tuple[str, int]]:
    by_item: dict[tuple, set] = {}
    for pid, arm, k in generations:
        by_item.setdefault((pid, k), set()).add(arm)
    return sorted(item for item, arms in by_item.items() if arms >= set(ARMS))


def kept_samples(triples: list[tuple], executions: dict[tuple, dict]) -> list[tuple]:
    kept = []
    for pid, k in triples:
        recs = [executions.get((pid, arm, k)) for arm in ARMS]
        if any(r is None for r in recs):
            continue
        if any(r["status"] == UNPARSEABLE for r in recs):
            continue
        kept.append((pid, k))
    return kept


def executed_triples(triples: list[tuple], executions: dict[tuple, dict]) -> list[tuple]:
    return [(pid, k) for pid, k in triples
            if all((pid, arm, k) in executions
                   and executions[(pid, arm, k)]["status"] != HARNESS_ERROR for arm in ARMS)]


def pass1_by_problem(samples: list[tuple], executions: dict[tuple, dict],
                     unparseable_is_fail: bool = False) -> dict[str, dict[str, float]]:
    hits: dict[str, dict[str, list[int]]] = {arm: {} for arm in ARMS}
    for pid, k in samples:
        for arm in ARMS:
            rec = executions[(pid, arm, k)]
            if rec["status"] == UNPARSEABLE and not unparseable_is_fail:
                continue
            hits[arm].setdefault(pid, []).append(1 if rec["status"] == "passed" else 0)
    return {arm: {pid: sum(v) / len(v) for pid, v in per.items() if v}
            for arm, per in hits.items()}


def paired_vectors(pass1: dict[str, dict[str, float]]) -> tuple[list[str], dict[str, np.ndarray]]:
    pids = sorted(set.intersection(*(set(pass1[arm]) for arm in ARMS))) if pass1["A"] else []
    return pids, {arm: np.asarray([pass1[arm][p] for p in pids], dtype=float) for arm in ARMS}


def pass1_summary(pass1: dict[str, dict[str, float]], margin: float = PASS1_MARGIN) -> dict:
    pids, vec = paired_vectors(pass1)
    if not pids:
        return {"n_problems": 0}
    out = {
        "n_problems": len(pids),
        "problem_ids": pids,
        "mean_by_arm": {arm: bootstrap_mean_ci(vec[arm]) for arm in ARMS},
    }
    for name, (left, right) in DELTAS.items():
        out[name] = bootstrap_mean_ci(vec[left] - vec[right])
        out[name]["definition"] = f"pass@1_{left} - pass@1_{right}, paired per problem"
    out["bound"] = {
        "statement": f"95% CI of mean(pass@1_A - pass@1_B1) within +/- {margin}",
        "margin": margin,
        "status": "PASS" if ci_within(out["delta_wm"], -margin, margin) else "FAIL",
    }
    out["control_covers_zero"] = ci_covers(out["delta_ctrl"], 0.0)
    out["per_problem_delta_wm"] = {p: float(a - b) for p, a, b in zip(pids, vec["A"], vec["B1"])}
    return out


def pass1_by_source(pass1: dict[str, dict[str, float]], source_of: dict[str, str]) -> dict:
    out = {}
    for source in sorted(set(source_of.values())):
        sub = {arm: {p: v for p, v in pass1[arm].items() if source_of.get(p) == source}
               for arm in ARMS}
        pids, vec = paired_vectors(sub)
        if not pids:
            continue
        out[source] = {"n_problems": len(pids),
                       "mean_by_arm": {arm: bootstrap_mean_ci(vec[arm]) for arm in ARMS},
                       "delta_wm": bootstrap_mean_ci(vec["A"] - vec["B1"]),
                       "delta_ctrl": bootstrap_mean_ci(vec["B2"] - vec["B1"])}
    return out



def exact_match(items: list[tuple], text_of) -> dict:
    n = len(items)
    if n == 0:
        return {"n": 0, "A_eq_B1": None, "B1_eq_B2": None, "A_eq_B1_count": 0,
                "B1_eq_B2_count": 0}
    a_b1 = sum(1 for pid, k in items if text_of(pid, "A", k) == text_of(pid, "B1", k))
    b1_b2 = sum(1 for pid, k in items if text_of(pid, "B1", k) == text_of(pid, "B2", k))
    return {"n": n, "A_eq_B1": a_b1 / n, "B1_eq_B2": b1_b2 / n,
            "A_eq_B1_count": a_b1, "B1_eq_B2_count": b1_b2}



def filter_counts(generations: dict[tuple, dict], executions: dict[tuple, dict],
                  triples: list[tuple], executed: list[tuple], kept: list[tuple]) -> dict:
    per_arm = {}
    for arm in ARMS:
        gens = [r for (pid, a, k), r in generations.items() if a == arm]
        execs = [r for (pid, a, k), r in executions.items() if a == arm]
        statuses: dict[str, int] = {}
        methods: dict[str, int] = {}
        assembly: dict[str, int] = {}
        for r in execs:
            statuses[r["status"]] = statuses.get(r["status"], 0) + 1
            m = (r.get("extraction") or {}).get("method", "none")
            methods[m] = methods.get(m, 0) + 1
            if r.get("assembly"):
                assembly[r["assembly"]] = assembly.get(r["assembly"], 0) + 1
        per_arm[arm] = {
            "n_generated": len(gens),
            "n_executed": len(execs),
            "n_unparseable": statuses.get(UNPARSEABLE, 0),
            "n_harness_error": statuses.get(HARNESS_ERROR, 0),
            "n_length_truncated": sum(1 for r in gens if r.get("finish_reason") == "length"),
            "n_unclosed_fence": sum(1 for r in execs
                                    if (r.get("extraction") or {}).get("closed") is False),
            "status_counts": dict(sorted(statuses.items())),
            "extraction_method_counts": dict(sorted(methods.items())),
            "assembly_counts": dict(sorted(assembly.items())),
        }
    n_items = len({(pid, k) for pid, arm, k in generations})
    return {
        "n_samples_generated": len(generations),
        "n_items": n_items,
        "n_complete_triples": len(triples),
        "n_incomplete_triples": n_items - len(triples),
        "n_executed_triples": len(executed),
        "n_unexecuted_triples": len(triples) - len(executed),
        "n_kept_samples": len(kept),
        "n_dropped_samples": len(executed) - len(kept),
        "n_problems_kept": len({pid for pid, _ in kept}),
        "per_arm": per_arm,
        "rule": "a (problem, sample) triple is dropped from every arm if any arm's "
                "sample is unparseable; syntax errors inside an extracted fence are "
                "kept and count as failures",
    }



def fpr_threshold(null_scores, fpr: float) -> tuple[float, float]:
    null = np.sort(np.asarray([s for s in null_scores if s is not None], dtype=float))[::-1]
    n = null.size
    if n == 0:
        return float("nan"), float("nan")
    m = int(math.floor(fpr * n))
    threshold = float(null[min(m, n - 1)])
    achieved = float(np.mean(null > threshold))
    return threshold, achieved


def tpr_at(wm_scores, threshold: float) -> float:
    wm = np.asarray([s for s in wm_scores if s is not None], dtype=float)
    return float(np.mean(wm > threshold)) if wm.size else float("nan")


def auroc(wm_scores, null_scores) -> float:
    wm = np.asarray([s for s in wm_scores if s is not None], dtype=float)
    null = np.asarray([s for s in null_scores if s is not None], dtype=float)
    if wm.size == 0 or null.size == 0:
        return float("nan")
    allv = np.concatenate([wm, null])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(allv.size, dtype=float)
    sorted_vals = allv[order]
    i = 0
    while i < allv.size:
        j = i
        while j + 1 < allv.size and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    r_wm = ranks[: wm.size].sum()
    return float((r_wm - wm.size * (wm.size + 1) / 2.0) / (wm.size * null.size))


def _score_at(rec: dict, length) -> float | None:
    if length == "full":
        return rec.get("score_full")
    p = (rec.get("prefix") or {}).get(str(length))
    return None if p is None else p.get("score")


def detection_curve(records: list[dict], prefix_lengths=PREFIX_LENGTHS,
                    fprs=FPR_TARGETS) -> list[dict]:
    rows = []
    for L in list(prefix_lengths) + ["full"]:
        wm = [_score_at(r, L) for r in records if r["arm"] == "A"]
        null = [_score_at(r, L) for r in records if r["arm"] in ("B1", "B2")]
        wm_v = [s for s in wm if s is not None]
        null_v = [s for s in null if s is not None]
        row = {"prefix_length": L, "n_wm": len(wm_v), "n_null": len(null_v),
               "n_null_B1": sum(1 for r in records if r["arm"] == "B1"
                                and _score_at(r, L) is not None),
               "n_null_B2": sum(1 for r in records if r["arm"] == "B2"
                                and _score_at(r, L) is not None),
               "mean_score_wm": float(np.mean(wm_v)) if wm_v else float("nan"),
               "mean_score_null": float(np.mean(null_v)) if null_v else float("nan"),
               "auroc": auroc(wm_v, null_v)}
        for fpr in fprs:
            thr, achieved = fpr_threshold(null_v, fpr)
            tag = f"fpr{int(round(fpr * 100)):02d}"
            row[f"threshold_{tag}"] = thr
            row[f"achieved_{tag}"] = achieved
            row[f"tpr_{tag}"] = tpr_at(wm_v, thr) if null_v else float("nan")
        rows.append(row)
    return rows


def curve_row(rows: list[dict], length) -> dict | None:
    for r in rows:
        if r["prefix_length"] == length:
            return r
    return None
