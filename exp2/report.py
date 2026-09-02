from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from exp1.common import (ARMS, load_run_config, log, read_jsonl, update_run_config, utcnow,
                         write_json_atomic)
from exp1.report import _fmt, _fmt_ci, _parity_markdown

from .analysis import (complete_triples, curve_row, detection_curve, exact_match,
                       executed_triples, filter_counts, kept_samples, pass1_by_problem,
                       pass1_by_source, pass1_summary)
from .detect import DET_KEY, DETECTION_FILE, DETECTOR_PATH, TOKENIZATION
from .execute import EXECUTION_FILE, index_records
from .extract import EXTRACTION_RULES_VERSION, RULES, extract_code
from .generate import GENERATIONS_FILE
from .parity import check_exp2_parity, print_parity
from .problems import load_problems
from .settings import (BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED, DETECTION_DOMAINS, FIGURE_DOMAINS,
                       FPR_TARGETS, N_OUTLIER_PROBLEMS, PASS1_MARGIN, PREFIX_LENGTHS, PRIMARY_FPR)

FIGURE_FILE = "detection_curves.png"
CURVES_FILE = "curves.csv"
DOMAIN_LABEL = {"prose": "prose (Exp 1 responses)", "code": "code (full response)",
                "code_extracted": "code (extracted fence only)",
                "code_ids": "code (generated token ids, no re-tokenisation)"}


def _fmt_pct(value, digits: int = 1) -> str:
    if value is None or not isinstance(value, (int, float)) or value != value:
        return "n/a"
    return f"{100 * value:.{digits}f}%"


def _fpr_tag(fpr: float) -> str:
    return f"fpr{int(round(fpr * 100)):02d}"


def load_artefacts(out_dir: Path, problems_path: str) -> dict:
    run_config = load_run_config(out_dir)
    if not run_config.get("stages", {}).get("generate"):
        raise SystemExit(f"{out_dir}/run_config.json has no generate stage -- run `generate`")
    generations = index_records(read_jsonl(out_dir / GENERATIONS_FILE))
    if not generations:
        raise SystemExit(f"no generations in {out_dir} -- run `generate`")
    executions = index_records(read_jsonl(out_dir / EXECUTION_FILE))
    if not executions:
        raise SystemExit(f"no executions in {out_dir} -- run `execute`")
    stale = {r.get("extraction_rules") for r in executions.values()} - {EXTRACTION_RULES_VERSION}
    if stale:
        raise SystemExit(f"execution.jsonl was produced under extraction rules {sorted(stale)}, "
                         f"this code implements {EXTRACTION_RULES_VERSION}; re-run `execute` "
                         "into a fresh --out directory")
    detections = list(index_records(read_jsonl(out_dir / DETECTION_FILE), DET_KEY).values())
    if not detections:
        raise SystemExit(f"no detection scores in {out_dir} -- run `detect`")
    payload = load_problems(problems_path, create_if_missing=False)
    problems = {p["problem_id"]: p for p in payload["problems"]}
    return {"run_config": run_config, "generations": generations, "executions": executions,
            "detections": detections, "problems": problems}


def _length_metrics(generations: dict[tuple, dict], triples: list[tuple]) -> dict:
    if not triples:
        return None
    counts = {arm: np.asarray([generations[(pid, arm, k)]["token_count"] for pid, k in triples],
                              dtype=float) for arm in ARMS}
    return {"n": len(triples), "unit": "generator tokens",
            "per_arm": {arm: {"mean": float(c.mean()), "median": float(np.median(c)),
                              "min": int(c.min()), "max": int(c.max())}
                        for arm, c in counts.items()}}


def compute_metrics(out_dir: Path, art: dict, parity: dict) -> dict:
    run_config, generations, executions = art["run_config"], art["generations"], art["executions"]
    detections, problems = art["detections"], art["problems"]
    source_of = {pid: p["source"] for pid, p in problems.items()}

    triples = complete_triples(generations)
    executed = executed_triples(triples, executions)
    kept = kept_samples(executed, executions)
    scored_code = {(r["item_id"], r["arm"], r["sample_idx"]) for r in detections
                   if r["domain"] == "code"}
    completeness = {
        "n_complete_triples": len(triples),
        "n_unexecuted_triples": len(triples) - len(executed),
        "n_generations_unscored": sum(1 for key in generations if key not in scored_code),
    }
    completeness["complete"] = (completeness["n_unexecuted_triples"] == 0
                                and completeness["n_generations_unscored"] == 0)

    pass1 = pass1_by_problem(kept, executions)
    summary = pass1_summary(pass1, PASS1_MARGIN)
    by_source = pass1_by_source(pass1, source_of)
    sensitivity = pass1_summary(pass1_by_problem(executed, executions, unparseable_is_fail=True),
                                PASS1_MARGIN)

    def raw_text(pid, arm, k):
        return generations[(pid, arm, k)]["text"]

    def code_text(pid, arm, k):
        code, _ = extract_code(generations[(pid, arm, k)]["text"],
                               problems[pid].get("entry_point"))
        return code

    exact = {"raw_text": exact_match(triples, raw_text),
             "extracted_code": exact_match(kept, code_text),
             "definition": "fraction of (problem, sample) triples whose two outputs are "
                           "byte-for-byte identical"}

    detection = {}
    for domain in DETECTION_DOMAINS:
        recs = [r for r in detections if r["domain"] == domain]
        if not recs:
            continue
        curve = detection_curve(recs, PREFIX_LENGTHS, FPR_TARGETS)
        entry = {
            "n_records": len(recs),
            "n_by_arm": {arm: sum(1 for r in recs if r["arm"] == arm) for arm in ARMS},
            "n_unscorable": sum(1 for r in recs if r.get("score_full") is None),
            "sources": sorted({r["source_run"] for r in recs}),
            "mean_tokens": float(np.mean([r["n_tokens"] for r in recs])),
            "full": curve_row(curve, "full"),
            "curve": curve,
        }
        rt = [r["roundtrip_ids_equal"] for r in recs if r.get("roundtrip_ids_equal") is not None]
        if rt:
            entry["roundtrip_ids_equal_fraction"] = float(np.mean(rt))
        detection[domain] = entry

    outliers = []
    if summary.get("n_problems"):
        ranked = sorted(summary["per_problem_delta_wm"].items(), key=lambda kv: -abs(kv[1]))
        for pid, d in ranked[:N_OUTLIER_PROBLEMS]:
            outliers.append({"problem_id": pid, "source": source_of.get(pid), "delta_wm": d,
                             "pass1": {arm: pass1[arm][pid] for arm in ARMS}})

    rlimit_as: dict[str, int] = {}
    for rec in executions.values():
        value = (rec.get("limits") or {}).get("RLIMIT_AS")
        if value is not None:
            rlimit_as[value] = rlimit_as.get(value, 0) + 1
    n_ran = sum(rlimit_as.values())
    harness = {"n_executed": len(executions),
               "n_ran": n_ran,
               "rlimit_as": rlimit_as,
               "mem_cap_enforced": (None if n_ran == 0
                                    else set(rlimit_as) == {"set"}),
               "note": "RLIMIT_AS 'refused' means the kernel rejected the cap (macOS); "
                       "'set' means it was in force for that candidate"}

    bound_status = summary["bound"]["status"] if summary.get("n_problems") else "FAIL"
    equivalence = {
        "pass1_delta": {"bound": f"95% CI of mean(pass@1_A - pass@1_B1) within +/- {PASS1_MARGIN}",
                        "status": bound_status,
                        "note": None if summary.get("n_problems") else "no kept problems"},
        "overall": bound_status,
    }

    return {
        "generated_at": utcnow(),
        "experiment": "exp2",
        "out_dir": str(out_dir),
        "bootstrap": {"resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED,
                      "method": "percentile, resampling problems with replacement"},
        "pass1": {"definition": "per problem per arm: fraction of kept samples that pass "
                                "the reference tests; arm means are over problems",
                  **summary, "by_source": by_source,
                  "sensitivity_unparseable_as_fail": sensitivity},
        "detection": {"detector": "mean g-value", "path": DETECTOR_PATH,
                      "tokenization": TOKENIZATION, "primary_fpr": PRIMARY_FPR,
                      "fpr_targets": list(FPR_TARGETS), "prefix_lengths": list(PREFIX_LENGTHS),
                      "null": "pooled unwatermarked scores, B1 + B2, per domain",
                      "threshold_rule": "the (m+1)-th largest null score with "
                                        "m = floor(FPR * n_null): the smallest threshold with "
                                        "at most m null scores strictly above it",
                      "domains": detection},
        "exact_match": exact,
        "completeness": completeness,
        "filter": filter_counts(generations, executions, triples, executed, kept),
        "length": _length_metrics(generations, triples),
        "harness": harness,
        "outliers": outliers,
        "equivalence": equivalence,
        "parity": parity,
    }


def _csv_rows(metrics: dict, run_config: dict) -> list[dict]:
    rows: list[dict] = []

    def add(metric, value, ci_low=None, ci_high=None, n=None, unit="", notes=""):
        rows.append({"metric": metric, "value": value, "ci_low": ci_low, "ci_high": ci_high,
                     "n": n, "unit": unit, "notes": notes})

    def add_stat(metric, stat, unit="", notes=""):
        add(metric, stat["mean"], stat["ci_low"], stat["ci_high"], stat["n"], unit, notes)

    gen = run_config.get("stages", {}).get("generate", {})
    add("model", gen.get("model"), notes="generator model")
    add("model_revision", gen.get("model_revision"))
    add("smoke", gen.get("smoke"), notes="true => wiring test, not a real run")
    add("n_samples_per_arm", gen.get("n_samples"))

    p = metrics["pass1"]
    if p.get("n_problems"):
        for arm in ARMS:
            add_stat(f"pass1_mean_{arm}", p["mean_by_arm"][arm], "", "mean over problems")
        add_stat("pass1_delta_A_minus_B1", p["delta_wm"], "", "primary: watermark, paired")
        add_stat("pass1_delta_B2_minus_B1", p["delta_ctrl"], "", "control: sampling noise")
        add("pass1_bound", p["bound"]["status"], notes=p["bound"]["statement"])
        add("pass1_control_covers_zero", p["control_covers_zero"])
        for source, block in p["by_source"].items():
            for arm in ARMS:
                add_stat(f"pass1_mean_{arm}_{source}", block["mean_by_arm"][arm])
            add_stat(f"pass1_delta_A_minus_B1_{source}", block["delta_wm"])
            add_stat(f"pass1_delta_B2_minus_B1_{source}", block["delta_ctrl"])
    s = p["sensitivity_unparseable_as_fail"]
    if s.get("n_problems"):
        add_stat("pass1_delta_A_minus_B1_unparseable_as_fail", s["delta_wm"], "",
                 "sensitivity: unparseable samples counted as failures instead of dropped")

    for domain, block in metrics["detection"]["domains"].items():
        full = block["full"]
        add(f"detect_{domain}_n_wm", full["n_wm"])
        add(f"detect_{domain}_n_null", full["n_null"])
        for fpr in FPR_TARGETS:
            tag = _fpr_tag(fpr)
            add(f"detect_{domain}_threshold_{tag}", full[f"threshold_{tag}"], n=full["n_null"])
            add(f"detect_{domain}_achieved_{tag}", full[f"achieved_{tag}"], n=full["n_null"])
            add(f"detect_{domain}_tpr_{tag}", full[f"tpr_{tag}"], n=full["n_wm"],
                notes="full length")
        add(f"detect_{domain}_mean_score_wm", full["mean_score_wm"], n=full["n_wm"])
        add(f"detect_{domain}_mean_score_null", full["mean_score_null"], n=full["n_null"])
        add(f"detect_{domain}_auroc", full["auroc"])
        for row in block["curve"]:
            add(f"detect_{domain}_tpr_{_fpr_tag(PRIMARY_FPR)}_at_{row['prefix_length']}",
                row[f"tpr_{_fpr_tag(PRIMARY_FPR)}"], n=row["n_wm"], unit="",
                notes=f"prefix {row['prefix_length']} tokens, n_null={row['n_null']}")

    e = metrics["exact_match"]
    for name in ("raw_text", "extracted_code"):
        add(f"exact_match_A_eq_B1_{name}", e[name]["A_eq_B1"], n=e[name]["n"])
        add(f"exact_match_B1_eq_B2_{name}", e[name]["B1_eq_B2"], n=e[name]["n"],
            notes="baseline")

    c = metrics["completeness"]
    add("run_complete", c["complete"], notes="every triple executed in all arms and every "
                                            "generation scored")
    add("n_unexecuted_triples", c["n_unexecuted_triples"])
    add("n_generations_unscored", c["n_generations_unscored"])

    f = metrics["filter"]
    for key in ("n_samples_generated", "n_complete_triples", "n_incomplete_triples",
                "n_unexecuted_triples", "n_kept_samples", "n_dropped_samples",
                "n_problems_kept"):
        add(f"filter_{key}", f[key])
    for arm, block in f["per_arm"].items():
        add(f"filter_unparseable_{arm}", block["n_unparseable"])
        add(f"filter_length_truncated_{arm}", block["n_length_truncated"])
        for status, count in block["status_counts"].items():
            add(f"exec_{status}_{arm}", count)

    add("harness_mem_cap_enforced", metrics["harness"]["mem_cap_enforced"],
        n=metrics["harness"]["n_executed"], notes=str(metrics["harness"]["rlimit_as"]))

    for name, check in metrics["parity"]["checks"].items():
        add(f"parity_{name}", check["status"], notes=check["detail"])
    add("parity_overall", metrics["parity"]["overall"])
    add("equivalence_overall", metrics["equivalence"]["overall"],
        notes=metrics["equivalence"]["pass1_delta"]["bound"])
    return rows


def _curve_rows(metrics: dict) -> list[dict]:
    rows = []
    for domain, block in metrics["detection"]["domains"].items():
        for row in block["curve"]:
            rows.append({"domain": domain, **row})
    return rows


def write_figure(metrics: dict, path: Path, model: str) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log("note: matplotlib not installed -- no figure written")
        return False

    domains = [d for d in FIGURE_DOMAINS if d in metrics["detection"]["domains"]]
    if not domains:
        log("note: no detection domains to plot")
        return False

    surface, ink, ink2, muted, grid, axis = ("#fcfcfb", "#0b0b0b", "#52514e", "#898781",
                                            "#e1e0d9", "#c3c2b7")
    series = {"prose": "#2a78d6", "code": "#eb6834"}
    lengths = list(PREFIX_LENGTHS) + ["full"]
    x = np.arange(len(lengths))
    tag = _fpr_tag(PRIMARY_FPR)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8), dpi=150)
    fig.patch.set_facecolor(surface)
    n_prefix = len(PREFIX_LENGTHS)
    for ax in (ax1, ax2):
        ax.set_facecolor(surface)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(axis)
            ax.spines[side].set_linewidth(0.8)
        ax.grid(axis="y", color=grid, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(colors=muted, labelsize=9, length=0)
        ax.set_xticks(x)
        ax.set_xticklabels([str(L) for L in lengths])
        ax.set_xlabel("prefix length (tokens); 'full' = every sample at its own length",
                      color=ink2, fontsize=9)
        ax.axvline(n_prefix - 0.5, color=grid, linewidth=0.8)
        ax.set_xlim(-0.3, len(lengths) - 0.5)

    mark = dict(marker="o", markersize=5.5, markeredgecolor=surface, markeredgewidth=1.2)
    hollow = dict(marker="o", markersize=6.5, markerfacecolor=surface, markeredgewidth=1.6,
                  linestyle="none")
    for domain in domains:
        curve = metrics["detection"]["domains"][domain]["curve"]
        by_len = {row["prefix_length"]: row for row in curve}

        def pick(field):
            return [by_len[L][field] if L in by_len else np.nan for L in lengths]

        wm, null, tpr = pick("mean_score_wm"), pick("mean_score_null"), pick(f"tpr_{tag}")
        color = series[domain]
        xp = x[:n_prefix]
        ax1.plot(xp, wm[:n_prefix], color=color, linewidth=1.5, label=f"{domain}: watermarked (A)",
                 **mark)
        ax1.plot(xp, null[:n_prefix], color=color, linewidth=1.0, linestyle=(0, (3, 2)),
                 alpha=0.75, label=f"{domain}: unwatermarked (B1 + B2)")
        ax1.plot([x[-1]], [wm[-1]], color=color, **hollow)
        ax1.plot([x[-1]], [null[-1]], color=color, alpha=0.75, **hollow)
        ax2.plot(xp, tpr[:n_prefix], color=color, linewidth=1.5, label=domain, **mark)
        ax2.plot([x[-1]], [tpr[-1]], color=color, **hollow)
        for ax, ys in ((ax1, wm), (ax2, tpr)):
            last = max((i for i, v in enumerate(ys) if v == v), default=None)
            if last is not None:
                ax.annotate(domain, (x[last], ys[last]), xytext=(7, 0),
                            textcoords="offset points", color=ink2, fontsize=9, va="center")

    ax1.set_title("Mean g-value score", color=ink, fontsize=10.5, loc="left")
    ax1.set_ylabel("mean g-value (0.5 = unwatermarked)", color=ink2, fontsize=9)
    ax2.set_title(f"TPR at {int(round(PRIMARY_FPR * 100))}% FPR (threshold re-set per prefix)",
                  color=ink, fontsize=10.5, loc="left")
    ax2.set_ylabel("true positive rate", color=ink2, fontsize=9)
    ax2.set_ylim(-0.02, 1.02)
    for ax, ncol in ((ax1, 2), (ax2, len(domains))):
        ax.legend(frameon=False, fontsize=8, labelcolor=ink2, loc="upper center",
                  bbox_to_anchor=(0.5, -0.2), ncol=ncol)
    fig.suptitle(f"SynthID mean g-value detection vs prefix length -- {model}", color=ink,
                 fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    fig.savefig(path, facecolor=surface)
    plt.close(fig)
    return True


def _markdown(metrics: dict, run_config: dict) -> str:
    gen = run_config.get("stages", {}).get("generate", {})
    ex = run_config.get("stages", {}).get("execute", {})
    p, d, e, f, q = (metrics["pass1"], metrics["detection"], metrics["exact_match"],
                     metrics["filter"], metrics["equivalence"])
    tag1, tag5 = _fpr_tag(PRIMARY_FPR), _fpr_tag(0.05)

    lines = [
        "# Experiment 2 - SynthID-Text on code: functional correctness and detectability",
        "",
        f"**Generator model:** `{gen.get('model')}`  ",
        f"**Model revision:** `{gen.get('model_revision')}`  ",
        f"**Run directory:** `{metrics['out_dir']}`  ",
        f"**Generated:** {metrics['generated_at']}  ",
    ]
    if gen.get("smoke"):
        lines += ["", f"> **SMOKE RUN** - {gen.get('n_problems_selected')} problems, "
                      f"{gen.get('n_samples')} samples, {gen.get('max_new_tokens')} new tokens. "
                      "Not a valid experimental result.", ""]
    c = metrics["completeness"]
    if not c["complete"]:
        lines += ["", f"> **PARTIAL RUN (written with --force)** - "
                      f"{c['n_unexecuted_triples']} complete sample triples have no execution "
                      f"outcome in some arm and {c['n_generations_unscored']} generations have no "
                      "detection score. Every number below is computed on the subset that "
                      "finished; the bound status is not valid.", ""]

    lines += [
        "",
        "## Result",
        "",
        f"**Overall: {q['overall']}**   (arm parity: {metrics['parity']['overall']})",
        "",
        "| Pre-registered bound | Result | Status |",
        "|---|---|---|",
        f"| {q['pass1_delta']['bound']} | "
        f"{_fmt_ci(p['delta_wm']) if p.get('n_problems') else 'n/a'} | "
        f"{q['pass1_delta']['status']} |",
        "",
        "## Arms",
        "",
        f"{gen.get('n_samples')} sample{'s' if gen.get('n_samples') != 1 else ''} per problem "
        f"per arm, identical sampling everywhere.",
        "",
        "| Arm | Watermark | Sampling seeds | Role |",
        "|---|---|---|---|",
        f"| A | SynthID (non-distortionary) | {gen.get('seeds', {}).get('arms', {}).get('A', {}).get('seed_indices')} | treatment |",
        f"| B1 | none | {gen.get('seeds', {}).get('arms', {}).get('B1', {}).get('seed_indices')} | paired control, same randomness stream as A |",
        f"| B2 | none | {gen.get('seeds', {}).get('arms', {}).get('B2', {}).get('seed_indices')} | sampling-noise control |",
        "",
        "## Functional correctness (pass@1)",
        "",
    ]
    if p.get("n_problems"):
        lines += [
            f"pass@1 per problem per arm is the fraction of its kept samples that pass the "
            f"reference tests; n = {p['n_problems']} problems, sampled (not greedy) so the "
            f"absolute level sits below greedy-decoding pass@1 in every arm alike.",
            "",
            "| Arm | Mean pass@1 | 95% CI |",
            "|---|---|---|",
        ]
        for arm in ARMS:
            st = p["mean_by_arm"][arm]
            lines.append(f"| {arm} | {_fmt(st['mean'], digits=4)} | "
                         f"[{_fmt(st['ci_low'], digits=4)}, {_fmt(st['ci_high'], digits=4)}] |")
        lines += [
            "",
            "| Paired difference | Mean | 95% CI | Check |",
            "|---|---|---|---|",
            f"| A - B1 (watermark) | {_fmt(p['delta_wm']['mean'], digits=4)} | "
            f"[{_fmt(p['delta_wm']['ci_low'], digits=4)}, {_fmt(p['delta_wm']['ci_high'], digits=4)}] | "
            f"within +/- {PASS1_MARGIN}: **{p['bound']['status']}** |",
            f"| B2 - B1 (sampling noise) | {_fmt(p['delta_ctrl']['mean'], digits=4)} | "
            f"[{_fmt(p['delta_ctrl']['ci_low'], digits=4)}, {_fmt(p['delta_ctrl']['ci_high'], digits=4)}] | "
            f"noise floor, CI {'covers' if p['control_covers_zero'] else 'MISSES'} 0 |",
            "",
        ]
        if p["by_source"]:
            lines += ["| Source | n | A | B1 | B2 | A - B1 [95% CI] | B2 - B1 [95% CI] |",
                      "|---|---|---|---|---|---|---|"]
            for source, block in p["by_source"].items():
                m = block["mean_by_arm"]
                lines.append(
                    f"| {source} | {block['n_problems']} | {_fmt(m['A']['mean'])} | "
                    f"{_fmt(m['B1']['mean'])} | {_fmt(m['B2']['mean'])} | "
                    f"{_fmt_ci(block['delta_wm'], digits=4)} | "
                    f"{_fmt_ci(block['delta_ctrl'], digits=4)} |")
            lines.append("")
        s = p["sensitivity_unparseable_as_fail"]
        if s.get("n_problems"):
            lines += [
                f"Sensitivity, unparseable samples counted as failures instead of dropped "
                f"(n = {s['n_problems']} problems): A - B1 = {_fmt_ci(s['delta_wm'], digits=4)}, "
                f"B2 - B1 = {_fmt_ci(s['delta_ctrl'], digits=4)}.",
                "",
            ]
    else:
        lines += ["_No kept problems: every sample triple was dropped or nothing was executed._",
                  ""]

    lines += [
        "## Detection",
        "",
        f"Detector: **mean g-value** with the saved keys. {d['path']}. Tokenisation: "
        f"{d['tokenization']}. Thresholds are set per domain at empirical FPR on the pooled "
        f"unwatermarked scores (B1 + B2); the rule is: {d['threshold_rule']}. Every code "
        f"sample of every arm is scored; prose is every Exp 1 generation of this model.",
        "",
    ]
    if d["domains"]:
        lines += [
            "| Domain | n wm | n null | Threshold @1% | Achieved FPR | **TPR @1% FPR** | "
            "TPR @5% FPR | Mean score wm / null | Mean tokens |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for domain, block in d["domains"].items():
            full = block["full"]
            lines.append(
                f"| {DOMAIN_LABEL.get(domain, domain)} | {full['n_wm']} | {full['n_null']} | "
                f"{_fmt(full[f'threshold_{tag1}'], digits=4)} | "
                f"{_fmt_pct(full[f'achieved_{tag1}'], 2)} | **{_fmt_pct(full[f'tpr_{tag1}'])}** | "
                f"{_fmt_pct(full[f'tpr_{tag5}'])} | {_fmt(full['mean_score_wm'], digits=4)} / "
                f"{_fmt(full['mean_score_null'], digits=4)} | {_fmt(block['mean_tokens'], digits=0)} |")
        lines.append("")
        if "prose" not in d["domains"]:
            lines += ["_No Exp 1 run for this model was found under the `--exp1-runs` glob, so "
                      "the prose domain is absent._", ""]
        for domain, block in d["domains"].items():
            if "roundtrip_ids_equal_fraction" in block:
                lines += [f"Tokenisation round trip ({DOMAIN_LABEL.get(domain, domain)}): "
                          f"{_fmt_pct(block['roundtrip_ids_equal_fraction'])} of samples "
                          f"re-tokenise to exactly the generated ids.", ""]
        lines += [
            "### Length curves",
            "",
            f"Scores rescored on token prefixes; a sample contributes to a prefix only if it "
            f"has at least that many tokens. Thresholds are re-set per prefix at "
            f"{int(round(PRIMARY_FPR * 100))}% FPR. Full data: `{CURVES_FILE}`; figure: "
            f"`{FIGURE_FILE}`.",
            "",
            "| Prefix | " + " | ".join(f"{DOMAIN_LABEL.get(dm, dm)}: TPR@1% (n wm / n null)"
                                       for dm in d["domains"]) + " |",
            "|---" * (len(d["domains"]) + 1) + "|",
        ]
        for L in list(PREFIX_LENGTHS) + ["full"]:
            cells = []
            for dm, block in d["domains"].items():
                row = curve_row(block["curve"], L)
                cells.append(f"{_fmt_pct(row[f'tpr_{tag1}'])} ({row['n_wm']} / {row['n_null']})"
                             if row else "n/a")
            lines.append(f"| {L} | " + " | ".join(cells) + " |")
        lines.append("")
    else:
        lines += ["_No detection scores._", ""]

    lines += [
        "## Exact-match rates",
        "",
        f"{e['definition'].capitalize()}. A == B1 measures how often the watermark left the "
        "sample untouched (same seed, same randomness stream); B1 == B2 is the baseline for "
        "two independent samples.",
        "",
        "| Text compared | Items | A == B1 | B1 == B2 (baseline) |",
        "|---|---|---|---|",
        f"| raw response | {e['raw_text']['n']} | {_fmt_pct(e['raw_text']['A_eq_B1'])} "
        f"({e['raw_text']['A_eq_B1_count']}) | {_fmt_pct(e['raw_text']['B1_eq_B2'])} "
        f"({e['raw_text']['B1_eq_B2_count']}) |",
        f"| extracted code | {e['extracted_code']['n']} | "
        f"{_fmt_pct(e['extracted_code']['A_eq_B1'])} ({e['extracted_code']['A_eq_B1_count']}) | "
        f"{_fmt_pct(e['extracted_code']['B1_eq_B2'])} ({e['extracted_code']['B1_eq_B2_count']}) |",
        "",
        "## Filtering and execution outcomes",
        "",
        f"{f['rule'].capitalize()}. Extraction rules ({EXTRACTION_RULES_VERSION}): {RULES}",
        "",
        "| Quantity | Count |",
        "|---|---|",
        f"| Samples generated | {f['n_samples_generated']} |",
        f"| Complete (A, B1, B2) sample triples | {f['n_complete_triples']} |",
        f"| Incomplete triples (skipped) | {f['n_incomplete_triples']} |",
        f"| Triples without an execution outcome in every arm | {f['n_unexecuted_triples']} |",
        f"| Triples dropped (unparseable in some arm) | {f['n_dropped_samples']} |",
        f"| Triples kept | {f['n_kept_samples']} |",
        f"| Problems with >= 1 kept triple | {f['n_problems_kept']} |",
        "",
        "| Arm | Generated | Unparseable | Length-truncated | Unclosed fence | Passed | Failed | "
        "Syntax error | Timeout | Harness error | Other |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for arm, block in f["per_arm"].items():
        sc = block["status_counts"]
        other = sum(v for k, v in sc.items()
                    if k not in ("passed", "failed", "syntax_error", "timeout", "unparseable",
                                 "harness_error"))
        lines.append(f"| {arm} | {block['n_generated']} | {block['n_unparseable']} | "
                     f"{block['n_length_truncated']} | {block['n_unclosed_fence']} | "
                     f"{sc.get('passed', 0)} | {sc.get('failed', 0)} | "
                     f"{sc.get('syntax_error', 0)} | {sc.get('timeout', 0)} | "
                     f"{block['n_harness_error']} | {other} |")
    lines.append("")

    if metrics["outliers"]:
        lines += ["Largest per-problem |A - B1|:", "",
                  "| Problem | Source | pass@1 A | B1 | B2 | A - B1 |", "|---|---|---|---|---|---|"]
        for o in metrics["outliers"]:
            lines.append(f"| `{o['problem_id']}` | {o['source']} | {_fmt(o['pass1']['A'], digits=2)} | "
                         f"{_fmt(o['pass1']['B1'], digits=2)} | {_fmt(o['pass1']['B2'], digits=2)} | "
                         f"{_fmt(o['delta_wm'], digits=2)} |")
        lines.append("")

    lines += _parity_markdown(metrics["parity"])

    h = ex.get("harness", {})
    lines += [
        "## Execution harness",
        "",
        f"- Each candidate runs in a fresh subprocess ({h.get('isolation')}).",
        f"- Timeout {ex.get('timeout_s')} s: {h.get('timeout')}.",
        f"- Memory: {h.get('limits')}; requested cap {ex.get('mem_mb')} MB, file-size cap "
        f"{ex.get('fsize_mb')} MB. **Address-space cap "
        f"{ {True: 'enforced on every candidate', False: 'NOT enforced', None: 'n/a, no candidate ran'}[metrics['harness']['mem_cap_enforced']] }** "
        f"(RLIMIT_AS outcomes over the {metrics['harness']['n_ran']} candidates that ran: "
        f"{metrics['harness']['rlimit_as']}).",
        f"- Network: {h.get('network')}"
        + (f"; exec prefix `{ex.get('exec_prefix')}`." if ex.get("exec_prefix") else "."),
        f"- Python `{h.get('python_version')}` at `{h.get('python')}`, {ex.get('workers')} "
        f"concurrent sandboxes.",
        f"- Program assembly: {ex.get('program_assembly')}.",
        "",
    ]

    sampling = gen.get("sampling", {})
    lines += [
        "## Reproducibility",
        "",
        f"- Sampling (all arms): `do_sample={sampling.get('do_sample')}, "
        f"temperature={sampling.get('temperature')}, top_k={sampling.get('top_k')}, "
        f"top_p={sampling.get('top_p')}, max_new_tokens={sampling.get('max_new_tokens')}`",
        f"- Seeds: {gen.get('seeds', {}).get('base_seed')}; per sample "
        f"{gen.get('seeds', {}).get('per_sample_derivation')}; B2 offset "
        f"{gen.get('seeds', {}).get('b2_offset')}",
        f"- Problems: `{gen.get('problems_path')}` (sha256 `{str(gen.get('problems_sha256'))[:16]}...`, "
        f"{gen.get('n_problems_selected')} selected)",
        f"- Watermark keys: `{gen.get('watermark_config_path')}` "
        f"(sha256 `{str(gen.get('watermark_config_sha256'))[:16]}...`)",
        f"- Bootstrap: {metrics['bootstrap']['resamples']} resamples, seed "
        f"{metrics['bootstrap']['seed']}, {metrics['bootstrap']['method']}",
        "",
        "Full parameters, library versions and model revisions: `run_config.json`.",
        "",
    ]
    return "\n".join(lines)


def run_report(args) -> None:
    out_dir = Path(args.out)
    art = load_artefacts(out_dir, args.problems)
    run_config = art["run_config"]

    triples = complete_triples(art["generations"])
    parity = check_exp2_parity(run_config, art["generations"], triples, art["executions"],
                               art["detections"])
    print_parity(parity)
    if parity["overall"] == "FAIL":
        raise SystemExit("arm parity failed -- no report written; fix the listed differences "
                         "and run report again")

    metrics = compute_metrics(out_dir, art, parity)
    c = metrics["completeness"]
    if not c["complete"]:
        msg = (f"run is incomplete: {c['n_unexecuted_triples']} complete sample triples lack an "
               f"execution outcome in some arm, {c['n_generations_unscored']} generations have "
               "no detection score. Run `execute` / `detect` to completion")
        if not getattr(args, "force", False):
            raise SystemExit(msg + ", or pass --force to write a report marked PARTIAL.")
        log("WARNING: " + msg + " -- writing a PARTIAL report because of --force")
    model = run_config.get("stages", {}).get("generate", {}).get("model", "")
    figure_written = write_figure(metrics, out_dir / FIGURE_FILE, model)
    metrics["figure"] = FIGURE_FILE if figure_written else None
    write_json_atomic(out_dir / "metrics.json", metrics)
    write_json_atomic(out_dir / "filter_report.json", metrics["filter"])

    (out_dir / "results.md").write_text(_markdown(metrics, run_config), encoding="utf-8")
    with open(out_dir / "results.csv", "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["metric", "value", "ci_low", "ci_high", "n",
                                                "unit", "notes"])
        writer.writeheader()
        writer.writerows(_csv_rows(metrics, run_config))
    curve_rows = _curve_rows(metrics)
    if curve_rows:
        with open(out_dir / CURVES_FILE, "w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(curve_rows[0].keys()))
            writer.writeheader()
            writer.writerows(curve_rows)

    update_run_config(out_dir, "report", {
        "generated_at": metrics["generated_at"],
        "bootstrap": metrics["bootstrap"],
        "equivalence": metrics["equivalence"],
        "parity": {"overall": parity["overall"],
                   "checks": {k: v["status"] for k, v in parity["checks"].items()}},
        "outputs": ["results.md", "results.csv", "metrics.json", "filter_report.json",
                    CURVES_FILE] + ([FIGURE_FILE] if figure_written else []),
    })

    p, d, e = metrics["pass1"], metrics["detection"], metrics["exact_match"]
    tag1 = _fpr_tag(PRIMARY_FPR)
    print()
    if not c["complete"]:
        print("  PARTIAL RUN -- numbers below are on the finished subset only")
    if p.get("n_problems"):
        print(f"  pass@1  A {p['mean_by_arm']['A']['mean']:.4f}  B1 {p['mean_by_arm']['B1']['mean']:.4f}"
              f"  B2 {p['mean_by_arm']['B2']['mean']:.4f}   (n = {p['n_problems']} problems)")
        print(f"  A - B1:  {_fmt_ci(p['delta_wm'], digits=4)}   -> {p['bound']['status']}  "
              f"({p['bound']['statement']})")
        print(f"  B2 - B1: {_fmt_ci(p['delta_ctrl'], digits=4)}   -> noise floor, CI "
              f"{'covers' if p['control_covers_zero'] else 'MISSES'} 0")
    else:
        print("  pass@1: no kept problems")
    for domain, block in d["domains"].items():
        full = block["full"]
        print(f"  detect {domain:15s} TPR@1%FPR {_fmt_pct(full[f'tpr_{tag1}'])}  "
              f"(thr {full[f'threshold_{tag1}']:.4f}, n wm {full['n_wm']}, n null {full['n_null']})")
    print(f"  exact match raw text: A==B1 {_fmt_pct(e['raw_text']['A_eq_B1'])}, "
          f"B1==B2 {_fmt_pct(e['raw_text']['B1_eq_B2'])}")
    print(f"  filtered: {metrics['filter']['n_kept_samples']} triples kept, "
          f"{metrics['filter']['n_dropped_samples']} dropped")
    print()
    print(f"  OVERALL: {metrics['equivalence']['overall']}   PARITY: {parity['overall']}")
    print()
    log(f"wrote {out_dir / 'results.md'}, {out_dir / 'results.csv'}, {out_dir / 'metrics.json'}"
        + (f", {out_dir / FIGURE_FILE}" if figure_written else ""))
