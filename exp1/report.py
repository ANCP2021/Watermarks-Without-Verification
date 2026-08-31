from __future__ import annotations

import csv
from pathlib import Path

from .common import (load_run_config, log, read_jsonl, update_run_config,
                     utcnow, write_json_atomic)
from .filters import load_valid_generations
from .settings import (BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED, DELTA_PPL_MARGIN_PCT,
                       WIN_RATE_CENTER, WIN_RATE_MARGIN)
from .stats import bootstrap_mean_ci, ci_within

WIN_SCORE = {"A": 1.0, "tie": 0.5, "B1": 0.0}


def _fmt(value: float, unit: str = "", digits: int = 3) -> str:
    if value != value:  # NaN
        return "n/a"
    if unit == "%":
        return f"{value:+.2f}%"
    return f"{value:.{digits}f}"


def _fmt_ci(stat: dict, unit: str = "", digits: int = 3) -> str:
    if not stat or stat["n"] == 0:
        return "n/a"
    return (f"{_fmt(stat['mean'], unit, digits)} "
            f"[{_fmt(stat['ci_low'], unit, digits)}, {_fmt(stat['ci_high'], unit, digits)}]")


def compute_metrics(out_dir: Path) -> dict:
    kept, filter_report = load_valid_generations(out_dir, write_report=True)
    analysis_ids = set(kept)

    metrics: dict = {
        "generated_at": utcnow(),
        "out_dir": str(out_dir),
        "filter": filter_report,
        "bootstrap": {"resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED,
                      "method": "percentile, resampling prompts with replacement"},
    }

    ppl_records = [r for r in read_jsonl(out_dir / "ppl.jsonl")
                   if r["prompt_id"] in analysis_ids and r.get("scorable", True)]
    if ppl_records:
        delta_pct = [100.0 * (r["ppl_A"] - r["ppl_B1"]) / r["ppl_B1"] for r in ppl_records]
        delta_pct_control = [100.0 * (r["ppl_B2"] - r["ppl_B1"]) / r["ppl_B1"]
                             for r in ppl_records]
        metrics["ppl"] = {
            "n": len(ppl_records),
            "scorer_model": ppl_records[0].get("scorer_model"),
            "delta_pct_A_vs_B1": bootstrap_mean_ci(delta_pct),
            "delta_pct_B2_vs_B1_control": bootstrap_mean_ci(delta_pct_control),
            "mean_ppl_A": bootstrap_mean_ci([r["ppl_A"] for r in ppl_records]),
            "mean_ppl_B1": bootstrap_mean_ci([r["ppl_B1"] for r in ppl_records]),
            "mean_ppl_B2": bootstrap_mean_ci([r["ppl_B2"] for r in ppl_records]),
            "mean_nll_A": sum(r["nll_A"] for r in ppl_records) / len(ppl_records),
            "mean_nll_B1": sum(r["nll_B1"] for r in ppl_records) / len(ppl_records),
        }
    else:
        metrics["ppl"] = None

    judge_records = [r for r in read_jsonl(out_dir / "judge.jsonl")
                     if r["prompt_id"] in analysis_ids]
    if judge_records:
        scores = [WIN_SCORE[r["winner"]] for r in judge_records]
        n_unparsed = sum(1 for r in judge_records
                         if "unparsed" in (r["verdict_order1"], r["verdict_order2"]))
        n_disagree = sum(1 for r in judge_records
                         if r["winner"] == "tie"
                         and not (r["verdict_order1"] == "tie" and r["verdict_order2"] == "tie"))
        metrics["judge"] = {
            "n": len(judge_records),
            "judge_model": judge_records[0].get("judge_model"),
            "win_rate_A": bootstrap_mean_ci(scores),
            "counts": {
                "A_wins": sum(1 for r in judge_records if r["winner"] == "A"),
                "B1_wins": sum(1 for r in judge_records if r["winner"] == "B1"),
                "ties": sum(1 for r in judge_records if r["winner"] == "tie"),
                "ties_from_order_disagreement": n_disagree,
                "pairs_with_an_unparsed_verdict": n_unparsed,
            },
            "win_rate_definition": "(A wins + 0.5 * ties) / n",
        }
    else:
        metrics["judge"] = None

    sim_records = [r for r in read_jsonl(out_dir / "sim.jsonl")
                   if r["prompt_id"] in analysis_ids]
    if sim_records:
        a_b1 = [r["sim_a_b1"] for r in sim_records]
        b1_b2 = [r["sim_b1_b2"] for r in sim_records]
        metrics["sim"] = {
            "n": len(sim_records),
            "embedding_model": sim_records[0].get("embedding_model"),
            "mean_sim_A_B1": bootstrap_mean_ci(a_b1),
            "mean_sim_B1_B2_control": bootstrap_mean_ci(b1_b2),
            "paired_delta": bootstrap_mean_ci([x - y for x, y in zip(a_b1, b1_b2)]),
        }
    else:
        metrics["sim"] = None

    win_lo, win_hi = WIN_RATE_CENTER - WIN_RATE_MARGIN, WIN_RATE_CENTER + WIN_RATE_MARGIN
    checks = {
        "win_rate": {
            "bound": f"95% CI within {WIN_RATE_CENTER} +/- {WIN_RATE_MARGIN} "
                     f"([{win_lo}, {win_hi}])",
            "status": ("PASS" if ci_within(metrics["judge"]["win_rate_A"], win_lo, win_hi)
                       else "FAIL") if metrics["judge"] else "INCOMPLETE",
        },
        "delta_ppl_pct": {
            "bound": f"95% CI within +/- {DELTA_PPL_MARGIN_PCT}%",
            "status": ("PASS" if ci_within(metrics["ppl"]["delta_pct_A_vs_B1"],
                                           -DELTA_PPL_MARGIN_PCT, DELTA_PPL_MARGIN_PCT)
                       else "FAIL") if metrics["ppl"] else "INCOMPLETE",
        },
    }
    statuses = [c["status"] for c in checks.values()]
    checks["overall"] = ("PASS" if all(s == "PASS" for s in statuses)
                         else "INCOMPLETE" if "INCOMPLETE" in statuses else "FAIL")
    metrics["equivalence"] = checks
    return metrics


def _csv_rows(metrics: dict, run_config: dict) -> list[dict]:
    rows: list[dict] = []

    def add(metric, value, ci_low=None, ci_high=None, n=None, unit="", notes=""):
        rows.append({"metric": metric, "value": value, "ci_low": ci_low,
                     "ci_high": ci_high, "n": n, "unit": unit, "notes": notes})

    def add_stat(metric, stat, unit="", notes=""):
        add(metric, stat["mean"], stat["ci_low"], stat["ci_high"], stat["n"], unit, notes)

    gen = run_config.get("stages", {}).get("generate", {})
    add("model", gen.get("model"), notes="generator model")
    add("model_revision", gen.get("model_revision"), notes="resolved commit sha")
    add("smoke", gen.get("smoke"), notes="true => wiring test, not a real run")

    p = metrics["ppl"]
    if p:
        add_stat("delta_ppl_pct_A_vs_B1", p["delta_pct_A_vs_B1"], "%",
                 "primary: mean per-prompt 100*(PPL_A-PPL_B1)/PPL_B1")
        add_stat("delta_ppl_pct_B2_vs_B1_control", p["delta_pct_B2_vs_B1_control"], "%",
                 "sampling-noise reference")
        add_stat("mean_ppl_A", p["mean_ppl_A"], "", "")
        add_stat("mean_ppl_B1", p["mean_ppl_B1"], "", "")
        add_stat("mean_ppl_B2", p["mean_ppl_B2"], "", "")
        add("ppl_scorer_model", p["scorer_model"])

    j = metrics["judge"]
    if j:
        add_stat("judge_win_rate_A", j["win_rate_A"], "", j["win_rate_definition"])
        for key, val in j["counts"].items():
            add(f"judge_{key}", val, n=j["n"])
        add("judge_model", j["judge_model"])

    s = metrics["sim"]
    if s:
        add_stat("mean_sim_A_B1", s["mean_sim_A_B1"], "", "treatment")
        add_stat("mean_sim_B1_B2", s["mean_sim_B1_B2_control"], "", "control (sampling noise)")
        add_stat("sim_paired_delta_A_B1_minus_B1_B2", s["paired_delta"], "",
                 "negative => watermark moves text further than reseeding does")
        add("sim_embedding_model", s["embedding_model"])

    f = metrics["filter"]
    add("filter_n_prompts_with_generations", f["n_prompts_with_generations"])
    add("filter_n_kept", f["n_kept"])
    add("filter_n_dropped", f["n_dropped"], notes="dropped symmetrically (whole triple)")
    add("filter_n_incomplete_triples", f["n_incomplete_triples"])
    for arm, reasons in f["dropped_by_arm_and_reason"].items():
        for reason, count in reasons.items():
            add(f"filter_{reason}_in_{arm}", count)

    e = metrics["equivalence"]
    add("equivalence_win_rate", e["win_rate"]["status"], notes=e["win_rate"]["bound"])
    add("equivalence_delta_ppl", e["delta_ppl_pct"]["status"], notes=e["delta_ppl_pct"]["bound"])
    add("equivalence_overall", e["overall"])
    return rows


def _markdown(metrics: dict, run_config: dict) -> str:
    gen = run_config.get("stages", {}).get("generate", {})
    p, j, s, f, e = (metrics["ppl"], metrics["judge"], metrics["sim"],
                     metrics["filter"], metrics["equivalence"])

    win_lo = WIN_RATE_CENTER - WIN_RATE_MARGIN
    win_hi = WIN_RATE_CENTER + WIN_RATE_MARGIN

    lines = [
        "# Experiment 1 - SynthID-Text and prose quality",
        "",
        f"**Generator model:** `{gen.get('model')}`  ",
        f"**Model revision:** `{gen.get('model_revision')}`  ",
        f"**Run directory:** `{metrics['out_dir']}`  ",
        f"**Generated:** {metrics['generated_at']}  ",
    ]
    if gen.get("smoke"):
        lines += ["", "> **SMOKE RUN** - reduced prompt count and/or smaller scorer "
                      "models. Not a valid experimental result.", ""]

    lines += [
        "",
        "## Result",
        "",
        f"**Overall: {e['overall']}**",
        "",
        "| Pre-registered bound | Result | Status |",
        "|---|---|---|",
        f"| Judge win rate 95% CI within [{win_lo}, {win_hi}] | "
        f"{_fmt_ci(j['win_rate_A']) if j else 'n/a'} | {e['win_rate']['status']} |",
        f"| Delta PPL 95% CI within +/- {DELTA_PPL_MARGIN_PCT}% | "
        f"{_fmt_ci(p['delta_pct_A_vs_B1'], '%') if p else 'n/a'} | "
        f"{e['delta_ppl_pct']['status']} |",
        "",
        "## Arms",
        "",
        "| Arm | Watermark | Sampling seed |",
        "|---|---|---|",
        "| A | SynthID (non-distortionary) | S1 |",
        "| B1 | none | S1 |",
        "| B2 | none | S2 (sampling-noise control) |",
        "",
        "## Perplexity",
        "",
    ]
    if p:
        lines += [
            f"Scorer: `{p['scorer_model']}`, mean per-token NLL of each output, "
            f"n = {p['n']} prompts.",
            "",
            "| Quantity | Mean | 95% CI |",
            "|---|---|---|",
            f"| Delta PPL, A vs B1 | {_fmt(p['delta_pct_A_vs_B1']['mean'], '%')} | "
            f"[{_fmt(p['delta_pct_A_vs_B1']['ci_low'], '%')}, "
            f"{_fmt(p['delta_pct_A_vs_B1']['ci_high'], '%')}] |",
            f"| Delta PPL, B2 vs B1 (control) | "
            f"{_fmt(p['delta_pct_B2_vs_B1_control']['mean'], '%')} | "
            f"[{_fmt(p['delta_pct_B2_vs_B1_control']['ci_low'], '%')}, "
            f"{_fmt(p['delta_pct_B2_vs_B1_control']['ci_high'], '%')}] |",
            f"| PPL, arm A | {_fmt(p['mean_ppl_A']['mean'])} | "
            f"[{_fmt(p['mean_ppl_A']['ci_low'])}, {_fmt(p['mean_ppl_A']['ci_high'])}] |",
            f"| PPL, arm B1 | {_fmt(p['mean_ppl_B1']['mean'])} | "
            f"[{_fmt(p['mean_ppl_B1']['ci_low'])}, {_fmt(p['mean_ppl_B1']['ci_high'])}] |",
            f"| PPL, arm B2 | {_fmt(p['mean_ppl_B2']['mean'])} | "
            f"[{_fmt(p['mean_ppl_B2']['ci_low'])}, {_fmt(p['mean_ppl_B2']['ci_high'])}] |",
            "",
        ]
    else:
        lines += ["_Not scored yet (`ppl` stage has produced no records)._", ""]

    lines += ["## Blinded pairwise preference (A vs B1)", ""]
    if j:
        c = j["counts"]
        lines += [
            f"Judge: `{j['judge_model']}`, temperature 0, every pair judged twice "
            f"with positions swapped; a win counts only when both orders agree. "
            f"n = {j['n']} pairs ({2 * j['n']} judge calls).",
            "",
            f"**Win rate for A: {_fmt_ci(j['win_rate_A'])}** "
            f"({j['win_rate_definition']})",
            "",
            "| Outcome | Count |",
            "|---|---|",
            f"| A preferred (both orders) | {c['A_wins']} |",
            f"| B1 preferred (both orders) | {c['B1_wins']} |",
            f"| Tie | {c['ties']} |",
            f"| ...of which from order disagreement | {c['ties_from_order_disagreement']} |",
            f"| Pairs with an unparsed verdict | {c['pairs_with_an_unparsed_verdict']} |",
            "",
        ]
    else:
        lines += ["_Not scored yet (`judge` stage has produced no records)._", ""]

    lines += ["## Semantic similarity", ""]
    if s:
        lines += [
            f"Embeddings: `{s['embedding_model']}` (CLS pooling, L2-normalised), "
            f"n = {s['n']} prompts.",
            "",
            "| Pair | Mean cosine | 95% CI |",
            "|---|---|---|",
            f"| sim(A, B1) - watermark effect | {_fmt(s['mean_sim_A_B1']['mean'], digits=4)} | "
            f"[{_fmt(s['mean_sim_A_B1']['ci_low'], digits=4)}, "
            f"{_fmt(s['mean_sim_A_B1']['ci_high'], digits=4)}] |",
            f"| sim(B1, B2) - sampling noise | "
            f"{_fmt(s['mean_sim_B1_B2_control']['mean'], digits=4)} | "
            f"[{_fmt(s['mean_sim_B1_B2_control']['ci_low'], digits=4)}, "
            f"{_fmt(s['mean_sim_B1_B2_control']['ci_high'], digits=4)}] |",
            f"| paired difference | {_fmt(s['paired_delta']['mean'], digits=4)} | "
            f"[{_fmt(s['paired_delta']['ci_low'], digits=4)}, "
            f"{_fmt(s['paired_delta']['ci_high'], digits=4)}] |",
            "",
            "sim(A, B1) is only interpretable against sim(B1, B2): two "
            "unwatermarked samples with different seeds are not identical either, "
            "so the control is the floor this comparison is measured from.",
            "",
        ]
    else:
        lines += ["_Not scored yet (`sim` stage has produced no records)._", ""]

    lines += [
        "## Filtering",
        "",
        "A prompt is dropped from every analysis if any one of its three outputs "
        "is empty or a refusal, so all arms are always scored on the same prompts.",
        "",
        "| Quantity | Count |",
        "|---|---|",
        f"| Prompts with generations | {f['n_prompts_with_generations']} |",
        f"| Incomplete triples (skipped) | {f['n_incomplete_triples']} |",
        f"| Dropped by filter | {f['n_dropped']} |",
        f"| Kept for analysis | {f['n_kept']} |",
        "",
        "| Arm | Empty | Refusal |",
        "|---|---|---|",
    ]
    for arm, reasons in f["dropped_by_arm_and_reason"].items():
        lines.append(f"| {arm} | {reasons['empty']} | {reasons['refusal']} |")

    sampling = gen.get("sampling", {})
    lines += [
        "",
        "## Reproducibility",
        "",
        f"- Sampling (all arms): `do_sample={sampling.get('do_sample')}, "
        f"temperature={sampling.get('temperature')}, top_k={sampling.get('top_k')}, "
        f"top_p={sampling.get('top_p')}, max_new_tokens={sampling.get('max_new_tokens')}`",
        f"- Seeds: S1 = {gen.get('seeds', {}).get('S1')}, "
        f"S2 = {gen.get('seeds', {}).get('S2')}, derived per prompt",
        f"- Prompts: `{gen.get('prompts_path')}` "
        f"(sha256 `{str(gen.get('prompts_sha256'))[:16]}...`, "
        f"{gen.get('n_prompts_selected')} selected)",
        f"- Watermark keys: `{gen.get('watermark_config_path')}` "
        f"(sha256 `{str(gen.get('watermark_config_sha256'))[:16]}...`)",
        f"- Bootstrap: {metrics['bootstrap']['resamples']} resamples, "
        f"seed {metrics['bootstrap']['seed']}, {metrics['bootstrap']['method']}",
        "",
        "Full parameters, library versions and model revisions: `run_config.json`.",
        "",
    ]
    return "\n".join(lines)


def run_report(args) -> None:
    out_dir = Path(args.out)
    run_config = load_run_config(out_dir)
    metrics = compute_metrics(out_dir)

    write_json_atomic(out_dir / "metrics.json", metrics)

    md = _markdown(metrics, run_config)
    (out_dir / "results.md").write_text(md, encoding="utf-8")

    rows = _csv_rows(metrics, run_config)
    with open(out_dir / "results.csv", "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["metric", "value", "ci_low",
                                                "ci_high", "n", "unit", "notes"])
        writer.writeheader()
        writer.writerows(rows)

    update_run_config(out_dir, "report", {
        "generated_at": metrics["generated_at"],
        "bootstrap": metrics["bootstrap"],
        "equivalence": metrics["equivalence"],
        "outputs": ["results.md", "results.csv", "metrics.json", "filter_report.json"],
    })

    e = metrics["equivalence"]
    print()
    print(f"  delta PPL (A vs B1): "
          f"{_fmt_ci(metrics['ppl']['delta_pct_A_vs_B1'], '%') if metrics['ppl'] else 'n/a'}"
          f"   -> {e['delta_ppl_pct']['status']}  ({e['delta_ppl_pct']['bound']})")
    print(f"  judge win rate (A):  "
          f"{_fmt_ci(metrics['judge']['win_rate_A']) if metrics['judge'] else 'n/a'}"
          f"   -> {e['win_rate']['status']}  ({e['win_rate']['bound']})")
    if metrics["sim"]:
        print(f"  sim(A,B1) = {metrics['sim']['mean_sim_A_B1']['mean']:.4f}   vs   "
              f"sim(B1,B2) = {metrics['sim']['mean_sim_B1_B2_control']['mean']:.4f}  (control)")
    print(f"  filtered: {metrics['filter']['n_kept']} kept, "
          f"{metrics['filter']['n_dropped']} dropped")
    print()
    print(f"  OVERALL: {e['overall']}")
    print()
    log(f"wrote {out_dir / 'results.md'}, {out_dir / 'results.csv'}, "
        f"{out_dir / 'metrics.json'}")
