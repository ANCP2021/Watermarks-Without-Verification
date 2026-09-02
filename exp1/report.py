from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

from .common import (ARMS, load_run_config, log, read_jsonl, update_run_config,
                     utcnow, write_json_atomic)
from .filters import load_valid_generations
from .parity import check_run_parity, print_parity
from .settings import (BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED, DELTA_NLL_MARGIN,
                       DELTA_NLL_MARGIN_PCT, N_OUTLIERS, OUTLIER_PREVIEW_CHARS,
                       TRIM_PROPORTION, WIN_RATE_CENTER, WIN_RATE_MARGIN)
from .stats import (bootstrap_ci, bootstrap_mean_ci, ci_covers, ci_within,
                    log_delta_to_pct)

WIN_SCORE = {"A": 1.0, "tie": 0.5, "B1": 0.0}
DELTAS = {"d_wm": ("A", "B1"), "d_ctrl": ("B2", "B1")}


def _fmt(value, unit: str = "", digits: int = 3) -> str:
    if not isinstance(value, (int, float)) or value != value:
        return "n/a"
    if unit == "%":
        return f"{value:+.2f}%"
    if unit == "nats":
        return f"{value:+.4f}"
    return f"{value:.{digits}f}"


def _fmt_ci(stat: dict, unit: str = "", digits: int = 3) -> str:
    if not stat or stat["n"] == 0:
        return "n/a"
    return (f"{_fmt(stat['mean'], unit, digits)} "
            f"[{_fmt(stat['ci_low'], unit, digits)}, {_fmt(stat['ci_high'], unit, digits)}]")


def _preview(text: str) -> str:
    return " ".join((text or "").split())[:OUTLIER_PREVIEW_CHARS]


def analysis_ppl_records(records: list[dict], analysis_ids: set) -> list[dict]:
    """Scorable records for kept prompts, with the two paired deltas attached."""
    out = []
    for rec in records:
        if rec["prompt_id"] not in analysis_ids or not rec.get("scorable", True):
            continue
        rec = dict(rec)
        for name, (left, right) in DELTAS.items():
            if rec.get(name) is None:
                rec[name] = rec[f"nll_{left}"] - rec[f"nll_{right}"]
        out.append(rec)
    return out


def top_outliers(records: list[dict], key: str, n: int = N_OUTLIERS) -> list[dict]:
    scored = [r for r in records if isinstance(r.get(key), (int, float))]
    return sorted(scored, key=lambda r: abs(r[key]), reverse=True)[:n]


def _delta_block(values: list[float]) -> dict:
    mean = bootstrap_ci(values, "mean")
    return {
        "mean": mean,
        "pct_equivalent": log_delta_to_pct(mean),
        "median": bootstrap_ci(values, "median"),
        "trimmed_mean": bootstrap_ci(values, "trimmed_mean"),
    }


def _length_metrics(kept: dict) -> dict:
    counts = {arm: np.asarray([kept[pid][arm]["token_count"] for pid in sorted(kept)],
                              dtype=float) for arm in ARMS}
    per_arm = {arm: {"mean": float(c.mean()), "median": float(np.median(c)),
                     "min": int(c.min()), "max": int(c.max())}
               for arm, c in counts.items()}
    return {
        "n": len(kept),
        "unit": "generator tokens (generations.jsonl token_count)",
        "per_arm": per_arm,
        "paired_delta_A_minus_B1": bootstrap_mean_ci(counts["A"] - counts["B1"]),
        "paired_delta_B2_minus_B1_control": bootstrap_mean_ci(counts["B2"] - counts["B1"]),
    }


def compute_metrics(out_dir: Path, run_config: dict, kept: dict, filter_report: dict,
                    ppl_records: list[dict], parity: dict) -> dict:
    analysis_ids = set(kept)
    ppl_config = run_config.get("stages", {}).get("ppl", {})

    metrics: dict = {
        "generated_at": utcnow(),
        "out_dir": str(out_dir),
        "filter": filter_report,
        "bootstrap": {"resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED,
                      "method": "percentile, resampling prompts with replacement"},
        "parity": parity,
    }

    if ppl_records:
        d_wm = [r["d_wm"] for r in ppl_records]
        d_ctrl = [r["d_ctrl"] for r in ppl_records]
        blocks = {"d_wm": _delta_block(d_wm), "d_ctrl": _delta_block(d_ctrl)}
        metrics["ppl"] = {
            "n": len(ppl_records),
            "scorer_model": ppl_records[0].get("scorer_model"),
            "conditioning": ppl_config.get("conditioning"),
            "context_template": ppl_config.get("context_template"),
            "statistic": "per-prompt difference of mean per-token NLL, in nats",
            "definition": {name: f"NLL_{left} - NLL_{right}"
                           for name, (left, right) in DELTAS.items()},
            "trim_proportion": TRIM_PROPORTION,
            "d_wm": blocks["d_wm"],
            "d_ctrl": blocks["d_ctrl"],
            "control_covers_zero": ci_covers(blocks["d_ctrl"]["mean"], 0.0),
            "mean_nll_by_arm": {arm: bootstrap_mean_ci([r[f"nll_{arm}"] for r in ppl_records])
                                for arm in ARMS},
            "n_truncated_by_arm": {arm: sum(1 for r in ppl_records if r.get(f"truncated_{arm}"))
                                   for arm in ARMS},
            "mean_scored_tokens_by_arm": {
                arm: (sum(r[f"n_scored_tokens_{arm}"] for r in ppl_records) / len(ppl_records))
                if all(r.get(f"n_scored_tokens_{arm}") is not None for r in ppl_records) else None
                for arm in ARMS},
        }
    else:
        metrics["ppl"] = None

    metrics["length"] = _length_metrics(kept) if kept else None

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
        "delta_nll": {
            "bound": f"95% CI of mean d_wm within +/- log(1.05) = "
                     f"+/- {DELTA_NLL_MARGIN:.4f} nats",
            "status": ("PASS" if ci_within(metrics["ppl"]["d_wm"]["mean"],
                                           -DELTA_NLL_MARGIN, DELTA_NLL_MARGIN)
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
        for name, (left, right) in DELTAS.items():
            block = p[name]
            role = "primary: watermark" if name == "d_wm" else "control: sampling noise"
            add_stat(f"{name}_mean", block["mean"], "nats",
                     f"{role}; mean per-prompt NLL_{left} - NLL_{right}")
            add_stat(f"{name}_pct_equivalent", block["pct_equivalent"], "%",
                     "100 * (exp(mean d) - 1)")
            add_stat(f"{name}_median", block["median"], "nats", "robustness")
            add_stat(f"{name}_trimmed_mean", block["trimmed_mean"], "nats",
                     f"robustness; {int(100 * TRIM_PROPORTION)}% trimmed from each tail")
        for arm in ARMS:
            add_stat(f"mean_nll_{arm}", p["mean_nll_by_arm"][arm], "nats", "")
            add(f"n_truncated_{arm}", p["n_truncated_by_arm"][arm], n=p["n"],
                notes="responses cut to fit the scorer context")
        add("ppl_scorer_model", p["scorer_model"])
        add("ppl_conditioning", p["conditioning"])
        add("ppl_control_covers_zero", p["control_covers_zero"],
            notes="d_ctrl CI must cover 0 for the noise floor to be credible")

    lengths = metrics["length"]
    if lengths:
        for arm in ARMS:
            stats = lengths["per_arm"][arm]
            for key in ("mean", "median", "min", "max"):
                add(f"length_{key}_{arm}", stats[key], n=lengths["n"], unit="tokens")
        add_stat("length_delta_A_minus_B1", lengths["paired_delta_A_minus_B1"], "tokens",
                 "paired mean difference")
        add_stat("length_delta_B2_minus_B1", lengths["paired_delta_B2_minus_B1_control"],
                 "tokens", "control")

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

    for name, check in metrics["parity"]["checks"].items():
        add(f"parity_{name}", check["status"], notes=check["detail"])
    add("parity_overall", metrics["parity"]["overall"])

    e = metrics["equivalence"]
    add("equivalence_win_rate", e["win_rate"]["status"], notes=e["win_rate"]["bound"])
    add("equivalence_delta_nll", e["delta_nll"]["status"], notes=e["delta_nll"]["bound"])
    add("equivalence_overall", e["overall"])
    return rows


def _parity_markdown(parity: dict) -> list[str]:
    lines = [
        "## Arm parity",
        "",
        "| Check | Status | Detail |",
        "|---|---|---|",
    ]
    for name, check in parity["checks"].items():
        lines.append(f"| {name} | {check['status']} | {check['detail']} |")
    lines.append("")

    kwargs_by_arm = parity.get("generation_kwargs_by_arm")
    if kwargs_by_arm:
        keys = sorted({k for arm in ARMS for k in kwargs_by_arm.get(arm, {})})
        lines += ["Generation kwargs, exactly as passed to `model.generate`:", "",
                  "| Argument | " + " | ".join(ARMS) + " |",
                  "|---" * (len(ARMS) + 1) + "|"]
        for key in keys:
            cells = " | ".join(f"`{kwargs_by_arm.get(arm, {}).get(key, '<absent>')}`"
                               for arm in ARMS)
            lines.append(f"| `{key}` | {cells} |")
        lines += ["",
                  "Temperature, top-k and top-p are 1.0/0/1.0 here on purpose: all "
                  "warping happens inside the logits processor chain below, so that "
                  "SynthID sees the same warped distribution the control arms sample "
                  "from. The effective sampling parameters are under Reproducibility.",
                  ""]
    else:
        lines += ["Generation kwargs were not recorded per arm by this run.", ""]

    chains = parity.get("logits_processor_chain_by_arm")
    decode = parity.get("decode_by_arm")
    if chains and decode:
        lines += ["| Arm | Logits processors | Decoding |", "|---|---|---|"]
        for arm in ARMS:
            rules = decode.get(arm, {})
            lines.append(f"| {arm} | `{', '.join(chains.get(arm, []))}` | "
                         f"`skip_special_tokens={rules.get('skip_special_tokens')}, "
                         f"strip={rules.get('strip')}` |")
        lines.append("")
    return lines


def _markdown(metrics: dict, run_config: dict) -> str:
    gen = run_config.get("stages", {}).get("generate", {})
    p, j, s, f, e = (metrics["ppl"], metrics["judge"], metrics["sim"],
                     metrics["filter"], metrics["equivalence"])
    lengths = metrics["length"]

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
        f"| Mean d_wm 95% CI within +/- log(1.05) | "
        f"{_fmt_ci(p['d_wm']['mean'], 'nats') if p else 'n/a'} | "
        f"{e['delta_nll']['status']} |",
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
            f"Scorer: `{p['scorer_model']}`. NLL is the mean per-token negative "
            f"log-likelihood of a response conditioned on its prompt; prompt tokens "
            f"are excluded from the mean. The statistic is the per-prompt difference "
            f"of NLLs, in nats, over n = {p['n']} prompts.",
            "",
            "| Quantity | Mean d | 95% CI | Percent equivalent |",
            "|---|---|---|---|",
        ]
        for name, label in (("d_wm", "d_wm = NLL_A - NLL_B1 (watermark)"),
                            ("d_ctrl", "d_ctrl = NLL_B2 - NLL_B1 (sampling noise)")):
            block = p[name]
            lines.append(
                f"| {label} | {_fmt(block['mean']['mean'], 'nats')} | "
                f"[{_fmt(block['mean']['ci_low'], 'nats')}, "
                f"{_fmt(block['mean']['ci_high'], 'nats')}] | "
                f"{_fmt_ci(block['pct_equivalent'], '%')} |")
        lines += [
            "",
            f"Percent equivalent is `100 * (exp(mean d) - 1)`: the geometric-mean "
            f"perplexity ratio implied by the mean log difference. The equivalence "
            f"bound lives on the log scale, `+/- log(1.05) = +/- "
            f"{DELTA_NLL_MARGIN:.4f}` nats, i.e. {DELTA_NLL_MARGIN_PCT}% in perplexity.",
            "",
            f"The control d_ctrl is the noise floor: two unwatermarked samples that "
            f"differ only in sampling seed. Its CI "
            f"{'covers' if p['control_covers_zero'] else 'does NOT cover'} 0"
            + ("." if p["control_covers_zero"] else
               " -- a null comparison should not be displaced, so treat d_wm with "
               "the same suspicion until this is explained."),
            "",
            "| Robustness | d_wm | d_ctrl |",
            "|---|---|---|",
            f"| Mean | {_fmt_ci(p['d_wm']['mean'], 'nats')} | "
            f"{_fmt_ci(p['d_ctrl']['mean'], 'nats')} |",
            f"| Median | {_fmt_ci(p['d_wm']['median'], 'nats')} | "
            f"{_fmt_ci(p['d_ctrl']['median'], 'nats')} |",
            f"| {int(100 * TRIM_PROPORTION)}% trimmed mean | "
            f"{_fmt_ci(p['d_wm']['trimmed_mean'], 'nats')} | "
            f"{_fmt_ci(p['d_ctrl']['trimmed_mean'], 'nats')} |",
            "",
            "| Arm | Mean NLL | 95% CI | exp(mean NLL) | Mean scored tokens | Truncated |",
            "|---|---|---|---|---|---|",
        ]
        for arm in ARMS:
            stat = p["mean_nll_by_arm"][arm]
            scored = p["mean_scored_tokens_by_arm"][arm]
            lines.append(
                f"| {arm} | {_fmt(stat['mean'], digits=4)} | "
                f"[{_fmt(stat['ci_low'], digits=4)}, {_fmt(stat['ci_high'], digits=4)}] | "
                f"{_fmt(math.exp(stat['mean']), digits=2)} | "
                f"{_fmt(scored, digits=1) if scored is not None else 'n/a'} | "
                f"{p['n_truncated_by_arm'][arm]} |")
        lines += [
            "",
            "Per-prompt differences, not per-prompt ratios: the ratio "
            "`PPL_A/PPL_B1 - 1` is bounded below by -100% and unbounded above, so "
            "averaging it reports a positive shift even when the two arms are "
            "exchangeable. The log difference is symmetric and its mean is the "
            "quantity the equivalence bound was written for.",
            "",
        ]
    else:
        lines += ["_Not scored yet (`ppl` stage has produced no records)._", ""]

    lines += ["## Output length", ""]
    if lengths:
        lines += [
            f"{lengths['unit']}, n = {lengths['n']} prompts.",
            "",
            "| Arm | Mean | Median | Min | Max |",
            "|---|---|---|---|---|",
        ]
        for arm in ARMS:
            stats = lengths["per_arm"][arm]
            lines.append(f"| {arm} | {_fmt(stats['mean'], digits=1)} | "
                         f"{_fmt(stats['median'], digits=1)} | {stats['min']} | "
                         f"{stats['max']} |")
        lines += [
            "",
            "| Paired difference | Mean | 95% CI |",
            "|---|---|---|",
            f"| A - B1 (watermark) | "
            f"{_fmt(lengths['paired_delta_A_minus_B1']['mean'], digits=2)} | "
            f"[{_fmt(lengths['paired_delta_A_minus_B1']['ci_low'], digits=2)}, "
            f"{_fmt(lengths['paired_delta_A_minus_B1']['ci_high'], digits=2)}] |",
            f"| B2 - B1 (sampling noise) | "
            f"{_fmt(lengths['paired_delta_B2_minus_B1_control']['mean'], digits=2)} | "
            f"[{_fmt(lengths['paired_delta_B2_minus_B1_control']['ci_low'], digits=2)}, "
            f"{_fmt(lengths['paired_delta_B2_minus_B1_control']['ci_high'], digits=2)}] |",
            "",
        ]
    else:
        lines += ["_No generations to measure._", ""]

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

    lines += _parity_markdown(metrics["parity"])

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
        "Per-prompt outliers on both deltas: `diagnostics.md`. Full parameters, "
        "library versions and model revisions: `run_config.json`.",
        "",
    ]
    return "\n".join(lines)


def _diagnostics_markdown(metrics: dict, run_config: dict, kept: dict,
                          ppl_records: list[dict]) -> str:
    gen = run_config.get("stages", {}).get("generate", {})
    lines = [
        "# Experiment 1 - outlier diagnostics",
        "",
        f"**Generator model:** `{gen.get('model')}`  ",
        f"**Run directory:** `{metrics['out_dir']}`  ",
        f"**Generated:** {metrics['generated_at']}  ",
        "",
        f"The {N_OUTLIERS} prompts with the largest absolute value of each paired "
        f"delta, out of {len(ppl_records)} scored. NLL is the mean per-token "
        f"negative log-likelihood of the response conditioned on its prompt. "
        f"Previews are the first {OUTLIER_PREVIEW_CHARS} characters, whitespace "
        f"collapsed.",
        "",
    ]

    for key, label in (("d_wm", "d_wm = NLL_A - NLL_B1 (watermark)"),
                       ("d_ctrl", "d_ctrl = NLL_B2 - NLL_B1 (sampling noise)")):
        lines += [f"## Largest |{key}| -- {label}", ""]
        outliers = top_outliers(ppl_records, key)
        if not outliers:
            lines += ["_No scored records._", ""]
            continue
        for rank, rec in enumerate(outliers, start=1):
            pid = rec["prompt_id"]
            arms = kept[pid]
            lines += [
                f"### {rank}. `{pid}`",
                "",
                f"d_wm = {_fmt(rec['d_wm'], 'nats')} "
                f"({_fmt(100 * (math.exp(rec['d_wm']) - 1), '%')}), "
                f"d_ctrl = {_fmt(rec['d_ctrl'], 'nats')} "
                f"({_fmt(100 * (math.exp(rec['d_ctrl']) - 1), '%')})",
                "",
                "| Arm | Generated tokens | Scored tokens | NLL | PPL |",
                "|---|---|---|---|---|",
            ]
            for arm in ARMS:
                scored = rec.get(f"n_scored_tokens_{arm}")
                lines.append(
                    f"| {arm} | {arms[arm]['token_count']} | "
                    f"{scored if scored is not None else 'n/a'} | "
                    f"{_fmt(rec[f'nll_{arm}'], digits=4)} | "
                    f"{_fmt(rec[f'ppl_{arm}'], digits=2)} |")
            lines.append("")
            for arm in ARMS:
                lines.append(f"- **{arm}:** {_preview(arms[arm]['text'])}")
            lines.append("")
    return "\n".join(lines)


def run_report(args) -> None:
    out_dir = Path(args.out)
    run_config = load_run_config(out_dir)

    kept, filter_report = load_valid_generations(out_dir)
    raw_ppl = read_jsonl(out_dir / "ppl.jsonl")
    ppl_records = analysis_ppl_records(raw_ppl, set(kept))

    parity = check_run_parity(run_config, kept, raw_ppl)
    print_parity(parity)
    if parity["overall"] == "FAIL":
        raise SystemExit("arm parity failed -- no report written; fix the listed "
                         "differences and run report again")

    metrics = compute_metrics(out_dir, run_config, kept, filter_report,
                              ppl_records, parity)
    write_json_atomic(out_dir / "metrics.json", metrics)

    (out_dir / "results.md").write_text(_markdown(metrics, run_config), encoding="utf-8")
    (out_dir / "diagnostics.md").write_text(
        _diagnostics_markdown(metrics, run_config, kept, ppl_records), encoding="utf-8")

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
        "parity": {"overall": parity["overall"],
                   "checks": {k: v["status"] for k, v in parity["checks"].items()}},
        "outputs": ["results.md", "results.csv", "diagnostics.md", "metrics.json",
                    "filter_report.json"],
    })

    p, e = metrics["ppl"], metrics["equivalence"]
    print()
    if p:
        print(f"  d_wm  (NLL_A - NLL_B1):  {_fmt_ci(p['d_wm']['mean'], 'nats')} nats"
              f"   = {_fmt_ci(p['d_wm']['pct_equivalent'], '%')}"
              f"   -> {e['delta_nll']['status']}  ({e['delta_nll']['bound']})")
        print(f"  d_ctrl (NLL_B2 - NLL_B1): {_fmt_ci(p['d_ctrl']['mean'], 'nats')} nats"
              f"   = {_fmt_ci(p['d_ctrl']['pct_equivalent'], '%')}"
              f"   -> noise floor, CI "
              f"{'covers' if p['control_covers_zero'] else 'MISSES'} 0")
    else:
        print(f"  d_wm: n/a   -> {e['delta_nll']['status']}")
    print(f"  judge win rate (A):  "
          f"{_fmt_ci(metrics['judge']['win_rate_A']) if metrics['judge'] else 'n/a'}"
          f"   -> {e['win_rate']['status']}  ({e['win_rate']['bound']})")
    if metrics["sim"]:
        print(f"  sim(A,B1) = {metrics['sim']['mean_sim_A_B1']['mean']:.4f}   vs   "
              f"sim(B1,B2) = {metrics['sim']['mean_sim_B1_B2_control']['mean']:.4f}  (control)")
    if metrics["length"]:
        print(f"  length A - B1: "
              f"{_fmt_ci(metrics['length']['paired_delta_A_minus_B1'], digits=2)} tokens")
    print(f"  filtered: {metrics['filter']['n_kept']} kept, "
          f"{metrics['filter']['n_dropped']} dropped")
    print()
    print(f"  OVERALL: {e['overall']}   PARITY: {parity['overall']}")
    print()
    log(f"wrote {out_dir / 'results.md'}, {out_dir / 'results.csv'}, "
        f"{out_dir / 'diagnostics.md'}, {out_dir / 'metrics.json'}")
