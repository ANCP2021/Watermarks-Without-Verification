from __future__ import annotations

import glob
import os
from pathlib import Path

import torch

from exp1.common import (Timer, append_jsonl, check_config_consistency, done_keys,
                         load_run_config, log, provenance, read_jsonl, sha256_file,
                         update_run_config, utcnow)
from exp1.modeling import load_tokenizer
from exp1.watermark_setup import (build_synthid_processor, ensure_watermark_config,
                                  load_sampling_table, sampling_table_path,
                                  sampling_table_sha256)

from .execute import EXECUTION_FILE, index_records
from .extract import extract_code, sha256_text
from .generate import GEN_KEY, GENERATIONS_FILE
from .problems import load_problems
from .settings import PREFIX_LENGTHS

DETECTION_FILE = "detection.jsonl"
DET_KEY = ("domain", "source_run", "item_id", "arm", "sample_idx")
DETECTOR_NAME = "mean"
DETECTOR_PATH = ("no-training path: mean g-value over unmasked n-grams of the "
                 "re-tokenised text, empirical threshold; not the deployed Bayesian detector")
TOKENIZATION = "generator tokenizer, add_special_tokens=False, on the decoded+stripped text"


class GValueScorer:

    def __init__(self, processor, tokenizer, prefix_lengths=PREFIX_LENGTHS):
        self.processor = processor
        self.tokenizer = tokenizer
        self.prefix_lengths = tuple(int(L) for L in prefix_lengths)
        self.ngram_len = int(processor.ngram_len)
        self.depth = int(processor.keys.shape[0])
        eos = tokenizer.eos_token_id
        self.eos_token_id = eos[0] if isinstance(eos, (list, tuple)) else eos

    def encode(self, text: str) -> list[int]:
        return list(self.tokenizer(text, add_special_tokens=False).input_ids)

    def strip_trailing_special(self, ids: list[int]) -> list[int]:
        """Generated ids minus the EOS/pad tail that decode(skip_special_tokens) drops."""
        special = set(getattr(self.tokenizer, "all_special_ids", None) or [])
        for attr in ("eos_token_id", "pad_token_id"):
            value = getattr(self.tokenizer, attr, None)
            special.update(value if isinstance(value, (list, tuple)) else [value])
        special.discard(None)
        out = list(ids)
        while out and out[-1] in special:
            out.pop()
        return out

    @torch.inference_mode()
    def score_ids(self, ids: list[int]) -> dict:
        n = len(ids)
        empty = {"n_tokens": n, "score_full": None, "n_scored_full": 0,
                 "prefix": {str(L): None for L in self.prefix_lengths}}
        if n < self.ngram_len:
            return empty
        t = torch.tensor([ids], dtype=torch.long, device=self.processor.device)
        g = self.processor.compute_g_values(t)[0].float()                    # (n-k+1, depth)
        mask = self.processor.compute_context_repetition_mask(t)[0]
        if self.eos_token_id is not None:
            eos_mask = self.processor.compute_eos_token_mask(t, self.eos_token_id)
            mask = mask * eos_mask[0, self.ngram_len - 1:]
        m = mask.float()
        per_ngram = g.mean(dim=-1)                                            # (n-k+1,)
        cum_num = torch.cumsum(per_ngram * m, dim=0)
        cum_den = torch.cumsum(m, dim=0)

        def at(L: int):
            idx = L - self.ngram_len                                          # last n-gram inside L tokens
            den = float(cum_den[idx])
            if den <= 0:
                return {"score": None, "n_scored": 0}
            return {"score": float(cum_num[idx] / den), "n_scored": int(den)}

        full = at(n)
        return {
            "n_tokens": n,
            "score_full": full["score"],
            "n_scored_full": full["n_scored"],
            "prefix": {str(L): (at(L) if n >= L else None) for L in self.prefix_lengths},
        }

    def score_text(self, text: str) -> dict:
        ids = self.encode(text)
        return {**self.score_ids(ids), "ids": ids}


def find_exp1_runs(pattern: str, model_id: str, this_out: Path, smoke: bool) -> list[dict]:
    found = []
    for d in sorted(glob.glob(pattern)):
        d = Path(os.path.normpath(d))
        if not d.is_dir() or d.resolve() == this_out.resolve():
            continue
        cfg = load_run_config(d)
        gen = cfg.get("stages", {}).get("generate", {})
        if not gen or cfg.get("experiment") == "exp2" or gen.get("experiment") == "exp2":
            continue
        if gen.get("model") != model_id:
            continue
        if not (d / GENERATIONS_FILE).exists():
            continue
        if bool(gen.get("smoke")) and not smoke:
            log(f"note: skipping Exp 1 smoke run {d} (this run is not --smoke)")
            continue
        found.append({"dir": str(d), "model": gen.get("model"),
                      "source_run": "exp1:" + str(cfg.get("out_dir") or d.name),
                      "model_revision": gen.get("model_revision"), "smoke": bool(gen.get("smoke")),
                      "watermark_config_sha256": gen.get("watermark_config_sha256"),
                      "generations_sha256": sha256_file(d / GENERATIONS_FILE)})
    return found


def run_detect(args) -> None:
    out_dir = Path(args.out)
    det_path = out_dir / DETECTION_FILE
    run_config = load_run_config(out_dir)
    gen_stage = run_config.get("stages", {}).get("generate", {})
    if not gen_stage:
        raise SystemExit(f"{out_dir}/run_config.json has no generate stage -- run `generate` first")
    model_id = gen_stage["model"]

    wm_cfg = ensure_watermark_config(args.watermark_config)
    wm_sha = sha256_file(args.watermark_config)
    if gen_stage.get("watermark_config_sha256") not in (None, wm_sha):
        raise SystemExit("watermark_config.json differs from the one used at generation "
                         f"({gen_stage.get('watermark_config_sha256')} vs {wm_sha}); "
                         "detection would use the wrong keys")

    loaded_table = load_sampling_table(args.watermark_config, wm_cfg, "cpu")
    if loaded_table is None:
        raise SystemExit(
            f"{sampling_table_path(args.watermark_config)} not found. Detection needs the g-value "
            "table that generation drew (cpu and cuda generators differ for the same seed); "
            "run `python run_exp2.py dump-table` on the generation machine (cuda) first.")
    table_sha = sampling_table_sha256(loaded_table[0])
    table_meta = loaded_table[1]

    generations = index_records(read_jsonl(out_dir / GENERATIONS_FILE))
    if not generations:
        raise SystemExit("no generations to score -- run `generate` first")
    executions = index_records(read_jsonl(out_dir / EXECUTION_FILE))
    payload_problems = load_problems(args.problems, create_if_missing=False)
    problems = {p["problem_id"]: p for p in payload_problems["problems"]}

    exp1_runs = find_exp1_runs(args.exp1_runs, model_id, out_dir, bool(gen_stage.get("smoke")))
    for r in exp1_runs:
        if r["watermark_config_sha256"] not in (None, wm_sha):
            raise SystemExit(f"Exp 1 run {r['dir']} used different watermark keys "
                             f"({r['watermark_config_sha256']}); refusing to score it")
    if len({r["source_run"] for r in exp1_runs}) != len(exp1_runs):
        raise SystemExit("two Exp 1 runs share a source key; give them distinct out dirs: "
                         f"{[(r['dir'], r['source_run']) for r in exp1_runs]}")
    if len(exp1_runs) > 1:
        log(f"WARNING: {len(exp1_runs)} Exp 1 runs match {model_id}; their prose is POOLED "
            "into one domain -- narrow --exp1-runs if that is not intended")
    if exp1_runs:
        log(f"prose: {len(exp1_runs)} Exp 1 run(s) for {model_id}: "
            f"{[r['dir'] for r in exp1_runs]}")
    else:
        log(f"prose: no Exp 1 run for {model_id} under {args.exp1_runs!r} -- "
            "the prose domain will be absent from the report")

    current = {
        "detector": DETECTOR_NAME, "detector_path": DETECTOR_PATH,
        "tokenizer_model": model_id, "tokenizer_revision": gen_stage.get("model_revision"),
        "tokenization": TOKENIZATION,
        "watermark_config_sha256": wm_sha, "ngram_len": wm_cfg["ngram_len"],
        "depth": len(wm_cfg["keys"]), "prefix_lengths": list(PREFIX_LENGTHS),
        "sampling_table_sha256": table_sha,
        "sampling_table_file": str(sampling_table_path(args.watermark_config)),
        "sampling_table_drawn_on": table_meta.get("drawn_on"),
        "sampling_table_cuda_emulation_agrees": table_meta.get("cuda_emulation_agrees"),
        "score": "sum(g * mask) / (depth * n_unmasked) == MarkLLM MeanDetector",
        "masks": "context-repetition mask (non-distortionary) and EOS mask, as in "
                 "MarkLLM SynthID.detect_watermark",
        "prefix_rule": "a sample contributes to prefix L only if it has >= L tokens; "
                       "'full' is every sample with >= ngram_len tokens",
        "exp1_source_runs": sorted(r["source_run"] for r in exp1_runs),
        "domains": {"prose": "Exp 1 response text", "code": "full response text",
                    "code_extracted": "the extracted fence only",
                    "code_ids": "the generated token ids themselves (no re-tokenisation; "
                                "trailing EOS/pad removed)"},
    }
    check_config_consistency(out_dir, "detect", current,
                             critical=("detector", "tokenizer_model", "tokenization",
                                       "watermark_config_sha256", "sampling_table_sha256",
                                       "ngram_len", "depth", "prefix_lengths",
                                       "exp1_source_runs"),
                             force=args.force)

    tokenizer = load_tokenizer(model_id, gen_stage.get("model_revision"))
    processor = build_synthid_processor(args.watermark_config, tokenizer, "cpu",
                                        require_table=True)
    if processor.sampling_table_sha256 != table_sha:
        raise SystemExit("internal error: processor table != loaded sampling_table.pt")
    scorer = GValueScorer(processor, tokenizer, PREFIX_LENGTHS)

    payload = {**current, "exp1_runs": exp1_runs, "exp1_runs_glob": args.exp1_runs,
               "device": "cpu", "started_at": utcnow(), "provenance": provenance()}
    update_run_config(out_dir, "detect", payload)

    self_run = "self" 
    items: list[tuple[tuple, dict]] = []
    for key in sorted(generations, key=lambda k: (k[0], k[2], k[1])):
        rec = generations[key]
        pid, arm, k = key
        items.append((("code", self_run, pid, arm, k),
                      {"text": rec["text"], "watermarked": bool(rec.get("watermarked")),
                       "stored_ids": rec.get("token_ids"), "source": rec.get("source")}))
        if rec.get("token_ids") is not None:
            items.append((("code_ids", self_run, pid, arm, k),
                          {"ids": rec["token_ids"], "watermarked": bool(rec.get("watermarked")),
                           "stored_ids": None, "source": rec.get("source")}))
        ex = executions.get(key)
        code, info = extract_code(rec["text"], problems.get(pid, {}).get("entry_point"))
        if code is not None:
            items.append((("code_extracted", self_run, pid, arm, k),
                          {"text": code, "watermarked": bool(rec.get("watermarked")),
                           "stored_ids": None, "source": rec.get("source"),
                           "execution_status": ex.get("status") if ex else None}))
    for r in exp1_runs:
        for rec in read_jsonl(Path(r["dir"]) / GENERATIONS_FILE):
            items.append((("prose", r["source_run"], rec["prompt_id"], rec["arm"], 0),
                          {"text": rec["text"], "watermarked": bool(rec.get("watermarked",
                                                                            rec["arm"] == "A")),
                           "stored_ids": None, "source": "dolly"}))

    already = done_keys(det_path, DET_KEY)
    todo = [(key, item) for key, item in items if key not in already]
    log(f"detect: {len(items)} items, {len(items) - len(todo)} done, {len(todo)} to score")

    timer = Timer()
    seen: set[tuple] = set()
    for i, (key, item) in enumerate(todo, start=1):
        if key in seen:
            continue
        seen.add(key)
        domain, source_run, item_id, arm, k = key
        if "ids" in item:
            ids = scorer.strip_trailing_special(item["ids"])
            scored = {**scorer.score_ids(ids), "ids": ids}
            text_sha = None
        else:
            scored = scorer.score_text(item["text"])
            text_sha = sha256_text(item["text"])
        roundtrip = None
        if item["stored_ids"] is not None:
            roundtrip = scorer.strip_trailing_special(item["stored_ids"]) == scored["ids"]
        append_jsonl(det_path, {
            "domain": domain, "source_run": source_run, "item_id": item_id, "arm": arm,
            "sample_idx": k, "watermarked": item["watermarked"], "source": item.get("source"),
            "n_tokens": scored["n_tokens"], "score_full": scored["score_full"],
            "n_scored_full": scored["n_scored_full"], "prefix": scored["prefix"],
            "roundtrip_ids_equal": roundtrip, "text_sha256": text_sha,
            "execution_status": item.get("execution_status"),
            "detector": DETECTOR_NAME, "tokenizer_model": model_id,
            "watermark_config_sha256": wm_sha, "sampling_table_sha256": table_sha,
            "timestamp": utcnow(),
        })
        if i % 200 == 0 or i == len(todo):
            log(f"  {i}/{len(todo)} scored  eta {timer.eta(i, len(todo))}")

    update_run_config(out_dir, "detect", {**payload, "finished_at": utcnow(),
                                          "n_scored_on_disk": len(done_keys(det_path, DET_KEY))})
    log(f"detection complete -> {det_path}")
