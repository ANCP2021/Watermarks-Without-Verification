
from __future__ import annotations

import math
from pathlib import Path

import torch

from .common import (Timer, append_jsonl, check_config_consistency, done_keys,
                     log, model_commit_hash, provenance, update_run_config, utcnow)
from .filters import load_valid_generations
from .modeling import load_model_and_tokenizer, resolve_device, resolve_dtype
from .settings import PPL_MODEL, SMOKE_PPL_MODEL

PPL_FILE = "ppl.jsonl"
DEFAULT_MAX_LENGTH = 2048


@torch.inference_mode()
def mean_nll(model, tokenizer, text: str, device: str, max_length: int):
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False,
                    truncation=True, max_length=max_length).input_ids.to(device)
    if ids.shape[1] < 2:
        return None, int(ids.shape[1])
    logits = model(ids).logits[:, :-1, :].float()
    targets = ids[:, 1:]
    nll = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="mean")
    return float(nll), int(targets.shape[1])


def run_ppl(args) -> None:
    out_dir = Path(args.out)
    ppl_path = out_dir / PPL_FILE

    kept, filter_report = load_valid_generations(out_dir)
    prompt_ids = sorted(kept)

    model_id = args.ppl_model or (SMOKE_PPL_MODEL if args.smoke else PPL_MODEL)
    check_config_consistency(out_dir, "ppl",
                             {"scorer_model": model_id, "max_length": args.ppl_max_length},
                             critical=("scorer_model", "max_length"), force=args.force)

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    model, tokenizer, attn_impl = load_model_and_tokenizer(model_id, device, dtype)

    payload = {
        "scorer_model": model_id,
        "scorer_model_revision": model_commit_hash(model, model_id),
        "dtype": str(dtype).replace("torch.", ""),
        "device": device,
        "attn_implementation": attn_impl,
        "max_length": args.ppl_max_length,
        "scoring": "output_only (not conditioned on the prompt); "
                   "mean cross-entropy over tokens 2..n",
        "filter": {k: filter_report[k] for k in
                   ("n_kept", "n_dropped", "n_incomplete_triples")},
        "started_at": utcnow(),
        "provenance": provenance(),
    }
    update_run_config(out_dir, "ppl", payload)

    already = {k[0] for k in done_keys(ppl_path, ("prompt_id",))}
    todo = [pid for pid in prompt_ids if pid not in already]
    log(f"ppl: {len(prompt_ids)} prompts to score, {len(already & set(prompt_ids))} done, "
        f"{len(todo)} to go")

    timer = Timer()
    for i, pid in enumerate(todo, start=1):
        rec: dict = {"prompt_id": pid, "scorer_model": model_id, "timestamp": utcnow()}
        ok = True
        for arm in ("A", "B1", "B2"):
            nll, n_tok = mean_nll(model, tokenizer, kept[pid][arm]["text"],
                                  device, args.ppl_max_length)
            rec[f"nll_{arm}"] = nll
            rec[f"ppl_{arm}"] = math.exp(nll) if nll is not None else None
            rec[f"n_scored_tokens_{arm}"] = n_tok
            ok = ok and nll is not None
        rec["scorable"] = ok
        if ok:
        
            rec["delta_pct_A_vs_B1"] = 100.0 * (rec["ppl_A"] - rec["ppl_B1"]) / rec["ppl_B1"]
            rec["delta_pct_B2_vs_B1"] = 100.0 * (rec["ppl_B2"] - rec["ppl_B1"]) / rec["ppl_B1"]
        if not ok:
            log(f"  WARNING: {pid} has an output too short to score; marked unscorable")
        append_jsonl(ppl_path, rec)

        if i % 25 == 0 or i == len(todo):
            log(f"  {i}/{len(todo)} scored  eta {timer.eta(i, len(todo))}")

    update_run_config(out_dir, "ppl", {**payload, "finished_at": utcnow(),
                                       "n_scored": len(done_keys(ppl_path, ("prompt_id",)))})
    log(f"ppl complete -> {ppl_path}")
