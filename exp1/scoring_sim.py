
from __future__ import annotations

from pathlib import Path

import torch

from .common import (Timer, append_jsonl, check_config_consistency, done_keys,
                     log, model_commit_hash, provenance,
                     update_run_config, utcnow)
from .filters import load_valid_generations
from .modeling import resolve_device
from .settings import SIM_MODEL, SMOKE_SIM_MODEL

SIM_FILE = "sim.jsonl"
MAX_LENGTH = 512 


def load_embedder(model_id: str, device: str):
    from transformers import AutoModel, AutoTokenizer

    log(f"loading embedder {model_id} (device={device}, dtype=float32)")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id, torch_dtype=torch.float32)
    model.to(device).eval()
    return model, tokenizer


@torch.inference_mode()
def embed(model, tokenizer, texts: list[str], device: str) -> torch.Tensor:
    """BGE's own recipe: CLS pooling, then L2 normalise."""
    batch = tokenizer(texts, padding=True, truncation=True, max_length=MAX_LENGTH,
                      return_tensors="pt").to(device)
    hidden = model(**batch).last_hidden_state[:, 0]
    return torch.nn.functional.normalize(hidden, p=2, dim=1)


def run_sim(args) -> None:
    out_dir = Path(args.out)
    sim_path = out_dir / SIM_FILE

    kept, filter_report = load_valid_generations(out_dir)
    prompt_ids = sorted(kept)

    model_id = args.sim_model or (SMOKE_SIM_MODEL if args.smoke else SIM_MODEL)
    check_config_consistency(out_dir, "sim", {"embedding_model": model_id},
                             critical=("embedding_model",), force=args.force)

    device = resolve_device(args.device)
    model, tokenizer = load_embedder(model_id, device)

    payload = {
        "embedding_model": model_id,
        "embedding_model_revision": model_commit_hash(model, model_id),
        "dtype": "float32",
        "device": device,
        "pooling": "cls",
        "normalize": True,
        "max_length": MAX_LENGTH,
        "pairs": {"treatment": "sim(A, B1)", "control": "sim(B1, B2)"},
        "filter": {k: filter_report[k] for k in
                   ("n_kept", "n_dropped", "n_incomplete_triples")},
        "started_at": utcnow(),
        "provenance": provenance(),
    }
    update_run_config(out_dir, "sim", payload)

    already = {k[0] for k in done_keys(sim_path, ("prompt_id",))}
    todo = [pid for pid in prompt_ids if pid not in already]
    log(f"sim: {len(prompt_ids)} prompts, {len(already & set(prompt_ids))} done, "
        f"{len(todo)} to go")

    timer = Timer()
    for i, pid in enumerate(todo, start=1):
        texts = [kept[pid][a]["text"] for a in ("A", "B1", "B2")]
        emb = embed(model, tokenizer, texts, device)
        append_jsonl(sim_path, {
            "prompt_id": pid,
            "sim_a_b1": float(torch.dot(emb[0], emb[1])),
            "sim_b1_b2": float(torch.dot(emb[1], emb[2])),
            "embedding_model": model_id,
            "timestamp": utcnow(),
        })
        if i % 50 == 0 or i == len(todo):
            log(f"  {i}/{len(todo)} embedded  eta {timer.eta(i, len(todo))}")

    update_run_config(out_dir, "sim", {**payload, "finished_at": utcnow(),
                                       "n_scored": len(done_keys(sim_path, ("prompt_id",)))})
    log(f"sim complete -> {sim_path}")
