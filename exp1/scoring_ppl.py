
from __future__ import annotations

import math
from pathlib import Path

import torch

from .common import (Timer, append_jsonl, check_config_consistency, done_keys,
                     log, model_commit_hash, provenance, sha256_file,
                     update_run_config, utcnow)
from .filters import load_valid_generations
from .modeling import load_model_and_tokenizer, resolve_device, resolve_dtype
from .prompts import load_prompts
from .settings import (PPL_CONDITIONING, PPL_CONTEXT_TEMPLATE,
                       PPL_MIN_RESPONSE_ROOM, PPL_MODEL, SMOKE_PPL_MODEL)

PPL_FILE = "ppl.jsonl"
DEFAULT_MAX_LENGTH = 2048


def encode_context(tokenizer, prompt_text: str, max_length: int) -> tuple[list[int], bool]:
    """The conditioning context, encoded once per prompt and shared by all arms."""
    ids = tokenizer(PPL_CONTEXT_TEMPLATE.format(prompt=prompt_text),
                    add_special_tokens=False).input_ids
    room = max(1, max_length - PPL_MIN_RESPONSE_ROOM)
    if len(ids) > room:
        return ids[-room:], True
    return ids, False


def encode_response(tokenizer, response_text: str, n_context: int,
                    max_length: int) -> tuple[list[int], int, bool]:
    ids = tokenizer(response_text, add_special_tokens=False).input_ids
    room = max(0, max_length - n_context)
    if len(ids) > room:
        return ids[:room], len(ids), True
    return ids, len(ids), False


@torch.inference_mode()
def response_nll(model, context_ids, response_ids, device: str) -> float | None:
    """Mean per-token NLL of the response, conditioned on the context.

    The context is fed to the model but its tokens are not in the mean: only
    positions len(context)..len(context)+len(response)-1 are scored, so the
    first response token is predicted from the prompt rather than skipped.
    """
    n_context = len(context_ids)
    if n_context < 1:
        raise ValueError("scoring context is empty; the response would be scored standalone")
    if not response_ids:
        return None
    ids = torch.tensor([list(context_ids) + list(response_ids)], device=device)
    logits = model(ids).logits[:, n_context - 1: -1, :].float()
    targets = ids[:, n_context:]
    nll = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="mean")
    return float(nll)


def _archive_previous(ppl_path: Path) -> None:
    if not ppl_path.exists():
        return
    backup = ppl_path.with_name(f"{ppl_path.name}.bak-{utcnow().replace(':', '')}")
    ppl_path.rename(backup)
    log(f"--rescore: archived the previous {ppl_path.name} as {backup.name}")


def run_ppl(args) -> None:
    out_dir = Path(args.out)
    ppl_path = out_dir / PPL_FILE

    kept, filter_report = load_valid_generations(out_dir)
    prompt_ids = sorted(kept)

    prompts_payload = load_prompts(args.prompts, create_if_missing=False)
    instructions = {p["prompt_id"]: p["text"] for p in prompts_payload["prompts"]}
    missing = [pid for pid in prompt_ids if pid not in instructions]
    if missing:
        raise SystemExit(f"{len(missing)} generated prompt ids are absent from "
                         f"{args.prompts} (e.g. {missing[:3]}) -- wrong prompts file?")

    model_id = args.ppl_model or (SMOKE_PPL_MODEL if args.smoke else PPL_MODEL)
    current = {"scorer_model": model_id, "max_length": args.ppl_max_length,
               "conditioning": PPL_CONDITIONING, "context_template": PPL_CONTEXT_TEMPLATE,
               "prompts_sha256": sha256_file(args.prompts)}
    if getattr(args, "rescore", False):
        _archive_previous(ppl_path)
    else:
        check_config_consistency(out_dir, "ppl", current,
                                 critical=tuple(current), force=args.force)

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    model, tokenizer, attn_impl = load_model_and_tokenizer(model_id, device, dtype)

    payload = {
        **current,
        "scorer_model_revision": model_commit_hash(model, model_id),
        "dtype": str(dtype).replace("torch.", ""),
        "device": device,
        "attn_implementation": attn_impl,
        "prompts_path": str(args.prompts),
        "min_response_room": PPL_MIN_RESPONSE_ROOM,
        "scoring": "mean cross-entropy over the response tokens of "
                   "'<prompt>\\n\\n<response>'; the prompt is conditioned on but "
                   "its tokens are excluded from the mean",
        "statistics": {"d_wm": "nll_A - nll_B1", "d_ctrl": "nll_B2 - nll_B1"},
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
        context_ids, context_truncated = encode_context(
            tokenizer, instructions[pid], args.ppl_max_length)

        rec: dict = {"prompt_id": pid, "scorer_model": model_id, "timestamp": utcnow(),
                     "n_context_tokens": len(context_ids),
                     "context_truncated": context_truncated}
        ok = True
        for arm in ("A", "B1", "B2"):
            response_ids, n_response, truncated = encode_response(
                tokenizer, kept[pid][arm]["text"], len(context_ids), args.ppl_max_length)
            nll = response_nll(model, context_ids, response_ids, device)
            rec[f"nll_{arm}"] = nll
            rec[f"ppl_{arm}"] = math.exp(nll) if nll is not None else None
            rec[f"n_context_tokens_{arm}"] = len(context_ids)
            rec[f"n_scored_tokens_{arm}"] = len(response_ids)
            rec[f"n_response_tokens_{arm}"] = n_response
            rec[f"truncated_{arm}"] = truncated
            ok = ok and nll is not None
        rec["scorable"] = ok
        if ok:
            rec["d_wm"] = rec["nll_A"] - rec["nll_B1"]
            rec["d_ctrl"] = rec["nll_B2"] - rec["nll_B1"]
        else:
            log(f"  WARNING: {pid} has an output with no response tokens; marked unscorable")
        append_jsonl(ppl_path, rec)

        if i % 25 == 0 or i == len(todo):
            log(f"  {i}/{len(todo)} scored  eta {timer.eta(i, len(todo))}")

    update_run_config(out_dir, "ppl", {**payload, "finished_at": utcnow(),
                                       "n_scored": len(done_keys(ppl_path, ("prompt_id",)))})
    log(f"ppl complete -> {ppl_path}")
