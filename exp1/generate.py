from __future__ import annotations

from pathlib import Path

import torch

from .common import (Timer, append_jsonl, check_config_consistency, derive_seed,
                     done_keys, log, model_commit_hash, provenance,
                     seed_everything, sha256_file, sha256_text,
                     update_run_config, utcnow)
from .modeling import (build_processor_chain, encode_rendered, generation_kwargs,
                       load_causal_lm, load_tokenizer, render_prompt, resolve_device,
                       resolve_dtype)
from .parity import assert_generation_parity
from .prompts import load_prompts, select_prompts
from .settings import (SAMPLING, SEED_S1, SEED_S2, SMOKE_MAX_NEW_TOKENS,
                       SMOKE_MODEL, SMOKE_N_PROMPTS, TARGET_MODELS)
from .watermark_setup import (build_synthid_processor, ensure_watermark_config,
                              reset_synthid_state)

GENERATIONS_FILE = "generations.jsonl"

ARM_SPEC = {
    "A":  (True,  SEED_S1, "S1"),
    "B1": (False, SEED_S1, "S1"),
    "B2": (False, SEED_S2, "S2"),
}

DECODE_KWARGS = {"skip_special_tokens": True}
DECODE_STRIP = True


def _check_model_allowed(model_id: str, smoke: bool) -> None:
    if model_id in TARGET_MODELS:
        return
    if smoke:
        if model_id != SMOKE_MODEL:
            log(f"note: --smoke with non-standard model {model_id}")
        return
    raise SystemExit(
        f"{model_id} is not one of the pre-registered generator models "
        f"{list(TARGET_MODELS)}.\nFor a wiring test use --smoke (which also allows "
        f"{SMOKE_MODEL})."
    )


def _finish_reason(gen_ids: torch.Tensor, tokenizer, max_new_tokens: int) -> str:
    if gen_ids.shape[0] >= max_new_tokens:
        return "length"
    eos_ids = tokenizer.eos_token_id
    eos_ids = eos_ids if isinstance(eos_ids, (list, tuple)) else [eos_ids]
    if gen_ids.numel() and int(gen_ids[-1]) in [e for e in eos_ids if e is not None]:
        return "eos"
    return "other"


def run_generate(args) -> None:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    gen_path = out_dir / GENERATIONS_FILE

    _check_model_allowed(args.model, args.smoke)

    prompts_payload = load_prompts(args.prompts)
    prompts = select_prompts(prompts_payload, args.smoke, SMOKE_N_PROMPTS)
    if args.limit:
        prompts = prompts[: args.limit]

    wm_cfg = ensure_watermark_config(args.watermark_config)
    wm_sha = sha256_file(args.watermark_config)
    prompts_sha = sha256_file(args.prompts)

    max_new_tokens = args.max_new_tokens
    if max_new_tokens is None:
        max_new_tokens = SMOKE_MAX_NEW_TOKENS if args.smoke else SAMPLING["max_new_tokens"]

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    tokenizer = load_tokenizer(args.model, args.revision)

    rendered = {p["prompt_id"]: render_prompt(tokenizer, p["text"])
                for p in prompts_payload["prompts"]}
    render_sha = sha256_text(
        "\x00".join(rendered[p["prompt_id"]] for p in prompts_payload["prompts"]))
    log("first rendered prompt (verbatim):\n" + "-" * 60 + "\n"
        + rendered[prompts[0]["prompt_id"]] + "\n" + "-" * 60)

    effective_sampling = dict(SAMPLING)
    effective_sampling["max_new_tokens"] = max_new_tokens

    synthid = build_synthid_processor(args.watermark_config, tokenizer, device)
    chains = {arm: build_processor_chain(synthid if watermarked else None)
              for arm, (watermarked, _, _) in ARM_SPEC.items()}
    gen_kwargs = generation_kwargs(tokenizer, max_new_tokens)

    kwargs_by_arm = {arm: dict(gen_kwargs) for arm in ARM_SPEC}
    processors_by_arm = {arm: [type(p).__name__ for p in chains[arm]] for arm in ARM_SPEC}
    decode_by_arm = {arm: {**DECODE_KWARGS, "strip": DECODE_STRIP} for arm in ARM_SPEC}
    parity = assert_generation_parity(kwargs_by_arm, processors_by_arm, decode_by_arm)

    payload = {
        "model": args.model,
        "dtype": str(dtype).replace("torch.", ""),
        "device": device,
        "smoke": bool(args.smoke),
        "max_new_tokens": max_new_tokens,
        "sampling": effective_sampling,
        "seeds": {
            "S1": SEED_S1,
            "S2": SEED_S2,
            "per_prompt_derivation": "int(sha256(f'{base_seed}|{prompt_id}').hexdigest()[:8], 16)",
            "arms": {a: {"watermarked": w, "base_seed": s, "label": lbl}
                     for a, (w, s, lbl) in ARM_SPEC.items()},
        },
        "prompts_path": str(args.prompts),
        "prompts_sha256": prompts_sha,
        "n_prompts_selected": len(prompts),
        "prompt_render_sha256": render_sha,
        "watermark_config_path": str(args.watermark_config),
        "watermark_config_sha256": wm_sha,
        "watermark": {
            "algorithm": "SynthID",
            "mode": wm_cfg["watermark_mode"],
            "ngram_len": wm_cfg["ngram_len"],
            "depth": len(wm_cfg["keys"]),
            "sampling_table_size": wm_cfg["sampling_table_size"],
            "sampling_table_seed": wm_cfg["sampling_table_seed"],
            "implementation": "MarkLLM watermark.synthid.SynthIDLogitsProcessor",
            "overrides": {
                "temperature": 1.0,
                "top_k": -1,
                "why": "processor is applied after temperature/top-k/top-p warping, "
                       "so it must not re-apply them",
            },
        },
        "generation_kwargs_by_arm": kwargs_by_arm,
        "logits_processor_chain_by_arm": processors_by_arm,
        "decode_by_arm": decode_by_arm,
        "arm_parity": parity["checks"],
        "batch_size": 1,
        "started_at": utcnow(),
    }
    check_config_consistency(
        out_dir, "generate", payload,
        critical=("model", "dtype", "max_new_tokens", "sampling", "seeds",
                  "prompts_sha256", "prompt_render_sha256",
                  "watermark_config_sha256"),
        force=args.force,
    )

    model, attn_impl = load_causal_lm(
        args.model, device, dtype, args.attn_impl, args.revision)
    payload["model_revision"] = model_commit_hash(model, args.model)
    payload["attn_implementation"] = attn_impl
    payload["provenance"] = provenance()

    update_run_config(out_dir, "generate", payload,
                      top_level={"out_dir": str(out_dir), "smoke": bool(args.smoke)})

    already = done_keys(gen_path, ("prompt_id", "arm"))
    tasks = [(p, arm) for p in prompts for arm in ARM_SPEC]
    todo = [(p, arm) for p, arm in tasks if (p["prompt_id"], arm) not in already]
    log(f"{len(tasks)} generations planned, {len(tasks) - len(todo)} already on disk, "
        f"{len(todo)} to run")

    timer = Timer()

    for i, (prompt, arm) in enumerate(todo, start=1):
        watermarked, base_seed, seed_label = ARM_SPEC[arm]
        prompt_id = prompt["prompt_id"]
        seed = derive_seed(base_seed, prompt_id)

        inputs = encode_rendered(tokenizer, rendered[prompt_id], device)
        n_prompt_tokens = int(inputs["input_ids"].shape[1])

        if watermarked:
            reset_synthid_state(synthid)
        seed_everything(seed)

        with torch.inference_mode():
            output = model.generate(
                **inputs,
                logits_processor=chains[arm],
                **gen_kwargs,
            )

        gen_ids = output[0, n_prompt_tokens:]
        text = tokenizer.decode(gen_ids, **DECODE_KWARGS)
        if DECODE_STRIP:
            text = text.strip()

        append_jsonl(gen_path, {
            "prompt_id": prompt_id,
            "arm": arm,
            "seed": seed,
            "text": text,
            "token_count": int(gen_ids.shape[0]),
            "model": args.model,
            "timestamp": utcnow(),
            # extras, for auditing
            "watermarked": watermarked,
            "seed_base": base_seed,
            "seed_label": seed_label,
            "n_prompt_tokens": n_prompt_tokens,
            "finish_reason": _finish_reason(gen_ids, tokenizer, max_new_tokens),
        })

        if i % 10 == 0 or i == len(todo):
            log(f"  {i}/{len(todo)} generations  (last: {prompt_id}/{arm}, "
                f"{int(gen_ids.shape[0])} tok)  eta {timer.eta(i, len(todo))}")

    update_run_config(out_dir, "generate", {**payload, "finished_at": utcnow(),
                                            "n_generations_on_disk": len(
                                                done_keys(gen_path, ("prompt_id", "arm")))})
    log(f"generation complete -> {gen_path}")
