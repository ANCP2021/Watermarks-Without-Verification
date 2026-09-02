from __future__ import annotations

from pathlib import Path

import torch

from exp1.common import (ARMS, Timer, append_jsonl, check_config_consistency, derive_seed,
                         done_keys, log, model_commit_hash, provenance, seed_everything,
                         sha256_file, sha256_text, update_run_config, utcnow)
from exp1.generate import DECODE_KWARGS, DECODE_STRIP, _finish_reason
from exp1.modeling import (build_processor_chain, encode_rendered, generation_kwargs,
                           load_causal_lm, load_tokenizer, render_prompt, resolve_device,
                           resolve_dtype)
from exp1.parity import assert_generation_parity
from exp1.watermark_setup import (build_synthid_processor, ensure_watermark_config,
                                  reset_synthid_state)

from .problems import load_problems, select_problems
from .prompting import user_message
from .settings import (HUMANEVAL_TEMPLATE, INSTRUCTION, MBPP_TEMPLATE, N_SAMPLES, SAMPLING,
                       SMOKE_MAX_NEW_TOKENS, SMOKE_MODEL, SMOKE_N_PROBLEMS, TARGET_MODELS,
                       b2_seed_offset, base_seed)

GENERATIONS_FILE = "generations.jsonl"
GEN_KEY = ("problem_id", "arm", "sample_idx")

WATERMARKED = {"A": True, "B1": False, "B2": False}


def seed_index(arm: str, sample_idx: int, n_samples: int) -> int:
    if arm in ("A", "B1"):
        return sample_idx + 1
    return b2_seed_offset(n_samples) + sample_idx + 1


def sample_seed(arm: str, sample_idx: int, n_samples: int, problem_id: str) -> tuple[int, str]:
    i = seed_index(arm, sample_idx, n_samples)
    return derive_seed(base_seed(i), problem_id), f"S{i}"


def seed_table(n_samples: int) -> dict:
    offset = b2_seed_offset(n_samples)
    return {f"S{i}": base_seed(i) for i in range(1, offset + n_samples + 1)}


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
        f"{SMOKE_MODEL}).")


def run_generate(args) -> None:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    gen_path = out_dir / GENERATIONS_FILE

    _check_model_allowed(args.model, args.smoke)

    payload_problems = load_problems(args.problems)
    problems = select_problems(payload_problems, args.smoke, SMOKE_N_PROBLEMS)
    if args.limit:
        problems = problems[: args.limit]

    wm_cfg = ensure_watermark_config(args.watermark_config)
    wm_sha = sha256_file(args.watermark_config)
    problems_sha = sha256_file(args.problems)

    n_samples = int(args.samples or N_SAMPLES)
    if n_samples < 1:
        raise SystemExit("--samples must be >= 1")
    max_new_tokens = args.max_new_tokens
    if max_new_tokens is None:
        max_new_tokens = SMOKE_MAX_NEW_TOKENS if args.smoke else SAMPLING["max_new_tokens"]

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    tokenizer = load_tokenizer(args.model, args.revision)

    messages = {p["problem_id"]: user_message(p) for p in payload_problems["problems"]}
    rendered = {pid: render_prompt(tokenizer, msg) for pid, msg in messages.items()}
    render_sha = sha256_text(
        "\x00".join(rendered[p["problem_id"]] for p in payload_problems["problems"]))
    log("first rendered prompt (verbatim):\n" + "-" * 60 + "\n"
        + rendered[problems[0]["problem_id"]] + "\n" + "-" * 60)

    effective_sampling = dict(SAMPLING)
    effective_sampling["max_new_tokens"] = max_new_tokens

    synthid = build_synthid_processor(args.watermark_config, tokenizer, device)
    chains = {arm: build_processor_chain(synthid if WATERMARKED[arm] else None)
              for arm in ARMS}
    kwargs_by_arm = {arm: generation_kwargs(tokenizer, max_new_tokens) for arm in ARMS}
    processors_by_arm = {arm: [type(p).__name__ for p in chains[arm]] for arm in ARMS}
    decode_by_arm = {arm: {**DECODE_KWARGS, "strip": DECODE_STRIP} for arm in ARMS}
    parity = assert_generation_parity(kwargs_by_arm, processors_by_arm, decode_by_arm)

    offset = b2_seed_offset(n_samples)
    payload = {
        "experiment": "exp2",
        "model": args.model,
        "dtype": str(dtype).replace("torch.", ""),
        "device": device,
        "smoke": bool(args.smoke),
        "max_new_tokens": max_new_tokens,
        "sampling": effective_sampling,
        "n_samples": n_samples,
        "seeds": {
            "table": seed_table(n_samples),
            "base_seed": "S_i = int(sha256(f'exp2|S{i}').hexdigest()[:8], 16)",
            "per_sample_derivation": "int(sha256(f'{S_i}|{problem_id}').hexdigest()[:8], 16)",
            "b2_offset": offset,
            "arms": {
                "A": {"watermarked": True, "seed_indices": f"S1..S{n_samples}"},
                "B1": {"watermarked": False, "seed_indices": f"S1..S{n_samples}"},
                "B2": {"watermarked": False,
                       "seed_indices": f"S{offset + 1}..S{offset + n_samples}"},
            },
        },
        "prompting": {"instruction": INSTRUCTION, "humaneval_template": HUMANEVAL_TEMPLATE,
                      "mbpp_template": MBPP_TEMPLATE,
                      "chat_template": "tokenizer.apply_chat_template, one user turn, "
                                       "add_generation_prompt=True"},
        "problems_path": str(args.problems),
        "problems_sha256": problems_sha,
        "n_problems_selected": len(problems),
        "problem_ids_selected": [p["problem_id"] for p in problems],
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
            "sampling_table_sha256": synthid.sampling_table_sha256,
            "sampling_table_source": synthid.sampling_table_source,
            "implementation": "MarkLLM watermark.synthid.SynthIDLogitsProcessor",
            "overrides": {"temperature": 1.0, "top_k": -1,
                          "why": "processor is applied after temperature/top-k/top-p "
                                 "warping, so it must not re-apply them"},
        },
        "generate_call": {
            "inputs": ["input_ids", "attention_mask"],
            "logits_processor": "logits_processor_chain_by_arm[arm]",
            "kwargs": "generation_kwargs_by_arm[arm], passed as **kwargs verbatim",
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
        critical=("model", "dtype", "device", "max_new_tokens", "sampling", "n_samples",
                  "seeds", "prompting", "problems_sha256", "prompt_render_sha256",
                  "watermark_config_sha256"),
        force=args.force,
    )

    model, attn_impl = load_causal_lm(args.model, device, dtype, args.attn_impl, args.revision)
    payload["model_revision"] = model_commit_hash(model, args.model)
    payload["attn_implementation"] = attn_impl
    payload["provenance"] = provenance()
    check_config_consistency(
        out_dir, "generate",
        {"model_revision": payload["model_revision"], "attn_implementation": attn_impl},
        critical=("model_revision", "attn_implementation"), force=args.force)

    update_run_config(out_dir, "generate", payload,
                      top_level={"experiment": "exp2", "out_dir": str(out_dir),
                                 "smoke": bool(args.smoke)})

    already = done_keys(gen_path, GEN_KEY)
    tasks = [(p, k, arm) for p in problems for k in range(n_samples) for arm in ARMS]
    todo = [t for t in tasks if (t[0]["problem_id"], t[2], t[1]) not in already]
    log(f"{len(tasks)} generations planned ({len(problems)} problems x {n_samples} samples "
        f"x {len(ARMS)} arms), {len(tasks) - len(todo)} already on disk, {len(todo)} to run")

    timer = Timer()
    for i, (problem, k, arm) in enumerate(todo, start=1):
        pid = problem["problem_id"]
        seed, seed_label = sample_seed(arm, k, n_samples, pid)

        inputs = encode_rendered(tokenizer, rendered[pid], device)
        n_prompt_tokens = int(inputs["input_ids"].shape[1])

        if WATERMARKED[arm]:
            reset_synthid_state(synthid)
        seed_everything(seed)

        with torch.inference_mode():
            output = model.generate(
                **inputs,
                logits_processor=chains[arm],
                **kwargs_by_arm[arm],
            )

        gen_ids = output[0, n_prompt_tokens:]
        text = tokenizer.decode(gen_ids, **DECODE_KWARGS)
        if DECODE_STRIP:
            text = text.strip()

        append_jsonl(gen_path, {
            "problem_id": pid,
            "arm": arm,
            "sample_idx": k,
            "seed": seed,
            "text": text,
            "token_ids": [int(t) for t in gen_ids.tolist()],
            "token_count": int(gen_ids.shape[0]),
            "model": args.model,
            "timestamp": utcnow(),
            # extras, for auditing
            "source": problem["source"],
            "watermarked": WATERMARKED[arm],
            "seed_label": seed_label,
            "n_prompt_tokens": n_prompt_tokens,
            "finish_reason": _finish_reason(gen_ids, tokenizer, max_new_tokens),
        })

        if i % 10 == 0 or i == len(todo):
            log(f"  {i}/{len(todo)} generations  (last: {pid}/{arm}/{k}, "
                f"{int(gen_ids.shape[0])} tok)  eta {timer.eta(i, len(todo))}")

    update_run_config(out_dir, "generate", {
        **payload, "finished_at": utcnow(),
        "n_generations_on_disk": len(done_keys(gen_path, GEN_KEY))})
    log(f"generation complete -> {gen_path}")
