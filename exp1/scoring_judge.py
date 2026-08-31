
from __future__ import annotations

import re
from pathlib import Path
import torch
from .common import (Timer, append_jsonl, check_config_consistency, done_keys,
                     log, model_commit_hash, provenance, seed_everything,
                     update_run_config, utcnow)
from .filters import load_valid_generations
from .modeling import load_model_and_tokenizer, resolve_device, resolve_dtype
from .prompts import load_prompts
from .settings import JUDGE_MODEL, SMOKE_JUDGE_MODEL

JUDGE_FILE = "judge.jsonl"
MAX_NEW_TOKENS = 8

SYSTEM_PROMPT = (
    "You are an impartial evaluator of writing quality. You compare two "
    "candidate responses to the same instruction and decide which one is "
    "better overall, considering helpfulness, accuracy, coherence and "
    "fluency. You are not influenced by the order in which the responses are "
    "presented, or by their length. You answer with a single token."
)

USER_TEMPLATE = """[Instruction]
{instruction}

[Response 1]
{response_1}

[Response 2]
{response_2}

Which response is better? Answer with exactly one word: "1" if Response 1 is \
better, "2" if Response 2 is better, or "Tie" if they are of equal quality. \
Answer:"""

_VERDICT_RE = re.compile(r"\b(tie|1|2)\b", re.IGNORECASE)


def parse_verdict(raw: str) -> str:
    match = _VERDICT_RE.search((raw or "").strip())
    if not match:
        return "unparsed"
    token = match.group(1).lower()
    return "tie" if token == "tie" else token


@torch.inference_mode()
def ask_judge(model, tokenizer, device, instruction: str, r1: str, r2: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(
            instruction=instruction, response_1=r1, response_2=r2)},
    ]
    rendered = tokenizer.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True)
    inputs = tokenizer(rendered, return_tensors="pt", add_special_tokens=False).to(device)
    output = model.generate(
        **inputs,
        do_sample=False,            # temperature 0
        num_beams=1,
        max_new_tokens=MAX_NEW_TOKENS,
        repetition_penalty=1.0,
        temperature=1.0,
        top_k=0,
        top_p=1.0,
        pad_token_id=tokenizer.pad_token_id,
    )
    return tokenizer.decode(output[0, inputs["input_ids"].shape[1]:],
                            skip_special_tokens=True).strip()


def decide(v1: str, v2: str) -> str:
    
    first = {"1": "A", "2": "B1"}.get(v1)
    second = {"1": "B1", "2": "A"}.get(v2)
    if first is not None and first == second:
        return first
    return "tie"


def run_judge(args) -> None:
    out_dir = Path(args.out)
    judge_path = out_dir / JUDGE_FILE

    kept, filter_report = load_valid_generations(out_dir)
    prompt_ids = sorted(kept)

    prompts_payload = load_prompts(args.prompts, create_if_missing=False)
    instructions = {p["prompt_id"]: p["text"] for p in prompts_payload["prompts"]}
    missing = [pid for pid in prompt_ids if pid not in instructions]
    if missing:
        raise SystemExit(f"{len(missing)} generated prompt ids are absent from "
                         f"{args.prompts} (e.g. {missing[:3]}) -- wrong prompts file?")

    model_id = args.judge_model or (SMOKE_JUDGE_MODEL if args.smoke else JUDGE_MODEL)
    check_config_consistency(out_dir, "judge",
                             {"judge_model": model_id, "system_prompt": SYSTEM_PROMPT,
                              "user_template": USER_TEMPLATE},
                             critical=("judge_model", "system_prompt", "user_template"),
                             force=args.force)

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    model, tokenizer, attn_impl = load_model_and_tokenizer(model_id, device, dtype)
    seed_everything(0)   # greedy decoding, but pin the RNG anyway

    payload = {
        "judge_model": model_id,
        "judge_model_revision": model_commit_hash(model, model_id),
        "dtype": str(dtype).replace("torch.", ""),
        "device": device,
        "attn_implementation": attn_impl,
        "decoding": {"do_sample": False, "temperature": 0,
                     "max_new_tokens": MAX_NEW_TOKENS},
        "protocol": {
            "comparison": "A vs B1",
            "blinded": True,
            "ties_allowed": True,
            "positions_swapped": True,
            "win_rule": "a win requires both orderings to agree; otherwise tie",
            "order_1": "Response 1 = A, Response 2 = B1",
            "order_2": "Response 1 = B1, Response 2 = A",
        },
        "system_prompt": SYSTEM_PROMPT,
        "user_template": USER_TEMPLATE,
        "filter": {k: filter_report[k] for k in
                   ("n_kept", "n_dropped", "n_incomplete_triples")},
        "started_at": utcnow(),
        "provenance": provenance(),
    }
    update_run_config(out_dir, "judge", payload)

    already = {k[0] for k in done_keys(judge_path, ("prompt_id",))}
    todo = [pid for pid in prompt_ids if pid not in already]
    log(f"judge: {len(prompt_ids)} pairs, {len(already & set(prompt_ids))} done, "
        f"{len(todo)} to go ({2 * len(todo)} judge calls)")

    timer = Timer()
    for i, pid in enumerate(todo, start=1):
        instruction = instructions[pid]
        text_a = kept[pid]["A"]["text"]
        text_b1 = kept[pid]["B1"]["text"]

        raw1 = ask_judge(model, tokenizer, device, instruction, text_a, text_b1)
        raw2 = ask_judge(model, tokenizer, device, instruction, text_b1, text_a)
        v1, v2 = parse_verdict(raw1), parse_verdict(raw2)

        append_jsonl(judge_path, {
            "prompt_id": pid,
            "verdict_order1": v1,
            "verdict_order2": v2,
            "winner": decide(v1, v2),
            "raw_order1": raw1,
            "raw_order2": raw2,
            "judge_model": model_id,
            "timestamp": utcnow(),
        })
        if i % 25 == 0 or i == len(todo):
            log(f"  {i}/{len(todo)} pairs judged  eta {timer.eta(i, len(todo))}")

    update_run_config(out_dir, "judge", {**payload, "finished_at": utcnow(),
                                         "n_judged": len(done_keys(judge_path, ("prompt_id",)))})
    log(f"judge complete -> {judge_path}")
