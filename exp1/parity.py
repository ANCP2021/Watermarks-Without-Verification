
from __future__ import annotations

from .common import ARMS, log
from .settings import PPL_CONDITIONING

SYNTHID_PROCESSOR = "SynthIDLogitsProcessor"
WATERMARKED_ARM = "A"
MAX_LISTED = 5


def _cap(items: list[str]) -> list[str]:
    if len(items) <= MAX_LISTED:
        return items
    return items[:MAX_LISTED] + [f"... and {len(items) - MAX_LISTED} more"]


def _check(ok: bool, detail: str, differs: list[str] | None = None) -> dict:
    return {"status": "PASS" if ok else "FAIL", "detail": detail,
            "differs": _cap(differs or [])}


def _incomplete(detail: str) -> dict:
    return {"status": "INCOMPLETE", "detail": detail, "differs": []}


def _overall(checks: dict) -> str:
    statuses = [c["status"] for c in checks.values()]
    if "FAIL" in statuses:
        return "FAIL"
    return "INCOMPLETE" if "INCOMPLETE" in statuses else "PASS"


def _diff_dicts(by_arm: dict, label: str) -> list[str]:
    ref_arm = ARMS[0]
    ref = by_arm[ref_arm]
    differs = []
    for arm in ARMS[1:]:
        other = by_arm[arm]
        for key in sorted(set(ref) | set(other)):
            a, b = ref.get(key, "<absent>"), other.get(key, "<absent>")
            if a != b:
                differs.append(f"{label}.{key}: {ref_arm}={a!r} {arm}={b!r}")
    return differs


def kwargs_check(kwargs_by_arm: dict | None) -> dict:
    if not kwargs_by_arm or any(a not in kwargs_by_arm for a in ARMS):
        return _incomplete("generation kwargs were not recorded per arm by this run")
    differs = _diff_dicts(kwargs_by_arm, "generation_kwargs")
    n_args = len(kwargs_by_arm[ARMS[0]])
    return _check(not differs,
                  f"{n_args} arguments, identical across {', '.join(ARMS)}" if not differs
                  else f"arms disagree on {len(differs)} of {n_args} arguments", differs)


def processor_check(processors_by_arm: dict | None) -> dict:
    if not processors_by_arm or any(a not in processors_by_arm for a in ARMS):
        return _incomplete("logits processors were not recorded per arm by this run")

    chains = {a: list(processors_by_arm[a]) for a in ARMS}
    base = {a: [p for p in chains[a] if p != SYNTHID_PROCESSOR] for a in ARMS}
    ref = base[ARMS[0]]

    differs = []
    for arm in ARMS[1:]:
        if base[arm] != ref:
            differs.append(f"non-SynthID chain: {ARMS[0]}={ref} {arm}={base[arm]}")
    for arm in ARMS:
        found = chains[arm].count(SYNTHID_PROCESSOR)
        expected = 1 if arm == WATERMARKED_ARM else 0
        if found != expected:
            differs.append(f"{arm}: expected {expected} {SYNTHID_PROCESSOR}, found {found}")
    if chains[WATERMARKED_ARM] != ref + [SYNTHID_PROCESSOR]:
        differs.append(f"{WATERMARKED_ARM} is not the shared chain plus a trailing "
                       f"{SYNTHID_PROCESSOR}: {chains[WATERMARKED_ARM]}")

    return _check(not differs,
                  f"{WATERMARKED_ARM} = shared chain + {SYNTHID_PROCESSOR}; "
                  f"shared chain = {ref}" if not differs
                  else "the arms differ by more than the SynthID processor", differs)


def decode_check(decode_by_arm: dict | None) -> dict:
    if not decode_by_arm or any(a not in decode_by_arm for a in ARMS):
        return _incomplete("decoding rules were not recorded per arm by this run")
    differs = _diff_dicts(decode_by_arm, "decode")
    ref = decode_by_arm[ARMS[0]]
    return _check(not differs,
                  f"skip_special_tokens={ref.get('skip_special_tokens')}, "
                  f"strip={ref.get('strip')}, identical across {', '.join(ARMS)}"
                  if not differs else "the arms were decoded by different rules",
                  differs)


def check_generation_parity(kwargs_by_arm: dict, processors_by_arm: dict,
                            decode_by_arm: dict) -> dict:
    checks = {
        "generation_kwargs": kwargs_check(kwargs_by_arm),
        "logits_processors": processor_check(processors_by_arm),
        "decode_rules": decode_check(decode_by_arm),
    }
    return {
        "checks": checks,
        "generation_kwargs_by_arm": kwargs_by_arm,
        "logits_processor_chain_by_arm": processors_by_arm,
        "decode_by_arm": decode_by_arm,
        "overall": _overall(checks),
    }


def prompt_check(kept: dict) -> dict:
    if not kept:
        return _incomplete("no complete generation triples")
    fields = ("n_prompt_tokens", "model")
    missing = [f for f in fields
               if any(f not in kept[pid][a] for pid in kept for a in ARMS)]
    if missing:
        return _incomplete(f"generations.jsonl predates the field(s) {missing}")

    differs = []
    for pid in sorted(kept):
        arms = kept[pid]
        tokens = {a: arms[a]["n_prompt_tokens"] for a in ARMS}
        if len(set(tokens.values())) > 1:
            differs.append(f"{pid}: prompt token counts differ across arms {tokens}")
        models = {arms[a]["model"] for a in ARMS}
        if len(models) > 1:
            differs.append(f"{pid}: generator models differ across arms {sorted(models)}")
    return _check(not differs,
                  f"all {len(kept)} prompts encode to one token count and one "
                  f"generator model across arms" if not differs
                  else f"{len(differs)} prompts were not put to the arms identically",
                  differs)


def seed_check(kept: dict) -> dict:
    if not kept:
        return _incomplete("no complete generation triples")
    missing = [f for f in ("seed", "watermarked")
               if any(f not in kept[pid][a] for pid in kept for a in ARMS)]
    if missing:
        return _incomplete(f"generations.jsonl predates the field(s) {missing}")

    differs = []
    for pid in sorted(kept):
        arms = kept[pid]
        if arms["A"]["seed"] != arms["B1"]["seed"]:
            differs.append(f"{pid}: A seed {arms['A']['seed']} != B1 seed "
                           f"{arms['B1']['seed']} (A and B1 are the paired arms)")
        if arms["B2"]["seed"] == arms["B1"]["seed"]:
            differs.append(f"{pid}: B2 seed equals B1 seed (no sampling-noise contrast)")
        if arms["A"]["watermarked"] is not True:
            differs.append(f"{pid}: arm A is not marked watermarked")
        for arm in ("B1", "B2"):
            if arms[arm]["watermarked"]:
                differs.append(f"{pid}: arm {arm} is marked watermarked")
    return _check(not differs,
                  "A and B1 share a sampling seed, B2 uses a different one, "
                  "and only A is watermarked" if not differs
                  else "the arm/seed/watermark assignment is not the pre-registered one",
                  differs)


def nll_check(run_config: dict, ppl_records: list[dict]) -> dict:
    if not ppl_records:
        return _incomplete("ppl.jsonl is empty -- run the ppl stage")

    recorded = run_config.get("stages", {}).get("ppl", {}).get("conditioning")
    if recorded != PPL_CONDITIONING:
        return _check(False,
                      f"stages.ppl.conditioning is {recorded!r}, required "
                      f"{PPL_CONDITIONING!r}",
                      [f"NLL was scored as {recorded!r}: the mean is not taken over "
                       f"response tokens conditioned on the prompt",
                       "rescore with: python run_exp1.py ppl --out <run dir> --rescore"])

    differs = []
    for rec in ppl_records:
        if not rec.get("scorable", True):
            continue
        pid = rec["prompt_id"]
        context = {a: rec.get(f"n_context_tokens_{a}") for a in ARMS}
        if any(v is None for v in context.values()):
            differs.append(f"{pid}: no context token counts recorded")
            continue
        if len(set(context.values())) > 1:
            differs.append(f"{pid}: arms were conditioned on different contexts {context}")
        if any(v == 0 for v in context.values()):
            differs.append(f"{pid}: empty context -- the response was scored standalone")
        for arm in ARMS:
            scored = rec.get(f"n_scored_tokens_{arm}")
            response = rec.get(f"n_response_tokens_{arm}")
            if scored is None or response is None:
                differs.append(f"{pid}/{arm}: no scored/response token counts recorded")
            elif scored != response and not rec.get(f"truncated_{arm}"):
                differs.append(f"{pid}/{arm}: scored {scored} of {response} response "
                               f"tokens without truncation")
    return _check(not differs,
                  f"{len(ppl_records)} prompts scored as {PPL_CONDITIONING}; every arm "
                  f"shares one context and contributes only response tokens"
                  if not differs
                  else f"{len(differs)} records are not scored as {PPL_CONDITIONING}",
                  differs)


def prompt_set_check(run_config: dict) -> dict:
    stages = run_config.get("stages", {})
    generated = stages.get("generate", {}).get("prompts_sha256")
    scored = stages.get("ppl", {}).get("prompts_sha256")
    if not generated or not scored:
        return _incomplete("prompts sha256 not recorded by both generate and ppl")
    return _check(generated == scored,
                  f"ppl conditioned on the prompts used for generation "
                  f"(sha256 {generated[:16]}...)" if generated == scored
                  else "ppl scored a different prompt file than generation used",
                  [] if generated == scored else
                  [f"generate={generated}", f"ppl={scored}"])


def check_run_parity(run_config: dict, kept: dict, ppl_records: list[dict]) -> dict:
    generate = run_config.get("stages", {}).get("generate", {})
    parity = check_generation_parity(generate.get("generation_kwargs_by_arm"),
                                     generate.get("logits_processor_chain_by_arm"),
                                     generate.get("decode_by_arm"))
    parity["checks"].update({
        "prompt_parity": prompt_check(kept),
        "seed_parity": seed_check(kept),
        "prompt_set": prompt_set_check(run_config),
        "nll_conditioning": nll_check(run_config, ppl_records),
    })
    parity["overall"] = _overall(parity["checks"])
    return parity


def _kwargs_table(kwargs_by_arm: dict | None) -> list[str]:
    if not kwargs_by_arm:
        return ["  (not recorded per arm by this run)"]
    keys = sorted({k for arm in ARMS for k in kwargs_by_arm.get(arm, {})})
    width = max(len(k) for k in keys) if keys else 0
    lines = ["  " + "argument".ljust(width) + "  " + "  ".join(a.ljust(24) for a in ARMS)]
    for key in keys:
        cells = [repr(kwargs_by_arm.get(arm, {}).get(key, "<absent>")).ljust(24)
                 for arm in ARMS]
        lines.append("  " + key.ljust(width) + "  " + "  ".join(cells))
    return lines


def print_parity(parity: dict) -> None:
    print()
    print("Arm parity")
    print()
    print("generation kwargs, exactly as passed to model.generate:")
    for line in _kwargs_table(parity.get("generation_kwargs_by_arm")):
        print(line)
    print()
    chains = parity.get("logits_processor_chain_by_arm")
    decode = parity.get("decode_by_arm")
    for arm in ARMS:
        chain = ", ".join(chains[arm]) if chains and arm in chains else "(not recorded)"
        print(f"  logits processors {arm}: {chain}")
    for arm in ARMS:
        rules = decode[arm] if decode and arm in decode else "(not recorded)"
        print(f"  decode {arm}: {rules}")
    print()
    for name, check in parity["checks"].items():
        print(f"  [{check['status']}] {name}: {check['detail']}")
        for line in check["differs"]:
            print(f"      - {line}")
    print()
    print(f"  PARITY: {parity['overall']}")
    print()


def assert_generation_parity(kwargs_by_arm: dict, processors_by_arm: dict,
                             decode_by_arm: dict) -> dict:
    parity = check_generation_parity(kwargs_by_arm, processors_by_arm, decode_by_arm)
    if parity["overall"] == "FAIL":
        print_parity(parity)
        raise SystemExit("arms differ by more than the SynthID logits processor "
                         "-- refusing to generate")
    for name, check in parity["checks"].items():
        log(f"parity {name}: {check['status']} -- {check['detail']}")
    return parity
