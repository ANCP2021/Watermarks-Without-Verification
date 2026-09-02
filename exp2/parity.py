
from __future__ import annotations

from exp1.common import ARMS
from exp1.parity import check_generation_parity, print_parity

MAX_LISTED = 5


def _cap(items: list[str]) -> list[str]:
    if len(items) <= MAX_LISTED:
        return items
    return items[:MAX_LISTED] + [f"... and {len(items) - MAX_LISTED} more"]


def _check(ok: bool, detail: str, differs: list[str] | None = None) -> dict:
    return {"status": "PASS" if ok else "FAIL", "detail": detail, "differs": _cap(differs or [])}


def _strict(check: dict) -> dict:
    if check["status"] == "INCOMPLETE":
        return {"status": "FAIL", "detail": "not recorded -- " + check["detail"],
                "differs": ["rerun `generate` with this code: it records the kwargs, "
                            "processor chain and decode rules per arm"]}
    return check


def sample_check(generations: dict[tuple, dict], triples: list[tuple]) -> dict:
    if not triples:
        return _check(False, "no complete (A, B1, B2) sample triples on disk")
    differs = []
    for pid, k in triples:
        recs = {arm: generations[(pid, arm, k)] for arm in ARMS}
        tag = f"{pid}/{k}"
        missing = [f for f in ("seed", "watermarked", "n_prompt_tokens", "model")
                   if any(f not in recs[a] for a in ARMS)]
        if missing:
            differs.append(f"{tag}: records lack {missing}")
            continue
        if recs["A"]["seed"] != recs["B1"]["seed"]:
            differs.append(f"{tag}: A seed {recs['A']['seed']} != B1 seed {recs['B1']['seed']}")
        if recs["B2"]["seed"] == recs["B1"]["seed"]:
            differs.append(f"{tag}: B2 seed equals B1 seed (no sampling-noise contrast)")
        if recs["A"]["watermarked"] is not True:
            differs.append(f"{tag}: arm A is not marked watermarked")
        for arm in ("B1", "B2"):
            if recs[arm]["watermarked"]:
                differs.append(f"{tag}: arm {arm} is marked watermarked")
        tokens = {a: recs[a]["n_prompt_tokens"] for a in ARMS}
        if len(set(tokens.values())) > 1:
            differs.append(f"{tag}: prompt token counts differ across arms {tokens}")
        models = {recs[a]["model"] for a in ARMS}
        if len(models) > 1:
            differs.append(f"{tag}: generator models differ across arms {sorted(models)}")
    return _check(not differs,
                  f"all {len(triples)} sample triples: A and B1 share a seed, B2 differs, "
                  f"only A is watermarked, one prompt encoding and one model per triple"
                  if not differs else
                  f"{len(differs)} sample triples break the arm/seed/prompt assignment",
                  differs)


def harness_check(executions: dict[tuple, dict], run_config: dict,
                  triples: list[tuple] | None = None) -> dict:
    if not executions:
        return _check(False, "no execution records on disk -- run `execute`")
    stage = run_config.get("stages", {}).get("execute", {})
    configs = {}
    for key, rec in executions.items():
        h = rec.get("harness") or {}
        sig = (h.get("timeout_s"), h.get("mem_mb"), h.get("fsize_mb"), h.get("exec_prefix"),
               h.get("python_version"), rec.get("extraction_rules"))
        configs.setdefault(sig, []).append(key)
    differs = []
    if len(configs) > 1:
        for sig, keys in configs.items():
            differs.append(f"{len(keys)} records under harness {sig} (e.g. {keys[0]})")
    sig = next(iter(configs))
    expected = (stage.get("timeout_s"), stage.get("mem_mb"), stage.get("fsize_mb"),
                stage.get("exec_prefix"), stage.get("python_version"),
                stage.get("extraction_rules"))
    if stage and sig != expected:
        differs.append(f"records {sig} != run_config execute stage {expected}")
    incomplete = [f"{pid}/{k}: executions missing or harness_error in "
                  f"{[a for a in ARMS if (pid, a, k) not in executions or executions[(pid, a, k)]['status'] == 'harness_error']}"
                  for pid, k in (triples or [])
                  if any((pid, a, k) not in executions
                         or executions[(pid, a, k)]["status"] == "harness_error" for a in ARMS)]
    differs += incomplete
    return _check(not differs,
                  f"{len(executions)} executions under one harness configuration "
                  f"(timeout {sig[0]}s, mem cap {sig[1]} MB, python {sig[4]}, "
                  f"rules {sig[5]}); every complete sample triple has an outcome in all "
                  f"three arms" if not differs
                  else ("executions were not run under one harness configuration"
                        if not incomplete else
                        f"{len(incomplete)} sample triples lack an outcome in some arm -- "
                        "run `execute` to completion"), differs)


def detector_check(detections: list[dict], run_config: dict) -> dict:
    if not detections:
        return _check(False, "no detection records on disk -- run `detect`")
    stage = run_config.get("stages", {}).get("detect", {})
    sigs = {(r.get("detector"), r.get("tokenizer_model"), r.get("watermark_config_sha256"),
             r.get("sampling_table_sha256"))
            for r in detections}
    differs = []
    if len(sigs) > 1:
        differs.append(f"{len(sigs)} detector configurations among the records: {sorted(sigs)}")
    sig = next(iter(sigs))
    expected = (stage.get("detector"), stage.get("tokenizer_model"),
                stage.get("watermark_config_sha256"), stage.get("sampling_table_sha256"))
    if stage and sig != expected:
        differs.append(f"records {sig} != run_config detect stage {expected}")
    if sig[3] is None:
        differs.append("records carry no sampling_table_sha256: scored with a table drawn at "
                       "detection time, not the persisted one (run `dump-table`, re-run `detect`)")
    gen_stage = run_config.get("stages", {}).get("generate", {})
    gen_sha = gen_stage.get("watermark_config_sha256")
    if gen_sha and sig[2] != gen_sha:
        differs.append(f"scored with keys {sig[2]} but generated with keys {gen_sha}")
    gen_table = gen_stage.get("watermark", {}).get("sampling_table_sha256")
    if gen_table and sig[3] != gen_table:
        differs.append(f"scored with sampling table {sig[3]} but generated with {gen_table}")
    return _check(not differs,
                  f"{len(detections)} scores from one detector ({sig[0]}) with one tokenizer "
                  f"({sig[1]}), the generation keys and the persisted sampling table"
                  if not differs
                  else "scores do not come from one detector configuration", differs)


def check_exp2_parity(run_config: dict, generations: dict[tuple, dict],
                      triples: list[tuple], executions: dict[tuple, dict],
                      detections: list[dict]) -> dict:
    gen = run_config.get("stages", {}).get("generate", {})
    parity = check_generation_parity(gen.get("generation_kwargs_by_arm"),
                                     gen.get("logits_processor_chain_by_arm"),
                                     gen.get("decode_by_arm"))
    checks = {name: _strict(c) for name, c in parity["checks"].items()}
    checks["sample_parity"] = sample_check(generations, triples)
    checks["harness_parity"] = harness_check(executions, run_config, triples)
    checks["detector_parity"] = detector_check(detections, run_config)
    parity["checks"] = checks
    parity["overall"] = "FAIL" if any(c["status"] != "PASS" for c in checks.values()) else "PASS"
    return parity

