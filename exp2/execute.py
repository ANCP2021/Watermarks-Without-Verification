from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from exp1.common import (Timer, append_jsonl, check_config_consistency, done_keys, log,
                         provenance, read_jsonl, sha256_file, update_run_config, utcnow)

from .extract import (ASSEMBLY_RULES, EXTRACTION_RULES_VERSION, RULES, build_program,
                      extract_code, sha256_text)
from .generate import GEN_KEY, GENERATIONS_FILE
from .harness import HARNESS_ERROR, harness_description, preflight, run_program
from .problems import load_problems
from .settings import EXEC_FSIZE_MB, EXEC_MEM_MB, EXEC_TIMEOUT_S, EXEC_WORKERS

EXECUTION_FILE = "execution.jsonl"
UNPARSEABLE = "unparseable"


def index_records(records: list[dict], key=GEN_KEY) -> dict[tuple, dict]:
    """First record per key; duplicates (crash + resume) are ignored."""
    out: dict[tuple, dict] = {}
    for rec in records:
        k = tuple(rec.get(f) for f in key)
        out.setdefault(k, rec)
    return out


def harness_config(args) -> dict:
    return {
        "timeout_s": float(args.timeout),
        "mem_mb": int(args.mem_mb),
        "fsize_mb": int(EXEC_FSIZE_MB),
        "exec_prefix": args.exec_prefix or None,
        "python_version": sys.version.split()[0],
        "extraction_rules": EXTRACTION_RULES_VERSION,
        "extraction_rules_text": RULES,
        "program_assembly": ASSEMBLY_RULES,
    }


def run_execute(args) -> None:
    out_dir = Path(args.out)
    exec_path = out_dir / EXECUTION_FILE
    gen_path = out_dir / GENERATIONS_FILE

    generations = index_records(read_jsonl(gen_path))
    if not generations:
        raise SystemExit(f"no generations in {gen_path} -- run `generate` first")

    payload_problems = load_problems(args.problems, create_if_missing=False)
    problems = {p["problem_id"]: p for p in payload_problems["problems"]}
    missing = sorted({k[0] for k in generations} - set(problems))
    if missing:
        raise SystemExit(f"{len(missing)} generated problem ids are absent from "
                         f"{args.problems} (e.g. {missing[:3]}) -- wrong problems file?")

    current = {**harness_config(args), "problems_sha256": sha256_file(args.problems)}
    check_config_consistency(
        out_dir, "execute", current,
        critical=("timeout_s", "mem_mb", "fsize_mb", "exec_prefix", "python_version",
                  "extraction_rules", "program_assembly", "problems_sha256"),
        force=args.force)

    workers = int(args.workers or EXEC_WORKERS)
    preflight(current["timeout_s"], current["mem_mb"], current["fsize_mb"], current["exec_prefix"])
    payload = {**current, "workers": workers, "harness": harness_description(),
               "problems_path": str(args.problems), "started_at": utcnow(),
               "provenance": provenance()}
    update_run_config(out_dir, "execute", payload)

    # harness errors are not outcomes of the candidate: they are redone on resume
    already = {tuple(r.get(f) for f in GEN_KEY) for r in read_jsonl(exec_path)
               if r.get("status") != HARNESS_ERROR}
    todo = [k for k in sorted(generations, key=lambda k: (k[0], k[2], k[1]))
            if k not in already]
    log(f"execute: {len(generations)} samples, {len(generations) - len(todo)} done, "
        f"{len(todo)} to run ({workers} workers, timeout {current['timeout_s']}s, "
        f"mem cap {current['mem_mb']} MB)")

    def record_base(key: tuple) -> dict:
        pid, arm, k = key
        return {"problem_id": pid, "arm": arm, "sample_idx": k,
                "source": problems[pid]["source"],
                "extraction_rules": EXTRACTION_RULES_VERSION,
                "harness": {"timeout_s": current["timeout_s"], "mem_mb": current["mem_mb"],
                            "fsize_mb": current["fsize_mb"],
                            "exec_prefix": current["exec_prefix"],
                            "python_version": current["python_version"]}}

    timer = Timer()
    n_done = 0
    n_unparseable = 0
    counts: dict[str, int] = {}

    def bump(status: str) -> None:
        counts[status] = counts.get(status, 0) + 1

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {}
        for key in todo:
            rec = generations[key]
            problem = problems[key[0]]
            code, info = extract_code(rec["text"], problem.get("entry_point"))
            if code is None:
                append_jsonl(exec_path, {
                    **record_base(key), "status": UNPARSEABLE, "extraction": info,
                    "assembly": None, "code_sha256": None, "program_sha256": None,
                    "exit_code": None, "elapsed_s": 0.0, "stderr_tail": "", "limits": {},
                    "timestamp": utcnow()})
                n_done += 1
                n_unparseable += 1
                bump(UNPARSEABLE)
                log(f"  unparseable: {key[0]}/{key[1]}/{key[2]} "
                    f"(fences={info['n_fences']}, preview={rec['text'][:80]!r})")
                continue
            program, mode = build_program(problem, code)
            fut = pool.submit(run_program, program, current["timeout_s"], current["mem_mb"],
                              current["fsize_mb"], current["exec_prefix"])
            futures[fut] = (key, info, mode, sha256_text(code), sha256_text(program))

        for fut in as_completed(futures):
            key, info, mode, code_sha, program_sha = futures[fut]
            result = fut.result()
            append_jsonl(exec_path, {
                **record_base(key), **result, "extraction": info, "assembly": mode,
                "code_sha256": code_sha, "program_sha256": program_sha,
                "timestamp": utcnow()})
            n_done += 1
            bump(result["status"])
            if n_done % 50 == 0 or n_done == len(todo):
                log(f"  {n_done}/{len(todo)} executed  {counts}  "
                    f"eta {timer.eta(n_done, len(todo))}")

    update_run_config(out_dir, "execute", {
        **payload, "finished_at": utcnow(), "status_counts_this_run": counts,
        "n_executed_on_disk": len(done_keys(exec_path, GEN_KEY))})
    if counts.get(HARNESS_ERROR):
        log(f"WARNING: {counts[HARNESS_ERROR]} candidates hit a harness error (the sandbox "
            "did not start); they are not outcomes and will be re-run on the next `execute`")
    log(f"execution complete -> {exec_path}")
