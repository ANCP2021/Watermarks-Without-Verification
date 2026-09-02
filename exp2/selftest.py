"""Harness validation: every canonical solution must pass its own tests."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

from exp1.common import Timer, log

from .extract import build_program
from .harness import run_program
from .problems import load_problems
from .settings import EXEC_FSIZE_MB, EXEC_WORKERS


def run_selftest(args) -> int:
    payload = load_problems(args.problems, create_if_missing=False)
    problems = payload["problems"]
    workers = int(args.workers or EXEC_WORKERS)
    log(f"selftest: {len(problems)} canonical solutions, {workers} workers, "
        f"timeout {args.timeout}s")

    timer = Timer()
    failures = []
    modes: dict[str, int] = {}
    limits_seen: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {}
        for p in problems:
            # HumanEval canonical solutions are bodies only, so they exercise prompt+code
            program, mode = build_program(p, p["canonical_solution"])
            modes[mode] = modes.get(mode, 0) + 1
            futures[pool.submit(run_program, program, float(args.timeout), int(args.mem_mb),
                                EXEC_FSIZE_MB, args.exec_prefix)] = p["problem_id"]
        for i, fut in enumerate(as_completed(futures), start=1):
            pid = futures[fut]
            result = fut.result()
            limits_seen = result.get("limits") or limits_seen
            if result["status"] != "passed":
                failures.append((pid, result["status"], result["stderr_tail"][-200:]))
            if i % 100 == 0 or i == len(futures):
                log(f"  {i}/{len(futures)}  eta {timer.eta(i, len(futures))}")

    print(f"\nselftest: {len(problems) - len(failures)}/{len(problems)} canonical solutions "
          f"passed  (assembly modes: {modes}; limits: {limits_seen})")
    for pid, status, tail in failures:
        print(f"  FAIL {pid}: {status}  {tail!r}")
    return 1 if failures else 0
