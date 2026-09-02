from __future__ import annotations

import builtins
import json
import re
from collections import Counter
from pathlib import Path

from exp1.common import REPO_ROOT, log, repo_revision, sha256_file, utcnow, write_json_atomic

from .settings import (HUMANEVAL_REPO, HUMANEVAL_SPLIT, MBPP_CONFIG, MBPP_N, MBPP_REPO,
                       MBPP_SPLIT)

DEFAULT_PROBLEMS_PATH = REPO_ROOT / "problems.json"
SOURCES = ("humaneval", "mbpp")

_CALL_RE = re.compile(r"(?<![\w.])([A-Za-z_]\w*)\s*\(")
_BUILTIN_NAMES = set(dir(builtins)) | {"math", "isclose", "np", "numpy", "re"}


def guess_entry_point(tests: list[str]) -> str | None:
    counts: Counter[str] = Counter()
    for line in tests:
        for name in _CALL_RE.findall(line):
            if name not in _BUILTIN_NAMES:
                counts[name] += 1
    return counts.most_common(1)[0][0] if counts else None


def build_problems(path: str | Path = DEFAULT_PROBLEMS_PATH,
                   n_mbpp: int = MBPP_N,
                   humaneval_revision: str | None = None,
                   mbpp_revision: str | None = None) -> dict:
    from datasets import load_dataset

    log(f"loading {HUMANEVAL_REPO} ({HUMANEVAL_SPLIT})")
    he = load_dataset(HUMANEVAL_REPO, split=HUMANEVAL_SPLIT, revision=humaneval_revision)
    problems = []
    for row in he:
        number = row["task_id"].split("/")[-1]
        problems.append({
            "problem_id": f"humaneval-{number}",
            "source": "humaneval",
            "source_id": row["task_id"],
            "prompt": row["prompt"],
            "entry_point": row["entry_point"],
            "test": row["test"],
            "canonical_solution": row["canonical_solution"],
        })
    n_he = len(problems)

    log(f"loading {MBPP_REPO} ({MBPP_CONFIG}/{MBPP_SPLIT})")
    mb = load_dataset(MBPP_REPO, MBPP_CONFIG, split=MBPP_SPLIT, revision=mbpp_revision)
    rows = sorted((dict(r) for r in mb), key=lambda r: int(r["task_id"]))
    rows = rows[:n_mbpp]
    for row in rows:
        problems.append({
            "problem_id": f"mbpp-{int(row['task_id'])}",
            "source": "mbpp",
            "source_id": int(row["task_id"]),
            "prompt": row["prompt"],
            "entry_point": guess_entry_point(list(row["test_list"])),
            "test_imports": list(row["test_imports"]),
            "test_list": list(row["test_list"]),
            "canonical_solution": row["code"],
        })

    payload = {
        "created_at": utcnow(),
        "sources": {
            "humaneval": {"dataset": HUMANEVAL_REPO, "split": HUMANEVAL_SPLIT,
                          "revision": humaneval_revision
                          or repo_revision(HUMANEVAL_REPO, repo_type="dataset"),
                          "n": n_he, "selection": "all"},
            "mbpp": {"dataset": MBPP_REPO, "config": MBPP_CONFIG, "split": MBPP_SPLIT,
                     "revision": mbpp_revision or repo_revision(MBPP_REPO, repo_type="dataset"),
                     "n": len(rows), "n_in_split": len(mb),
                     "selection": f"first {n_mbpp} by ascending task_id (all if fewer)"},
        },
        "n_problems": len(problems),
        "problems": problems,
    }
    write_json_atomic(path, payload)
    log(f"wrote {path} with {len(problems)} problems (sha256={sha256_file(path)[:16]}...)")
    return payload


def load_problems(path: str | Path = DEFAULT_PROBLEMS_PATH,
                  create_if_missing: bool = True,
                  humaneval_revision: str | None = None,
                  mbpp_revision: str | None = None) -> dict:
    path = Path(path)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        log(f"loaded {len(payload['problems'])} problems from {path} "
            f"(sha256={sha256_file(path)[:16]}...)")
        return payload
    if not create_if_missing:
        raise SystemExit(f"{path} does not exist and creation was not requested")
    return build_problems(path, humaneval_revision=humaneval_revision,
                          mbpp_revision=mbpp_revision)


def select_problems(payload: dict, smoke: bool, n_smoke: int) -> list[dict]:
    problems = payload["problems"]
    if not smoke:
        return list(problems)
    n_he = (n_smoke + 1) // 2
    he = [p for p in problems if p["source"] == "humaneval"][:n_he]
    mb = [p for p in problems if p["source"] == "mbpp"][: n_smoke - len(he)]
    picked = he + mb
    if len(picked) < n_smoke:
        rest = [p for p in problems if p not in picked]
        picked += rest[: n_smoke - len(picked)]
    return picked
