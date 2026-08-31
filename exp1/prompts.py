from __future__ import annotations

import json
from pathlib import Path

from .common import (DEFAULT_PROMPTS_PATH, hf_token, log, repo_revision,
                     sha256_file, utcnow, write_json_atomic)

DOLLY_REPO = "databricks/databricks-dolly-15k"
CATEGORY = "open_qa"
N_PROMPTS = 500
PROMPT_SAMPLE_SEED = 20240517


def build_prompts(path: str | Path = DEFAULT_PROMPTS_PATH,
                  n: int = N_PROMPTS,
                  seed: int = PROMPT_SAMPLE_SEED,
                  revision: str | None = None) -> dict:
    import numpy as np
    from datasets import load_dataset

    log(f"loading {DOLLY_REPO} (category={CATEGORY})")
    ds = load_dataset(DOLLY_REPO, split="train", revision=revision, token=hf_token())

    rows = [
        {"row_index": i, "instruction": r["instruction"], "context": r.get("context", "")}
        for i, r in enumerate(ds)
        if r["category"] == CATEGORY
    ]
    log(f"{len(rows)} rows in category '{CATEGORY}'")

    rows = [r for r in rows if not (r["context"] or "").strip()]
    if len(rows) < n:
        raise SystemExit(f"only {len(rows)} usable {CATEGORY} rows, need {n}")

    rng = np.random.default_rng(seed)
    picked = rng.choice(len(rows), size=n, replace=False)

    prompts = [
        {
            "prompt_id": f"dolly-{rows[int(j)]['row_index']}",
            "text": rows[int(j)]["instruction"].strip(),
            "source_row_index": int(rows[int(j)]["row_index"]),
        }
        for j in picked
    ]

    payload = {
        "created_at": utcnow(),
        "dataset": DOLLY_REPO,
        "dataset_revision": revision or repo_revision(DOLLY_REPO, repo_type="dataset"),
        "category": CATEGORY,
        "n_prompts": n,
        "sample_seed": seed,
        "sampler": "numpy.random.default_rng(seed).choice(n_category_rows, size=n, replace=False)",
        "n_category_rows": len(rows),
        "prompts": prompts,
    }
    write_json_atomic(path, payload)
    log(f"wrote {path} with {n} prompts (sha256={sha256_file(path)[:16]}...)")
    return payload


def load_prompts(path: str | Path = DEFAULT_PROMPTS_PATH,
                 create_if_missing: bool = True,
                 revision: str | None = None) -> dict:
    path = Path(path)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        log(f"loaded {len(payload['prompts'])} prompts from {path} "
            f"(sha256={sha256_file(path)[:16]}...)")
        return payload
    if not create_if_missing:
        raise SystemExit(f"{path} does not exist and creation was not requested")
    return build_prompts(path, revision=revision)


def select_prompts(payload: dict, smoke: bool, n_smoke: int = 10) -> list[dict]:

    prompts = payload["prompts"]
    return prompts[:n_smoke] if smoke else prompts
