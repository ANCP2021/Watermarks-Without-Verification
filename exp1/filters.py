
from __future__ import annotations

import re
from pathlib import Path

from .common import ARMS, log, read_jsonl, write_json_atomic

MIN_CHARS = 3
REFUSAL_SCAN_CHARS = 400

REFUSAL_PATTERNS = [
    r"\bi (?:can ?not|can't|cannot|am unable to|'m unable to|am not able to|'m not able to)\b[^.]{0,60}\b(?:help|assist|provide|answer|comply|fulfil|fulfill|complete|do that|with that)\b",
    r"\bi (?:must|have to|will) (?:decline|refuse)\b",
    r"\bi (?:won't|will not) (?:be able to |)(?:help|assist|provide|answer|comply)\b",
    r"\bas an ai(?: language)?(?: model)?\b[^.]{0,80}\b(?:can ?not|can't|cannot|unable)\b",
    r"\bi'?m sorry[^.]{0,40}\b(?:can ?not|can't|cannot|unable|not able)\b",
    r"\bi (?:do not|don't) (?:have the ability|feel comfortable)\b",
    r"\bsorry, (?:but )?i (?:can ?not|can't|cannot)\b",
]
_REFUSAL_RE = re.compile("|".join(REFUSAL_PATTERNS), re.IGNORECASE)


def classify(text: str) -> str | None:
    stripped = (text or "").strip()
    if len(stripped) < MIN_CHARS:
        return "empty"
    if _REFUSAL_RE.search(stripped[:REFUSAL_SCAN_CHARS]):
        return "refusal"
    return None


def load_valid_generations(out_dir: str | Path, write_report: bool = True) -> tuple[dict, dict]:
    out_dir = Path(out_dir)
    records = read_jsonl(out_dir / "generations.jsonl")
    if not records:
        raise SystemExit(f"no generations found in {out_dir / 'generations.jsonl'} "
                         "-- run `generate` first")

    by_prompt: dict[str, dict[str, dict]] = {}
    n_duplicates = 0
    for rec in records:
        slot = by_prompt.setdefault(rec["prompt_id"], {})
        if rec["arm"] in slot:
            n_duplicates += 1 
            continue
        slot[rec["arm"]] = rec

    incomplete = sorted(pid for pid, arms in by_prompt.items()
                        if any(a not in arms for a in ARMS))
    incomplete_set = set(incomplete)

    dropped: list[dict] = []
    kept: dict[str, dict[str, dict]] = {}
    for pid, arms in by_prompt.items():
        if pid in incomplete_set:
            continue
        reasons = {a: classify(arms[a]["text"]) for a in ARMS}
        failed = {a: r for a, r in reasons.items() if r}
        if failed:
            dropped.append({
                "prompt_id": pid,
                "failed_arms": failed,
                "preview": {a: (arms[a]["text"] or "")[:160] for a in failed},
            })
        else:
            kept[pid] = {a: arms[a] for a in ARMS}

    per_arm: dict[str, dict[str, int]] = {a: {"empty": 0, "refusal": 0} for a in ARMS}
    for d in dropped:
        for arm, reason in d["failed_arms"].items():
            per_arm[arm][reason] += 1

    report = {
        "n_prompts_with_generations": len(by_prompt),
        "n_incomplete_triples": len(incomplete),
        "incomplete_prompt_ids": incomplete[:50],
        "n_duplicate_records_ignored": n_duplicates,
        "n_dropped": len(dropped),
        "n_kept": len(kept),
        "dropped_by_arm_and_reason": per_arm,
        "dropped_prompt_ids": sorted(d["prompt_id"] for d in dropped),
        "filter_rules": {
            "min_chars": MIN_CHARS,
            "refusal_scan_chars": REFUSAL_SCAN_CHARS,
            "refusal_patterns": REFUSAL_PATTERNS,
            "symmetric": "a prompt is dropped if ANY of its three outputs fails",
        },
    }

    log(f"filter: {len(kept)} prompts kept, {len(dropped)} dropped "
        f"({sum(v['empty'] for v in per_arm.values())} empty, "
        f"{sum(v['refusal'] for v in per_arm.values())} refusal across arms), "
        f"{len(incomplete)} incomplete triples skipped")
    if n_duplicates:
        log(f"WARNING: ignored {n_duplicates} duplicate (prompt_id, arm) records")

    if write_report:
        write_json_atomic(out_dir / "filter_report.json",
                          {**report, "dropped_detail": dropped[:200]})
    return kept, report
