"""Exp 2 wiring tests: no network, no real checkpoints.

    python -m pytest tests/test_exp2_wiring.py -v
    python tests/test_exp2_wiring.py
"""

from __future__ import annotations

import json
import platform
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from exp1.common import ARMS, write_json_atomic
from exp1.watermark_setup import build_synthid_processor, ensure_watermark_config
from exp2.analysis import (auroc, complete_triples, detection_curve, exact_match,
                           executed_triples, filter_counts, fpr_threshold, kept_samples,
                           pass1_by_problem, pass1_summary, tpr_at)
from exp2.detect import GValueScorer, find_exp1_runs
from exp2.extract import build_program, extract_code, find_fences
from exp2.generate import sample_seed, seed_index, seed_table
from exp2.harness import run_program
from exp2.parity import check_exp2_parity
from exp2.problems import select_problems
from exp2.prompting import user_message
from exp2.settings import N_SAMPLES, PREFIX_LENGTHS, b2_seed_offset

VOCAB = 512


class StubTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def __len__(self) -> int:
        return VOCAB

    def __call__(self, text, add_special_tokens=False):
        class R:
            input_ids = [int(t) for t in text.split()]
        return R()


HUMANEVAL = {
    "problem_id": "humaneval-0", "source": "humaneval", "entry_point": "add",
    "prompt": "from typing import List\n\n\ndef add(a: int, b: int) -> int:\n"
              "    \"\"\"Add two ints.\n    >>> add(1, 2)\n    3\n    \"\"\"\n",
    "test": "\n\nMETADATA = {}\n\n\ndef check(candidate):\n    assert candidate(1, 2) == 3\n"
            "    assert candidate(-1, 1) == 0\n",
}
MBPP = {
    "problem_id": "mbpp-1", "source": "mbpp", "entry_point": "twice",
    "prompt": "Write a function that doubles a number.",
    "test_imports": ["import math"],
    "test_list": ["assert twice(2) == 4", "assert twice(0) == 0"],
}


def test_seed_pairing():
    n = N_SAMPLES
    for k in range(n):
        sa, la = sample_seed("A", k, n, "humaneval-3")
        sb1, lb1 = sample_seed("B1", k, n, "humaneval-3")
        sb2, lb2 = sample_seed("B2", k, n, "humaneval-3")
        assert sa == sb1 and la == lb1 == f"S{k + 1}"
        assert sb2 != sb1 and lb2 == f"S{n + k + 1}"
        assert sample_seed("A", k, n, "humaneval-4")[0] != sa
    assert len({sample_seed("A", k, n, "p")[0] for k in range(n)}) == n
    assert seed_index("B2", 0, 3) == b2_seed_offset(3) + 1 == 11      # offset never below 10
    assert seed_index("B2", 0, 12) == 13
    assert len(seed_table(10)) == 20 and len(set(seed_table(10).values())) == 20


def test_fence_extraction():
    code, info = extract_code("Sure:\n```python\ndef add(a, b):\n    return a + b\n```\nDone.", "add")
    assert code == "def add(a, b):\n    return a + b" and info["method"] == "fence_python_entry"
    assert info["closed"] is True

    text = "```python\nprint('demo')\n```\n\n```py\ndef add(a, b):\n    return a + b\n```"
    code, info = extract_code(text, "add")
    assert code.startswith("def add") and info["method"] == "fence_python_entry"
    code, info = extract_code(text, None)
    assert code == "print('demo')" and info["method"] == "fence_python"

    code, info = extract_code("```\nx = 1\n```", "f")
    assert code == "x = 1" and info["method"] == "fence_untagged"

    code, info = extract_code("```Python\ndef f():\n    return 1\n```", "f")
    assert info["method"] == "fence_python_entry"

    code, info = extract_code("```python\ndef add(a, b):\n    return a +", "add")
    assert code == "def add(a, b):\n    return a +" and info["closed"] is False

    code, info = extract_code("def add(a, b):\n    return a + b\n", "add")
    assert info["method"] == "raw_parses" and code.startswith("def add")

    for text in ("I cannot help with that.", "Use ```python``` blocks for code.", "", "   "):
        code, info = extract_code(text, "add")
        assert code is None and info["method"] == "none", text

    assert len(find_fences("a\n```json\n{}\n```\nb\n```python\nx=1\n```")) == 2


def test_program_assembly():
    program, mode = build_program(HUMANEVAL, "def add(a: int, b: int) -> int:\n    return a + b")
    assert mode == "prompt_then_code" and program.startswith("from typing import List")
    assert program.rstrip().endswith("check(add)")
    assert program.count("def add(") == 2          # the prompt's stub, then the model's def

    program, mode = build_program(HUMANEVAL, "    return a + b")   # body only
    assert mode == "prompt+code" and "def add(a: int, b: int) -> int:" in program
    assert program.count("def add(") == 1

    program, mode = build_program(MBPP, "def twice(x):\n    return 2 * x")
    assert mode == "code" and program.startswith("import math")
    assert program.rstrip().endswith("assert twice(0) == 0")



def test_harness_outcomes():
    ok = run_program(build_program(MBPP, "def twice(x):\n    return 2 * x")[0])
    assert ok["status"] == "passed", ok
    assert run_program(build_program(MBPP, "def twice(x):\n    return x")[0])["status"] == "failed"
    assert run_program("def twice(x:\n    return x\n")["status"] == "syntax_error"
    assert run_program("import sys\nsys.exit(0)\n")["status"] == "failed"
    assert run_program("import os\nos._exit(0)\n")["status"] == "failed"
    assert run_program("import subprocess\nsubprocess.run(['true'])\n")["status"] == "failed"
    assert run_program("import os\nos.system('true')\n")["status"] == "failed"

    slow = run_program("while True:\n    pass\n", timeout_s=1.0)
    assert slow["status"] == "timeout" and slow["elapsed_s"] < 4.0

    net = run_program("import socket\nsocket.create_connection(('127.0.0.1', 9), timeout=1)\n")
    assert net["status"] == "failed" and "sandbox" in net["stderr_tail"]
    net = run_program("import urllib.request\nurllib.request.urlopen('http://127.0.0.1:9/', timeout=1)\n")
    assert net["status"] == "failed"

    limits = ok["limits"]
    assert limits.get("RLIMIT_FSIZE") == "set"
    if platform.system() == "Linux":
        assert limits.get("RLIMIT_AS") == "set"
        big = run_program("x = bytearray(3 * 1024 * 1024 * 1024)\n", mem_mb=512)
        assert big["status"] in ("failed", "killed"), big


def test_program_assembly_runs_end_to_end():
    program, mode = build_program(HUMANEVAL, "    return a + b")
    assert mode == "prompt+code"
    assert run_program(program)["status"] == "passed"
    program, _ = build_program(HUMANEVAL, "def add(a, b):\n    return a - b")
    assert run_program(program)["status"] == "failed"

    helper = dict(HUMANEVAL, entry_point="twice_add",
                  prompt="def helper(x):\n    return 2 * x\n\n\ndef twice_add(a, b):\n"
                         "    \"\"\"helper(a) + helper(b)\"\"\"\n",
                  test="\ndef check(candidate):\n    assert candidate(1, 2) == 6\n")
    program, mode = build_program(helper, "def twice_add(a, b):\n    return helper(a) + helper(b)")
    assert mode == "prompt_then_code"
    assert run_program(program)["status"] == "passed"


def test_harness_error_is_not_an_outcome():
    from exp2.harness import HARNESS_ERROR, preflight
    broken = run_program("assert True\n", exec_prefix="false")      # sandbox never starts
    assert broken["status"] == HARNESS_ERROR and broken["limits"] == {}
    try:
        preflight(exec_prefix="false")
    except SystemExit as exc:
        assert "preflight" in str(exc)
    else:
        raise AssertionError("preflight accepted a sandbox that cannot run a trivial program")
    assert preflight()["status"] == "passed"


def _mean_detector(processor, ids: list[int], eos: int) -> float | None:
    """MarkLLM's MeanDetector on the full ids, written out longhand."""
    t = torch.tensor([ids])
    g = processor.compute_g_values(t).float()
    rep = processor.compute_context_repetition_mask(t)
    eos_mask = processor.compute_eos_token_mask(t, eos)[:, processor.ngram_len - 1:]
    mask = (rep * eos_mask).float()
    if mask.sum() == 0:
        return None
    return float((g * mask.unsqueeze(-1)).sum() / (g.shape[-1] * mask.sum()))


def test_prefix_scores_match_mean_detector_on_truncated_ids():
    with tempfile.TemporaryDirectory() as tmp:
        keys = Path(tmp) / "watermark_config.json"
        ensure_watermark_config(keys)
        tok = StubTokenizer()
        processor = build_synthid_processor(keys, tok, "cpu")
        scorer = GValueScorer(processor, tok, PREFIX_LENGTHS)
        rng = np.random.default_rng(3)
        ids = [int(x) for x in rng.integers(2, VOCAB, size=130)]
        ids[60:70] = ids[10:20]                                   # repeated contexts get masked
        scored = scorer.score_ids(ids)
        assert scored["n_tokens"] == 130
        assert abs(scored["score_full"] - _mean_detector(processor, ids, 1)) < 1e-6
        for L in PREFIX_LENGTHS:
            got = scored["prefix"][str(L)]
            if L > len(ids):
                assert got is None
                continue
            assert abs(got["score"] - _mean_detector(processor, ids[:L], 1)) < 1e-6, L
        assert scorer.score_ids([5, 6, 7])["score_full"] is None
        text_scored = scorer.score_text(" ".join(str(i) for i in ids))
        assert text_scored["ids"] == ids and text_scored["score_full"] == scored["score_full"]
        assert scorer.strip_trailing_special(ids + [1, 0, 1]) == ids     # eos/pad tail dropped
        assert scorer.strip_trailing_special([1, 5, 1, 6]) == [1, 5, 1, 6]


def test_fpr_threshold_and_tpr():
    rng = np.random.default_rng(0)
    null = rng.normal(0.5, 0.02, size=1000)
    wm = rng.normal(0.56, 0.02, size=500)
    for fpr in (0.01, 0.05):
        thr, achieved = fpr_threshold(null, fpr)
        assert achieved <= fpr + 1e-12
        assert np.mean(null > thr) == achieved
        assert 0.5 < tpr_at(wm, thr) <= 1.0
    thr, achieved = fpr_threshold([0.5] * 50, 0.01)          # fewer than 100 nulls: max null
    assert thr == 0.5 and achieved == 0.0
    assert abs(auroc([1, 2, 3], [0, 0, 0]) - 1.0) < 1e-12
    assert abs(auroc([1, 1], [1, 1]) - 0.5) < 1e-12
    assert 0.9 < auroc(wm, null) < 1.0


def _det_record(domain, arm, n_tokens, score):
    return {"domain": domain, "source_run": "r", "item_id": "p", "arm": arm, "sample_idx": 0,
            "n_tokens": n_tokens, "score_full": score,
            "prefix": {str(L): ({"score": score, "n_scored": L - 4} if n_tokens >= L else None)
                       for L in PREFIX_LENGTHS}}


def test_detection_curve_prefix_rule():
    recs = ([_det_record("code", "A", 60, 0.6)] * 10 + [_det_record("code", "A", 300, 0.6)] * 5
            + [_det_record("code", "B1", 60, 0.5)] * 10 + [_det_record("code", "B2", 500, 0.5)] * 10)
    rows = {r["prefix_length"]: r for r in detection_curve(recs)}
    assert rows[20]["n_wm"] == 15 and rows[20]["n_null"] == 20
    assert rows[100]["n_wm"] == 5 and rows[100]["n_null"] == 10
    assert rows[400]["n_wm"] == 0 and rows[400]["n_null"] == 10
    assert rows["full"]["n_wm"] == 15 and rows["full"]["tpr_fpr01"] == 1.0
    assert rows["full"]["achieved_fpr01"] == 0.0


def _gen(pid, arm, k, text, seed=1, watermarked=None):
    return {"problem_id": pid, "arm": arm, "sample_idx": k, "text": text, "seed": seed,
            "token_count": 5, "model": "m", "n_prompt_tokens": 7, "finish_reason": "eos",
            "watermarked": (arm == "A") if watermarked is None else watermarked}


def _exec(pid, arm, k, status, method="fence_python"):
    return {"problem_id": pid, "arm": arm, "sample_idx": k, "status": status,
            "extraction": {"method": "none" if status == "unparseable" else method,
                           "closed": True},
            "assembly": None if status == "unparseable" else "code",
            "extraction_rules": "exp2-extract-v2",
            "harness": {"timeout_s": 10.0, "mem_mb": 4096, "fsize_mb": 64, "exec_prefix": None,
                        "python_version": "3.11.4"}}


def _fixture():
    gens, execs = {}, {}
    outcomes = {
        # pid: {k: (A, B1, B2)}
        "p1": {0: ("passed", "passed", "failed"), 1: ("failed", "passed", "passed")},
        "p2": {0: ("passed", "unparseable", "passed"), 1: ("passed", "passed", "passed")},
        "p3": {0: ("unparseable", "failed", "failed"), 1: ("unparseable", "passed", "failed")},
    }
    for pid, per_k in outcomes.items():
        for k, statuses in per_k.items():
            for arm, status in zip(ARMS, statuses):
                text = f"{pid}-{k}-same" if arm in ("A", "B1") and pid != "p3" else f"{pid}-{k}-{arm}"
                seed = 11 if arm in ("A", "B1") else 22
                gens[(pid, arm, k)] = _gen(pid, arm, k, text, seed=seed)
                execs[(pid, arm, k)] = _exec(pid, arm, k, status)
    gens[("p4", "A", 0)] = _gen("p4", "A", 0, "orphan")          # incomplete triple
    return gens, execs


def test_pass1_aggregation_with_symmetric_drop():
    gens, execs = _fixture()
    triples = complete_triples(gens)
    assert triples == [("p1", 0), ("p1", 1), ("p2", 0), ("p2", 1), ("p3", 0), ("p3", 1)]
    executed = executed_triples(triples, execs)
    kept = kept_samples(executed, execs)
    assert kept == [("p1", 0), ("p1", 1), ("p2", 1)]              # p2/0 and both p3 dropped

    pass1 = pass1_by_problem(kept, execs)
    assert pass1["A"] == {"p1": 0.5, "p2": 1.0}
    assert pass1["B1"] == {"p1": 1.0, "p2": 1.0}
    assert pass1["B2"] == {"p1": 0.5, "p2": 1.0}
    summary = pass1_summary(pass1)
    assert summary["n_problems"] == 2
    assert abs(summary["delta_wm"]["mean"] - (-0.25)) < 1e-12
    assert abs(summary["delta_ctrl"]["mean"] - (-0.25)) < 1e-12
    assert summary["bound"]["status"] == "FAIL"

    strict = pass1_by_problem(executed, execs, unparseable_is_fail=True)
    assert strict["A"]["p3"] == 0.0 and strict["B1"]["p3"] == 0.5
    assert strict["B1"]["p2"] == 0.5

    f = filter_counts(gens, execs, triples, executed, kept)
    assert f["n_incomplete_triples"] == 1 and f["n_dropped_samples"] == 3
    assert f["n_unexecuted_triples"] == 0
    assert f["n_kept_samples"] == 3 and f["n_problems_kept"] == 2
    assert f["per_arm"]["A"]["n_unparseable"] == 2 and f["per_arm"]["B1"]["n_unparseable"] == 1

    # a harness error is not an outcome: the triple is unexecuted, not failed
    broken = dict(execs)
    broken[("p1", "B2", 0)] = _exec("p1", "B2", 0, "harness_error")
    assert executed_triples(triples, broken) == [("p1", 1), ("p2", 0), ("p2", 1), ("p3", 0), ("p3", 1)]
    assert filter_counts(gens, broken, triples, executed_triples(triples, broken),
                         kept_samples(executed_triples(triples, broken), broken))[
        "n_unexecuted_triples"] == 1


def test_exact_match_rates():
    gens, execs = _fixture()
    triples = complete_triples(gens)
    em = exact_match(triples, lambda pid, arm, k: gens[(pid, arm, k)]["text"])
    assert em["n"] == 6 and em["A_eq_B1_count"] == 4 and em["B1_eq_B2_count"] == 0
    assert abs(em["A_eq_B1"] - 4 / 6) < 1e-12 and em["B1_eq_B2"] == 0.0
    assert exact_match([], lambda *a: "")["A_eq_B1"] is None


def _kwargs():
    return {arm: {"do_sample": True, "temperature": 1.0, "top_k": 0, "top_p": 1.0,
                  "num_beams": 1, "max_new_tokens": 512, "min_length": 0,
                  "repetition_penalty": 1.0, "no_repeat_ngram_size": 0, "pad_token_id": 0}
            for arm in ARMS}


WARPERS = ["Float32LogitsProcessor", "TemperatureLogitsWarper", "TopKLogitsWarper",
           "TopPLogitsWarper"]


def _run_config(**overrides):
    cfg = {"stages": {
        "generate": {"generation_kwargs_by_arm": _kwargs(),
                     "logits_processor_chain_by_arm": {"A": WARPERS + ["SynthIDLogitsProcessor"],
                                                       "B1": list(WARPERS), "B2": list(WARPERS)},
                     "decode_by_arm": {a: {"skip_special_tokens": True, "strip": True}
                                       for a in ARMS},
                     "watermark_config_sha256": "k1"},
        "execute": {"timeout_s": 10.0, "mem_mb": 4096, "fsize_mb": 64, "exec_prefix": None,
                    "python_version": "3.11.4", "extraction_rules": "exp2-extract-v2"},
        "detect": {"detector": "mean", "tokenizer_model": "m", "watermark_config_sha256": "k1"},
    }}
    for key, value in overrides.items():
        cfg["stages"][key] = value
    return cfg


def _detections(n=3, sha="k1"):
    return [{"domain": "code", "arm": a, "detector": "mean", "tokenizer_model": "m",
             "watermark_config_sha256": sha} for a in ARMS for _ in range(n)]


def test_exp2_parity_is_pass_or_fail_never_incomplete():
    gens, execs = _fixture()
    triples = complete_triples(gens)
    healthy = check_exp2_parity(_run_config(), gens, triples, execs, _detections())
    assert healthy["overall"] == "PASS", healthy["checks"]
    assert all(c["status"] in ("PASS", "FAIL") for c in healthy["checks"].values())

    unrecorded = check_exp2_parity({"stages": {"generate": {}}}, gens, triples, execs,
                                   _detections())
    assert unrecorded["overall"] == "FAIL"
    for name in ("generation_kwargs", "logits_processors", "decode_rules"):
        assert unrecorded["checks"][name]["status"] == "FAIL"
        assert "not recorded" in unrecorded["checks"][name]["detail"]
    assert "INCOMPLETE" not in {c["status"] for c in unrecorded["checks"].values()}

    tampered = dict(gens)
    tampered[("p1", "A", 0)] = _gen("p1", "A", 0, "x", seed=99)
    assert check_exp2_parity(_run_config(), tampered, triples, execs, _detections())[
        "checks"]["sample_parity"]["status"] == "FAIL"

    mixed = dict(execs)
    mixed[("p1", "A", 0)] = {**execs[("p1", "A", 0)],
                             "harness": {**execs[("p1", "A", 0)]["harness"], "timeout_s": 5.0}}
    assert check_exp2_parity(_run_config(), gens, triples, mixed, _detections())[
        "checks"]["harness_parity"]["status"] == "FAIL"

    partial = dict(execs)
    del partial[("p2", "B2", 1)]                       # execute stopped before this arm
    incomplete = check_exp2_parity(_run_config(), gens, triples, partial, _detections())
    assert incomplete["checks"]["harness_parity"]["status"] == "FAIL"
    assert "lack an outcome" in incomplete["checks"]["harness_parity"]["detail"]
    orphan = dict(execs)
    orphan[("p4", "A", 0)] = _exec("p4", "A", 0, "passed")   # an orphan arm is not a defect
    assert check_exp2_parity(_run_config(), gens, triples, orphan, _detections())[
        "checks"]["harness_parity"]["status"] == "PASS"

    wrong_keys = check_exp2_parity(_run_config(), gens, triples, execs, _detections(sha="k2"))
    assert wrong_keys["checks"]["detector_parity"]["status"] == "FAIL"

    empty = check_exp2_parity(_run_config(), gens, triples, {}, [])
    assert empty["checks"]["harness_parity"]["status"] == "FAIL"
    assert empty["checks"]["detector_parity"]["status"] == "FAIL"
    assert empty["overall"] == "FAIL"


def test_smoke_selection_and_prompts():
    payload = {"problems": [{"problem_id": f"humaneval-{i}", "source": "humaneval"}
                            for i in range(10)]
               + [{"problem_id": f"mbpp-{i}", "source": "mbpp"} for i in range(10)]}
    picked = select_problems(payload, smoke=True, n_smoke=5)
    assert [p["problem_id"] for p in picked] == ["humaneval-0", "humaneval-1", "humaneval-2",
                                                 "mbpp-0", "mbpp-1"]
    assert len(select_problems(payload, smoke=False, n_smoke=5)) == 20

    msg = user_message(HUMANEVAL)
    assert "```python\nfrom typing import List" in msg and "one ```python code block" in msg
    msg = user_message(MBPP)
    assert "import math\nassert twice(2) == 4" in msg and MBPP["prompt"] in msg


def test_find_exp1_runs_matches_model_and_skips_exp2_dirs():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)

        def make(name, model, experiment=None, smoke=False):
            d = root / name
            d.mkdir()
            cfg = {"stages": {"generate": {"model": model, "smoke": smoke,
                                           "watermark_config_sha256": "k1"}}}
            if experiment:
                cfg["experiment"] = experiment
            write_json_atomic(d / "run_config.json", cfg)
            (d / "generations.jsonl").write_text('{"prompt_id": "p", "arm": "A", "text": "t"}\n')
            return d

        make("exp1-m", "m")
        make("exp1-other", "other")
        make("exp2-m", "m", experiment="exp2")
        make("exp1-smoke", "m", smoke=True)
        this_out = make("exp2-this", "m", experiment="exp2")
        found = find_exp1_runs(str(root / "*"), "m", this_out, smoke=False)
        assert [Path(r["dir"]).name for r in found] == ["exp1-m"]
        # the prose key does not depend on how the glob was spelt
        absolute = find_exp1_runs(str(root.resolve() / "*"), "m", this_out, smoke=False)
        assert absolute[0]["source_run"] == found[0]["source_run"] == "exp1:exp1-m"
        found = find_exp1_runs(str(root / "*"), "m", this_out, smoke=True)
        assert [Path(r["dir"]).name for r in found] == ["exp1-m", "exp1-smoke"]


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL  {fn.__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    sys.exit(1 if failures else 0)
