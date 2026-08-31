"""Wiring tests that need no network and no real checkpoints.

    python -m pytest tests/test_wiring.py -v
    python tests/test_wiring.py            # same checks, no pytest needed
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from exp1.common import derive_seed, seed_everything
from exp1.filters import classify, load_valid_generations
from exp1.modeling import build_processor_chain, generation_kwargs
from exp1.scoring_judge import decide, parse_verdict
from exp1.stats import bootstrap_mean_ci, ci_within
from exp1.watermark_setup import (build_synthid_processor, ensure_watermark_config, reset_synthid_state)

VOCAB = 512


class StubTokenizer:
    pad_token_id = 0
    eos_token_id = None

    def __len__(self) -> int:
        return VOCAB


def tiny_model():
    config = GPT2Config(vocab_size=VOCAB, n_positions=1024, n_embd=32, n_layer=2,
                        n_head=2, bos_token_id=0, eos_token_id=0)
    torch.manual_seed(0)
    model = GPT2LMHeadModel(config).eval()
    model.generation_config.eos_token_id = None
    model.config.eos_token_id = None
    return model


def _keys_file(tmp: Path) -> Path:
    path = tmp / "watermark_config.json"
    ensure_watermark_config(path)
    return path


def _generate(model, chain, prompt_ids, seed, max_new_tokens=128):
    seed_everything(seed)
    with torch.inference_mode():
        return model.generate(input_ids=prompt_ids, logits_processor=chain,
                              **generation_kwargs(StubTokenizer(), max_new_tokens))



def test_keys_written_once():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "watermark_config.json"
        first = ensure_watermark_config(path)
        mtime = path.stat().st_mtime_ns
        second = ensure_watermark_config(path)
        assert first["keys"] == second["keys"]
        assert path.stat().st_mtime_ns == mtime, "config was rewritten on reload"
        assert first["ngram_len"] == 5
        assert first["watermark_mode"] == "non-distortionary"
        assert len(first["keys"]) == 30


def test_seed_derivation_is_stable_and_distinct():
    assert derive_seed(1234, "dolly-7") == derive_seed(1234, "dolly-7")
    assert derive_seed(1234, "dolly-7") != derive_seed(5678, "dolly-7")
    assert derive_seed(1234, "dolly-7") != derive_seed(1234, "dolly-8")


def test_chain_is_the_only_warping():
    model = tiny_model()
    prompt = torch.tensor([[1, 2, 3, 4, 5]])
    chain = build_processor_chain(None)
    chain[2].top_k = 1  # TopKLogitsWarper
    a = _generate(model, chain, prompt, seed=11, max_new_tokens=32)
    b = _generate(model, chain, prompt, seed=99, max_new_tokens=32)
    assert torch.equal(a, b), "top_k=1 should be deterministic regardless of seed"


def test_same_seed_reproduces_generation():
    model = tiny_model()
    prompt = torch.tensor([[1, 2, 3, 4, 5]])
    chain = build_processor_chain(None)
    a = _generate(model, chain, prompt, seed=1234, max_new_tokens=48)
    b = _generate(model, chain, prompt, seed=1234, max_new_tokens=48)
    c = _generate(model, chain, prompt, seed=4321, max_new_tokens=48)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_watermark_shifts_g_values():
    """Watermarked text must carry a mean g-value above the 0.5 null."""
    with tempfile.TemporaryDirectory() as tmp:
        keys = _keys_file(Path(tmp))
        model = tiny_model()
        processor = build_synthid_processor(keys, StubTokenizer(), "cpu")

        wm_chain = build_processor_chain(processor)
        plain_chain = build_processor_chain(None)

        wm_scores, plain_scores = [], []
        for i in range(4):
            prompt = torch.tensor([[1, 2, 3, 4, 5 + i]])
            reset_synthid_state(processor)
            wm = _generate(model, wm_chain, prompt, seed=100 + i, max_new_tokens=192)
            plain = _generate(model, plain_chain, prompt, seed=100 + i, max_new_tokens=192)
            for ids, bucket in ((wm, wm_scores), (plain, plain_scores)):
                new = ids[:, prompt.shape[1]:]
                g = processor.compute_g_values(new)
                mask = processor.compute_context_repetition_mask(new)
                bucket.append(float((g.float() * mask.unsqueeze(-1)).sum()
                                    / (g.shape[-1] * mask.sum())))

        wm_mean = sum(wm_scores) / len(wm_scores)
        plain_mean = sum(plain_scores) / len(plain_scores)
        assert wm_mean > 0.53, f"watermark not detectable: mean g-value {wm_mean:.4f}"
        assert abs(plain_mean - 0.5) < 0.05, f"unwatermarked baseline off: {plain_mean:.4f}"
        assert wm_mean > plain_mean + 0.02


def test_state_reset_makes_generations_independent():
    """Without the reset, a second generation inherits the first one's context."""
    with tempfile.TemporaryDirectory() as tmp:
        keys = _keys_file(Path(tmp))
        model = tiny_model()
        processor = build_synthid_processor(keys, StubTokenizer(), "cpu")
        chain = build_processor_chain(processor)
        prompt = torch.tensor([[1, 2, 3, 4, 5]])

        reset_synthid_state(processor)
        first = _generate(model, chain, prompt, seed=7, max_new_tokens=64)
        reset_synthid_state(processor)
        again = _generate(model, chain, prompt, seed=7, max_new_tokens=64)
        assert torch.equal(first, again), "reset generations must be identical"

        stale = _generate(model, chain, prompt, seed=7, max_new_tokens=64)  # no reset
        assert not torch.equal(first, stale), (
            "carrying watermark state across prompts should change the output -- "
            "if this fails the reset is not doing anything")


def test_filter_classification():
    assert classify("") == "empty"
    assert classify("   \n ") == "empty"
    assert classify("I cannot help with that request.") == "refusal"
    assert classify("I'm sorry, but I can't provide that information.") == "refusal"
    assert classify("As an AI language model, I cannot browse the web.") == "refusal"
    # narrow on purpose: these are real answers, not refusals
    assert classify("I'm sorry to hear that. The capital of France is Paris.") is None
    assert classify("The answer cannot be determined from the question alone.") is None
    assert classify("Yes.") is None


def test_filter_is_symmetric_and_drops_whole_triples():
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        rows = []
        for pid, texts in {
            "p1": ("good A text", "good B1 text", "good B2 text"),
            "p2": ("", "good B1 text", "good B2 text"),                    # empty in A
            "p3": ("good A", "I cannot assist with that.", "good B2"),     # refusal in B1
            "p4": ("good A", "good B1", "good B2"),
        }.items():
            for arm, text in zip(("A", "B1", "B2"), texts):
                rows.append({"prompt_id": pid, "arm": arm, "text": text,
                             "seed": 1, "token_count": 3, "model": "m", "timestamp": "t"})
        rows.append({"prompt_id": "p5", "arm": "A", "text": "orphan", "seed": 1,
                     "token_count": 1, "model": "m", "timestamp": "t"})  # incomplete
        with open(out / "generations.jsonl", "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")

        kept, report = load_valid_generations(out, write_report=False)
        assert set(kept) == {"p1", "p4"}
        assert report["n_dropped"] == 2
        assert report["n_incomplete_triples"] == 1
        assert report["dropped_by_arm_and_reason"]["A"]["empty"] == 1
        assert report["dropped_by_arm_and_reason"]["B1"]["refusal"] == 1


def test_torn_final_line_is_survivable():
    from exp1.common import read_jsonl
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "x.jsonl"
        path.write_text('{"a": 1}\n{"a": 2}\n{"a": 3, "b": ')
        assert [r["a"] for r in read_jsonl(path)] == [1, 2]


def test_judge_two_order_rule():
    assert parse_verdict("1") == "1"
    assert parse_verdict("Response 2 is better") == "2"
    assert parse_verdict(" Tie ") == "tie"
    assert parse_verdict("hmm") == "unparsed"
    # A first in order 1, second in order 2
    assert decide("1", "2") == "A"      # both orders picked A
    assert decide("2", "1") == "B1"     # both orders picked B1
    assert decide("1", "1") == "tie"    # judge just prefers position 1
    assert decide("tie", "1") == "tie"
    assert decide("1", "unparsed") == "tie"


def test_bootstrap_ci():
    stat = bootstrap_mean_ci([0.5] * 100)
    assert stat["mean"] == 0.5 and stat["ci_low"] == 0.5 and stat["ci_high"] == 0.5
    stat = bootstrap_mean_ci(list(range(100)), n_resamples=2000, seed=1)
    assert stat["ci_low"] < stat["mean"] < stat["ci_high"]
    assert bootstrap_mean_ci([0.5] * 50, seed=1) == bootstrap_mean_ci([0.5] * 50, seed=1)
    assert ci_within({"ci_low": -1.0, "ci_high": 1.0}, -5.0, 5.0)
    assert not ci_within({"ci_low": -6.0, "ci_high": 1.0}, -5.0, 5.0)


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
