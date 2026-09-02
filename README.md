# Watermarks-Without-Verification

## Experiment 1 - does SynthID-Text cost prose quality?

One resumable CLI pipeline, run once per generator model, that measures whether Google's SynthID-Text watermark degrades generation quality. Three arms per prompt isolate the watermark's effect from ordinary sampling noise:

| Arm | Watermark | Sampling seed | Role |
|-----|-----------|---------------|------|
| `A`  | SynthID (non-distortionary) | S1 | treatment |
| `B1` | none | S1 | paired control — same randomness stream as A |
| `B2` | none | S2 | sampling-noise floor |

The watermark comes from the vendored [MarkLLM](MarkLLM/) implementation (`watermark.synthid`), in its default non-distortionary configuration.

### Setup

```bash
pip install -r requirements-exp1.txt
export HF_TOKEN=...          # both generator models are gated
```

### Running

```bash
# generation
python run_exp1.py generate --model google/gemma-2-9b-it              --out runs/gemma2-9b
python run_exp1.py generate --model meta-llama/Llama-3.1-8B-Instruct  --out runs/llama31-8b

# scoring
python run_exp1.py ppl    --out runs/gemma2-9b   # --rescore to redo an existing ppl.jsonl
python run_exp1.py sim    --out runs/gemma2-9b
python run_exp1.py judge  --out runs/gemma2-9b
python run_exp1.py report --out runs/gemma2-9b
```

```bash
python run_exp1.py all --model Qwen/Qwen2.5-0.5B-Instruct --out runs/smoke --smoke
python tests/test_wiring.py     # 17 offline checks, no downloads
```

### Write-once shared artefacts

Two files at the repo root are created on first use and then only ever read. Both are tracked in git; both are hashed into every `run_config.json`.

| File | Contents | Why it must not be regenerated |
|------|----------|--------------------------------|
| `prompts.json` | 500 `open_qa` prompts from `databricks/databricks-dolly-15k`, sampled once with a fixed RNG seed | both models, and every later experiment, must see the identical prompt set |
| `watermark_config.json` | SynthID keys (depth 30), `ngram_len` 5, non-distortionary | the same keys are needed by the later code experiment and by detection scoring (mean g-value); new keys would make existing text undetectable |

`watermark_config.json` has no regenerate flag by design, replacing the keys requires deleting the file by hand. Both files can also be created explicitly:

```bash
python run_exp1.py keys
python run_exp1.py prompts
```

### Sampling

Identical for every arm: `do_sample=True, temperature=0.7, top_k=40, top_p=0.95, max_new_tokens=512`. 

### Restartability

| Stage | File | Skip key |
|-------|------|----------|
| `generate` | `generations.jsonl` | `(prompt_id, arm)` |
| `ppl` | `ppl.jsonl` | `prompt_id` |
| `sim` | `sim.jsonl` | `prompt_id` |
| `judge` | `judge.jsonl` | `prompt_id` |


### Scoring

**`ppl`**: mean per-token NLL of each output under `Qwen/Qwen2.5-7B`. The scorer sees `"<prompt>\n\n<response>"` and averages over the response tokens only: the prompt is conditioned on but contributes nothing to the mean, and the first response token is scored rather than skipped. `--rescore` archives an existing `ppl.jsonl` and scores every prompt again, which is required whenever the conditioning or the context template changes.

The reported statistic is the per-prompt **difference** of mean NLLs, in nats:

| Statistic | Definition | Role |
|-----------|------------|------|
| `d_wm`   | `NLL_A - NLL_B1`  | watermark effect |
| `d_ctrl` | `NLL_B2 - NLL_B1` | sampling-noise floor; its CI should cover 0 |

Each is reported as a bootstrap mean with a 95% CI, plus the percent equivalent `100 * (exp(mean d) - 1)`, with median and 10%-trimmed mean as robustness rows. Not a mean of per-prompt ratios: `PPL_A/PPL_B1 - 1` is bounded below by -100% and unbounded above, so averaging it shifts a null comparison upwards — the off-vs-off control read +8% on the Gemma run under the old statistic.

**`sim`**: cosine similarity under `BAAI/bge-large-en-v1.5` (CLS pooling, L2-normalised, fp32). `sim(A, B1)` is reported against the control distribution `sim(B1, B2)`.

**`judge`**: blinded pairwise preference between A and B1 under `Qwen/Qwen2.5-7B-Instruct` at temperature 0, ties allowed. 

**Exact-match rates** (`report`): the fraction of kept prompts whose A and B1 responses are byte-for-byte identical, next to the B1 == B2 baseline. A and B1 share a randomness stream, so an identical pair means the watermark never changed a sampled token.

### Equivalence bounds

| Quantity | Bound |
|----------|-------|
| judge win rate for A | 95% CI within 0.50 +/- 0.05 |
| mean `d_wm` | 95% CI within +/- `log(1.05)` = +/- 0.0488 nats, i.e. 5% in perplexity |

### Arm parity

`generate` asserts, before it emits a single token, that the three arms differ only by the SynthID logits processor: identical generation kwargs, identical decoding rules, and a processor chain that is the shared warper stack plus a trailing `SynthIDLogitsProcessor` in arm A alone. The per-arm kwargs, chains and decode rules are written into `run_config.json`.

`report` re-checks all of that from the stored artefacts and adds four data checks: every prompt encodes to one token count and one generator model across arms; A and B1 share a sampling seed while B2 does not; `ppl` was conditioned on the same prompt file that generation used; and the NLLs were scored over response tokens conditioned on the prompt. A failing check prints exactly what differs and writes no report.

### Outputs

Per run directory: `generations.jsonl`, `ppl.jsonl`, `sim.jsonl`, `judge.jsonl`, `filter_report.json`, `metrics.json`, `results.md`, `results.csv`, `diagnostics.md`, `run_config.json`.

`diagnostics.md` lists the 10 prompts with the largest `|d_wm|` and the 10 with the largest `|d_ctrl|`, each with per-arm token counts, NLLs and a 200-character preview of all three outputs.

## Experiment 2 - SynthID-Text on code: functional correctness and detectability

One resumable CLI, run once per generator model, that measures on the *same* outputs (a) whether the watermark costs functional correctness and (b) how detectable those outputs are, for code next to Exp 1's prose. Same MarkLLM SynthID integration, same `watermark_config.json`, same sampling as Exp 1.

### Setup

```bash
pip install -r requirements-exp2.txt      # Exp 1 requirements + matplotlib
export HF_TOKEN=...
python run_exp2.py selftest               # every canonical solution through the harness
```

### Running

```bash
python run_exp2.py all --model google/gemma-2-9b-it             --out runs/exp2-gemma2-9b
python run_exp2.py all --model meta-llama/Llama-3.1-8B-Instruct --out runs/exp2-llama31-8b

# or stage by stage - each is resumable and order-free after generate
python run_exp2.py generate --model google/gemma-2-9b-it --out runs/exp2-gemma2-9b
python run_exp2.py execute  --out runs/exp2-gemma2-9b     # --workers, --timeout, --mem-mb, --exec-prefix
python run_exp2.py detect   --out runs/exp2-gemma2-9b     # --exp1-runs "runs/*"
python run_exp2.py report   --out runs/exp2-gemma2-9b

python run_exp2.py all --model Qwen/Qwen2.5-0.5B-Instruct --out runs/exp2-smoke --smoke   # 5 problems, 256 tokens
python tests/test_exp2_wiring.py                                                          # 13 offline checks
```

`detect` scores the prose of every Exp 1 run under `--exp1-runs` whose generator model matches, so run Exp 1 first (or accept a report without the prose rows).

### Problems

`problems.json` (repo root, write-once, hashed into every `run_config.json`): all 164 HumanEval problems plus the first 200 MBPP *sanitized* test problems by `task_id` (mbpp-11 .. mbpp-417), with prompts, reference tests and canonical solutions. Each problem is put to the model as one user turn through its chat template, with a minimal instruction to return only the solution in one ```python fence. HumanEval shows the function signature and docstring; MBPP shows the task description and its three asserts (the function name is only knowable from them).

### Arms

| Arm | Watermark | Sampling seeds | Role |
|-----|-----------|----------------|------|
| `A`  | SynthID (non-distortionary) | S1..S10 | treatment |
| `B1` | none | S1..S10 | paired control - sample k shares A's randomness stream |
| `B2` | none | S11..S20 | sampling-noise floor |

10 samples per problem per arm (`--samples` overrides; B2 then uses S_{max(n,10)+1}..). Base seeds are `S_i = sha256("exp2|S{i}")[:8]`, per-sample seeds are derived from `(S_i, problem_id)` as in Exp 1. Sampling is identical everywhere: `do_sample=True, temperature=0.7, top_k=40, top_p=0.95, max_new_tokens=512` - matched to Exp 1, so absolute pass@1 sits below greedy decoding in every arm alike.

### Statistics

pass@1 per problem per arm is the fraction of kept samples that pass. The report gives mean pass@1 per arm, the paired per-problem `A - B1` with a percentile bootstrap 95% CI (10k resamples, seed 987654321, resampling problems), `B2 - B1` as the noise floor, a HumanEval/MBPP split, and byte-for-byte exact-match rates (A == B1, B1 == B2) for raw responses and extracted code.

**Pre-registered bound:** the 95% CI of mean(`A - B1`) lies within +/- 0.05.

### Detection

Mean g-value with the saved keys on the re-tokenised text, `sum(g * mask) / (depth * n_unmasked)` with MarkLLM's context-repetition and EOS masks: the **no-training path**, not the deployed Bayesian detector. Domains: `prose` (Exp 1 responses) and `code` (full response), plus `code_extracted` (the fence alone) and `code_ids` (generated token ids, no re-tokenisation). Per model and domain the threshold is the `(m+1)`-th largest pooled unwatermarked score (B1 + B2) with `m = floor(0.01 * n_null)`, i.e. empirical FPR 1%; TPR of the watermarked outputs at it is the headline row, TPR at 5% FPR the secondary row. Prefixes of 20, 50, 100, 200, 400 tokens and the full length are rescored, with thresholds re-set per prefix and a sample counted only if it is at least that long, into `curves.csv` and `detection_curves.png`.

### Arm parity

`generate` records per arm the exact kwargs passed to `model.generate`, the logits-processor chain and the decode rules in `run_config.json` before emitting a token, and refuses to run if the arms differ by more than the trailing `SynthIDLogitsProcessor` in A. `report` re-checks them (an unrecorded field is a FAIL, never INCOMPLETE) and adds: per sample, A and B1 share a seed, B2 differs, only A is watermarked, one prompt encoding per triple; one harness configuration and an outcome in all three arms for every triple; one detector configuration over every score. A failing check prints what differs and writes no report; an incomplete `execute` or `detect` is refused too, unless `--force` writes a report marked PARTIAL.

### Restartability

| Stage | File | Skip key |
|-------|------|----------|
| `generate` | `generations.jsonl` | `(problem_id, arm, sample_idx)` |
| `execute` | `execution.jsonl` | `(problem_id, arm, sample_idx)` |
| `detect` | `detection.jsonl` | `(domain, source_run, item_id, arm, sample_idx)` |

Outputs per run: `results.md`, `results.csv`, `metrics.json`, `curves.csv`, `detection_curves.png`, `filter_report.json`, `run_config.json`.
