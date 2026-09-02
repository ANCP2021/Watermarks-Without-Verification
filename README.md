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