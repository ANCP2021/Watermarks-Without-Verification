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
python run_exp1.py ppl    --out runs/gemma2-9b
python run_exp1.py sim    --out runs/gemma2-9b
python run_exp1.py judge  --out runs/gemma2-9b
python run_exp1.py report --out runs/gemma2-9b
```

```bash
python run_exp1.py all --model Qwen/Qwen2.5-0.5B-Instruct --out runs/smoke --smoke
python tests/test_wiring.py     # 11 offline checks, no downloads
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

**`ppl`**: mean per-token NLL of each output under `Qwen/Qwen2.5-7B`, scored standalone (not conditioned on the prompt). 

**`sim`**: cosine similarity under `BAAI/bge-large-en-v1.5` (CLS pooling, L2-normalised, fp32). `sim(A, B1)` is reported against the control distribution `sim(B1, B2)`.

**`judge`**: blinded pairwise preference between A and B1 under `Qwen/Qwen2.5-7B-Instruct` at temperature 0, ties allowed. 


### Outputs

Per run directory: `generations.jsonl`, `ppl.jsonl`, `sim.jsonl`, `judge.jsonl`, `filter_report.json`, `metrics.json`, `results.md`, `results.csv`, `run_config.json`.