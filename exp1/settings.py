
from __future__ import annotations

SAMPLING = {
    "do_sample": True,
    "temperature": 0.7,
    "top_k": 40,
    "top_p": 0.95,
    "max_new_tokens": 512,
    "repetition_penalty": 1.0,
    "no_repeat_ngram_size": 0,
    "num_beams": 1,
}

SMOKE_MAX_NEW_TOKENS = 128  
SMOKE_N_PROMPTS = 10
SEED_S1 = 1234
SEED_S2 = 5678

BOOTSTRAP_SEED = 987654321
BOOTSTRAP_RESAMPLES = 10_000

TARGET_MODELS = (
    "google/gemma-2-9b-it",
    "meta-llama/Llama-3.1-8B-Instruct",
)
SMOKE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

PPL_MODEL = "Qwen/Qwen2.5-7B"
JUDGE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
SIM_MODEL = "BAAI/bge-large-en-v1.5"

SMOKE_PPL_MODEL = "Qwen/Qwen2.5-0.5B"
SMOKE_JUDGE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
SMOKE_SIM_MODEL = "BAAI/bge-small-en-v1.5"

WIN_RATE_CENTER = 0.5
WIN_RATE_MARGIN = 0.05        # CI within 0.50 +/- 0.05
DELTA_PPL_MARGIN_PCT = 5.0    # CI within +/- 5%

CHAT_TEMPLATE_DATE = "26 Jul 2024"
