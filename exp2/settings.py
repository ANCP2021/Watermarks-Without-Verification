from __future__ import annotations

import hashlib
import os

from exp1.settings import (BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED, CHAT_TEMPLATE_DATE,  # noqa: F401
                           SAMPLING, SMOKE_MODEL, TARGET_MODELS)

N_SAMPLES = 10                       # per problem per arm; --samples overrides
SMOKE_N_PROBLEMS = 5                 # 3 HumanEval + 2 MBPP, so both harness paths run
SMOKE_MAX_NEW_TOKENS = 256

# Seeds.  S_i is a fixed function of i; the per-sample seed is derived from
# (S_i, problem_id) exactly as Exp 1 derives its per-prompt seeds.
#   A  sample k -> S_{k+1}            (S1..S10)
#   B1 sample k -> S_{k+1}            (S1..S10, paired with A)
#   B2 sample k -> S_{offset+k+1}     (S11..S20; offset = max(n_samples, 10))
SEED_TABLE_TAG = "exp2"


def base_seed(i: int) -> int:
    """S_i for i >= 1."""
    digest = hashlib.sha256(f"{SEED_TABLE_TAG}|S{i}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def b2_seed_offset(n_samples: int) -> int:
    return max(n_samples, N_SAMPLES)


HUMANEVAL_REPO = "openai/openai_humaneval"
HUMANEVAL_SPLIT = "test"
MBPP_REPO = "google-research-datasets/mbpp"
MBPP_CONFIG = "sanitized"
MBPP_SPLIT = "test"
MBPP_N = 200 

INSTRUCTION = ("Return only the solution, in one ```python code block, "
               "with no explanation.")
HUMANEVAL_TEMPLATE = ("Complete the following Python function. {instruction}\n\n"
                      "```python\n{prompt}\n```")
MBPP_TEMPLATE = ("Write a Python function for the following task. {instruction}\n\n"
                 "{prompt}\n\nYour solution must pass these tests:\n"
                 "```python\n{tests}\n```")

# Execution harness
EXEC_TIMEOUT_S = 10.0
EXEC_MEM_MB = 4096
EXEC_FSIZE_MB = 64
EXEC_WORKERS = max(1, min(8, os.cpu_count() or 1))

PREFIX_LENGTHS = (20, 50, 100, 200, 400)
FPR_TARGETS = (0.01, 0.05)
PRIMARY_FPR = 0.01
DETECTION_DOMAINS = ("prose", "code", "code_extracted", "code_ids")
FIGURE_DOMAINS = ("prose", "code")

# Pre-registered bound
PASS1_MARGIN = 0.05                  # 95% CI of mean(pass@1_A - pass@1_B1) within +/- 0.05

# Report
N_OUTLIER_PROBLEMS = 10
