
from __future__ import annotations

import json
from pathlib import Path

from .common import (DEFAULT_WATERMARK_CONFIG_PATH, ensure_markllm_on_path, log,
                     sha256_file, write_json_atomic)

KEY_GEN_SEED = 20240517
WATERMARK_DEPTH = 30      # number of keys == SynthID watermarking depth
NGRAM_LEN = 5


def ensure_watermark_config(path: str | Path = DEFAULT_WATERMARK_CONFIG_PATH) -> dict:

    path = Path(path)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        _validate(cfg, path)
        log(f"loaded watermark config {path} (sha256={sha256_file(path)[:16]}..., "
            f"depth={len(cfg['keys'])}, ngram_len={cfg['ngram_len']})")
        return cfg

    import numpy as np

    rng = np.random.default_rng(KEY_GEN_SEED)
    keys = [int(k) for k in rng.integers(0, 2**31 - 1, size=WATERMARK_DEPTH)]

    cfg = {
        "algorithm_name": "SynthID",
        "ngram_len": NGRAM_LEN,
        "keys": keys,
        "sampling_table_size": 65536,
        "sampling_table_seed": 0,
        "watermark_mode": "non-distortionary",   # MarkLLM's default arm
        "num_leaves": 2,
        "context_history_size": 1024,
        "detector_type": "mean",
        "threshold": 0.52,
        "_provenance": {
            "key_gen_seed": KEY_GEN_SEED,
            "key_gen": "numpy.random.default_rng(seed).integers(0, 2**31-1, size=depth)",
            "note": "Write-once. Reused by both generator models, the code "
                    "experiment, and detection scoring. Never regenerate.",
        },
    }
    write_json_atomic(path, cfg)
    log(f"created {path} (sha256={sha256_file(path)[:16]}..., depth={WATERMARK_DEPTH})")
    return cfg


def _validate(cfg: dict, path: Path) -> None:
    if cfg.get("algorithm_name") != "SynthID":
        raise SystemExit(f"{path}: algorithm_name must be 'SynthID', got {cfg.get('algorithm_name')!r}")
    if cfg.get("ngram_len") != NGRAM_LEN:
        raise SystemExit(f"{path}: ngram_len must be {NGRAM_LEN}, got {cfg.get('ngram_len')!r}")
    if cfg.get("watermark_mode") != "non-distortionary":
        raise SystemExit(f"{path}: watermark_mode must be 'non-distortionary', "
                         f"got {cfg.get('watermark_mode')!r}")
    if not cfg.get("keys"):
        raise SystemExit(f"{path}: no keys")


def build_synthid_processor(config_path: str | Path, tokenizer, device: str):

    ensure_markllm_on_path()
    from utils.transformers_config import TransformersConfig
    from watermark.synthid.synthid import (SynthIDConfig, SynthIDLogitsProcessor, SynthIDUtils)

    cfg_path = Path(config_path)
    if not cfg_path.exists():
        raise SystemExit(f"watermark config not found: {cfg_path}")

    tf_config = TransformersConfig(model=None, tokenizer=tokenizer, device=device)
    synthid_config = SynthIDConfig(str(cfg_path), tf_config)

    synthid_config.temperature = 1.0
    synthid_config.top_k = -1

    utils = SynthIDUtils(synthid_config)
    processor = SynthIDLogitsProcessor(synthid_config, utils)
    log(f"SynthID processor ready: mode={synthid_config.watermark_mode}, "
        f"ngram_len={synthid_config.ngram_len}, depth={len(synthid_config.keys)}, "
        f"sampling_table_size={synthid_config.sampling_table_size}")
    return processor


def reset_synthid_state(processor) -> None:
    processor.state = None
