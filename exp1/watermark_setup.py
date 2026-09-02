
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from .common import (DEFAULT_WATERMARK_CONFIG_PATH, ensure_markllm_on_path, log,
                     sha256_file, utcnow, write_json_atomic)

KEY_GEN_SEED = 20240517
WATERMARK_DEPTH = 30      # number of keys == SynthID watermarking depth
NGRAM_LEN = 5

# The SynthID g-value lookup table. MarkLLM draws it in SynthIDLogitsProcessor.__init__ as
# torch.randint(0, 2, (size,), generator=torch.Generator(device).manual_seed(seed), device):
# the CPU and CUDA generators produce different streams for the same seed, so a table drawn
# on cpu at detection time is NOT the table generation drew on cuda. The table is therefore
# persisted once, next to watermark_config.json, and loaded everywhere.
SAMPLING_TABLE_FILE = "sampling_table.pt"


def ensure_watermark_config(path: str | Path = DEFAULT_WATERMARK_CONFIG_PATH) -> dict:

    path = Path(path)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        _validate(cfg, path)
        log(f"loaded watermark config {path} (sha256={sha256_file(path)[:16]}..., "
            f"depth={len(cfg['keys'])}, ngram_len={cfg['ngram_len']})")
        return cfg

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


# ----------------------------------------------------------------------------- sampling table

def sampling_table_path(config_path: str | Path) -> Path:
    return Path(config_path).with_name(SAMPLING_TABLE_FILE)


def sampling_table_sha256(table: torch.Tensor) -> str:
    """sha256 of the table as little-endian int64 -- the dtype MarkLLM draws it in."""
    arr = table.detach().to("cpu", torch.int64).contiguous().numpy().astype("<i8")
    return hashlib.sha256(arr.tobytes()).hexdigest()


def load_sampling_table(config_path: str | Path, cfg: dict, device: str):
    """(table on `device` as int64, file metadata) or None if the file is absent."""
    path = sampling_table_path(config_path)
    if not path.exists():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=True)
    table = payload["sampling_table"]
    for field in ("sampling_table_seed", "sampling_table_size"):
        if payload.get(field) != cfg[field]:
            raise SystemExit(f"{path}: {field}={payload.get(field)!r} but {config_path} says "
                             f"{cfg[field]!r}; the table does not belong to this config")
    if table.dim() != 1 or table.numel() != int(cfg["sampling_table_size"]):
        raise SystemExit(f"{path}: table has shape {tuple(table.shape)}, expected "
                         f"({cfg['sampling_table_size']},)")
    if not set(torch.unique(table).tolist()) <= {0, 1}:
        raise SystemExit(f"{path}: table values are not all 0/1")
    return table.to(device=device, dtype=torch.int64), payload


def draw_sampling_table_like_markllm(config_path: str | Path, device: str) -> torch.Tensor:
    """The table MarkLLM's SynthIDLogitsProcessor draws in __init__ on `device`, through
    MarkLLM's own code path (SynthIDConfig -> SynthIDUtils.rng -> torch.randint)."""
    ensure_markllm_on_path()
    from utils.transformers_config import TransformersConfig
    from watermark.synthid.synthid import (SynthIDConfig, SynthIDLogitsProcessor, SynthIDUtils)

    tf_config = TransformersConfig(model=None, tokenizer=None, vocab_size=0, device=device)
    synthid_config = SynthIDConfig(str(config_path), tf_config)
    processor = SynthIDLogitsProcessor(synthid_config, SynthIDUtils(synthid_config))
    return processor.sampling_table


def emulate_cuda_sampling_table(seed: int, size: int) -> torch.Tensor:
    """numpy re-implementation of torch.randint(0, 2, (size,), generator=g, device='cuda')
    for a CUDA generator g seeded with `seed` and at philox offset 0.

    PyTorch's CUDA randint (range < 2**32) launches distribution_elementwise_grid_stride_kernel
    with 256 threads/block, unroll 4 and grid = ceil(size/256) blocks (capped at
    SMs * 8, which does not bind for size 65536 on any GPU with >= 32 SMs). Thread t runs
    curand_init(seed, subsequence=t, offset=0), i.e. Philox4x32-10 with counter (0, 0, t, 0)
    and key (seed_lo, seed_hi); element t takes the .x word of its first curand4() call and
    maps it with (rand % 2) + 0. `dump-table --device cuda` checks this against the real draw.
    """
    if size > 256 * 256 * 4:
        raise SystemExit("emulation only covers tables that fit one grid-stride pass")
    M0, M1 = np.uint64(0xD2511F53), np.uint64(0xCD9E8D57)
    W0, W1 = np.uint32(0x9E3779B9), np.uint32(0xBB67AE85)
    idx = np.arange(size, dtype=np.uint64)
    c0 = np.zeros(size, dtype=np.uint32)
    c1 = np.zeros(size, dtype=np.uint32)
    c2 = (idx & np.uint64(0xFFFFFFFF)).astype(np.uint32)
    c3 = (idx >> np.uint64(32)).astype(np.uint32)
    k0 = np.uint32(seed & 0xFFFFFFFF)
    k1 = np.uint32((seed >> 32) & 0xFFFFFFFF)
    mask32 = np.uint64(0xFFFFFFFF)
    with np.errstate(over="ignore"):
        for r in range(10):
            p0 = c0.astype(np.uint64) * M0
            p1 = c2.astype(np.uint64) * M1
            hi0, lo0 = (p0 >> np.uint64(32)).astype(np.uint32), (p0 & mask32).astype(np.uint32)
            hi1, lo1 = (p1 >> np.uint64(32)).astype(np.uint32), (p1 & mask32).astype(np.uint32)
            c0, c1, c2, c3 = hi1 ^ c1 ^ k0, lo1, hi0 ^ c3 ^ k1, lo0
            if r < 9:
                k0 = np.uint32(k0 + W0)
                k1 = np.uint32(k1 + W1)
    return torch.from_numpy((c0 % np.uint32(2)).astype(np.int64))


def _save_sampling_table(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def dump_sampling_table(args) -> int:
    """`run_exp2.py dump-table`: draw the table exactly as generation did and persist it."""
    cfg_path = Path(args.watermark_config)
    cfg = ensure_watermark_config(cfg_path)
    seed, size = int(cfg["sampling_table_seed"]), int(cfg["sampling_table_size"])
    dest = sampling_table_path(cfg_path)

    emulated = emulate_cuda_sampling_table(seed, size)
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("dump-table must run where generation ran: cuda is not available here "
                             "(use --device emulate-cuda only for a provisional table)")
        table = draw_sampling_table_like_markllm(cfg_path, "cuda").to("cpu", torch.int64)
        drawn_on = f"cuda ({torch.cuda.get_device_name(0)})"
        emulation_agrees = bool(torch.equal(table, emulated))
    else:
        table = emulated
        drawn_on = ("emulate-cuda: numpy Philox4x32-10 re-implementation of torch CUDA randint; "
                    "provisional until `dump-table --device cuda` confirms it")
        emulation_agrees = None
    sha = sampling_table_sha256(table)

    if dest.exists():
        old = torch.load(dest, map_location="cpu", weights_only=True)
        old_sha = sampling_table_sha256(old["sampling_table"])
        if old_sha != sha:
            msg = (f"{dest} DIFFERS: existing sha256 {old_sha} (drawn on {old.get('drawn_on')}), "
                   f"drawn now {sha} (on {drawn_on})")
            if not args.force:
                print(msg + "\nrefusing to overwrite; pass --force, then re-run `detect` and "
                            "`report` for every run scored with the old table")
                return 1
            log("WARNING: " + msg + " -- overwriting because of --force")
        else:
            log(f"{dest} already holds this table (sha256 {sha}); refreshing provenance only")
            if args.device != "cuda":
                print(f"sampling_table.pt sha256 {sha}  unchanged  drawn_on {old.get('drawn_on')}")
                return 0

    payload = {
        "sampling_table": table,
        "sampling_table_seed": seed,
        "sampling_table_size": size,
        "sampling_table_sha256": sha,
        "watermark_config_sha256": sha256_file(cfg_path),
        "drawn_on": drawn_on,
        "cuda_emulation_agrees": emulation_agrees,
        "torch_version": str(torch.__version__),
        "created_at": utcnow(),
        "how": "MarkLLM SynthIDLogitsProcessor.__init__: torch.randint(0, 2, (size,), "
               "generator=torch.Generator(device).manual_seed(seed), device)",
        "note": "Write-once. Generation and detection both load this table; never redraw.",
    }
    _save_sampling_table(dest, payload)
    print(f"sampling_table.pt sha256 {sha}  drawn_on {drawn_on}  "
          f"cuda_emulation_agrees {emulation_agrees}  -> {dest}")
    return 0


def build_synthid_processor(config_path: str | Path, tokenizer, device: str,
                            require_table: bool = False):

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

    table_cfg = {"sampling_table_seed": synthid_config.sampling_table_seed,
                 "sampling_table_size": synthid_config.sampling_table_size}
    loaded = load_sampling_table(cfg_path, table_cfg, device)
    if loaded is not None:
        table, meta = loaded
        processor.sampling_table = table
        processor.sampling_table_source = f"file:{SAMPLING_TABLE_FILE} ({meta.get('drawn_on')})"
    elif require_table:
        raise SystemExit(
            f"{sampling_table_path(cfg_path)} not found. Detection needs the g-value table that "
            "generation drew (cpu and cuda generators differ for the same seed). Run "
            "`python run_exp2.py dump-table` on the generation machine (cuda) and put the file "
            "next to watermark_config.json.")
    else:
        processor.sampling_table_source = f"drawn:{device}"
        log(f"WARNING: SynthID sampling table drawn on {device} and not persisted; a run on "
            f"another device type draws a different table. Run `run_exp2.py dump-table` first.")
    processor.sampling_table_sha256 = sampling_table_sha256(processor.sampling_table)
    log(f"SynthID processor ready: mode={synthid_config.watermark_mode}, "
        f"ngram_len={synthid_config.ngram_len}, depth={len(synthid_config.keys)}, "
        f"sampling_table_size={synthid_config.sampling_table_size}, "
        f"sampling_table={processor.sampling_table_sha256[:12]} "
        f"[{processor.sampling_table_source}]")
    return processor


def reset_synthid_state(processor) -> None:
    processor.state = None
