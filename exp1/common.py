from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent
MARKLLM_DIR = REPO_ROOT / "MarkLLM"

# Shared, write-once artifacts.  Both model runs -- and the later code
# experiment and detection scoring -- must read these exact files.
DEFAULT_PROMPTS_PATH = REPO_ROOT / "prompts.json"
DEFAULT_WATERMARK_CONFIG_PATH = REPO_ROOT / "watermark_config.json"

ARMS = ("A", "B1", "B2")

def log(msg: str) -> None:
    """Timestamped line on stderr, flushed, so HPC job logs stay readable."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"[{ts}] {msg}", file=sys.stderr, flush=True)


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

def ensure_markllm_on_path() -> None:
    if not MARKLLM_DIR.is_dir():
        raise FileNotFoundError(
            f"MarkLLM not found at {MARKLLM_DIR}. Clone it into the repo root first."
        )
    p = str(MARKLLM_DIR)
    if p not in sys.path:
        sys.path.append(p)

def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json_atomic(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, sort_keys=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

def append_jsonl(path: str | Path, record: dict) -> None:
    """Append one record and force it to disk before returning."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(path: str | Path) -> list[dict]:
   
    path = Path(path)
    if not path.exists():
        return []
    records: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            if i == len(lines) - 1:
                log(f"WARNING: dropping torn final line of {path} (crash during write)")
            else:
                raise
    return records


def done_keys(path: str | Path, fields: Iterable[str]) -> set[tuple]:
    """Set of already-completed keys in a jsonl file, for skip-on-restart."""
    fields = tuple(fields)
    return {tuple(r.get(f) for f in fields) for r in read_jsonl(path)}


def derive_seed(base_seed: int, prompt_id: str) -> int:
    digest = hashlib.sha256(f"{base_seed}|{prompt_id}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def seed_everything(seed: int) -> None:
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)          # seeds CUDA generators too
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _git_commit(path: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def _git_dirty(path: Path) -> bool | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(path), "status", "--porcelain"],
            capture_output=True, text=True, timeout=10,
        )
        return bool(out.stdout.strip())
    except Exception:
        return None


def library_versions() -> dict:
    versions: dict[str, str | None] = {"python": sys.version.split()[0]}
    for mod in ("torch", "transformers", "datasets", "numpy", "huggingface_hub",
                "tokenizers", "accelerate"):
        try:
            versions[mod] = __import__(mod).__version__
        except Exception:
            versions[mod] = None
    return versions


def hardware_info() -> dict:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "processor": platform.processor(),
    }
    try:
        import torch

        info["torch_cuda_version"] = torch.version.cuda
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_count"] = torch.cuda.device_count()
            info["gpu_capability"] = list(torch.cuda.get_device_capability(0))
    except Exception:
        pass
    return info


def provenance() -> dict:
    return {
        "captured_at": utcnow(),
        "repo_commit": _git_commit(REPO_ROOT),
        "repo_dirty": _git_dirty(REPO_ROOT),
        "markllm_commit": _git_commit(MARKLLM_DIR),
        "libraries": library_versions(),
        "hardware": hardware_info(),
        "env": {
            k: os.environ.get(k)
            for k in ("HF_HOME", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "CUDA_VISIBLE_DEVICES")
            if os.environ.get(k) is not None
        },
    }


def hf_token() -> str | None:

    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        tok = os.environ.get(var)
        if tok:
            return tok
    return None


def require_hf_token(model_id: str) -> str | None:
    tok = hf_token()
    gated_prefixes = ("google/gemma", "meta-llama/")
    if tok is None and model_id.startswith(gated_prefixes):
        raise SystemExit(
            f"{model_id} is a gated repo but HF_TOKEN is not set.\n"
            "  export HF_TOKEN=...   (and accept the model licence on huggingface.co)"
        )
    return tok


def repo_revision(repo_id: str, repo_type: str = "model") -> str | None:
    try:
        from huggingface_hub import HfApi

        api = HfApi(token=hf_token())
        if repo_type == "dataset":
            return api.dataset_info(repo_id).sha
        return api.model_info(repo_id).sha
    except Exception as exc:  # offline, rate limited, no access -- not fatal
        log(f"note: could not resolve revision for {repo_id} ({type(exc).__name__})")
        return None


def model_commit_hash(model, fallback_repo_id: str) -> str | None:
    sha = getattr(getattr(model, "config", None), "_commit_hash", None)
    if sha:
        return sha
    return repo_revision(fallback_repo_id)

def load_run_config(out_dir: str | Path) -> dict:
    path = Path(out_dir) / "run_config.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def update_run_config(out_dir: str | Path, stage: str, payload: dict, top_level: dict | None = None) -> dict:

    cfg = load_run_config(out_dir)
    cfg.setdefault("created_at", utcnow())
    cfg["updated_at"] = utcnow()
    if top_level:
        cfg.update(top_level)
    cfg.setdefault("stages", {})
    cfg["stages"][stage] = payload
    write_json_atomic(Path(out_dir) / "run_config.json", cfg)
    return cfg


def check_config_consistency(out_dir: str | Path, stage: str, current: dict, critical: Iterable[str], force: bool = False) -> None:
  
    previous = load_run_config(out_dir).get("stages", {}).get(stage)
    if not previous:
        return
    diffs = []
    for key in critical:
        old, new = previous.get(key, "<absent>"), current.get(key, "<absent>")
        if old != new:
            diffs.append(f"  {key}: existing={old!r}  requested={new!r}")
    if diffs:
        msg = (
            f"Refusing to resume stage '{stage}' in {out_dir}: parameters changed.\n"
            + "\n".join(diffs)
            + "\nUse a fresh --out directory, or pass --force to append anyway "
              "(this mixes incomparable records)."
        )
        if force:
            log("WARNING: " + msg)
        else:
            raise SystemExit(msg)


class Timer:
    def __init__(self) -> None:
        self.t0 = time.time()

    @property
    def elapsed(self) -> float:
        return time.time() - self.t0

    def eta(self, done: int, total: int) -> str:
        if done <= 0:
            return "?"
        remaining = self.elapsed / done * (total - done)
        return f"{remaining / 60:.1f}m"
