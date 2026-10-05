"""Run plumbing: config loading, seeding, device selection, raw logs and manifests.

All paths are resolved relative to the repo root so nothing personal is hard-coded.
"""
import hashlib
import json
import logging
import platform
import random
import subprocess
import sys
from datetime import datetime
from importlib import metadata
from pathlib import Path

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
MEMBER = "kruthika"
TASK = "task3"
RAW_LOG_DIR = REPO_ROOT / "reproducibility" / "raw_logs"
MANIFEST_DIR = REPO_ROOT / "reproducibility" / "manifests"


def repo_path(p):
    """Resolve a config path (relative to repo root) to an absolute Path."""
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


def rel(p):
    """Repo-relative string form of a path, for logs/manifests (no personal paths)."""
    try:
        return str(Path(p).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def load_config(path):
    with open(repo_path(path)) as f:
        return yaml.safe_load(f)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pick_device(name="auto"):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_name(device):
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    if device.type == "mps":
        return f"Apple MPS ({_mac_chip()})"
    return platform.processor() or platform.machine()


def _mac_chip():
    try:
        return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    except Exception:
        return platform.machine()


def sync(device):
    """Wait for queued GPU work so wall-clock timings are honest."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def reset_peak_memory(device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def peak_memory_mb(device):
    """Peak accelerator memory (CUDA: true peak; MPS: current driver allocation) + process peak RSS."""
    out = {}
    if device.type == "cuda":
        out["peak_gpu_mem_mb"] = torch.cuda.max_memory_allocated(device) / 2**20
    elif device.type == "mps":
        out["mps_driver_mem_mb"] = torch.mps.driver_allocated_memory() / 2**20
    try:
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        out["peak_rss_mb"] = rss / 2**20 if sys.platform == "darwin" else rss / 2**10
    except ImportError:  # Windows
        pass
    return out


def run_id(run_name):
    return f"{TASK}_{MEMBER}_{run_name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"


def get_logger(rid):
    """Logger that writes to stdout and an append-only raw log file. Never edit these files."""
    RAW_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = RAW_LOG_DIR / f"{rid}.log"
    logger = logging.getLogger(rid)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(log_path)):
        h.setFormatter(fmt)
        logger.addHandler(h)
    return logger, log_path


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(*args):
    try:
        return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def environment_info(device):
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device": str(device),
        "device_name": device_name(device),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "packages": sorted(f"{d.metadata['Name']}=={d.version}" for d in metadata.distributions()),
    }


def write_manifest(rid, cfg, device, log_path, checkpoints=(), metrics=None, extra=None):
    """Record exactly what produced a result: env, config, log, checkpoint hashes, metrics."""
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_id": rid,
        "created": datetime.now().isoformat(timespec="seconds"),
        "environment": environment_info(device),
        "config": cfg,
        "raw_log": rel(log_path),
        "checkpoints": [{"path": rel(c), "sha256": sha256(c)} for c in checkpoints],
        "metrics": metrics or {},
        **(extra or {}),
    }
    path = MANIFEST_DIR / f"{rid}.json"
    path.write_text(json.dumps(manifest, indent=2, default=str))
    return path
