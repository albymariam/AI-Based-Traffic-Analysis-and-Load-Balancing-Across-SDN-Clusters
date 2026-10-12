"""Shared config, seeding and run-directory helpers (no hard-coded paths)."""
import json
import os
import random
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"


def load_yaml(name: str) -> dict:
    """load_yaml('topology') -> dict from configs/topology.yaml"""
    with open(CONFIG_DIR / f"{name}.yaml", "r") as f:
        return yaml.safe_load(f)


def load_all() -> dict:
    return {n: load_yaml(n) for n in ("topology", "traffic", "experiment")}


def path_from(cfg_paths_key: str, exp_cfg: dict | None = None) -> Path:
    exp_cfg = exp_cfg or load_yaml("experiment")
    p = ROOT / exp_cfg["paths"][cfg_paths_key]
    p.mkdir(parents=True, exist_ok=True)
    return p


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def git_hash() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL
        )
        dirty = subprocess.call(
            ["git", "diff", "--quiet"], cwd=ROOT, stderr=subprocess.DEVNULL
        )
        return out.decode().strip() + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def make_run_dir(method: str, scenario: str, seed: int) -> Path:
    """results/runs/<timestamp>_<method>_<scenario>_s<seed>/ with config snapshot + meta."""
    exp = load_yaml("experiment")
    ts = time.strftime("%Y%m%d-%H%M%S")
    run = path_from("runs", exp) / f"{ts}_{method}_{scenario}_s{seed}"
    run.mkdir(parents=True, exist_ok=True)
    for name in ("topology", "traffic", "experiment"):
        shutil.copy(CONFIG_DIR / f"{name}.yaml", run / f"{name}.yaml")
    meta = {
        "method": method,
        "scenario": scenario,
        "seed": seed,
        "git_hash": git_hash(),
        "started": ts,
    }
    (run / "meta.json").write_text(json.dumps(meta, indent=2))
    return run


def in_wsl_or_root_warning() -> None:
    if os.name == "nt":
        raise SystemExit("Run this on Linux/WSL, not native Windows.")