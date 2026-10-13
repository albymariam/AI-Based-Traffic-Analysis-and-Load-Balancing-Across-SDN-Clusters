"""Leak-free dataset from REAL telemetry. Split BY SEED (whole runs held out); normalisation from train only."""
import json

import numpy as np
import pandas as pd

from common.config import ROOT
from common.features import FEATURES

POOL_MIN_SEED = 100      # training pool; evaluation experiments use seeds < 100


def _pool_runs():
    runs = {}
    for d in sorted((ROOT / "results" / "runs").glob("*/")):
        mp = d / "meta.json"
        if not (mp.exists() and (d / "telemetry.csv").exists()):
            continue
        m = json.loads(mp.read_text())
        if "finished" not in m or m["seed"] < POOL_MIN_SEED or m["method"] == "lstm_hysteresis":
            continue
        runs.setdefault(m["seed"], []).append(d)
    return runs


def _link_arrays(d, warmup):
    df = pd.read_csv(d / "telemetry.csv")
    df = df[df.t >= warmup].sort_values("t")
    return [g[FEATURES].to_numpy(dtype=np.float32) for _, g in df.groupby("link_id")]


def _windows(arrays, W, H, u_idx):
    X, Y = [], []
    for a in arrays:
        for i in range(len(a) - W - H + 1):
            X.append(a[i:i + W])
            Y.append(a[i + W:i + W + H, u_idx])
    return np.array(X, np.float32), np.array(Y, np.float32)


def normalize(X, norm):
    return (X - np.array(norm["lo"], np.float32)) / np.array(norm["rng"], np.float32)


def build_splits(W, H, warmup):
    runs = _pool_runs()
    seeds = sorted(runs)
    if len(seeds) < 3:
        raise SystemExit("need >=3 training-pool seeds (>=%d); found %s" % (POOL_MIN_SEED, seeds))
    split = {"test": seeds[-1:], "val": seeds[-2:-1], "train": seeds[:-2]}
    u = FEATURES.index("utilization")
    S = {"split_seeds": split}
    for name, ss in split.items():
        arrs = [a for s in ss for d in runs[s] for a in _link_arrays(d, warmup)]
        S["X_" + name], S["Y_" + name] = _windows(arrs, W, H, u)
    flat = S["X_train"].reshape(-1, len(FEATURES))
    lo, hi = flat.min(0), flat.max(0)
    S["norm"] = {"lo": lo.tolist(), "rng": np.where(hi - lo == 0, 1.0, hi - lo).tolist()}
    return S