"""Read the Abilene X*.gz files. Layout and units are VERIFIED by scripts/verify_data.py, never assumed."""
import gzip
import re
from pathlib import Path

import numpy as np

N_NODES, N_OD, N_EST = 12, 144, 5
EST_NAMES = ["real", "simple_gravity", "simple_tomogravity", "general_gravity", "general_tomogravity"]
LAYOUTS = ("pair_major", "estimate_major")


def week_files(raw_dir):
    fs = sorted(Path(raw_dir).glob("X[0-9][0-9].gz"), key=lambda p: int(re.findall(r"\d+", p.name)[0]))
    if not fs:
        raise FileNotFoundError("no X??.gz in %s -- run scripts/download_data.py" % raw_dir)
    return fs


def read_week(path):
    """-> (T, 720) float64. Raises on ragged rows instead of padding."""
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as f:
        rows = [np.array(l.split(), dtype=np.float64) for l in f if l.strip()]
    if len({len(r) for r in rows}) != 1:
        raise ValueError("%s: ragged rows %s" % (path, sorted({len(r) for r in rows})))
    return np.vstack(rows)


def split_layout(raw, layout):
    """(T,720) -> (T,144,5). pair_major: [od0e0..od0e4, od1e0..]; estimate_major: [e0 od0..od143, e1 ..]."""
    T = raw.shape[0]
    if raw.shape[1] != N_OD * N_EST:
        raise ValueError("expected %d columns, got %d" % (N_OD * N_EST, raw.shape[1]))
    if layout == "pair_major":
        return raw.reshape(T, N_OD, N_EST)
    if layout == "estimate_major":
        return raw.reshape(T, N_EST, N_OD).transpose(0, 2, 1)
    raise ValueError(layout)


def rank1_residual(m):
    s = np.linalg.svd(m, compute_uv=False)
    return float(s[1] / s[0]) if s[0] > 0 else np.inf


def detect_layout(raw, n=40):
    """Simple-gravity estimates (slot 1) are rank-1 as 12x12 matrices; the wrong layout scrambles that.
    Returns (layout, scores). Raises if inconclusive. Does NOT resolve src-major vs dst-major (transpose of rank-1 is rank-1)."""
    idx = np.linspace(0, raw.shape[0] - 1, min(n, raw.shape[0])).astype(int)
    scores = {}
    for lay in LAYOUTS:
        d = split_layout(raw[idx], lay)[:, :, 1]
        scores[lay] = float(np.median([rank1_residual(x.reshape(N_NODES, N_NODES)) for x in d]))
    best, other = sorted(scores, key=scores.get)
    if not (scores[best] < 0.05 and scores[other] > 4 * scores[best]):
        raise ValueError("layout detection inconclusive: %s" % scores)
    return best, scores


def load_real(raw_dir, layout="auto"):
    """-> {week_no: (T,144) real OD traffic in file units}, layout used. Missing weeks are simply absent."""
    out, lay = {}, layout
    for p in week_files(raw_dir):
        raw = read_week(p)
        if lay == "auto":
            lay, _ = detect_layout(raw)
        out[int(re.findall(r"\d+", p.name)[0])] = split_layout(raw, lay)[:, :, 0].copy()
    return out, lay
