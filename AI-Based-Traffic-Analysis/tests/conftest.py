"""SYNTHETIC fixtures for unit tests ONLY (same file format as the real X*.gz). Never used for reported results."""
import gzip
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def make_week(T=2016, seed=0, layout="pair_major"):
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    base = np.random.default_rng(123).lognormal(3, 1, size=(12, 12)) * 1e3   # same OD structure every week
    real = base[None] * (1.2 + np.sin(2 * np.pi * t / 288))[:, None, None] * rng.lognormal(0, .1, (T, 12, 12))
    out = np.zeros((T, 12, 12))
    sg = real.sum(2)[:, :, None] * real.sum(1)[:, None, :] / real.sum((1, 2))[:, None, None]   # rank-1 gravity
    est = [real, sg, sg * 1.01, sg * 0.99, sg * 1.02]
    E = np.stack([e.reshape(T, 144) for e in est], axis=2)             # (T,144,5)
    return E.reshape(T, 720) if layout == "pair_major" else E.transpose(0, 2, 1).reshape(T, 720)


def write_weeks(d, n=6, layout="pair_major"):
    d.mkdir(parents=True, exist_ok=True)
    for k in range(1, n + 1):
        with gzip.open(d / ("X%02d.gz" % k), "wt") as f:
            for row in make_week(seed=k, layout=layout):
                f.write(" ".join("%.4f" % v for v in row) + "\n")


@pytest.fixture
def weeks_dir(tmp_path):
    write_weeks(tmp_path / "raw")
    return tmp_path / "raw"
