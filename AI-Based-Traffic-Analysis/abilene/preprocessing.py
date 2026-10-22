"""Units, masking, chronological week split, train-only scaling, leak-free windows. Nothing is imputed."""
import numpy as np


def to_mbps(x, bytes_per_unit=100, interval_s=300):
    return np.asarray(x, np.float64) * bytes_per_unit * 8.0 / interval_s / 1e6


def clean(weeks):
    """Negative / non-finite values -> NaN (kept as NaN, windows touching them are dropped). Returns weeks, n_invalid."""
    out, bad = {}, 0
    for k, a in weeks.items():
        a = a.astype(np.float64).copy()
        m = ~np.isfinite(a) | (a < 0)
        bad += int(m.sum())
        a[m] = np.nan
        out[k] = a
    return out, bad


def od_columns(drop_diagonal=True, n=12):
    idx = np.arange(n * n)
    return idx[(idx // n) != (idx % n)] if drop_diagonal else idx


def chrono_split(week_ids, train=0.70, val=0.15):
    w = sorted(week_ids)
    n = len(w)
    ntr, nva = max(1, round(train * n)), max(1, round(val * n))
    if ntr + nva >= n:
        raise ValueError("need >=3 weeks for a train/val/test split, got %d" % n)
    return {"train": w[:ntr], "val": w[ntr:ntr + nva], "test": w[ntr + nva:]}


class Scaler:
    """z = (log1p(Mbps) - mu) / sd per OD pair; fitted on TRAIN weeks only."""
    def fit(self, arrays):
        z = np.log1p(np.vstack(arrays))
        self.mu, self.sd = np.nanmean(z, 0), np.nanstd(z, 0) + 1e-6
        return self

    def transform(self, x):
        return (np.log1p(x) - self.mu) / self.sd

    def inverse(self, z):
        return np.clip(np.expm1(z * self.sd + self.mu), 0.0, None)

    def state(self):
        return {"mu": self.mu.tolist(), "sd": self.sd.tolist()}


def windows(series, W, H):
    """series: {week: (T,P)} -> X (N,W,P), Y (N,H,P), meta (N,2)=(week, start). Windows never cross a week or contain NaN."""
    X, Y, M = [], [], []
    for wk in sorted(series):
        a = series[wk]
        ok = ~np.isnan(a).any(1)
        for s in range(len(a) - W - H + 1):
            if ok[s:s + W + H].all():
                X.append(a[s:s + W]); Y.append(a[s + W:s + W + H]); M.append((wk, s))
    if not X:
        return np.empty((0, W, 0)), np.empty((0, H, 0)), np.empty((0, 2), int)
    return np.stack(X), np.stack(Y), np.array(M)
