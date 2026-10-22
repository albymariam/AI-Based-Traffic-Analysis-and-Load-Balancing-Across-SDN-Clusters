"""Offline traffic-engineering evaluation on the REAL Abilene topology/capacities (fluid model, NOT Mininet).
Every policy decides from the information it is allowed to have; its routing is then scored on the ACTUAL demand.
   python -m abilene.te_offline --forecast lstm_s1 tcn_s1
Policies: shortest_path, ecmp (hop-count), round_robin (equal split over K paths; fluid limit of per-flow RR),
          reactive (last observed demand + hysteresis), <model>_heuristic, <model>_opt (CVXPY min-max), oracle_opt."""
import argparse
import time
from pathlib import Path

import cvxpy as cp
import networkx as nx
import numpy as np
import pandas as pd
import yaml

from abilene.topology_mapping import build_digraph, k_paths, parse_topology

ROOT = Path(__file__).resolve().parents[1]


class TE:
    def __init__(self, G, pairs, K):
        self.G, self.pairs, self.K = G, pairs, K
        self.links = list(G.edges)
        self.cap = np.array([G.edges[e]["capacity_mbps"] for e in self.links])
        lidx = {e: i for i, e in enumerate(self.links)}
        self.paths = [k_paths(G, s, d, K) for s, d in pairs]
        self.npath = np.array([len(p) for p in self.paths])
        self.off = np.concatenate([[0], np.cumsum(self.npath)])
        A = np.zeros((len(self.links), self.off[-1]))
        for k, ps in enumerate(self.paths):
            for j, p in enumerate(ps):
                for e in zip(p, p[1:]):
                    A[lidx[e], self.off[k] + j] = 1.0
        self.A = A
        self.hops = np.array([len(p) - 1 for ps in self.paths for p in ps])
        # fixed splits
        self.x_sp = self._onehot(lambda k: 0)
        self.x_rr = np.concatenate([np.full(n, 1.0 / n) for n in self.npath])
        ecmp = []
        for k, (s, d) in enumerate(pairs):
            sp = [p for p in nx.all_shortest_paths(G, s, d)]               # equal HOP count
            x = np.zeros(self.npath[k])
            for j, p in enumerate(self.paths[k]):
                if p in sp:
                    x[j] = 1.0
            ecmp.append(x / x.sum() if x.sum() else np.eye(self.npath[k])[0])
        self.x_ecmp = np.concatenate(ecmp)
        # CVXPY DPP problem: d is a Parameter, so one compile serves every time step
        self.d = cp.Parameter(self.off[-1], nonneg=True)
        self.x = cp.Variable(self.off[-1], nonneg=True)
        self.U = cp.Variable(nonneg=True)
        cons = [self.A @ cp.multiply(self.d, self.x) <= cp.multiply(self.cap, self.U)]
        for k in range(len(pairs)):
            cons.append(cp.sum(self.x[self.off[k]:self.off[k + 1]]) == 1)
        self.prob = cp.Problem(cp.Minimize(self.U + 1e-4 * (self.hops @ cp.multiply(self.d, self.x)) / self.cap.max()), cons)

    def _onehot(self, f):
        x = np.zeros(self.off[-1])
        for k in range(len(self.pairs)):
            x[self.off[k] + f(k)] = 1.0
        return x

    def expand(self, dem):
        return np.repeat(dem, self.npath)

    def util(self, x, dem):
        return (self.A @ (self.expand(dem) * x)) / self.cap

    def mlu(self, x, dem):
        return float(self.util(x, dem).max())

    def solve(self, dem):
        """Fractional min-max; returns (x, U, ms). Falls back to shortest path if the solver fails."""
        self.d.value = self.expand(dem)
        t0 = time.perf_counter()
        try:
            self.prob.solve(solver=cp.CLARABEL)
            ok = self.prob.status in ("optimal", "optimal_inaccurate") and self.x.value is not None
        except cp.error.SolverError:
            ok = False
        ms = (time.perf_counter() - t0) * 1000
        if not ok:
            return self.x_sp, self.mlu(self.x_sp, dem), ms, False
        x = np.clip(self.x.value, 0, 1)
        for k in range(len(self.pairs)):                       # renormalise tiny solver drift
            sl = slice(self.off[k], self.off[k + 1]); x[sl] /= x[sl].sum()
        return x, float(self.U.value), ms, True

    def integral(self, x):
        """One path per OD pair (what plain OpenFlow routes can install): argmax of the fractional split."""
        y = np.zeros_like(x)
        for k in range(len(self.pairs)):
            sl = slice(self.off[k], self.off[k + 1]); y[self.off[k] + int(np.argmax(x[sl]))] = 1.0
        return y

    def greedy(self, dem, x0):
        """Reactive/heuristic: move the largest-demand ODs to the path with lowest resulting MLU (one pass)."""
        x = x0.copy()
        for k in np.argsort(-dem)[:20]:
            best, bj = None, None
            for j in range(self.npath[k]):
                y = x.copy(); y[self.off[k]:self.off[k + 1]] = 0; y[self.off[k] + j] = 1
                v = self.mlu(y, dem)
                if best is None or v < best - 1e-12:
                    best, bj = v, j
            x[self.off[k]:self.off[k + 1]] = 0; x[self.off[k] + bj] = 1
        return x


def run_policy(te, name, decide_dem, actual, meta, eta, integral=True):
    """decide_dem[i]: the demand estimate the policy may use at step i. Hysteresis state resets at week boundaries."""
    n = len(actual)
    mlu, ms, moves, ok = np.zeros(n), np.zeros(n), np.zeros(n), np.ones(n, bool)
    x, prev_wk = te.x_sp.copy(), None
    for i in range(n):
        if meta[i][0] != prev_wk:
            x, prev_wk = te.x_sp.copy(), meta[i][0]
        if name in ("shortest_path", "ecmp", "round_robin"):
            xn = {"shortest_path": te.x_sp, "ecmp": te.x_ecmp, "round_robin": te.x_rr}[name]
            moves[i] = 0
        else:
            dem = decide_dem[i]
            t0 = time.perf_counter()
            if name.endswith("_heuristic") or name == "reactive":
                xn = te.greedy(dem, x); ms[i] = (time.perf_counter() - t0) * 1000
            else:
                xf, _, ms[i], ok[i] = te.solve(dem)
                xn = te.integral(xf) if integral else xf
            if te.mlu(x, dem) - te.mlu(xn, dem) <= eta:          # hysteresis: keep the current routing
                xn = x
            moves[i] = np.abs(xn - x).sum() / 2 if integral else float((np.abs(xn - x) > 1e-6).any())
        x = xn
        mlu[i] = te.mlu(x, actual[i])
    return mlu, ms, moves, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--forecast", nargs="+", default=["lstm_s1", "tcn_s1"], help="names of pred_test_<name>.npy")
    ap.add_argument("--fractional", action="store_true", help="score fractional splits (not installable as plain OpenFlow routes)")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(ROOT / "configs" / "abilene.yaml"))
    tc, dc = cfg["te"], cfg["data"]
    fdir = ROOT / "results" / "abilene" / "forecast"
    actual_all = np.load(fdir / "actual_test.npy"); meta = np.load(fdir / "meta_test.npy")
    import json
    cols = np.array(json.load(open(fdir / "run_config.json"))["od_columns"])
    r, l = parse_topology(ROOT / dc["raw_dir"] / "topo-2003-04-10.txt")
    G, _ = build_digraph(r, l)
    pairs = [(c // 12 + 1, c % 12 + 1) for c in cols]
    te = TE(G, pairs, tc["k_paths"])
    # demand scale: TRAIN-set peak MLU under shortest path -> target (calibrated on train, applied to test)
    from abilene.forecasting import prepare
    D, sc, split, mb, _ = prepare(cfg)
    peak_train = max(te.mlu(te.x_sp, D["train"]["Y"][i, 0]) for i in range(0, len(D["train"]["Y"]), 7))
    scale = tc["target_peak_mlu"] / peak_train
    print("raw train peak MLU (shortest path) %.4f -> demand scale x%.1f (ASSUMPTION: stress scaling)" % (peak_train, scale))
    sel = lambda arr: arr[:, 0] if tc["agg"] == "first" else arr.max(1)
    act = actual_all[:, 0] * scale
    runs = {"shortest_path": None, "ecmp": None, "round_robin": None,
            "reactive": scale * np.vstack([np.zeros((1, act.shape[1])), act[:-1]]),   # last observed step
            "oracle_opt": act}
    for f in a.forecast:
        P = np.load(fdir / ("pred_test_%s.npy" % f))
        runs[f + "_heuristic"] = scale * sel(P)
        runs[f + "_opt"] = scale * sel(P)
    rows = []
    for name, dem in runs.items():
        mlu, ms, mv, ok = run_policy(te, name, dem, act, meta, tc["eta"], not a.fractional)
        rows.append(dict(policy=name, mean_MLU=float(mlu.mean()), p95_MLU=float(np.percentile(mlu, 95)), max_MLU=float(mlu.max()),
                         congested_frac=float((mlu > tc["congestion_mlu"]).mean()), mean_solve_ms=float(ms.mean()),
                         moves_per_step=float(mv.mean()), solver_fallbacks=int((~ok).sum()), n_steps=len(act)))
        np.save(fdir / ("mlu_%s.npy" % name), mlu)
        print(rows[-1])
    (ROOT / "results" / "tables").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).round(5).to_csv(ROOT / "results" / "tables" / "abilene_te_offline.csv", index=False)


if __name__ == "__main__":
    main()
