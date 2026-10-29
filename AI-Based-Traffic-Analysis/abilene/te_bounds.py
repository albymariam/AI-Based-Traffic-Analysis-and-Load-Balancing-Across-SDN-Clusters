"""Routing-structure bounds for the Abilene fluid model (faculty fix #3).

Answers: how far is the K-path LP used in the paper from the best possible min-max routing, and how does it compare
with demand-oblivious and semi-oblivious routing?  All policies see the TRUE demand of the step they are scored on
(oracle), so the differences below come from the routing structure only, not from forecasting.

  mcf_optimal    full multi-commodity-flow LP over ALL paths (fractional)  -> lower bound on MLU
  kpath_K        min-max LP over K candidate paths (fractional)             K in --ks
  kpath_K_int    the same, rounded to ONE path per OD (what plain OpenFlow can install)
  oblivious_opt  optimal oblivious routing (Applegate & Cohen 2003), fixed splits, never looks at demand
  smore_K        SMORE-style semi-oblivious: K paths taken from the oblivious routing, rates re-optimised per step
  vlb            Valiant load balancing (oblivious, uniform detour through every node)
  shortest_path  single OSPF shortest path

   python -m abilene.te_bounds                       # real test demand (needs results/abilene/forecast/)
   python -m abilene.te_bounds --synthetic --steps 200   # quick self-test without the dataset
"""
import argparse
import json
import time
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import yaml
from scipy import sparse
from scipy.optimize import linprog

from abilene.topology_mapping import build_digraph, k_paths, parse_topology

ROOT = Path(__file__).resolve().parents[1]
TOPO = "data/raw/abilene/topo-2003-04-10.txt"


# ------------------------------------------------------------------ graph helpers
class Net:
    def __init__(self, G, pairs):
        self.G, self.pairs = G, pairs
        self.nodes = sorted(G.nodes)
        self.nidx = {v: i for i, v in enumerate(self.nodes)}
        self.links = list(G.edges)
        self.lidx = {e: i for i, e in enumerate(self.links)}
        self.cap = np.array([G.edges[e]["capacity_mbps"] for e in self.links], dtype=float)
        self.n, self.m, self.P = len(self.nodes), len(self.links), len(pairs)
        # node-link incidence: +1 where the link leaves the node, -1 where it enters
        inc = np.zeros((self.n, self.m))
        for j, (u, v) in enumerate(self.links):
            inc[self.nidx[u], j] += 1.0
            inc[self.nidx[v], j] -= 1.0
        self.inc = inc


def path_matrix(net, paths_by_pair):
    """-> sparse A (links x paths), offsets, flat path list."""
    cols, rows, flat, off = [], [], [], [0]
    c = 0
    for ps in paths_by_pair:
        for p in ps:
            for e in zip(p, p[1:]):
                rows.append(net.lidx[e]); cols.append(c)
            flat.append(p); c += 1
        off.append(c)
    A = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(net.m, c))
    return A, np.array(off), flat


# ------------------------------------------------------------------ 1. full multi-commodity-flow LP
class MCF:
    """min theta s.t. sum_s f_s(e) <= theta*cap(e); flow conservation per source (source-aggregated: m*n variables)."""

    def __init__(self, net):
        self.net = net
        n, m = net.n, net.m
        self.nv = n * m + 1
        A_ub = sparse.hstack([sparse.hstack([sparse.identity(m)] * n), sparse.csr_matrix(-net.cap[:, None])]).tocsr()
        A_eq = sparse.hstack([sparse.block_diag([sparse.csr_matrix(net.inc)] * n), sparse.csr_matrix((n * n, 1))]).tocsr()
        self.A_ub, self.A_eq = A_ub, A_eq
        self.c = np.zeros(self.nv); self.c[-1] = 1.0
        self.b_ub = np.zeros(m)

    def solve(self, D):
        """D: (n x n) demand matrix in Mbps indexed by node position. Returns theta."""
        n = self.net.n
        b = np.zeros(n * n)
        for s in range(n):
            b[s * n:(s + 1) * n] = -D[s]
            b[s * n + s] = D[s].sum()           # source emits everything, every other node absorbs its demand
        res = linprog(self.c, A_ub=self.A_ub, b_ub=self.b_ub, A_eq=self.A_eq, b_eq=b,
                      bounds=[(0, None)] * self.nv, method="highs")
        return float(res.x[-1]) if res.success else np.nan


def demand_matrix(net, dem):
    D = np.zeros((net.n, net.n))
    for k, (s, d) in enumerate(net.pairs):
        D[net.nidx[s], net.nidx[d]] = dem[k]
    return D


# ------------------------------------------------------------------ 2. K-path LP (fractional) + rounding
class KPath:
    def __init__(self, net, paths_by_pair):
        self.net = net
        self.A, self.off, self.flat = path_matrix(net, paths_by_pair)
        self.npath = np.diff(self.off)
        self.np_ = self.A.shape[1]
        self.hops = np.array([len(p) - 1 for p in self.flat], dtype=float)
        rows, cols = [], []
        for k in range(len(paths_by_pair)):
            for j in range(self.off[k], self.off[k + 1]):
                rows.append(k); cols.append(j)
        self.Aeq = sparse.hstack([sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(paths_by_pair), self.np_)),
                                  sparse.csr_matrix((len(paths_by_pair), 1))]).tocsr()
        self.beq = np.ones(len(paths_by_pair))
        self.x_sp = np.zeros(self.np_); self.x_sp[self.off[:-1]] = 1.0

    def expand(self, dem):
        return np.repeat(dem, self.npath)

    def util(self, x, dem):
        return (self.A @ (self.expand(dem) * x)) / self.net.cap

    def solve(self, dem):
        d = self.expand(dem)
        S = sparse.diags(1.0 / self.net.cap) @ self.A @ sparse.diags(d)         # link x path, per-unit-split utilisation
        A_ub = sparse.hstack([S, sparse.csr_matrix(-np.ones((self.net.m, 1)))]).tocsr()
        cost = np.concatenate([1e-4 * self.hops * d / self.net.cap.max(), [1.0]])
        res = linprog(cost, A_ub=A_ub, b_ub=np.zeros(self.net.m), A_eq=self.Aeq, b_eq=self.beq,
                      bounds=[(0, 1)] * self.np_ + [(0, None)], method="highs")
        if not res.success:
            return self.x_sp.copy()
        x = np.clip(res.x[:-1], 0, 1)
        for k in range(len(self.npath)):
            sl = slice(self.off[k], self.off[k + 1]); x[sl] /= max(x[sl].sum(), 1e-12)
        return x

    def integral(self, x):
        y = np.zeros_like(x)
        for k in range(len(self.npath)):
            sl = slice(self.off[k], self.off[k + 1]); y[self.off[k] + int(np.argmax(x[sl]))] = 1.0
        return y


# ------------------------------------------------------------------ 3. optimal oblivious routing (Applegate & Cohen)
def oblivious_routing(net, verbose=True):
    """Fixed per-pair flow fractions f[ab, e] minimising the worst-case ratio  r = max_D MLU_obl(D) / MLU_opt(D).
    Dual of the inner 'worst demand' problem for every link e (potentials q, link lengths pi).
    Returns (f array P x m, r)."""
    n, m, P = net.n, net.m, net.P
    cap = net.cap
    # variable layout: [ f (P*m) | pi (m*m) | q (m*P*n) | r ]
    nf, npi, nq = P * m, m * m, m * P * n
    iF = lambda k, e: k * m + e
    iPi = lambda e, l: nf + e * m + l
    iQ = lambda e, k, v: nf + npi + (e * P + k) * n + v
    iR = nf + npi + nq
    nv = iR + 1
    rows_eq, cols_eq, vals_eq, b_eq = [], [], [], []
    r_ = 0
    for k, (a, b) in enumerate(net.pairs):                   # unit flow a->b for every pair
        ia, ib = net.nidx[a], net.nidx[b]
        for v in range(n):
            for e in range(m):
                c = net.inc[v, e]
                if c != 0:
                    rows_eq.append(r_); cols_eq.append(iF(k, e)); vals_eq.append(c)
            b_eq.append(1.0 if v == ia else (-1.0 if v == ib else 0.0))
            r_ += 1
    n_eq = r_
    rows_u, cols_u, vals_u, b_u = [], [], [], []
    r_ = 0
    for e in range(m):                                       # sum_l cap_l * pi_e(l) <= r
        for l in range(m):
            rows_u.append(r_); cols_u.append(iPi(e, l)); vals_u.append(cap[l])
        rows_u.append(r_); cols_u.append(iR); vals_u.append(-1.0); b_u.append(0.0); r_ += 1
    for e in range(m):
        for k, (a, b) in enumerate(net.pairs):
            ia, ib = net.nidx[a], net.nidx[b]
            for l, (u, v) in enumerate(net.links):           # pi_e(l) + q_i - q_j >= 0   ->  -pi - q_i + q_j <= 0
                iu, iv = net.nidx[u], net.nidx[v]
                rows_u += [r_, r_, r_]; cols_u += [iPi(e, l), iQ(e, k, iu), iQ(e, k, iv)]; vals_u += [-1.0, -1.0, 1.0]
                b_u.append(0.0); r_ += 1
            # q_b - q_a >= f_ab(e)/cap_e   ->  f/cap_e - q_b + q_a <= 0
            rows_u += [r_, r_, r_]; cols_u += [iF(k, e), iQ(e, k, ib), iQ(e, k, ia)]
            vals_u += [1.0 / cap[e], -1.0, 1.0]; b_u.append(0.0); r_ += 1
    A_eq = sparse.csr_matrix((vals_eq, (rows_eq, cols_eq)), shape=(n_eq, nv))
    A_ub = sparse.csr_matrix((vals_u, (rows_u, cols_u)), shape=(r_, nv))
    bounds = [(0, None)] * (nf + npi) + [(None, None)] * nq + [(0, None)]
    for e in range(m):                                       # gauge: q_b = 0
        for k, (a, b) in enumerate(net.pairs):
            ib = net.nidx[b]
            bounds[iQ(e, k, ib)] = (0, 0)
    c = np.zeros(nv); c[iR] = 1.0
    t0 = time.perf_counter()
    res = linprog(c, A_ub=A_ub, b_ub=np.array(b_u), A_eq=A_eq, b_eq=np.array(b_eq), bounds=bounds, method="highs")
    if verbose:
        print("oblivious LP: %d vars, %d rows, status=%s, %.1f s, ratio r=%.4f" %
              (nv, A_ub.shape[0] + n_eq, res.status, time.perf_counter() - t0, res.x[iR] if res.success else float("nan")))
    if not res.success:
        raise RuntimeError("oblivious LP failed: " + res.message)
    f = np.clip(res.x[:nf].reshape(P, m), 0, None)
    return f, float(res.x[iR])


def flow_paths(net, k, fk, k_keep, eps=1e-7):
    s, t = net.pairs[k]
    f = fk.copy()
    out = []
    while f.max() > eps and len(out) < 50:
        H = nx.DiGraph()
        for e, (u, v) in enumerate(net.links):
            if f[e] > eps:
                H.add_edge(u, v, w=f[e], e=e)
        if not nx.has_path(H, s, t):
            break
        # widest path: Dijkstra on 1/w gives a decent approximation; refine by bottleneck
        p = nx.shortest_path(H, s, t, weight=lambda u, v, d: 1.0 / d["w"])
        es = [H[u][v]["e"] for u, v in zip(p, p[1:])]
        w = min(f[e] for e in es)
        for e in es:
            f[e] -= w
        out.append((w, p))
    out.sort(key=lambda x: -x[0])
    out = out[:k_keep]
    tot = sum(w for w, _ in out) or 1.0
    return [p for _, p in out], np.array([w / tot for w, _ in out])


# ------------------------------------------------------------------ 4. Valiant load balancing
def vlb_fractions(net):
    """f[ab, e]: route a->m->b through every node m (uniform), each leg on the hop-count shortest path."""
    G = nx.DiGraph(); G.add_edges_from(net.links)
    sp = dict(nx.all_pairs_shortest_path(G))
    f = np.zeros((net.P, net.m))
    for k, (a, b) in enumerate(net.pairs):
        for mid in net.nodes:
            for s, t in ((a, mid), (mid, b)):
                if s == t:
                    continue
                for e in zip(sp[s][t], sp[s][t][1:]):
                    f[k, net.lidx[e]] += 1.0 / net.n
    return f


# ------------------------------------------------------------------ evaluation
def evaluate(net, act, ks, steps_stride=1, include_obl=True, verbose=True):
    idx = np.arange(0, len(act), steps_stride)
    act = act[idx]
    T = len(act)
    mcf = MCF(net)
    theta_opt = np.array([mcf.solve(demand_matrix(net, d)) for d in act])
    res = {"mcf_optimal": theta_opt}
    kp = {}
    for K in ks:
        paths = [k_paths(net.G, s, d, K) for s, d in net.pairs]
        kp[K] = KPath(net, paths)
        xs = [kp[K].solve(d) for d in act]
        res["kpath_%d" % K] = np.array([kp[K].util(x, d).max() for x, d in zip(xs, act)])
        res["kpath_%d_int" % K] = np.array([kp[K].util(kp[K].integral(x), d).max() for x, d in zip(xs, act)])
    res["shortest_path"] = np.array([kp[ks[0]].util(kp[ks[0]].x_sp, d).max() for d in act])
    if include_obl:
        f_obl, r = oblivious_routing(net, verbose)
        res["oblivious_opt"] = (f_obl.T @ act.T / net.cap[:, None]).max(0)
        f_vlb = vlb_fractions(net)
        res["vlb"] = (f_vlb.T @ act.T / net.cap[:, None]).max(0)
        for K in (4,):
            paths, wts = [], []
            for k in range(net.P):
                ps, w = flow_paths(net, k, f_obl[k], K)
                if not ps:                                   # numerically empty flow: fall back to the shortest path
                    ps, w = k_paths(net.G, *net.pairs[k], 1), np.array([1.0])
                paths.append(ps); wts.append(w)
            sm = KPath(net, paths)
            res["smore_%d" % K] = np.array([sm.util(sm.solve(d), d).max() for d in act])
            x_obl = np.concatenate(wts)
            res["oblivious_paths_static"] = np.array([sm.util(x_obl, d).max() for d in act])
    else:
        r = float("nan")
    rows = []
    for name, v in res.items():
        ratio = v / theta_opt
        rows.append(dict(policy=name, mean_MLU=v.mean(), p95_MLU=np.percentile(v, 95), max_MLU=v.max(),
                         mean_ratio_to_opt=np.mean(ratio), max_ratio_to_opt=ratio.max(), n_steps=T))
    return pd.DataFrame(rows), r


def synthetic_demand(net, T, seed=0, peak_mlu=0.8):
    rng = np.random.default_rng(seed)
    base = rng.lognormal(0, 1.2, net.P)
    t = np.arange(T)
    act = base[None] * (1 + 0.5 * np.sin(2 * np.pi * t / 288))[:, None] * rng.lognormal(0, 0.3, (T, net.P))
    sp = KPath(net, [k_paths(net.G, s, d, 1) for s, d in net.pairs])
    peak = max(sp.util(sp.x_sp, d).max() for d in act)
    return act * peak_mlu / peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", nargs="+", type=int, default=[1, 2, 3, 4, 6, 8])
    ap.add_argument("--steps", type=int, default=None, help="use only the first N test steps")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--synthetic", action="store_true", help="synthetic demand on the fixture topology (self-test only)")
    ap.add_argument("--no-oblivious", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "results" / "tables" / "abilene_te_bounds.csv"))
    a = ap.parse_args()
    cfg = yaml.safe_load(open(ROOT / "configs" / "abilene.yaml"))
    if a.synthetic:
        r_, l_ = parse_topology(ROOT / "tests" / "fixtures" / "topo-2003-04-10.txt")
        G, _ = build_digraph(r_, l_)
        pairs = [(i, j) for i in range(1, 13) for j in range(1, 13) if i != j]
        net = Net(G, pairs)
        act = synthetic_demand(net, a.steps or 300)
    else:
        fdir = ROOT / "results" / "abilene" / "forecast"
        cols = np.array(json.load(open(fdir / "run_config.json"))["od_columns"])
        r_, l_ = parse_topology(ROOT / cfg["data"]["raw_dir"] / "topo-2003-04-10.txt")
        G, _ = build_digraph(r_, l_)
        pairs = [(c // 12 + 1, c % 12 + 1) for c in cols]
        net = Net(G, pairs)
        # identical demand scaling to abilene.te_offline: train-set peak MLU under shortest path -> target
        from abilene.forecasting import prepare
        D, sc, split, mb, _ = prepare(cfg)
        sp = KPath(net, [k_paths(G, s, d, 1) for s, d in pairs])
        peak_train = max(sp.util(sp.x_sp, D["train"]["Y"][i, 0]).max() for i in range(0, len(D["train"]["Y"]), 7))
        scale = cfg["te"]["target_peak_mlu"] / peak_train
        print("demand scale x%.1f (same rule as te_offline)" % scale)
        act = np.load(fdir / "actual_test.npy")[:, 0] * scale
        if a.steps:
            act = act[:a.steps]
    df, r = evaluate(net, act, a.ks, a.stride, not a.no_oblivious)
    print("worst-case oblivious ratio (LP value): %.3f" % r)
    pd.set_option("display.width", 160)
    print(df.round(4).to_string(index=False))
    if not a.synthetic:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        df.round(5).to_csv(a.out, index=False)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
