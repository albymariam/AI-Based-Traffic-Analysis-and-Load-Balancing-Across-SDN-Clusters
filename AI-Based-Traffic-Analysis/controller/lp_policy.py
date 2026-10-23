"""Methodology 2 online policy: Transformer/TCN forecast -> min-max LP -> rounded per-flow path pinning.
Only ELEPHANT flows are controlled (mice follow default routes, they count as background load)."""
import csv
import threading
import time

import numpy as np
import torch

from common.config import ROOT
from common.features import FEATURES
from models.dataset import normalize
from models.transformer_predictor import build_model
from optimization.minmax_routing import solve_minmax
from topology.mininet_topology import get_k_shortest_paths


class LPReroute(threading.Thread):
    def __init__(self, api, tele, runner, G, topo_cfg, exp_cfg, run_dir, t0, arch="transformer"):
        super(LPReroute, self).__init__(daemon=True)
        torch.set_num_threads(1)
        ck = torch.load(ROOT / "models" / "checkpoints" / ("%s.pt" % arch), map_location="cpu")
        self.W, self.norm = ck["window"], ck["norm"]
        self.model = build_model(arch, len(FEATURES), ck["window"], ck["horizon"],
                                 FEATURES.index("utilization"), ck["pc"])
        self.model.load_state_dict(ck["state"])
        self.model.eval()
        self.lp = exp_cfg["lp"]
        self.api, self.tele, self.runner, self.G, self.t0 = api, tele, runner, G, t0
        self.run_dir = run_dir
        self.cap = float(topo_cfg["emulated_capacity_mbps"])
        self.k = topo_cfg["k_paths"]
        self.links = [(u, v) for u, v in topo_cfg["links"]] + [(v, u) for u, v in topo_cfg["links"]]
        self.last, self.cache = {}, {}
        self.fh = open(run_dir / "reroutes.csv", "w", newline="")
        self.w = csv.writer(self.fh)
        self.w.writerow(["t", "flow_id", "src", "dst", "old_path", "new_path", "cur_cost", "new_cost",
                         "infer_ms", "decision_ms", "pin_ms", "lp_ms"])
        self.fp = open(run_dir / "plans.csv", "w", newline="")
        self.wp = csv.writer(self.fp)
        self.wp.writerow(["t", "n_elephants", "n_commodities", "theta_cur", "theta_lp_frac", "theta_plan",
                          "infer_ms", "lp_ms", "applied", "n_moves"])
        self._halt = threading.Event()

    def _cands(self, s, d):
        if (s, d) not in self.cache:
            self.cache[(s, d)] = get_k_shortest_paths(self.G, s, d, self.k)
        return self.cache[(s, d)]

    def run(self):
        torch.set_num_threads(1)     # the thread count is per thread: set it in the thread that runs inference
        nxt = time.time() + 5
        while not self._halt.is_set():
            if self._halt.wait(max(0.0, nxt - time.time())):
                break
            nxt += self.lp["interval_s"]
            if self._halt.is_set():
                break
            try:
                self.tick()
            except Exception as ex:
                print("[lp_policy] tick failed:", repr(ex))

    def _theta(self, ubg, els, paths_by_id):
        load = dict(ubg)
        for e in els:
            p = paths_by_id[e["id"]]
            for l in zip(p, p[1:]):
                load[l] += e["rate"] / self.cap
        return max(load.values())

    def _contrib(self, els):
        c = {}
        for e in els:
            for l in zip(e["path"], e["path"][1:]):
                c[l] = c.get(l, 0.0) + e["rate"] / self.cap
        return c

    def _windows(self):
        hist = self.tele.hist
        if any(len(hist.get(l, ())) < self.W for l in self.links):
            return None
        return np.array([list(hist[l])[-self.W:] for l in self.links], dtype=np.float32)

    def _forecast(self, X):
        with torch.no_grad():
            P = self.model(torch.from_numpy(normalize(X, self.norm))).numpy()
        agg = P.mean(1) if self.lp["aggregate"] == "mean" else P.max(1)
        return {l: float(agg[i]) for i, l in enumerate(self.links)}

    def tick(self):
        t_in = time.perf_counter()
        X = self._windows()
        if X is None:
            return
        els = self.runner.elephants()
        if not els:
            return
        uhat = self._forecast(X)
        infer_ms = (time.perf_counter() - t_in) * 1000
        now = time.time() - self.t0
        contrib = self._contrib(els)
        ubg = {l: max(0.0, uhat[l] - contrib.get(l, 0.0)) for l in self.links}
        self._plan_and_apply(els, ubg, infer_ms, t_in, now)

    def _plan_and_apply(self, els, ubg, infer_ms, t_in, now, extra=()):
        groups = {}
        for e in els:
            groups.setdefault((e["src"], e["dst"]), []).append(e)
        keys, coms = list(groups), []
        for key in keys:
            paths = [list(p) for p in self._cands(*key)]
            for e in groups[key]:
                if e["path"] not in paths:
                    paths.append(e["path"])
            coms.append(dict(demand=sum(e["rate"] for e in groups[key]), paths=paths))
        sol = solve_minmax(self.links, coms, ubg, self.cap, self.lp["hop_epsilon"])
        if sol is None:
            return
        theta_frac, xs, lp_ms = sol

        # integer realisation: frozen (cooldown) flows first, then largest-deficit path with stickiness
        new = {e["id"]: e["path"] for e in els}
        for key, com, x in zip(keys, coms, xs):
            target, assigned = x * com["demand"], np.zeros(len(com["paths"]))
            fl = sorted(groups[key], key=lambda e: -e["rate"])
            frozen = [e for e in fl if now - self.last.get(e["id"], -1e9) < self.lp["cooldown_s"]]
            for e in frozen:
                assigned[com["paths"].index(e["path"])] += e["rate"]
            for e in fl:
                if e in frozen:
                    continue
                cur = com["paths"].index(e["path"])
                score = target - assigned
                score[cur] += self.lp["stickiness"] * e["rate"]
                j = int(np.argmax(score))
                assigned[j] += e["rate"]
                new[e["id"]] = com["paths"][j]

        cur_paths = {e["id"]: e["path"] for e in els}
        theta_cur = self._theta(ubg, els, cur_paths)
        theta_plan = self._theta(ubg, els, new)
        moves = sorted([e for e in els if new[e["id"]] != e["path"]], key=lambda e: -e["rate"])
        applied = int(theta_cur - theta_plan > self.lp["eta"] and len(moves) > 0)   # hysteresis
        n_moves = 0
        if applied:
            for e in moves[: self.lp["max_moves_per_tick"]]:
                dec_ms = (time.perf_counter() - t_in) * 1000
                try:
                    pin_ms = self.api.pin(e["src"], e["dst"], e["port"], e["protos"], new[e["id"]])
                except Exception as ex:
                    print("[lp_policy] pin failed:", ex)
                    continue
                self.runner.set_path(e["id"], new[e["id"]])
                self.last[e["id"]] = now
                n_moves += 1
                self.w.writerow([round(now, 2), e["id"], e["src"], e["dst"], "-".join(map(str, e["path"])),
                                 "-".join(map(str, new[e["id"]])), round(theta_cur, 3), round(theta_plan, 3),
                                 round(infer_ms, 2), round(dec_ms, 2), round(pin_ms, 2), round(lp_ms, 2)])
            self.fh.flush()
        self.wp.writerow([round(now, 2), len(els), len(coms), round(theta_cur, 4), round(theta_frac, 4),
                          round(theta_plan, 4), round(infer_ms, 2), round(lp_ms, 2), applied, n_moves]
                         + list(extra))
        self.fp.flush()

    def stop(self):
        self._halt.set()
        self.join(timeout=3)
        self.fh.close()
        self.fp.close()