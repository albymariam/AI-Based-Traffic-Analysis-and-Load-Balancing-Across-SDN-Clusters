"""Distributed-controller emulation with state staleness.
C logical controllers (one per domain in topology.yaml), each owning the elephants whose source switch is in its
domain. Controller k sees FRESH telemetry/flow state for links it owns and state that is `stale_s` seconds OLD for
remote links (stale telemetry window + stale knowledge of its own flows' placement at that time). Controllers act
staggered and independently, so they cannot see each other's decisions -> herding when stale_s grows.
stale_s = 0 -> distributed but with perfectly fresh state (isolates the effect of decentralisation)."""
import csv
import time
from collections import deque

import numpy as np
import torch

from controller.lp_policy import LPReroute


class DistLPReroute(LPReroute):
    def __init__(self, api, tele, runner, G, topo_cfg, exp_cfg, run_dir, t0, arch="transformer", stale_s=0.0):
        super(DistLPReroute, self).__init__(api, tele, runner, G, topo_cfg, exp_cfg, run_dir, t0, arch)
        self.fp.close()                                  # re-open plans.csv with two extra columns
        self.fp = open(run_dir / "plans.csv", "w", newline="")
        self.wp = csv.writer(self.fp)
        self.wp.writerow(["t", "n_elephants", "n_commodities", "theta_cur", "theta_lp_frac", "theta_plan",
                          "infer_ms", "lp_ms", "applied", "n_moves", "ctrl", "view_age_s"])
        self.stale_s = float(stale_s)
        names = sorted(topo_cfg["domains"])
        self.C = len(names)
        self.dom_of = {s: i for i, n in enumerate(names) for s in topo_cfg["domains"][n]}
        assert set(self.dom_of) == set(topo_cfg["switches"]), "domains must cover all switches"
        self.own_mask = [np.array([self.dom_of[a] == k for (a, b) in self.links]) for k in range(self.C)]
        self.ring = deque(maxlen=256)

    def run(self):
        torch.set_num_threads(1)     # the thread count is per thread: set it in the thread that runs inference
        period = self.lp["interval_s"] / self.C           # staggered: one controller acts every period
        nxt, i = time.time() + 5.0, 0
        while not self._halt.is_set():
            self._halt.wait(max(0.0, nxt - time.time()))
            nxt += period
            try:
                self._ctrl_tick(i % self.C)
            except Exception as ex:
                print("[dist_lp] tick failed:", repr(ex))
            i += 1

    def _ctrl_tick(self, k):
        t_in = time.perf_counter()
        X = self._windows()
        if X is None:
            return
        now = time.time() - self.t0
        els_all = self.runner.elephants()
        self.ring.append(dict(t=now, X=X, els=els_all))   # global snapshot history (stale views come from here)
        if self.stale_s <= 0:
            st = self.ring[-1]
        else:
            cand = [s for s in self.ring if s["t"] <= now - self.stale_s]
            if not cand:
                return                                    # not enough history yet: controller idles
            st = cand[-1]
        own_now = [e for e in els_all if self.dom_of[e["src"]] == k]
        if not own_now:
            return
        mask = self.own_mask[k]
        Xv = np.where(mask[:, None, None], X, st["X"])    # fresh for own links, stale for remote links
        uhat = self._forecast(Xv)
        infer_ms = (time.perf_counter() - t_in) * 1000
        c_now = self._contrib(own_now)
        c_old = self._contrib([e for e in st["els"] if self.dom_of[e["src"]] == k])
        ubg = {l: max(0.0, uhat[l] - (c_now if mask[i] else c_old).get(l, 0.0))
               for i, l in enumerate(self.links)}
        self._plan_and_apply(own_now, ubg, infer_ms, t_in, now, extra=[k, round(now - st["t"], 2)])