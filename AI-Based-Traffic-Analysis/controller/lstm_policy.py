"""M1 online policy: LSTM forecast of every link -> hysteresis rerouting of ELEPHANT flows via real FlowMods.
Mice follow default routes. A flow moves only if predicted bottleneck improves by > eta, and not within cooldown."""
import csv
import json
import threading
import time

import numpy as np
import torch

from common.config import ROOT
from common.features import FEATURES
from models.dataset import normalize
from models.lstm_predictor import LSTMTrafficPredictor
from topology.mininet_topology import get_k_shortest_paths


class LSTMReroute(threading.Thread):
    def __init__(self, api, tele, runner, G, topo_cfg, exp_cfg, run_dir, t0):
        super(LSTMReroute, self).__init__(daemon=True)
        torch.set_num_threads(1)
        ck = ROOT / "models" / "checkpoints"
        meta = json.loads((ck / "norm.json").read_text())
        pc, rc = exp_cfg["predictor"], exp_cfg["reroute"]
        self.W, self.norm = meta["window"], meta["norm"]
        self.model = LSTMTrafficPredictor(input_dim=len(FEATURES), hidden_dim=pc["hidden_dim"],
                                          num_layers=pc["num_layers"], forecast_horizon=meta["horizon"],
                                          dropout=pc["dropout"])
        self.model.load_state_dict(torch.load(ck / "lstm.pt", map_location="cpu"))
        self.model.eval()
        self.api, self.tele, self.runner, self.G, self.t0 = api, tele, runner, G, t0
        self.cap = float(topo_cfg["emulated_capacity_mbps"])
        self.k = topo_cfg["k_paths"]
        self.links = [(u, v) for u, v in topo_cfg["links"]] + [(v, u) for u, v in topo_cfg["links"]]
        self.thr, self.eta, self.cooldown = rc["util_threshold"], rc["hysteresis_eta"], rc["cooldown_s"]
        self.interval = float(exp_cfg["telemetry"]["poll_interval_s"])
        self.last, self.cache = {}, {}
        self.fh = open(run_dir / "reroutes.csv", "w", newline="")
        self.w = csv.writer(self.fh)
        self.w.writerow(["t", "flow_id", "src", "dst", "old_path", "new_path", "cur_cost", "new_cost",
                         "infer_ms", "decision_ms", "pin_ms"])
        self._halt = threading.Event()

    def _cands(self, s, d):
        if (s, d) not in self.cache:
            self.cache[(s, d)] = get_k_shortest_paths(self.G, s, d, self.k)
        return self.cache[(s, d)]

    def run(self):
        torch.set_num_threads(1)     # the thread count is per thread: set it in the thread that runs inference
        nxt = time.time() + 5
        while not self._halt.is_set():
            self._halt.wait(max(0.0, nxt - time.time()))
            nxt += self.interval
            try:
                self.tick()
            except Exception as ex:
                print("[lstm_policy] tick failed:", ex)

    def tick(self):
        t_in = time.perf_counter()
        hist = self.tele.hist
        if any(len(hist.get(l, ())) < self.W for l in self.links):
            return
        X = np.array([list(hist[l])[-self.W:] for l in self.links], dtype=np.float32)
        with torch.no_grad():
            P = self.model(torch.from_numpy(normalize(X, self.norm))).numpy()
        uhat = {l: float(P[i].max()) for i, l in enumerate(self.links)}   # worst predicted util over horizon
        infer_ms = (time.perf_counter() - t_in) * 1000
        now = time.time() - self.t0
        hot = {l for l, u in uhat.items() if u > self.thr}
        if not hot:
            return
        flows = [e for e in self.runner.elephants()
                 if any(l in hot for l in zip(e["path"], e["path"][1:]))
                 and now - self.last.get(e["id"], -1e9) >= self.cooldown]
        flows.sort(key=lambda e: -e["rate"])
        for e in flows:
            cur = list(zip(e["path"], e["path"][1:]))
            if not any(uhat[l] > self.thr for l in cur):
                continue                                   # already relieved by an earlier move this tick
            r = e["rate"] / self.cap
            cur_cost = max(uhat[l] for l in cur)
            best, best_cost = None, cur_cost
            for p in self._cands(e["src"], e["dst"]):
                if p == e["path"]:
                    continue
                cost = max(uhat[l] + (0.0 if l in cur else r) for l in zip(p, p[1:]))
                if cost < best_cost:
                    best, best_cost = p, cost
            if best is None or cur_cost - best_cost <= self.eta:     # hysteresis
                continue
            dec_ms = (time.perf_counter() - t_in) * 1000
            try:
                pin_ms = self.api.pin(e["src"], e["dst"], e["port"], e["protos"], best)
            except Exception as ex:
                print("[lstm_policy] pin failed:", ex)
                continue
            new = list(zip(best, best[1:]))
            for l in cur:
                if l not in new:
                    uhat[l] -= r
            for l in new:
                if l not in cur:
                    uhat[l] += r
            self.runner.set_path(e["id"], best)
            self.last[e["id"]] = now
            self.w.writerow([round(now, 2), e["id"], e["src"], e["dst"], "-".join(map(str, e["path"])),
                             "-".join(map(str, best)), round(cur_cost, 3), round(best_cost, 3),
                             round(infer_ms, 2), round(dec_ms, 2), round(pin_ms, 2)])
            self.fh.flush()

    def stop(self):
        self._halt.set()
        self.join(timeout=3)
        self.fh.close()