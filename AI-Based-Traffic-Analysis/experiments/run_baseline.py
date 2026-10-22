"""Run ONE (method, load, profile, seed) experiment on the real Mininet+Ryu testbed.

  sudo -E env PYTHONPATH=. ~/venvs/ml/bin/python experiments/run_baseline.py \
        --method ecmp --load moderate --profile steady --seed 1
Methods: static_shortest | static_tracked | ecmp | round_robin
(static_tracked = same physical routes as static_shortest, but every flow is pinned explicitly, so
 active_flows telemetry is available -> use it for collecting predictor training data.)
"""
import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from traffic.generator import FlowRunner, build_pairs, build_schedule, start_mouse_servers


import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.config import load_all, make_run_dir, set_seed
from controller.rest_client import TEClient
from controller.telemetry_logger import TelemetryLogger
from evaluation.summarize_run import summarize
from topology import portmap as pm
from topology.mininet_net import build_network
from topology.mininet_topology import create_nsfnet_topology, get_k_shortest_paths
from traffic.generator import FlowRunner, build_pairs, build_schedule

METHODS = {"static_shortest": ("static", None), "static_tracked": ("static", "shortest"),
           "ecmp": ("ecmp", None), "round_robin": ("static", "rr"),
           "lstm_hysteresis": ("static", "shortest"),
           "transformer_minmax_lp": ("static", "shortest"),
           "tcn_minmax_lp": ("static", "shortest")}

for _s in (0, 1, 2, 5, 10, 20):
    METHODS["dist_lp_s%d" % _s] = ("static", "shortest")

class PathPolicy:
    def __init__(self, kind, G, cfg):
        self.kind, self.G, self.cfg = kind, G, cfg
        self.nh, self.cache, self.ctr = pm.next_hops(cfg), {}, {}

    def path(self, s, d):
        if self.kind == "shortest":
            return pm.static_path(self.nh, s, d)
        if (s, d) not in self.cache:
            self.cache[(s, d)] = get_k_shortest_paths(self.G, s, d, self.cfg["k_paths"])
        ps, i = self.cache[(s, d)], self.ctr.get((s, d), 0)
        self.ctr[(s, d)] = i + 1
        return ps[i % len(ps)]                          # per-pair round-robin over K paths


def start_probes(net, pairs, policy, api, total_s, interval):
    procs = []
    for s, d in pairs:
        if policy:
            p = policy.path(s, d)
            api.pin(s, d, None, [1], p)
            api.pin(d, s, None, [1], p[::-1])           # pin the echo-reply direction too
        n = int(total_s / interval)
        procs.append(((s, d), net.get("h%d" % s).popen(
            ["ping", "-D", "-i", str(interval), "-c", str(n), pm.host_ip(d)])))
    return procs


def collect_probes(procs, run_dir, t0):
    with open(run_dir / "probes.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["t", "src", "dst", "rtt_ms"])
        for (s, d), p in procs:
            try:
                out, _ = p.communicate(timeout=30)
            except Exception:
                p.kill()
                out, _ = p.communicate()
            for m in re.finditer(r"\[(\d+\.\d+)\].*?time=([\d.]+) ms", out.decode(errors="ignore")):
                wr.writerow([round(float(m.group(1)) - t0, 3), s, d, m.group(2)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=list(METHODS))
    ap.add_argument("--load", default="moderate")
    ap.add_argument("--profile", default="steady")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--duration", type=int, default=None, help="override traffic duration (s)")
    ap.add_argument("--check", action="store_true", help="only build net + pingall")
    ap.add_argument("--schedule", default=None, help="CSV flow schedule (e.g. Abilene replay) instead of the synthetic generator")
    a = ap.parse_args()
    if os.geteuid() != 0:
        sys.exit("Run with sudo (Mininet needs root).")

    cfgs = load_all()
    topo, tcfg, exp = cfgs["topology"], cfgs["traffic"], cfgs["experiment"]
    mode, pol_kind = METHODS[a.method]
    set_seed(a.seed)
    api = TEClient(exp["controller"]["rest_host"], exp["controller"]["rest_port"])

    net = None
    try:
        net, pmap = build_network(topo, exp["controller"]["rest_host"], exp["controller"]["of_port"])
        api.wait_switches(len(topo["switches"]))
        api.set_mode(mode)
        time.sleep(2)
        if a.check:
            loss = net.pingAll()
            print("pingAll dropped: %.1f%%" % loss)
            return

        warm = tcfg["warmup_s"]
        total = warm + (a.duration or tcfg["duration_s"])
        run = make_run_dir(a.method, "%s_%s" % (a.load, a.profile), a.seed)
        if a.schedule:
            from abilene.replay import load_schedule
            sched, pairs, w, info = load_schedule(a.schedule, topo)
        else:
            sched, pairs, w, info = build_schedule(topo, tcfg, a.load, a.profile, total, a.seed)
        print("run dir:", run, "| flows:", info["n_flows"], "| offered %.1f Mbps | avg hops %.2f"
              % (info["demand_mbps"], info["avg_hops"]))
        with open(run / "schedule.csv", "w", newline="") as fh:
            all_keys = []
            for f in sched:
                for k in f.keys():
                    if k not in all_keys:
                        all_keys.append(k)
            wr = csv.DictWriter(fh, fieldnames=all_keys)
            wr.writeheader()
            wr.writerows(sched)

        G = create_nsfnet_topology(topo)
        policy = PathPolicy(pol_kind, G, topo) if pol_kind else None
        mservers = start_mouse_servers(net, topo)
        probe_pairs = [pairs[i] for i in np.argsort(-w)[:tcfg["n_probes"]]]

        t0 = time.time()
        tele = TelemetryLogger(TEClient(exp["controller"]["rest_host"], exp["controller"]["rest_port"]),
                               topo, pmap, run, t0, exp["telemetry"]["poll_interval_s"])
        runner = FlowRunner(net, sched, topo, TEClient(exp["controller"]["rest_host"],
                            exp["controller"]["rest_port"]), policy.path if policy else None,
                            run, t0)
        tele.start()
        probes = start_probes(net, probe_pairs, policy, api, total, tcfg["probe_interval_s"])
        runner.start()
        ctl = None
        mk_api = lambda: TEClient(exp["controller"]["rest_host"], exp["controller"]["rest_port"])
        if a.method == "lstm_hysteresis":
            from controller.lstm_policy import LSTMReroute
            ctl = LSTMReroute(mk_api(), tele, runner, G, topo, exp, run, t0)
            ctl.start()
        elif a.method in ("transformer_minmax_lp", "tcn_minmax_lp"):
            from controller.lp_policy import LPReroute
            ctl = LPReroute(mk_api(), tele, runner, G, topo, exp, run, t0,
                            arch=a.method.split("_")[0])
            ctl.start()
        elif a.method.startswith("dist_lp_s"):
            from controller.dist_lp_policy import DistLPReroute
            ctl = DistLPReroute(mk_api(), tele, runner, G, topo, exp, run, t0, arch="transformer",
                                stale_s=float(a.method[len("dist_lp_s"):]))
            ctl.start()

        while time.time() - t0 < total:
            time.sleep(1)
            print("\r  t=%3.0fs / %ds  active flows=%d" % (time.time() - t0, total, len(runner.active)),
                  end="", flush=True)
        print()
        runner.finish()
        if ctl:
            ctl.stop()
        collect_probes(probes, run, t0)
        tele.stop()
        for p in mservers:
            p.kill()

        meta = json.loads((run / "meta.json").read_text())
        meta.update(finished=time.strftime("%Y%m%d-%H%M%S"), **info)
        (run / "meta.json").write_text(json.dumps(meta, indent=2))
        print(json.dumps(summarize(run), indent=2))
    finally:
        if net:
            net.stop()
        subprocess.call(["mn", "-c"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


if __name__ == "__main__":
    main()