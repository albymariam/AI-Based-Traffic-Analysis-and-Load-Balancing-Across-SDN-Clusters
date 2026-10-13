"""Seeded traffic schedule + iperf3 flow runner (runs inside the Mininet driver process)."""
import csv
import json
import threading
import time
import subprocess
from pathlib import Path

import numpy as np

from topology import portmap as pm

BASE_PORT = 5201


# ---------------------------------------------------------------- schedule
def rate_factor(profile, t, total):
    shape = profile["shape"]
    if shape == "constant":
        return 1.0
    if shape == "on_off":                    # mean-preserving on/off modulation
        period, duty, mult = profile["period_s"], profile["duty"], profile["burst_multiplier"]
        low = max(0.0, (1 - duty * mult) / (1 - duty))
        return mult if (t % period) < duty * period else low
    if shape == "ramp":
        a, b = profile["start_factor"], profile["end_factor"]
        return a + (b - a) * min(1.0, t / total)
    raise ValueError(shape)


def build_pairs(topo_cfg, traffic_cfg, rng):
    sw, pc = topo_cfg["switches"], traffic_cfg["pairs"]
    allp = [(s, d) for s in sw for d in sw if s != d]
    if pc["mode"] == "all_to_all":
        return allp, np.ones(len(allp)) / len(allp)
    nw = dict(zip(sw, rng.lognormal(0.0, pc.get("gravity_sigma", 0.5), size=len(sw))))
    if pc["mode"] == "hotspot":
        for h in pc["hotspot_nodes"]:
            nw[h] *= 5.0
    pw = np.array([nw[s] * nw[d] for s, d in allp])
    idx = rng.choice(len(allp), size=min(pc["num_pairs"], len(allp)), replace=False, p=pw / pw.sum())
    w = pw[idx]
    return [allp[i] for i in idx], w / w.sum()


def _make_flow(rng, t, pairs, w, tc, total, kind=None, dur=None):
    mix = tc["flow_mix"]
    s, d = pairs[rng.choice(len(pairs), p=w)]
    if kind is None:
        kind = "mice" if rng.random() < mix["mice"]["share"] else "elephant"
    if kind == "mice":
        lo, hi = mix["mice"]["size_kb"]
        return dict(t_start=t, src=s, dst=d, kind="mice", proto="tcp",
                    size_bytes=int(rng.uniform(lo, hi) * 1000))
    rlo, rhi = mix["elephant"]["rate_mbps"]
    dlo, dhi = mix["elephant"]["duration_s"]
    dur = dur if dur is not None else rng.uniform(dlo, dhi)
    dur = min(dur, total - t - 1.0)
    if dur < 3.0:
        return None
    proto = "tcp" if rng.random() < tc["tcp_fraction"] else "udp"
    return dict(t_start=t, src=s, dst=d, kind="elephant", proto=proto,
                rate_mbps=round(float(rng.uniform(rlo, rhi)), 3), dur_s=round(float(dur), 1))


def build_schedule(topo_cfg, tc, load, profile_name, total_s, seed):
    rng = np.random.default_rng(seed)
    pairs, w = build_pairs(topo_cfg, tc, rng)
    dist = pm.hop_dist(topo_cfg)
    avg_hops = sum(wi * dist[s][d] for (s, d), wi in zip(pairs, w))
    cap = topo_cfg["emulated_capacity_mbps"]
    # load level = target MLU under static shortest-path routing
    nh = pm.next_hops(topo_cfg)
    share = {}
    for (s, d), wi in zip(pairs, w):
        p = pm.static_path(nh, s, d)
        for a, b in zip(p, p[1:]):
            share[(a, b)] = share.get((a, b), 0.0) + wi
    demand_mbps = tc["load_levels"][load] * cap / max(share.values())   # aggregate offered Mbps
    demand_mbps *= tc.get("load_calibration", 1.0)   # tune once so static MLU hits the target
    mix = tc["flow_mix"]
    mean_mice = np.mean(mix["mice"]["size_kb"]) * 0.008                                   # Mbit
    mean_dur = np.mean(mix["elephant"]["duration_s"])
    mean_eleph = np.mean(mix["elephant"]["rate_mbps"]) * mean_dur                         # Mbit
    mean_flow = mix["mice"]["share"] * mean_mice + mix["elephant"]["share"] * mean_eleph
    lam0 = demand_mbps / mean_flow                                                        # flows/s

    prof = tc["profiles"][profile_name]
    fmax = max(rate_factor(prof, t, total_s) for t in np.linspace(0, total_s, 2001))
    flows, t = [], 0.0
    while True:                                   # non-homogeneous Poisson via thinning
        t += rng.exponential(1.0 / (lam0 * fmax))
        if t >= total_s:
            break
        if rng.random() <= rate_factor(prof, t, total_s) / fmax:
            f = _make_flow(rng, t, pairs, w, tc, total_s)
            if f:
                flows.append(f)
    # pre-fill steady-state elephants so the load is reached without a long ramp-up
    n0 = rng.poisson(lam0 * rate_factor(prof, 0, total_s) * mix["elephant"]["share"] * mean_dur)
    for _ in range(n0):
        f = _make_flow(rng, float(rng.uniform(0, 2)), pairs, w, tc, total_s, kind="elephant",
                       dur=float(rng.uniform(5, mix["elephant"]["duration_s"][1])))
        if f:
            flows.append(f)
    flows.sort(key=lambda f: f["t_start"])
    for i, f in enumerate(flows):
        f["id"] = i
    info = dict(avg_hops=avg_hops, demand_mbps=demand_mbps, lambda0=lam0, n_flows=len(flows))
    return flows, pairs, w, info


# ---------------------------------------------------------------- flow execution
MOUSE_PORT = 6000
MOUSE_PY = str(Path(__file__).resolve().parent / "mouse.py")
CONNECT_DELAY = 0.5


def start_mouse_servers(net, topo_cfg):
    procs = [net.get("h%d" % s).popen(["python3", MOUSE_PY, "server", str(MOUSE_PORT)],
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
             for s in topo_cfg["switches"]]
    time.sleep(1.0)
    return procs


def parse_iperf(out, proto):
    try:
        j = json.loads(out)
    except ValueError:
        return {"ok": False, "error": "badjson"}
    if "error" in j:
        return {"ok": False, "error": str(j["error"])[:60]}
    e = j["end"]
    if proto == "tcp":
        r = e.get("sum_received") or {}
        return {"ok": True, "bps": r.get("bits_per_second"), "bytes": r.get("bytes"), "xfer_s": r.get("seconds")}
    s = e["sum"]
    return {"ok": True, "bps": s["bits_per_second"], "bytes": s["bytes"], "xfer_s": s["seconds"],
            "jitter_ms": s["jitter_ms"], "lost_percent": s["lost_percent"],
            "lost_packets": s["lost_packets"], "packets": s["packets"]}


def parse_mouse(out):
    try:
        j = json.loads(out.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {"ok": False, "error": "nooutput"}
    if not j.get("ok"):
        return {"ok": False, "error": j.get("error", "fail")}
    return {"ok": True, "bytes": j["bytes"], "xfer_s": j["fct_s"]}


class FlowRunner(threading.Thread):
    """Mice: python client/server, follow default routes (never pinned).
    Elephants: iperf3, pinned to a path by the policy (this is the TE-controlled traffic)."""

    def __init__(self, net, schedule, topo_cfg, api, path_fn, run_dir, t0):
        super(FlowRunner, self).__init__(daemon=True)
        self.net, self.schedule, self.api, self.path_fn = net, schedule, api, path_fn
        self.run_dir, self.t0 = run_dir, t0
        self.lock = threading.Lock()
        self.active, self.rows = {}, []
        self._halt = threading.Event()

    def run(self):
        i, sched = 0, self.schedule
        while not self._halt.is_set():
            now = time.time() - self.t0
            while i < len(sched) and sched[i]["t_start"] <= now:
                self._launch(sched[i], now)
                i += 1
            self._reap()
            time.sleep(0.02)

    def _launch(self, f, now):
        port = BASE_PORT + f["id"]
        mouse = f["kind"] == "mice"
        row = dict(flow_id=f["id"], kind=f["kind"], proto=f["proto"], src=f["src"], dst=f["dst"],
                   t_start=round(f["t_start"], 3), t_launch=round(now, 3), rate_mbps=f.get("rate_mbps"),
                   size_bytes=f.get("size_bytes"), dur_s=f.get("dur_s"),
                   dport=MOUSE_PORT if mouse else port, pin_ms=None, path="", n_reroutes=0, status="pending")
        protos = [6] if f["proto"] == "tcp" else [6, 17]
        path, srv = None, None
        src_h = self.net.get("h%d" % f["src"])
        if mouse:
            cli = src_h.popen(["python3", MOUSE_PY, "client", pm.host_ip(f["dst"]),
                               str(MOUSE_PORT), str(f["size_bytes"])])
        else:
            path = self.path_fn(f["src"], f["dst"]) if self.path_fn else None
            if path:
                try:
                    row["pin_ms"] = round(self.api.pin(f["src"], f["dst"], port, protos, path), 2)
                except Exception:
                    row["status"] = "pin_failed"
                    self.rows.append(row)
                    return
                row["path"] = "-".join(map(str, path))
            srv = self.net.get("h%d" % f["dst"]).popen(
                ["iperf3", "-s", "-1", "-p", str(port)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            cmd = ["iperf3", "-c", pm.host_ip(f["dst"]), "-p", str(port), "-J", "-i", "0",
                   "-b", "%gM" % f["rate_mbps"], "-t", str(int(f["dur_s"]))]
            if f["proto"] == "udp":
                cmd.append("-u")
            cli = src_h.popen(["sh", "-c", 'sleep %.1f; exec timeout -k 2 %d "$@"'
                               % (CONNECT_DELAY, int(f["dur_s"]) + 15), "sh"] + cmd)
        with self.lock:
            self.active[cli] = dict(row=row, f=f, t=time.time(), port=port, path=path, protos=protos, srv=srv)

    def _reap(self, force=False):
        with self.lock:
            items = list(self.active.items())
        for proc, e in items:
            f, row = e["f"], e["row"]
            killed = False
            if proc.poll() is None:
                if force or time.time() - e["t"] > (f.get("dur_s") or 0) + 30:
                    proc.kill()
                    proc.wait()
                    killed = True
                else:
                    continue
            out = b"" if killed else proc.communicate()[0]     # never communicate() on a killed proc
            if e["srv"] is not None:
                if e["srv"].poll() is None:
                    e["srv"].kill()
                e["srv"].wait()
            text = out.decode(errors="ignore")
            res = parse_mouse(text) if f["kind"] == "mice" else parse_iperf(text, f["proto"])
            row.update(res)
            row["wall_s"] = round(time.time() - e["t"], 3)
            row["status"] = "killed" if killed else ("ok" if res.get("ok") else "failed")
            if e["path"]:
                try:
                    self.api.unpin(f["src"], f["dst"], e["port"], e["protos"])
                except Exception:
                    pass
            self.rows.append(row)
            with self.lock:
                del self.active[proc]

    # --- used by the online TE policy ---
    def elephants(self):
        with self.lock:
            return [dict(id=e["f"]["id"], src=e["f"]["src"], dst=e["f"]["dst"], port=e["port"],
                         protos=e["protos"], path=list(e["path"]), rate=e["f"]["rate_mbps"])
                    for e in self.active.values() if e["f"]["kind"] == "elephant" and e["path"]]

    def set_path(self, flow_id, path):
        with self.lock:
            for e in self.active.values():
                if e["f"]["id"] == flow_id:
                    e["path"] = list(path)
                    e["row"]["n_reroutes"] += 1
                    e["row"]["path"] += ">" + "-".join(map(str, path))

    def finish(self, grace_s=10):
        self._halt.set()
        self.join(timeout=2)
        t = time.time()
        while self.active and time.time() - t < grace_s:
            self._reap()
            time.sleep(0.2)
        self._reap(force=True)
        keys = []
        for r in self.rows:
            keys += [k for k in r if k not in keys]
        with open(self.run_dir / "flows.csv", "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=keys)
            wr.writeheader()
            wr.writerows(sorted(self.rows, key=lambda r: r["flow_id"]))