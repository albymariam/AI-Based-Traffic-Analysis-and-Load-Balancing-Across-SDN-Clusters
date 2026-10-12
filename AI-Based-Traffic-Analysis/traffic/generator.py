"""Seeded traffic schedule + iperf3 flow runner (runs inside the Mininet driver process)."""
import csv
import json
import threading
import time

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
    nw = dict(zip(sw, rng.lognormal(0.0, 0.8, size=len(sw))))
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
    cap, ndir = topo_cfg["emulated_capacity_mbps"], 2 * len(topo_cfg["links"])
    demand_mbps = tc["load_levels"][load] * cap * ndir / avg_hops   # aggregate offered load

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


# ---------------------------------------------------------------- iperf3 helpers
def start_servers(net, topo_cfg, n_ports):
    procs = []
    for s in topo_cfg["switches"]:
        h = net.get("h%d" % s)
        for k in range(n_ports):
            procs.append(h.popen(["iperf3", "-s", "-p", str(BASE_PORT + k)]))
    time.sleep(1.5)
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


class FlowRunner(threading.Thread):
    def __init__(self, net, schedule, topo_cfg, api, path_fn, run_dir, t0, ports_per_host):
        super(FlowRunner, self).__init__(daemon=True)
        self.net, self.schedule, self.api, self.path_fn = net, schedule, api, path_fn
        self.run_dir, self.t0 = run_dir, t0
        self.free = {s: list(range(BASE_PORT, BASE_PORT + ports_per_host)) for s in topo_cfg["switches"]}
        self.active, self.rows = {}, []
        self._halt = threading.Event()

    def _cmd(self, f, port):
        c = ["iperf3", "-c", pm.host_ip(f["dst"]), "-p", str(port), "-J", "-i", "0"]
        if f["kind"] == "mice":
            return c + ["-n", str(f["size_bytes"])]
        c += ["-b", "%gM" % f["rate_mbps"], "-t", str(int(f["dur_s"]))]
        return c + (["-u"] if f["proto"] == "udp" else [])

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
        row = dict(flow_id=f["id"], kind=f["kind"], proto=f["proto"], src=f["src"], dst=f["dst"],
                   t_start=round(f["t_start"], 3), t_launch=round(now, 3),
                   rate_mbps=f.get("rate_mbps"), size_bytes=f.get("size_bytes"), dur_s=f.get("dur_s"),
                   dport=None, pin_ms=None, path="", status="skipped")
        if not self.free[f["dst"]]:
            self.rows.append(row)               # no free iperf3 server slot -> counted as blocked
            return
        port = self.free[f["dst"]].pop(0)
        row["dport"] = port
        path = self.path_fn(f["src"], f["dst"]) if self.path_fn else None
        protos = [6] if f["proto"] == "tcp" else [6, 17]
        if path:
            try:
                row["pin_ms"] = round(self.api.pin(f["src"], f["dst"], port, protos, path), 2)
            except Exception as e:
                row["status"] = "pin_failed"
                self.free[f["dst"]].append(port)
                self.rows.append(row)
                return
            row["path"] = "-".join(map(str, path))
        proc = self.net.get("h%d" % f["src"]).popen(self._cmd(f, port))
        self.active[proc] = (row, f, time.time(), port, path, protos)

    def _reap(self, force=False):
        for proc in list(self.active):
            row, f, t_l, port, path, protos = self.active[proc]
            limit = (f.get("dur_s") or 30) + 20
            killed = False
            if proc.poll() is None:
                if force or time.time() - t_l > limit:
                    proc.kill()
                    killed = True
                else:
                    continue
            out, _ = proc.communicate()
            res = parse_iperf(out.decode(errors="ignore"), f["proto"])
            row["wall_s"] = round(time.time() - t_l, 3)
            row.update(res)
            row["status"] = "killed" if killed else ("ok" if res.get("ok") else "failed")
            if path:
                try:
                    self.api.unpin(f["src"], f["dst"], port, protos)
                except Exception:
                    pass
            self.free[f["dst"]].append(port)
            self.rows.append(row)
            del self.active[proc]

    def finish(self, grace_s=15):
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