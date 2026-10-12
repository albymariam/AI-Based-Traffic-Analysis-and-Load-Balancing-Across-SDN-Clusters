"""Per-second link telemetry from OpenFlow port stats (via Ryu REST) + tc queue stats.
Writes <run>/telemetry.csv (one row per directed link per second) and <run>/ctrl.csv."""
import csv
import re
import subprocess
import threading
import time

from topology import portmap as pm

FIELDS = ["timestep", "t", "link_id", "src", "dst", "byte_throughput", "packet_rate", "active_flows",
          "utilization", "utilization_raw", "queue_length", "delay_ms", "packet_loss"]


def read_tc():
    """{iface: {dropped, backlog_b, backlog_p}} using the leaf qdisc (last block per device)."""
    out = subprocess.run(["tc", "-s", "qdisc", "show"], capture_output=True, text=True).stdout
    res = {}
    for chunk in re.split(r"(?m)^(?=qdisc )", out):
        m = re.search(r"dev (\S+)", chunk)
        if not m:
            continue
        d = re.search(r"dropped (\d+)", chunk)
        b = re.search(r"backlog (\d+)b (\d+)p", chunk)
        res[m.group(1)] = dict(dropped=int(d.group(1)) if d else 0,
                               backlog_b=int(b.group(1)) if b else 0,
                               backlog_p=int(b.group(2)) if b else 0)
    return res


class TelemetryLogger(threading.Thread):
    def __init__(self, api, topo_cfg, pmap, run_dir, t0, interval=1.0):
        super(TelemetryLogger, self).__init__(daemon=True)
        self.api, self.cfg, self.pmap, self.t0, self.interval = api, topo_cfg, pmap, t0, interval
        self.links = [(u, v) for u, v in topo_cfg["links"]] + [(v, u) for u, v in topo_cfg["links"]]
        self.cap_bps = topo_cfg["emulated_capacity_mbps"] * 1e6
        self.prop_ms = topo_cfg["link_delay_ms"]
        self.f1 = open(run_dir / "telemetry.csv", "w", newline="")
        self.w1 = csv.DictWriter(self.f1, fieldnames=FIELDS)
        self.w1.writeheader()
        self.f2 = open(run_dir / "ctrl.csv", "w", newline="")
        self.w2 = csv.writer(self.f2)
        self.w2.writerow(["t", "flowmods_total", "pinned_flows", "mode"])
        self._halt = threading.Event()
        self.prev = {}

    def run(self):
        nxt, k = time.time(), 0
        while not self._halt.is_set():
            try:
                self._tick(k)
            except Exception as e:
                print("[telemetry] tick %d failed: %s" % (k, e))
            k += 1
            nxt += self.interval
            self._halt.wait(max(0.0, nxt - time.time()))

    def _tick(self, k):
        snap, tc = self.api.ports(), read_tc()
        st, lf = self.api.stats(), self.api.linkflows()
        t = time.time() - self.t0
        self.w2.writerow([round(t, 3), st["flowmods"], st["pinned"], st["mode"]])
        for u, v in self.links:
            sw = snap["switches"].get(str(u))
            port = self.pmap[u][v]
            cur = sw and sw["ports"].get(str(port))
            if not cur:
                continue
            q = tc.get(pm.iface_name(self.pmap, u, v), {})
            now = (sw["ts"], cur["tx_bytes"], cur["tx_packets"], q.get("dropped", 0))
            pv, self.prev[(u, v)] = self.prev.get((u, v)), now
            if pv is None or now[0] <= pv[0]:
                continue
            dt = now[0] - pv[0]
            bps = (now[1] - pv[1]) / dt                       # bytes/s
            raw = bps * 8 / self.cap_bps
            qb = q.get("backlog_b", 0)
            self.w1.writerow(dict(
                timestep=k, t=round(t, 3), link_id="L_%d_%d" % (u, v), src=u, dst=v,
                byte_throughput=round(bps, 1), packet_rate=round((now[2] - pv[2]) / dt, 2),
                active_flows=lf.get("%d-%d" % (u, v), 0),
                utilization=round(min(raw, 1.0), 5), utilization_raw=round(raw, 5),
                queue_length=q.get("backlog_p", 0),
                delay_ms=round(self.prop_ms + qb * 8 / self.cap_bps * 1000, 4),
                packet_loss=now[3] - pv[3]))
        self.f1.flush()
        self.f2.flush()

    def stop(self):
        self._halt.set()
        self.join(timeout=3)
        self.f1.close()
        self.f2.close()