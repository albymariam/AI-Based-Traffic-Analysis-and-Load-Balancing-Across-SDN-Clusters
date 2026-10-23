"""M5: replay the TE controller's flow installs on a real ONOS (single node or cluster) and log the control-plane cost.
  .venv/bin/python experiments/onos_m5_replay.py --probe --nodes localhost:8182          (one pin/unpin, prints the raw REST answers)
  .venv/bin/python experiments/onos_m5_replay.py --events data/onos_m5/events_s1.csv --nodes localhost:8182,localhost:8183,localhost:8184 --out results/onos_m5/B
Events (scripts/make_onos_events.py): t,op,flow_id,src,dst,dport,protos,path     op = pin | reroute | unpin
Same rules as controller/ryu_te_app.py: priority 100, match ipv4 src/dst + proto (+ L4 dst port), output to the next hop.
Needs the Abilene Mininet network attached to ONOS (experiments/onos_m5_net.py) so that the 12 devices exist."""
import argparse
import csv
import json
import os
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

os.environ.setdefault("T11_TOPOLOGY", "abilene12")
from topology import portmap as pm                                    # noqa: E402

PRIO = 100
APP = "org.onosproject.rest"
AUTH = ("onos", "rocks")
TL = threading.local()
N_REQ = [0]
LOCK = threading.Lock()


def sess():
    if not hasattr(TL, "s"):
        TL.s = requests.Session()
        TL.s.auth = AUTH
    return TL.s


class Onos:
    def __init__(self, node):
        self.base = "http://%s/onos/v1" % node
        self.pmap = pm.port_map(pm.load_topology_cfg())

    def call(self, method, path, **kw):
        with LOCK:
            N_REQ[0] += 1
        return sess().request(method, self.base + path, timeout=30, **kw)

    def rule(self, s, src, dst, proto, dport, out):
        crit = [{"type": "ETH_TYPE", "ethType": "0x0800"},
                {"type": "IPV4_SRC", "ip": pm.host_ip(src) + "/32"},
                {"type": "IPV4_DST", "ip": pm.host_ip(dst) + "/32"},
                {"type": "IP_PROTO", "protocol": proto}]
        if dport is not None and proto == 6:
            crit.append({"type": "TCP_DST", "tcpPort": dport})
        elif dport is not None and proto == 17:
            crit.append({"type": "UDP_DST", "udpPort": dport})
        return {"priority": PRIO, "timeout": 0, "isPermanent": True, "deviceId": "of:" + pm.dpid(s),
                "treatment": {"instructions": [{"type": "OUTPUT", "port": str(out)}]},
                "selector": {"criteria": crit}}

    def out_port(self, path, i):
        return pm.HOST_PORT if i == len(path) - 1 else self.pmap[path[i]][path[i + 1]]

    def add(self, rules):
        """rules: [(switch, proto, out, rule_dict)] -> {(switch, proto): (deviceId, flowId, out)}; one batch POST."""
        r = self.call("POST", "/flows?appId=%s" % APP, json={"flows": [x[3] for x in rules]})
        if r.status_code not in (200, 201):
            raise RuntimeError("POST flows -> %s %s" % (r.status_code, r.text[:200]))
        ids = [(f["deviceId"], f["flowId"]) for f in r.json().get("flows", [])]
        if len(ids) != len(rules):
            raise RuntimeError("POST flows returned %d ids for %d rules: %s" % (len(ids), len(rules), r.text[:200]))
        return {(x[0], x[1]): (d, f, x[2]) for x, (d, f) in zip(rules, ids)}

    def delete(self, dev, fid):
        r = self.call("DELETE", "/flows/%s/%s" % (dev, fid))
        if r.status_code not in (200, 204):
            raise RuntimeError("DELETE %s %s -> %s" % (dev, fid, r.status_code))

    def added(self, dev, fid):
        r = self.call("GET", "/flows/%s/%s" % (dev, fid))
        return r.status_code == 200 and all(f.get("state") == "ADDED" for f in r.json().get("flows", []))


class Replayer:
    def __init__(self, onos, protos_sample):
        self.o, self.rules, self.rows, self.sample = onos, {}, [], protos_sample
        self.n_pin = 0

    def pin(self, e, sample):
        path, protos = e["path"], e["protos"]
        want = [(s, p, self.o.out_port(path, i), self.o.rule(s, e["src"], e["dst"], p, e["dport"], self.o.out_port(path, i)))
                for i, s in enumerate(path) for p in protos]
        t = time.perf_counter()
        got = self.o.add(want)
        rest_ms = (time.perf_counter() - t) * 1000
        self.rules[e["flow_id"]] = got
        added_ms = None
        if sample:
            while not all(self.o.added(d, f) for d, f, _ in got.values()):
                if time.perf_counter() - t > 30:
                    break
                time.sleep(0.01)
            else:
                added_ms = (time.perf_counter() - t) * 1000
        return len(want), rest_ms, added_ms

    def unpin(self, e):
        got = self.rules.pop(e["flow_id"], {})
        t = time.perf_counter()
        for d, f, _ in got.values():
            self.o.delete(d, f)
        return len(got), (time.perf_counter() - t) * 1000, None

    def reroute(self, e):
        old, path, protos = self.rules.get(e["flow_id"], {}), e["path"], e["protos"]
        new_out = {(s, p): self.o.out_port(path, i) for i, s in enumerate(path) for p in protos}
        t = time.perf_counter()
        n = 0
        keep = {}
        for k, (d, f, out) in old.items():
            if new_out.get(k) == out:
                keep[k] = (d, f, out)
            else:
                self.o.delete(d, f)
                n += 1
        add = [(s, p, out, self.o.rule(s, e["src"], e["dst"], p, e["dport"], out))
               for (s, p), out in new_out.items() if (s, p) not in keep]
        if add:
            keep.update(self.o.add(add))
            n += len(add)
        self.rules[e["flow_id"]] = keep
        return n, (time.perf_counter() - t) * 1000, None

    def run_flow(self, evts, t0, speed):
        for e in evts:
            due = t0 + e["t"] / speed
            time.sleep(max(0.0, due - time.time()))
            row = dict(op=e["op"], flow_id=e["flow_id"], t_plan=round(e["t"] / speed, 3),
                       lag_s=round(time.time() - due, 3), n_rules=0, rest_ms=None, added_ms=None, ok=1, err="")
            try:
                if e["op"] == "pin":
                    with LOCK:
                        self.n_pin += 1
                        sample = self.n_pin % self.sample == 0
                    n, rest, added = self.pin(e, sample)
                elif e["op"] == "unpin":
                    n, rest, added = self.unpin(e)
                else:
                    n, rest, added = self.reroute(e)
                row.update(n_rules=n, rest_ms=round(rest, 2), added_ms=None if added is None else round(added, 2))
            except Exception as ex:
                row.update(ok=0, err=str(ex)[:120])
            with LOCK:
                self.rows.append(row)


def load_events(path):
    by_flow = {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            by_flow.setdefault(r["flow_id"], []).append(dict(
                t=float(r["t"]), op=r["op"], flow_id=r["flow_id"], src=int(r["src"]), dst=int(r["dst"]),
                dport=int(r["dport"]), protos=[int(x) for x in r["protos"].split("-")],
                path=[int(x) for x in r["path"].split("-")] if r["path"] else []))
    flows = [sorted(v, key=lambda e: e["t"]) for v in by_flow.values()]
    return sorted(flows, key=lambda v: v[0]["t"])


def q(x, p):
    x = sorted(x)
    return x[min(len(x) - 1, int(round(p * (len(x) - 1))))] if x else None


def summarize(rows, dur_s):
    out = {"duration_s": round(dur_s, 1), "n_ops": len(rows), "ok_pct": round(100.0 * sum(r["ok"] for r in rows) / max(1, len(rows)), 2),
           "rest_requests": N_REQ[0], "rest_requests_per_s": round(N_REQ[0] / dur_s, 2) if dur_s else None,
           "replay_lag_p95_s": q([r["lag_s"] for r in rows], 0.95)}
    for op in ("pin", "reroute", "unpin"):
        x = [r["rest_ms"] for r in rows if r["op"] == op and r["ok"]]
        out[op] = dict(n=len(x), rest_ms_median=q(x, 0.5), rest_ms_p95=q(x, 0.95))
    ad = [r["added_ms"] for r in rows if r["added_ms"] is not None]
    out["pin_added_ms"] = dict(n=len(ad), median=q(ad, 0.5), p95=q(ad, 0.95))
    return out


def probe(onos):
    print("-- one pin (switch path 1-2 or the first link in the topology), raw answers --")
    cfg = pm.load_topology_cfg()
    a, b = cfg["links"][0]
    rules = [(a, 17, onos.out_port([a, b], 0), onos.rule(a, a, b, 17, 5999, onos.out_port([a, b], 0)))]
    r = onos.call("POST", "/flows?appId=%s" % APP, json={"flows": [x[3] for x in rules]})
    print("POST", r.status_code, r.text[:300], dict(r.headers).get("Location"))
    ids = r.json().get("flows", [])
    for f in ids:
        t = time.perf_counter()
        while not onos.added(f["deviceId"], f["flowId"]) and time.perf_counter() - t < 10:
            time.sleep(0.01)
        print("GET state", onos.call("GET", "/flows/%s/%s" % (f["deviceId"], f["flowId"])).text[:300])
        print("ADDED after %.0f ms" % ((time.perf_counter() - t) * 1000))
        onos.delete(f["deviceId"], f["flowId"])
        print("DELETE ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", required=True, help="comma list; the first node receives all REST calls (the writer)")
    ap.add_argument("--events", default="data/onos_m5/events_s1.csv")
    ap.add_argument("--out", default=None)
    ap.add_argument("--speed", type=float, default=1.0, help="replay speed (1 = the original timing)")
    ap.add_argument("--workers", type=int, default=600)
    ap.add_argument("--sample-added", type=int, default=10, help="time until state ADDED for every Nth pin")
    ap.add_argument("--probe", action="store_true")
    a = ap.parse_args()
    onos = Onos(a.nodes.split(",")[0])
    if a.probe:
        return probe(onos)
    flows = load_events(a.events)
    t_first = min(f[0]["t"] for f in flows)
    for f in flows:
        for e in f:
            e["t"] -= t_first
    rp = Replayer(onos, a.sample_added)
    print("replaying %d flows / %d events on %s (speed %gx) ..." % (len(flows), sum(map(len, flows)), a.nodes.split(",")[0], a.speed))
    t0 = time.time() + 1.0
    with ThreadPoolExecutor(a.workers) as ex:
        for f in flows:
            ex.submit(rp.run_flow, f, t0, a.speed)
    dur = time.time() - t0
    s = summarize(rp.rows, dur)
    print(json.dumps(s, indent=1))
    if a.out:
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "ops.csv", "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rp.rows[0]))
            wr.writeheader()
            wr.writerows(sorted(rp.rows, key=lambda r: r["t_plan"]))
        (out / "replay_summary.json").write_text(json.dumps(s, indent=1))
    left = sum(len(v) for v in rp.rules.values())
    if left:
        print("note: %d rules from still-pinned flows remain on the devices (removed when the cluster is torn down)" % left)


if __name__ == "__main__":
    main()
